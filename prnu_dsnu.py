'''
DSNU / PRNU correction of RGB TIFF images (linear raw 14-bit DN):
    1. DSNU correction : DN_dsnu = max(0, max(0, DN - Dark) - Offset), cropped to the usable window
    2. PRNU correction : Radiance = DN_dsnu / Gain
                         DN_out   = rint(Radiance / Saturation_Radiance x 16383)
Output: <OUTPUT_FOLDER>/<SATELLITE_NAME>/<SATELLITE_NAME>_<name>_DSNU.tif      (float32, after step 1)
        <OUTPUT_FOLDER>/<SATELLITE_NAME>/<SATELLITE_NAME>_<name>_RADIANCE.tif  (float32, after step 2)
        <OUTPUT_FOLDER>/<SATELLITE_NAME>/<SATELLITE_NAME>_<name>_DN.tif        (uint16, radiance -> 14-bit DN)

Usage:
    python3 prnu_dsnu.py                    (PAYLOAD, INPUT_FOLDER, OUTPUT_FOLDER below)
    python3 prnu_dsnu.py --payload BRAVO
'''

import os
import json
import time
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import tifffile

import config


PAYLOAD = "BRAVO"
INPUT_FOLDER = "/home/tarun/Documents/oct_8/bravo/@2000_averaged/demosaiced_rgb_cv2_ea"
OUTPUT_FOLDER = "/home/tarun/Documents/oct_8/bravo"

logger = logging.getLogger(__name__)

TIFF_EXT = (".tif", ".tiff")


class DSNU_PRNU_Correction:
    '''
    Applies the DSNU and PRNU correction to the RGB images of one payload.
    '''

    BANDS = config.BANDS

    def __init__(self, satellite_name, output_dir, dark_frame_path=None):
        self.sat_config = config.get_config_by_satellite_name(satellite_name)
        self.satellite_name = self.sat_config["satellite_name"]
        self.dark_frame_path = dark_frame_path or self.sat_config["dark_frame_path"]
        self.output_dir = os.path.join(output_dir, self.satellite_name)

        self.saturation_radiance = {}
        for band in self.BANDS:
            sat_radiance = self.sat_config["saturation_radiance"][band]
            if sat_radiance is None or sat_radiance <= 0:
                raise ValueError(f"saturation_radiance for band '{band}' of "
                                 f"{self.satellite_name} is not set in config.py")
            self.saturation_radiance[band] = float(sat_radiance)

        self.sensor_shape = (self.sat_config["sensor_height"], self.sat_config["sensor_width"])
        self.usable_window = self.sat_config["usable_window"]  # (xmin, xmax, ymin, ymax) or None
        if self.usable_window is not None:
            xmin, xmax, ymin, ymax = self.usable_window
            self.map_shape = (ymax - ymin, xmax - xmin)
        else:
            self.map_shape = self.sensor_shape

        for band in self.BANDS:
            for kind in ("gain_path", "offset_path"):
                path = self.sat_config[kind][band]
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{kind} missing for band {band}: {path}")
                shape = self._tiff_shape(path)
                if shape != self.map_shape:
                    raise ValueError(f"{kind} for band {band} has shape {shape}, expected "
                                     f"the usable-window shape {self.map_shape}")

        if not os.path.exists(self.dark_frame_path):
            raise FileNotFoundError(f"Dark frame missing: {self.dark_frame_path}")
        self.dark_shape = self._tiff_shape(self.dark_frame_path)
        if len(self.dark_shape) != 3 or self.dark_shape[2] != 3:
            raise ValueError(f"Dark frame must be an RGB image (H, W, 3), got {self.dark_shape}")
        self.window = self._image_window(self.dark_shape)

        self.dn_scale = {b: config.OUTPUT_DN_MAX / self.saturation_radiance[b] for b in self.BANDS}
        self.dark = None       # dark frame, cropped to the usable window
        self.dsnu_map = None   # Offset, per band (NaN -> 0)
        self.prnu_map = None   # 1 / Gain, per band (0 where Gain <= 0 or NaN)
        self._maps_dir = os.path.join(self.output_dir, "_maps")
        self._pool = ThreadPoolExecutor(config.WORKERS)

        logger.info("%s: exposure %s us, maps %d x %d, usable window %s, dark frame %s",
                    self.satellite_name, self.sat_config["exposure_us"], self.map_shape[1],
                    self.map_shape[0], self.usable_window, self.dark_frame_path)
        logger.info("Saturation radiance (%s): %s", config.RADIANCE_UNIT, self.saturation_radiance)

    # ------------------------------------------------------------------
    @staticmethod
    def _tiff_shape(path):
        with tifffile.TiffFile(path) as tif:
            return tuple(tif.series[0].shape)

    # ------------------------------------------------------------------
    def _image_window(self, shape):
        '''
        (row0, row1, col0, col1) of an image that lines up with the gain/offset maps.
        '''
        h, w = shape[:2]
        if (h, w) == self.map_shape:
            return (0, h, 0, w)
        if (h, w) == self.sensor_shape:
            if self.usable_window is None:
                return (0, h, 0, w)
            xmin, xmax, ymin, ymax = self.usable_window
            return (ymin, ymax, xmin, xmax)
        raise ValueError(f"Image shape {shape} is neither the sensor frame {self.sensor_shape} "
                         f"nor the usable window {self.map_shape} of {self.satellite_name}")

    # ------------------------------------------------------------------
    @staticmethod
    def _open(path):
        try:
            return tifffile.memmap(path, mode="r")
        except ValueError:  # compressed TIFF
            return tifffile.imread(path)

    # ------------------------------------------------------------------
    @staticmethod
    def _mem_available():
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return 0

    # ------------------------------------------------------------------
    @staticmethod
    def _decode_map(path, out, transform):
        '''
        Decodes a single-band map tile by tile on all cores into `out`,
        applying transform(tile, row, col) to each tile.
        '''
        with tifffile.TiffFile(path) as tif:
            page = tif.pages.first
            h, w = out.shape

            def place(decoded):
                segment, (_, _, r, c, _), shape = decoded
                if segment is None:
                    tile = np.full((min(shape[1], h - r), min(shape[2], w - c)), np.nan, np.float32)
                else:
                    tile = segment[0, :shape[1], :shape[2], 0][:h - r, :w - c]
                out[r:r + tile.shape[0], c:c + tile.shape[1]] = transform(tile, r, c)

            for _ in page.segments(func=place, maxworkers=os.cpu_count(), sort=True,
                                   buffersize=64 << 20):
                pass

    # ------------------------------------------------------------------
    def load_calibration_maps(self):
        '''
        Reads the dark frame (cropped to the usable window) and the gain/offset maps once,
        into one float32 plane per band:
            dsnu_map = Offset
            prnu_map = 1 / Gain
        Kept in RAM, or memory-mapped from <output>/_maps when RAM is low.
        '''
        r0, r1, c0, c1 = self.window
        dark = self._open(self.dark_frame_path)
        # memory-mapped (uncompressed) darks stay on disk; compressed ones keep only the window
        self.dark = dark[r0:r1, c0:c1] if isinstance(dark, np.memmap) \
            else np.ascontiguousarray(dark[r0:r1, c0:c1])
        del dark

        h, w = self.map_shape
        need = 2 * len(self.BANDS) * h * w * 4
        in_ram = self._mem_available() - need > config.RAM_MARGIN_BYTES
        logger.info("Loading calibration maps: %.1f GB %s", need / 1e9,
                    "in RAM" if in_ram else f"memory-mapped from {self._maps_dir}")

        def plane(name):
            if in_ram:
                return np.empty((h, w), np.float32)
            os.makedirs(self._maps_dir, exist_ok=True)
            return np.lib.format.open_memmap(os.path.join(self._maps_dir, f"{name}.npy"),
                                             "w+", np.float32, (h, w))

        self.dsnu_map, self.prnu_map = {}, {}
        for band in self.BANDS:
            t0, n_bad = time.time(), []

            def dsnu_tile(tile, r, c):
                v = tile.astype(np.float32)
                bad = ~np.isfinite(v)
                if bad.any():
                    v[bad] = 0
                    n_bad.append(int(bad.sum()))
                return v

            self.dsnu_map[band] = plane(f"{band}_dsnu")
            self._decode_map(self.sat_config["offset_path"][band], self.dsnu_map[band], dsnu_tile)
            if n_bad:
                logger.warning("Band %s: %d pixels have a NaN offset - set to 0", band, sum(n_bad))
            n_bad.clear()

            def prnu_tile(tile, r, c):
                v = tile.astype(np.float32)
                bad = ~np.isfinite(v) | (v <= 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    np.divide(1.0, v, out=v)
                if bad.any():
                    v[bad] = 0
                    n_bad.append(int(bad.sum()))
                return v

            self.prnu_map[band] = plane(f"{band}_prnu")
            self._decode_map(self.sat_config["gain_path"][band], self.prnu_map[band], prnu_tile)
            if n_bad:
                logger.warning("Band %s: %d pixels have a gain <= 0 or NaN - output 0",
                               band, sum(n_bad))
            logger.info("Band %s maps loaded in %.1fs", band, time.time() - t0)

    # ------------------------------------------------------------------
    def apply_dsnu_correction(self, raw, band, rows, out):
        '''
        Step 1: DN_dsnu = max(0, max(0, DN - Dark) - Offset)
        Same as darkframe_sub.py (clip at 0) followed by the offset of tk.dn_to_radiance.
        '''
        np.subtract(raw, self.dark[rows, :, config.BAND_IDS[band]], out=out, dtype=np.float32)
        np.maximum(out, 0, out=out)
        out -= self.dsnu_map[band][rows]
        np.maximum(out, 0, out=out)
        return out

    # ------------------------------------------------------------------
    def apply_prnu_correction(self, dn_dsnu, band, rows):
        '''
        Step 2: Radiance = DN_dsnu / Gain, in place.
        '''
        dn_dsnu *= self.prnu_map[band][rows]
        return dn_dsnu

    # ------------------------------------------------------------------
    def radiance_to_dn(self, radiance, band):
        '''
        DN_out = rint(Radiance / Saturation_Radiance x OUTPUT_DN_MAX), clipped, in place.
        Returns the number of pixels clipped at OUTPUT_DN_MAX.
        '''
        radiance *= self.dn_scale[band]
        clipped = int(np.count_nonzero(radiance > config.OUTPUT_DN_MAX))
        np.minimum(radiance, config.OUTPUT_DN_MAX, out=radiance)
        np.rint(radiance, out=radiance)
        return clipped

    # ------------------------------------------------------------------
    def process_image(self, image_path):
        '''
        Applies both steps to one image, block by block on config.WORKERS threads.
        '''
        if self.dsnu_map is None:
            self.load_calibration_maps()

        name = os.path.splitext(os.path.basename(image_path))[0]
        image = self._open(image_path)
        if image.shape != self.dark_shape:
            raise ValueError(f"{name}: image shape {image.shape} != dark frame shape "
                             f"{self.dark_shape}")
        r0, r1, c0, c1 = self.window
        h, w = r1 - r0, c1 - c0

        os.makedirs(self.output_dir, exist_ok=True)
        info = {
            "satellite_name": self.satellite_name,
            "exposure_us": self.sat_config["exposure_us"],
            "source": os.path.basename(image_path),
            "dark_frame": os.path.basename(self.dark_frame_path),
            "usable_window": self.usable_window,
            "bands": list(self.BANDS),
        }
        products = {
            "DSNU": (np.float32, {"product": "NETRA DSNU corrected DN max(0, max(0, DN - Dark) - Offset)"}),
            "RADIANCE": (np.float32, {"product": "NETRA radiance (DN_dsnu / Gain)",
                                      "radiance_unit": config.RADIANCE_UNIT}),
            "DN": (config.OUTPUT_DTYPE, {"product": "NETRA DSNU / PRNU corrected 14-bit DN",
                                         "radiance_unit": config.RADIANCE_UNIT,
                                         "radiance_per_dn": {b: 1.0 / self.dn_scale[b]
                                                             for b in self.BANDS}}),
        }
        paths, outputs = {}, {}
        for key, (dtype, desc) in products.items():
            paths[key] = os.path.join(self.output_dir, f"{self.satellite_name}_{name}_{key}.tif")
            outputs[key] = tifffile.memmap(paths[key] + ".partial", shape=(h, w, 3), dtype=dtype,
                                           photometric="rgb", metadata=None,
                                           description=json.dumps({**desc, **info}))
        dsnu_out, radiance_out, dn_out = outputs["DSNU"], outputs["RADIANCE"], outputs["DN"]

        def correct_block(b0, b1):
            # per band: raw-saturated, sum after DSNU, sum of radiance, clipped, sum of output DN
            stats = np.zeros((5, len(self.BANDS)))
            rows = slice(b0, b1)
            raw_block = image[r0 + b0:r0 + b1, c0:c1]
            buf = np.empty((b1 - b0, w), np.float32)
            for i, band in enumerate(self.BANDS):
                ch = config.BAND_IDS[band]
                raw = raw_block[..., ch]
                stats[0, i] = np.count_nonzero(raw >= config.SATURATION_DN)
                self.apply_dsnu_correction(raw, band, rows, buf)
                stats[1, i] = buf.sum(dtype=np.float64)
                dsnu_out[b0:b1, :, ch] = buf
                self.apply_prnu_correction(buf, band, rows)
                stats[2, i] = buf.sum(dtype=np.float64)
                radiance_out[b0:b1, :, ch] = buf
                stats[3, i] = self.radiance_to_dn(buf, band)
                stats[4, i] = buf.sum(dtype=np.float64)
                dn_out[b0:b1, :, ch] = buf
            return stats

        t0 = time.time()
        futures = [self._pool.submit(correct_block, b0, min(b0 + config.BLOCK_ROWS, h))
                   for b0 in range(0, h, config.BLOCK_ROWS)]
        try:
            stats = sum(f.result() for f in futures)
        except BaseException:   # also Ctrl+C: leave no half-written file
            for f in futures:
                f.cancel()
            wait(futures)
            del outputs, dsnu_out, radiance_out, dn_out
            for path in paths.values():
                if os.path.exists(path + ".partial"):
                    os.remove(path + ".partial")
            raise
        del outputs, dsnu_out, radiance_out, dn_out
        for path in paths.values():
            os.replace(path + ".partial", path)

        n = h * w
        logger.info("%s -> DSNU, RADIANCE, DN (%.1fs)  R/G/B mean: DSNU %.0f/%.0f/%.0f DN, "
                    "radiance %.3f/%.3f/%.3f, DN %.0f/%.0f/%.0f", name, time.time() - t0,
                    *(stats[1] / n), *(stats[2] / n), *(stats[4] / n))
        for i, band in enumerate(self.BANDS):
            if stats[0, i] / n > 0.001:
                logger.warning("%s band %s: %.2f%% pixels saturated in the raw image",
                               name, band, stats[0, i] / n * 100)
            if stats[3, i] / n > 0.001:
                logger.warning("%s band %s: %.2f%% pixels above the saturation radiance",
                               name, band, stats[3, i] / n * 100)
        return paths["DN"]

    # ------------------------------------------------------------------
    def process(self, input_path):
        '''
        Corrects one RGB TIFF, or every TIFF in a folder. Returns the output paths.
        '''
        if os.path.isdir(input_path):
            images = sorted(os.path.join(input_path, f) for f in os.listdir(input_path)
                            if f.lower().endswith(TIFF_EXT))
        elif os.path.isfile(input_path):
            images = [input_path]
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")
        if not images:
            raise FileNotFoundError(f"No TIFF images in {input_path}")
        logger.info("%d image(s) from %s", len(images), input_path)

        outputs, failed = [], []
        try:
            for k, image_path in enumerate(images):
                if k + 1 < len(images):
                    self._read_ahead(images[k + 1])
                try:
                    outputs.append(self.process_image(image_path))
                except (ValueError, OSError) as e:
                    failed.append(image_path)
                    logger.error("%s", e)
        finally:
            self.close()
        if failed:
            logger.error("%d of %d image(s) failed", len(failed), len(images))
        return outputs

    # ------------------------------------------------------------------
    @staticmethod
    def _read_ahead(path):
        '''
        Loads the next image into the page cache in the background.
        '''
        def go():
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                finally:
                    os.close(fd)
            except (OSError, AttributeError):
                pass
        threading.Thread(target=go, daemon=True).start()

    # ------------------------------------------------------------------
    def close(self):
        self._pool.shutdown()
        self.dark = self.dsnu_map = self.prnu_map = None
        if os.path.isdir(self._maps_dir):
            for f in os.listdir(self._maps_dir):
                os.remove(os.path.join(self._maps_dir, f))
            os.rmdir(self._maps_dir)


def main():
    p = argparse.ArgumentParser(description="DSNU / PRNU correction: RGB TIFF -> corrected 14-bit DN TIFF")
    p.add_argument("--payload", default=PAYLOAD, type=str.upper,
                   choices=list(config.SATELLITE_CONFIG_BY_NAME))
    p.add_argument("--input", default=INPUT_FOLDER, help="One RGB TIFF or a folder of them")
    p.add_argument("--output", default=OUTPUT_FOLDER, help="Output folder")
    p.add_argument("--dark", default=None, help="Dark frame instead of the one in config.py")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    file_log = logging.FileHandler(os.path.join(args.output, "prnu_dsnu.log"))
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_log)

    t0 = time.time()
    correction = DSNU_PRNU_Correction(args.payload, output_dir=args.output,
                                      dark_frame_path=args.dark)
    try:
        outputs = correction.process(args.input)
    except KeyboardInterrupt:
        logger.warning("Stopped by user")
        raise SystemExit(130)

    result = {
        "satellite_name"    : correction.satellite_name,
        "output_dir"        : correction.output_dir,
        "images"            : len(outputs),
        "seconds"           : round(time.time() - t0, 1),
    }
    logger.info("DSNU / PRNU correction complete: %s", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
