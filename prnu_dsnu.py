'''
DSNU / PRNU correction of RGB TIFF images (linear raw 14-bit DN).
    1. DSNU : DN_dsnu = max(0, DN - Dark), cropped to the usable window
    2. PRNU : Radiance = max(0, DN_dsnu - Offset) / Gain
              DN_out   = rint(Radiance / Saturation_Radiance x 16383)

The image, dark frame and offset / gain maps are read block by block (config.BLOCK_ROWS
rows at a time, only the strips / tiles of those rows are decoded), so no full frame is
held in RAM.

Usage:
    python3 prnu_dsnu.py
    -> asks for the payload and the input RGB TIFF
    -> writes to <folder of the TIFF>/dsnu_prnu_<PAYLOAD>/
'''

import os
import time
import logging

import numpy as np
import tifffile

import config

logger = logging.getLogger(__name__)


class TiffBlockReader:
    '''
    Reads rows [r0, r1) of a TIFF by decoding only the strips / tiles that cover them.
    '''

    def __init__(self, path):
        self.path = path
        self.tif = tifffile.TiffFile(path)
        self.page = self.tif.pages.first
        if self.page.planarconfig == tifffile.PLANARCONFIG.SEPARATE:
            raise ValueError(f"Planar (separate) TIFFs are not supported: {path}")
        self.shape = self.page.shape
        self.dtype = self.page.dtype
        # uncompressed files: rows are sliced straight from a memory map
        self.memmap = self.page.asarray(out="memmap") if self.page.is_memmappable else None
        if self.page.is_tiled:
            self.seg_rows, self.seg_cols = self.page.tilelength, self.page.tilewidth
        else:
            self.seg_rows, self.seg_cols = self.page.rowsperstrip, self.page.imagewidth
        self.segs_per_row = -(-self.page.imagewidth // self.seg_cols)

    def read(self, r0, r1):
        if self.memmap is not None:
            return self.memmap[r0:r1]
        out = np.zeros((r1 - r0,) + self.shape[1:], self.dtype)
        fh, page = self.tif.filehandle, self.page
        for seg_row in range(r0 // self.seg_rows, -(-r1 // self.seg_rows)):
            for seg_col in range(self.segs_per_row):
                index = seg_row * self.segs_per_row + seg_col
                fh.seek(page.dataoffsets[index])
                data = fh.read(page.databytecounts[index])
                segment, (_, _, r, c, _), shape = page.decode(data, index,
                                                              jpegtables=page.jpegtables)
                if segment is None:   # empty strip / tile
                    continue
                # overlap of this segment with rows [r0, r1), clipped at the image's right edge
                s0, s1 = max(r, r0), min(r + shape[1], r1)
                cols = min(shape[2], self.shape[1] - c)
                block = segment[0, s0 - r:s1 - r, :cols]
                out[s0 - r0:s1 - r0, c:c + cols] = block.reshape((s1 - s0, cols) + self.shape[2:])
        return out

    def close(self):
        self.memmap = None
        self.tif.close()


def image_window(shape, sat_config):
    '''
    (row0, col0, height, width) of the part of an image that is processed:
    the usable window (xmin, xmax, ymin, ymax) of a full sensor frame, else the whole image.
    '''
    window = sat_config["usable_window"]
    if window is not None and shape[:2] == (sat_config["sensor_height"], sat_config["sensor_width"]):
        xmin, xmax, ymin, ymax = window
        return ymin, xmin, ymax - ymin, xmax - xmin
    return 0, 0, shape[0], shape[1]


def dsnu(raw, dark):
    '''
    DN_dsnu = max(0, DN - Dark), all bands of one block, computed in float32
    (a uint16 subtraction would wrap around wherever DN < Dark).
    '''
    out = np.subtract(raw, dark, dtype=np.float32)
    np.maximum(out, 0, out=out)
    return out


def to_uint16(dn):
    '''
    Rounds a float DN block and clips it to [0, 65535] for saving as uint16.
    '''
    return np.clip(np.rint(dn), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def prnu(dn_dsnu, offset, gain, sat_config):
    '''
    Radiance = max(0, DN_dsnu - Offset) / Gain,
    then DN_out = rint(Radiance / Saturation_Radiance x 16383), per band of one block.
    '''
    out = np.empty(dn_dsnu.shape, config.OUTPUT_DTYPE)

    for band in config.BANDS:
        ch = config.BAND_IDS[band]
        band_offset = np.nan_to_num(offset[band].astype(np.float32))
        band_gain = gain[band].astype(np.float32)
        bad = ~np.isfinite(band_gain) | (band_gain <= 0)
        band_gain[bad] = np.inf   # -> radiance 0

        radiance = dn_dsnu[..., ch] - band_offset
        np.maximum(radiance, 0, out=radiance)
        radiance /= band_gain
        radiance *= config.OUTPUT_DN_MAX / sat_config["saturation_radiance"][band]
        np.clip(radiance, 0, config.OUTPUT_DN_MAX, out=radiance)
        out[..., ch] = np.rint(radiance)
    return out


def correct_image(image_path, output_dir, dark, offset, gain, sat_config):
    '''
    Applies DSNU and PRNU to one image block by block, writing <name>_DSNU.tif and <name>_PRNU.tif.
    '''
    name = os.path.splitext(os.path.basename(image_path))[0]
    image = TiffBlockReader(image_path)
    try:
        if image.shape != dark.shape:
            logger.warning("Skipping %s: shape %s != dark frame %s", name, image.shape, dark.shape)
            return
        row0, col0, h, w = image_window(image.shape, sat_config)
        logger.info("%s: %s %s -> window rows %d:%d, cols %d:%d (%d x %d)", name, image.shape,
                    image.dtype, row0, row0 + h, col0, col0 + w, w, h)
        t0 = time.time()
        for band in config.BANDS:
            if offset[band].shape != (h, w) or gain[band].shape != (h, w):
                raise ValueError(f"Offset {offset[band].shape} / gain {gain[band].shape} map {band} "
                                 f"do not match the image window {(h, w)}")

        dsnu_out = tifffile.memmap(os.path.join(output_dir, f"{name}_DSNU.tif"),
                                   shape=(h, w, 3), dtype=np.uint16, photometric="rgb")
        prnu_out = tifffile.memmap(os.path.join(output_dir, f"{name}_PRNU.tif"),
                                   shape=(h, w, 3), dtype=config.OUTPUT_DTYPE, photometric="rgb")

        dsnu_sum = np.zeros(3)
        prnu_sum = np.zeros(3)
        for b0 in range(0, h, config.BLOCK_ROWS):
            b1 = min(b0 + config.BLOCK_ROWS, h)
            raw = image.read(row0 + b0, row0 + b1)[:, col0:col0 + w]
            dark_block = dark.read(row0 + b0, row0 + b1)[:, col0:col0 + w]

            dn_dsnu = dsnu(raw, dark_block)
            dsnu_out[b0:b1] = to_uint16(dn_dsnu)

            offset_block = {band: offset[band].read(b0, b1) for band in config.BANDS}
            gain_block = {band: gain[band].read(b0, b1) for band in config.BANDS}
            dn_out = prnu(dn_dsnu, offset_block, gain_block, sat_config)
            prnu_out[b0:b1] = dn_out

            dsnu_sum += dn_dsnu.sum(axis=(0, 1), dtype=np.float64)
            prnu_sum += dn_out.sum(axis=(0, 1), dtype=np.float64)
            logger.info("%s: rows %d / %d (%.0f%%)", name, b1, h, 100 * b1 / h)

        dsnu_out.flush()
        prnu_out.flush()
        del dsnu_out, prnu_out
        n = h * w
        logger.info("%s done in %.1fs. R/G/B mean: DSNU %.1f / %.1f / %.1f, PRNU %.1f / %.1f / %.1f",
                    name, time.time() - t0, *(dsnu_sum / n), *(prnu_sum / n))
    finally:
        image.close()


def main():
    payload = input(f"Payload {list(config.SATELLITE_CONFIG_BY_NAME)}: ").strip().upper()
    sat_config = config.get_config_by_satellite_name(payload)

    image_path = os.path.abspath(os.path.expanduser(input("Input RGB TIFF: ").strip().strip("'\"")))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input TIFF not found: {image_path}")

    output_dir = os.path.join(os.path.dirname(image_path), f"dsnu_prnu_{payload}")
    os.makedirs(output_dir, exist_ok=True)

    # log to the console and to <output_dir>/prnu_dsnu.log
    log_format = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console = logging.StreamHandler()
    log_file = logging.FileHandler(os.path.join(output_dir, "prnu_dsnu.log"))
    for handler in (console, log_file):
        handler.setFormatter(log_format)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Payload %s, exposure %s us, usable window %s", payload,
                sat_config["exposure_us"], sat_config["usable_window"])
    logger.info("Dark frame: %s", sat_config["dark_frame_path"])
    for band in config.BANDS:
        logger.info("Band %s: offset %s, gain %s, saturation radiance %s %s", band,
                    sat_config["offset_path"][band], sat_config["gain_path"][band],
                    sat_config["saturation_radiance"][band], config.RADIANCE_UNIT)
    logger.info("%s -> %s, %d rows per block", image_path, output_dir, config.BLOCK_ROWS)

    t0 = time.time()
    dark = TiffBlockReader(sat_config["dark_frame_path"])
    offset = {band: TiffBlockReader(sat_config["offset_path"][band]) for band in config.BANDS}
    gain = {band: TiffBlockReader(sat_config["gain_path"][band]) for band in config.BANDS}
    try:
        correct_image(image_path, output_dir, dark, offset, gain, sat_config)
        logger.info("All done in %.1fs", time.time() - t0)
    finally:
        for reader in [dark, *offset.values(), *gain.values()]:
            reader.close()


if __name__ == "__main__":
    main()
