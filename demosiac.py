'''
Demosaicing of Bayer TIFF images (single-channel raw DN) with OpenCV:
    Default method is OpenCV's Edge-Aware (EA) demosaicing, which interpolates along edges
    instead of across them (less zipper / color fringing than bilinear).
    Works on uint8 and uint16 data; float input (e.g. averaged frames) is converted to uint16 first.
Output: <OUTPUT_FOLDER>/<name>.tif   (RGB, H x W x 3, same dtype as the cv2 input)
        OUTPUT_FOLDER defaults to <INPUT_FOLDER>/demosaiced_rgb_cv2_<algorithm>,
        which is the INPUT_FOLDER of prnu_dsnu.py.

Usage:
    python3 demosiac.py                    (PAYLOAD, INPUT_FOLDER below, settings from config.py)
    python3 demosiac.py --payload ALPHA --algorithm bilinear
'''

import os
import json
import time
import logging
import argparse

import cv2
import numpy as np
import tifffile

import config


PAYLOAD = "BRAVO"
INPUT_FOLDER = "/home/tarun/Documents/oct_8/bravo/@2000_averaged"
OUTPUT_FOLDER = None   # None => <INPUT_FOLDER>/demosaiced_rgb_cv2_<algorithm>

logger = logging.getLogger(__name__)


class Demosaic:
    '''
    Demosaics the Bayer images of one payload to RGB.
    '''

    def __init__(self, satellite_name, algorithm=None, pattern=None, float_scale=None,
                 border_fix=None):
        self.sat_config = config.get_config_by_satellite_name(satellite_name)
        self.satellite_name = self.sat_config["satellite_name"]
        self.algorithm = algorithm or config.DEMOSAIC_ALGORITHM
        self.pattern = (pattern or config.BAYER_PATTERN).upper()
        self.float_scale = float_scale if float_scale is not None else config.DEMOSAIC_FLOAT_SCALE
        self.border_fix = config.DEMOSAIC_BORDER_FIX if border_fix is None else border_fix

        if self.algorithm not in config.DEMOSAIC_CV2_CODES:
            raise ValueError(f"Unknown demosaic algorithm {self.algorithm!r}, "
                             f"expected one of {list(config.DEMOSAIC_CV2_CODES)}")
        if self.pattern not in config.BAYER_PATTERNS:
            raise ValueError(f"Unknown Bayer pattern {self.pattern!r}, "
                             f"expected one of {config.BAYER_PATTERNS}")
        self.cv2_code = getattr(cv2, config.DEMOSAIC_CV2_CODES[self.algorithm][self.pattern])
        self.sensor_shape = (self.sat_config["sensor_height"], self.sat_config["sensor_width"])

        logger.info("%s: demosaic %s, pattern %s, float scale %s, border fix %s",
                    self.satellite_name, self.algorithm, self.pattern,
                    "auto" if self.float_scale is None else self.float_scale, self.border_fix)

    # ------------------------------------------------------------------
    def to_cv2_input(self, bayer, name):
        '''
        cv2 Bayer conversion only accepts uint8 / uint16.
        '''
        if bayer.dtype in (np.uint8, np.uint16):
            return bayer

        if np.issubdtype(bayer.dtype, np.integer):
            # e.g. uint32 / int32 - clip into uint16 range
            return np.clip(bayer, 0, config.CV2_INPUT_MAX).astype(np.uint16)

        # Float input (e.g. averaged frames)
        float_scale = self.float_scale
        if float_scale is None:
            float_scale = float(config.CV2_INPUT_MAX) if float(np.nanmax(bayer)) <= 1.0 else 1.0
        logger.info("%s: float input, scaling by %g -> uint16", name, float_scale)

        data = bayer.astype(np.float32, copy=True)
        if float_scale != 1.0:
            data *= float_scale
        np.nan_to_num(data, copy=False, nan=0.0)
        np.clip(data, 0, config.CV2_INPUT_MAX, out=data)
        np.rint(data, out=data)
        return data.astype(np.uint16)

    # ------------------------------------------------------------------
    @staticmethod
    def fix_border(rgb, b=config.DEMOSAIC_BORDER_PX):
        '''
        Replicates the inner pixels into the outer b-pixel ring, in place.
        '''
        rgb[:b] = rgb[b]
        rgb[-b:] = rgb[-b - 1]
        rgb[:, :b] = rgb[:, b:b + 1]
        rgb[:, -b:] = rgb[:, -b - 1:-b]
        return rgb

    # ------------------------------------------------------------------
    def demosaic(self, bayer, name=""):
        bayer = self.to_cv2_input(bayer, name)
        rgb = cv2.cvtColor(bayer, self.cv2_code)
        if self.border_fix:
            rgb = self.fix_border(rgb)
        return rgb

    # ------------------------------------------------------------------
    def process_image(self, image_path, output_dir):
        '''
        Demosaics one Bayer TIFF into <output_dir>/<same filename>. Returns the output path.
        '''
        filename = os.path.basename(image_path)
        bayer = tifffile.imread(image_path)
        if bayer.ndim != 2:
            raise ValueError(f"{filename}: not a single-channel Bayer image, shape={bayer.shape}")
        if bayer.shape != self.sensor_shape:
            logger.warning("%s: shape %s is not the %s sensor frame %s - make sure the crop "
                           "starts on an even row/column or the Bayer pattern is wrong",
                           filename, bayer.shape, self.satellite_name, self.sensor_shape)
        logger.info("%s: shape=%s, dtype=%s, min=%s, max=%s", filename, bayer.shape, bayer.dtype,
                    bayer.min(), bayer.max())

        t0 = time.time()
        rgb = self.demosaic(bayer, filename)
        del bayer

        output_path = os.path.join(output_dir, filename)
        desc = {
            "product": f"NETRA demosaiced RGB (OpenCV {self.algorithm})",
            "satellite_name": self.satellite_name,
            "source": filename,
            "bayer_pattern": self.pattern,
            "border_fix": self.border_fix,
        }
        try:
            tifffile.imwrite(output_path + ".partial", rgb, photometric="rgb", metadata=None,
                             description=json.dumps(desc))
            os.replace(output_path + ".partial", output_path)
        except BaseException:   # also Ctrl+C: leave no half-written file
            if os.path.exists(output_path + ".partial"):
                os.remove(output_path + ".partial")
            raise
        logger.info("%s -> %s (%s, %s) %.1fs", filename, output_path, rgb.dtype, rgb.shape,
                    time.time() - t0)
        return output_path

    # ------------------------------------------------------------------
    @staticmethod
    def list_images(input_path):
        if os.path.isdir(input_path):
            return sorted(os.path.join(input_path, f) for f in os.listdir(input_path)
                          if f.lower().endswith(config.TIFF_EXT)
                          and os.path.isfile(os.path.join(input_path, f)))
        if os.path.isfile(input_path):
            return [input_path]
        raise FileNotFoundError(f"Input not found: {input_path}")

    # ------------------------------------------------------------------
    def resolve_output_dir(self, input_path, output_dir=None):
        '''
        output_dir, or <input folder>/demosaiced_rgb_cv2_<algorithm>. Never the input folder.
        '''
        input_dir = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
        output_dir = output_dir or os.path.join(input_dir,
                                                f"{config.DEMOSAIC_DIR_PREFIX}{self.algorithm}")
        if os.path.abspath(output_dir) == os.path.abspath(input_dir):
            raise ValueError("Output folder must differ from the input folder "
                             "(would overwrite the sources)")
        return output_dir

    # ------------------------------------------------------------------
    def process(self, input_path, output_dir=None):
        '''
        Demosaics one Bayer TIFF, or every TIFF in a folder. Returns the output paths.
        '''
        images = self.list_images(input_path)
        if not images:
            raise FileNotFoundError(f"No TIFF images in {input_path}")
        output_dir = self.resolve_output_dir(input_path, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info("%d image(s) from %s -> %s", len(images), input_path, output_dir)

        outputs, failed = [], []
        for image_path in images:
            try:
                outputs.append(self.process_image(image_path, output_dir))
            except (ValueError, OSError) as e:
                failed.append(image_path)
                logger.error("%s", e)
        if failed:
            logger.error("%d of %d image(s) failed", len(failed), len(images))
        return outputs

def main():
    p = argparse.ArgumentParser(description="Demosaic Bayer TIFFs -> RGB TIFFs with OpenCV")
    p.add_argument("--payload", default=PAYLOAD, type=str.upper,
                   choices=list(config.SATELLITE_CONFIG_BY_NAME))
    p.add_argument("--input", default=INPUT_FOLDER, help="One Bayer TIFF or a folder of them")
    p.add_argument("--output", default=OUTPUT_FOLDER,
                   help=f"Output folder. Default: <input>/{config.DEMOSAIC_DIR_PREFIX}<algorithm>")
    p.add_argument("--algorithm", default=config.DEMOSAIC_ALGORITHM,
                   choices=list(config.DEMOSAIC_CV2_CODES),
                   help="ea = edge-aware, bilinear = plain bilinear")
    p.add_argument("--pattern", default=None, type=str.upper, choices=config.BAYER_PATTERNS,
                   help="Bayer pattern instead of the one in config.py")
    p.add_argument("--float-scale", type=float, default=config.DEMOSAIC_FLOAT_SCALE,
                   help="Multiplier for float input before converting to uint16. "
                        "Default: 65535 if max <= 1.0, otherwise 1.0")
    p.add_argument("--border-fix", action="store_true", default=config.DEMOSAIC_BORDER_FIX,
                   help=f"Replicate inner pixels into the outer {config.DEMOSAIC_BORDER_PX}-pixel "
                        "ring (only if you see bad pixels at the image edges)")
    args = p.parse_args()

    demosaic = Demosaic(args.payload, algorithm=args.algorithm, pattern=args.pattern,
                        float_scale=args.float_scale, border_fix=args.border_fix)
    output_dir = demosaic.resolve_output_dir(args.input, args.output)

    os.makedirs(output_dir, exist_ok=True)
    file_log = logging.FileHandler(os.path.join(output_dir, "demosaic.log"))
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_log)

    t0 = time.time()
    try:
        outputs = demosaic.process(args.input, output_dir=output_dir)
    except KeyboardInterrupt:
        logger.warning("Stopped by user")
        raise SystemExit(130)

    result = {
        "satellite_name"    : demosaic.satellite_name,
        "algorithm"         : demosaic.algorithm,
        "bayer_pattern"     : demosaic.pattern,
        "output_dir"        : output_dir,
        "images"            : len(outputs),
        "seconds"           : round(time.time() - t0, 1),
    }
    logger.info("Demosaicing complete: %s", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
