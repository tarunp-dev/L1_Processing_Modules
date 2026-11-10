'''
The module will read an .H5 File and does the following:
    1. Extract the Bayer Image Data from the .H5 file ( H5 Bayer Extraction )
    2. Convert the Bayer Image Data to RGB Image ( Demosaicing )
    3. Apply DSNU Correction to the RGB Image ( Dark Frame Subtraction )
    4. Apply PRNU Correction to the RGB Image ( Gain and Offset Correction )
    5. Save the Bayer, RGB, DSNU and PRNU images as .tiff files with the naming convention.

Naming Convention (L0 spec):
    Input  : <SAT>_<PAYLOAD>_<SENSOR>_<YYYYMMDDTHHMMSS>_O<orbit>_S<session>_F<frame>_RAW.h5
    Output : same name, only the product type RAW replaced by BAYER / DEM / DSNU / PRNU,
             extension .tiff
    Example: ABC_ALPHA_RGB_20261014T053211_O001250_S02_F0005_RAW.h5
             -> ABC_ALPHA_RGB_20261014T053211_O001250_S02_F0005_BAYER.tiff
             -> ABC_ALPHA_RGB_20261014T053211_O001250_S02_F0005_DEM.tiff    (demosaiced RGB)
             -> ABC_ALPHA_RGB_20261014T053211_O001250_S02_F0005_DSNU.tiff
             -> ABC_ALPHA_RGB_20261014T053211_O001250_S02_F0005_PRNU.tiff

    The payload (ALPHA / BRAVO) is taken from the <PAYLOAD> field of the input file name and
    must match the root attribute PayloadName.
    Outputs are written next to the input .h5.


INPUT HDF5 LAYOUT ("L0 HDF5 Product Structure: One File per Frame"), parts used here :

    /                                (attrs) : ProductID, SatelliteId, PayloadName,
                                                SensorType, ProductLevel, ...
    /Payload                         (attrs) : FrameId, ImageRows, ImageColumns, BitsPerPixel,
                                                BayerPattern, ExposureTime (s),
                                                FrameQualityFlag, B1..B3_Lmin / _Lmax, ...
    /Payload/Image                   (M x N) : uint16 raw Bayer frame, full sensor size

    /Payload/MissingPixelMask, Attitude/, Orbit/, Geolocation/, SolarGeometry/,
    ViewingGeometry/, ImagingSession/ and Pass/ are not used. Missing pixels are
    neither masked nor corrected.


CORRECTIONS :

    DSNU : DN_dsnu  = max(0, DN - Dark), cropped to the usable window
    PRNU : Radiance = max(0, DN_dsnu - Offset) / Gain
           DN_out   = rint(Radiance / Saturation_Radiance x OUTPUT_DN_MAX)
'''

import os
import re
import json
import time
import logging
import argparse
import warnings
import contextlib

import numpy as np
import cv2
import h5py
import tifffile
import rasterio
from rasterio.windows import Window
from rasterio.errors import NotGeoreferencedWarning

import config


logger = logging.getLogger(__name__)

# Calibration / intermediate TIFFs are plain images without geo-referencing.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


class Level_1_Processing:
    '''
    The module will perform Level 1 processing on the Bayer Image Data from the .H5 file. The processing steps include:
        1. Extracting the Bayer Image Data from the .H5 file
        2. Converting the Bayer Image Data to RGB Image (Demosaicing)
        3. Applying DSNU Correction to the RGB Image
        4. Applying PRNU Correction to the RGB Image
        5. Saving the Bayer, RGB, DSNU and PRNU images as .tiff files with the specified naming convention.
    '''

    BANDS = config.BANDS

    def __init__(self, input_hdf5_file):
        self.input_hdf5_file = os.path.abspath(input_hdf5_file)
        self.input_name = os.path.basename(self.input_hdf5_file)
        self.product_id = os.path.splitext(self.input_name)[0]

        if not os.path.exists(self.input_hdf5_file):
            raise FileNotFoundError(f"The {self.input_hdf5_file} file is missing.")

        # ---- Parse the file name ----
        self.name_fields = self._parse_product_name(self.input_name)
        if self.name_fields is None:
            raise ValueError(
                f"Input file name {self.input_name!r} does not match "
                f"<SAT>_<PAYLOAD>_<SENSOR>_<YYYYMMDDTHHMMSS>_O<orbit>_S<session>_F<frame>_RAW.h5"
            )
        if self.name_fields["product"].upper() != config.INPUT_PRODUCT_TYPE:
            raise ValueError(
                f"Input product type {self.name_fields['product']!r} is not "
                f"{config.INPUT_PRODUCT_TYPE!r} -- expected an L0 RAW .h5 file."
            )
        self.satellite_name = self.name_fields["payload"].upper()

        # ---- Resolve satellite-specific calibration config (Alpha/Bravo) ----
        self.sat_config = config.get_config_by_satellite_name(self.satellite_name)
        self.sensor_shape = (self.sat_config["sensor_height"], self.sat_config["sensor_width"])
        logger.info(
            "Processing %s: satellite_id=%s (%s), orbit=%s, session=%s, frame=%s",
            self.input_name, self.sat_config["satellite_id"], self.satellite_name,
            self.name_fields["orbit"], self.name_fields["session"], self.name_fields["frame"],
        )

        # ---- Demosaicing settings ----
        self.bayer_pattern = config.BAYER_PATTERN.upper()
        cv2_codes = config.DEMOSAIC_CV2_CODES.get(config.DEMOSAIC_ALGORITHM)
        if cv2_codes is None:
            raise ValueError(f"Unknown demosaic algorithm {config.DEMOSAIC_ALGORITHM!r}, "
                             f"expected one of {list(config.DEMOSAIC_CV2_CODES)}")
        if self.bayer_pattern not in cv2_codes:
            raise ValueError(f"Unknown Bayer pattern {self.bayer_pattern!r}, "
                             f"expected one of {list(cv2_codes)}")
        self.cv2_code = getattr(cv2, cv2_codes[self.bayer_pattern])

        # ---- Read the L0 metadata + validate the image dataset ----
        with h5py.File(self.input_hdf5_file, "r") as f:
            if config.H5_IMAGE_DATASET not in f:
                raise KeyError(f"Input HDF5 is missing dataset '{config.H5_IMAGE_DATASET}'.")
            self.image_shape = f[config.H5_IMAGE_DATASET].shape
            self.h5_info = self._read_h5_metadata(f)

        if self.image_shape != self.sensor_shape:
            raise ValueError(
                f"Bayer image shape {self.image_shape} does not match the sensor frame "
                f"{self.sensor_shape} for satellite_name={self.satellite_name}."
            )
        self._check_h5_metadata()

        # ---- Usable window: (row0, col0, height, width) of the corrected images ----
        usable_window = self.sat_config["usable_window"]  # (xmin, xmax, ymin, ymax) or None
        if usable_window is not None:
            xmin, xmax, ymin, ymax = usable_window
            self.window = (ymin, xmin, ymax - ymin, xmax - xmin)
        else:
            self.window = (0, 0) + self.sensor_shape
        expected_shape = self.window[2:]

        # ---- Verify dark / gain / offset calibration files exist and match shape ----
        dark_path = self.sat_config["dark_frame_path"]
        if not os.path.exists(dark_path):
            raise FileNotFoundError(f"Dark frame missing: {dark_path}")
        with rasterio.open(dark_path) as src:
            dshape = (src.count, src.height, src.width)
        if dshape != (3,) + self.sensor_shape:
            raise ValueError(
                f"Dark frame shape (bands, rows, cols) {dshape} does not match the "
                f"3-band sensor frame {(3,) + self.sensor_shape}."
            )

        for band in self.BANDS:

            gpath = self.sat_config["gain_path"][band]
            opath = self.sat_config["offset_path"][band]

            if not os.path.exists(gpath):
                raise FileNotFoundError(f"Gain file missing for band {band}: {gpath}")
            if not os.path.exists(opath):
                raise FileNotFoundError(f"Offset file missing for band {band}: {opath}")

            for kind, path in (("Gain", gpath), ("Offset", opath)):
                with rasterio.open(path) as src:
                    shape = (src.height, src.width)
                if shape != expected_shape:
                    raise ValueError(
                        f"{kind} map shape {shape} for band {band} does not match "
                        f"expected scene shape {expected_shape}. Gain/offset TIFFs "
                        f"must already be cropped to the usable window."
                    )

            if self.sat_config["saturation_radiance"][band] is None:
                raise ValueError(
                    f"saturation_radiance for band '{band}' is not set in "
                    f"config.py -- required for DN normalization."
                )

        # ---- Build output filenames ----
        # Same name as the input, only the product type RAW replaced; next to the input.
        self.output_dir = os.path.dirname(self.input_hdf5_file)
        self.bayer_tiff_file = os.path.join(self.output_dir, self._product_filename(config.PRODUCT_BAYER))
        self.dem_tiff_file = os.path.join(self.output_dir, self._product_filename(config.PRODUCT_DEM))
        self.dsnu_tiff_file = os.path.join(self.output_dir, self._product_filename(config.PRODUCT_DSNU))
        self.prnu_tiff_file = os.path.join(self.output_dir, self._product_filename(config.PRODUCT_PRNU))

        self.bayer = None   # raw Bayer frame, read on first use
        logger.info("Output folder: %s", self.output_dir)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_product_name(filename):
        '''
        Fields of a product file name (sat, payload, sensor, datetime, orbit, session, frame,
        product), or None if the name does not follow config.PRODUCT_NAME_REGEX.
        '''
        stem = os.path.splitext(os.path.basename(filename))[0]
        match = re.match(config.PRODUCT_NAME_REGEX, stem)
        return match.groupdict() if match else None

    # ------------------------------------------------------------------
    def _product_filename(self, product_type):
        '''
        Input file name with only the product type (RAW) replaced by product_type,
        extension config.OUTPUT_EXT.
        '''
        match = re.match(config.PRODUCT_NAME_REGEX, self.product_id)
        return (self.product_id[:match.start("product")] + product_type
                + self.product_id[match.end("product"):] + config.OUTPUT_EXT)

    # ------------------------------------------------------------------
    @classmethod
    def _read_h5_metadata(cls, f):
        '''
        Root (file identification) and /Payload (sensor, frame, radiometric) attributes
        listed in config.H5_ROOT_ATTRS / config.H5_PAYLOAD_ATTRS, as plain Python values.
        '''
        payload = f[config.H5_PAYLOAD_GROUP].attrs if config.H5_PAYLOAD_GROUP in f else {}
        info = {k: cls._attr_value(f.attrs[k]) for k in config.H5_ROOT_ATTRS if k in f.attrs}
        info.update({k: cls._attr_value(payload[k]) for k in config.H5_PAYLOAD_ATTRS if k in payload})
        return info

    # ------------------------------------------------------------------
    @staticmethod
    def _attr_value(value):
        '''
        H5 attribute -> plain Python value (bytes decoded, numpy scalars unwrapped) for JSON / logging.
        '''
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    # ------------------------------------------------------------------
    def _check_h5_metadata(self):
        '''
        Checks the L0 attributes against the file name, the image and the payload config.

        Errors (the frame cannot be processed correctly):
            - PayloadName differs from the <PAYLOAD> of the file name (wrong calibration)
            - ImageRows / ImageColumns differ from the Image dataset shape
            - BayerPattern differs from config.BAYER_PATTERN (wrong colours)
        Warnings:
            - missing attributes, ProductID / FrameId differ from the file name,
              BitsPerPixel differs from config.ADC_BIT_DEPTH, ExposureTime differs from the
              payload's calibrated exposure, FrameQualityFlag is not good, NoOfBands != 3
        '''
        info = self.h5_info

        missing = [k for k in ("ProductID", "PayloadName", "ImageRows", "ImageColumns",
                               "BitsPerPixel", "BayerPattern", "ExposureTime", "FrameQualityFlag")
                   if k not in info]
        if missing:
            logger.warning("Input HDF5 is missing attribute(s): %s", ", ".join(missing))

        # ---- File identification ----
        if "ProductID" in info and str(info["ProductID"]) != self.product_id:
            logger.warning("ProductID %s differs from the file name %s (renamed file?)",
                           info["ProductID"], self.product_id)

        if "PayloadName" in info and str(info["PayloadName"]).strip().upper() != self.satellite_name:
            raise ValueError(
                f"PayloadName attribute {info['PayloadName']!r} does not match the payload "
                f"{self.satellite_name!r} of the file name."
            )

        if "FrameId" in info and int(info["FrameId"]) != int(self.name_fields["frame"]):
            logger.warning("FrameId %s differs from the file name frame F%s",
                           info["FrameId"], self.name_fields["frame"])

        # ---- Image ----
        if "ImageRows" in info and "ImageColumns" in info:
            attr_shape = (int(info["ImageRows"]), int(info["ImageColumns"]))
            if attr_shape != tuple(self.image_shape):
                raise ValueError(
                    f"ImageRows x ImageColumns {attr_shape} does not match the "
                    f"{config.H5_IMAGE_DATASET} shape {self.image_shape}."
                )

        if "NoOfBands" in info and int(info["NoOfBands"]) != len(self.BANDS):
            logger.warning("NoOfBands is %s, expected %d", info["NoOfBands"], len(self.BANDS))

        pattern = str(info.get("BayerPattern", "")).strip().upper()
        if not pattern:
            logger.warning("No BayerPattern attribute, assuming %s", self.bayer_pattern)
        elif pattern != self.bayer_pattern:
            raise ValueError(
                f"H5 BayerPattern {pattern} differs from config.BAYER_PATTERN "
                f"{self.bayer_pattern} -- demosaicing would swap the colours."
            )

        if "BitsPerPixel" in info and int(info["BitsPerPixel"]) != config.ADC_BIT_DEPTH:
            logger.warning("BitsPerPixel %s differs from config.ADC_BIT_DEPTH %d",
                           info["BitsPerPixel"], config.ADC_BIT_DEPTH)

        # ---- Exposure: dark frame / gain / offset are calibrated for one exposure ----
        if "ExposureTime" in info:
            exposure_us = round(float(info["ExposureTime"]) * 1e6)
            if exposure_us != self.sat_config["exposure_us"]:
                logger.warning(
                    "Frame exposure %d us differs from the %s calibration exposure %s us -- "
                    "dark frame / gain / offset may not apply", exposure_us,
                    self.satellite_name, self.sat_config["exposure_us"],
                )

        # 0 = good frame (L0 spec)
        if "FrameQualityFlag" in info and int(info["FrameQualityFlag"]) != 0:
            logger.warning("FrameQualityFlag is %s (not good)", info["FrameQualityFlag"])

    # ------------------------------------------------------------------
    def _read_bayer(self):
        '''
        Reads the full raw Bayer frame from the input HDF5.
        '''
        with h5py.File(self.input_hdf5_file, "r") as f:
            bayer = f[config.H5_IMAGE_DATASET][()]
        logger.info("Bayer image %s %s, min %s, max %s", bayer.shape, bayer.dtype,
                    bayer.min(), bayer.max())
        return bayer

    # ------------------------------------------------------------------
    def _description(self, product, **extra):
        '''
        TIFF description: product, source file, payload and the L0 attributes.
        '''
        return {"product": product, "source": self.input_name,
                "satellite_name": self.satellite_name, **extra, "l0_attributes": self.h5_info}

    # ------------------------------------------------------------------
    @staticmethod
    def _write_tiff(output_path, data, desc, photometric):
        '''
        Writes a TIFF via <path>.partial, so no half-written file is left behind.
        '''
        try:
            tifffile.imwrite(output_path + ".partial", data, photometric=photometric,
                             metadata=None, description=json.dumps(desc))
            os.replace(output_path + ".partial", output_path)
        except BaseException:   # also Ctrl+C
            if os.path.exists(output_path + ".partial"):
                os.remove(output_path + ".partial")
            raise

    # ------------------------------------------------------------------
    def apply_h5_bayer_extraction(self):
        '''
        Extracts the raw Bayer frame from the input HDF5 and writes it as a single-channel
        TIFF, same dtype / size as the H5, values as is (no masking or correction).

        Warns when the image dtype is too small for BitsPerPixel or pixel values exceed
        2**BitsPerPixel - 1.

        OUTPUT: self.bayer_tiff_file
        '''
        t0 = time.time()
        self.bayer = self._read_bayer()
        self._check_bit_range(self.bayer, self.h5_info.get("BitsPerPixel"))

        desc = self._description("Bayer frame extracted from L0 RAW H5", dataset=config.H5_IMAGE_DATASET)
        self._write_tiff(self.bayer_tiff_file, self.bayer, desc, photometric="minisblack")

        logger.info("H5 Bayer extraction complete in %.1fs: %s", time.time() - t0, self.bayer_tiff_file)
        return self.bayer_tiff_file

    # ------------------------------------------------------------------
    @staticmethod
    def _check_bit_range(bayer, bits_per_pixel):
        '''
        Warns when the image dtype cannot hold BitsPerPixel or values exceed its range.
        Never modifies the data.
        '''
        if bits_per_pixel is None or not np.issubdtype(bayer.dtype, np.integer):
            return
        bpp = int(bits_per_pixel)
        dtype_bits = np.iinfo(bayer.dtype).bits
        if dtype_bits < bpp:
            logger.warning("BitsPerPixel %d but image is stored as %s (%d bits), "
                           "data may be truncated", bpp, bayer.dtype, dtype_bits)
        elif int(bayer.max()) > (1 << bpp) - 1:
            logger.warning("Max DN %d exceeds the %d-bit range (%d)", int(bayer.max()), bpp,
                           (1 << bpp) - 1)

    # ------------------------------------------------------------------
    def apply_demosaicing(self):
        '''
        Demosaics the Bayer frame to RGB with OpenCV (config.DEMOSAIC_ALGORITHM, default
        edge-aware: interpolates along edges instead of across them, less zipper / colour
        fringing than bilinear) and writes the RGB TIFF (H x W x 3, full sensor frame).

        INPUT : Bayer frame from apply_h5_bayer_extraction (read from the H5 if it did not run).
        OUTPUT: self.dem_tiff_file
        '''
        if self.bayer is None:
            self.bayer = self._read_bayer()
        t0 = time.time()

        # cv2 Bayer conversion only takes uint8 / uint16 (raw L0 frames are uint16)
        if self.bayer.dtype not in (np.uint8, np.uint16):
            raise TypeError(f"Bayer image dtype {self.bayer.dtype} is not supported, "
                            f"expected uint8 / uint16 raw DN")
        rgb = cv2.cvtColor(self.bayer, self.cv2_code)
        self.bayer = None   # free the raw frame, the RGB image is 3x its size

        desc = self._description(f"Demosaiced RGB (OpenCV {config.DEMOSAIC_ALGORITHM})",
                                 bayer_pattern=self.bayer_pattern)
        self._write_tiff(self.dem_tiff_file, rgb, desc, photometric="rgb")
        logger.info("Demosaicing complete in %.1fs: %s (%s, %s)", time.time() - t0,
                    self.dem_tiff_file, rgb.dtype, rgb.shape)
        return self.dem_tiff_file

    # ------------------------------------------------------------------
    def apply_dsnu_prnu_correction(self):
        '''
        DSNU and PRNU correction of the RGB image, cropped to the usable window, processed in
        row-blocks (config.BLOCK_ROWS) to bound memory usage on full-frame imagery.

            DN_dsnu  = max(0, DN - Dark)
            Radiance = max(0, DN_dsnu - Offset) / Gain
            DN_out   = rint(Radiance / Saturation_Radiance x OUTPUT_DN_MAX)

        Gain <= 0 / NaN pixels give radiance 0, NaN offsets are treated as 0.

        INPUT : self.dem_tiff_file (written by apply_demosaicing).
        OUTPUT: self.dsnu_tiff_file and self.prnu_tiff_file.
        '''
        if not os.path.exists(self.dem_tiff_file):
            raise FileNotFoundError(f"RGB image missing (run apply_demosaicing first): {self.dem_tiff_file}")

        row0, col0, n_rows, n_cols = self.window
        block_rows = config.BLOCK_ROWS
        t0 = time.time()
        logger.info(
            "Applying DSNU / PRNU correction: window rows %d:%d, cols %d:%d (%d x %d), block_rows=%d",
            row0, row0 + n_rows, col0, col0 + n_cols, n_cols, n_rows, block_rows,
        )

        window_desc = {"window_row0": row0, "window_col0": col0}
        dsnu_desc = self._description("DSNU corrected RGB (dark frame subtracted)",
                                      dark_frame=self.sat_config["dark_frame_path"], **window_desc)
        prnu_desc = self._description("PRNU corrected RGB (gain / offset, DN scaled to saturation radiance)",
                                      saturation_radiance=self.sat_config["saturation_radiance"],
                                      radiance_unit=config.RADIANCE_UNIT,
                                      output_dn_max=config.OUTPUT_DN_MAX, **window_desc)
        dsnu_out = tifffile.memmap(self.dsnu_tiff_file, shape=(n_rows, n_cols, 3),
                                   dtype=config.OUTPUT_DTYPE, photometric="rgb",
                                   metadata=None, description=json.dumps(dsnu_desc))
        prnu_out = tifffile.memmap(self.prnu_tiff_file, shape=(n_rows, n_cols, 3),
                                   dtype=config.OUTPUT_DTYPE, photometric="rgb",
                                   metadata=None, description=json.dumps(prnu_desc))
        dsnu_sum = np.zeros(3)
        prnu_sum = np.zeros(3)

        with contextlib.ExitStack() as stack:
            rgb_src = stack.enter_context(rasterio.open(self.dem_tiff_file))
            dark_src = stack.enter_context(rasterio.open(self.sat_config["dark_frame_path"]))
            gain_src = {band: stack.enter_context(rasterio.open(self.sat_config["gain_path"][band]))
                        for band in self.BANDS}
            offset_src = {band: stack.enter_context(rasterio.open(self.sat_config["offset_path"][band]))
                          for band in self.BANDS}

            for row_start in range(0, n_rows, block_rows):
                row_end = min(row_start + block_rows, n_rows)
                # image / dark frame are full sensor size, gain / offset are window size
                image_window = Window(col_off=col0, row_off=row0 + row_start,
                                      width=n_cols, height=row_end - row_start)
                calib_window = Window(col_off=0, row_off=row_start,
                                      width=n_cols, height=row_end - row_start)

                dn_block = rgb_src.read(window=image_window)       # (3, rows, cols)
                dark_block = dark_src.read(window=image_window)
                dn_dsnu = self._dsnu(dn_block, dark_block)
                del dn_block, dark_block
                dsnu_out[row_start:row_end] = self._to_output_dtype(dn_dsnu).transpose(1, 2, 0)

                for band in self.BANDS:
                    ch = config.BAND_IDS[band]
                    dn_out = self._prnu(
                        dn_dsnu[ch],
                        offset_src[band].read(1, window=calib_window),
                        gain_src[band].read(1, window=calib_window),
                        self.sat_config["saturation_radiance"][band],
                    )
                    prnu_out[row_start:row_end, :, ch] = dn_out
                    prnu_sum[ch] += dn_out.sum(dtype=np.float64)

                dsnu_sum += dn_dsnu.sum(axis=(1, 2), dtype=np.float64)
                logger.info("DSNU / PRNU rows %d / %d (%.0f%%)", row_end, n_rows,
                            100 * row_end / n_rows)

        dsnu_out.flush()
        prnu_out.flush()
        del dsnu_out, prnu_out

        n = n_rows * n_cols
        logger.info("DSNU / PRNU correction complete in %.1fs. R/G/B mean: "
                    "DSNU %.1f / %.1f / %.1f, PRNU %.1f / %.1f / %.1f",
                    time.time() - t0, *(dsnu_sum / n), *(prnu_sum / n))
        return self.dsnu_tiff_file, self.prnu_tiff_file

    # ------------------------------------------------------------------
    @staticmethod
    def _dsnu(dn_block, dark_block):
        '''
        DN_dsnu = max(0, DN - Dark), computed in float32 (a uint16 subtraction would wrap
        around wherever DN < Dark).
        '''
        out = np.subtract(dn_block, dark_block, dtype=np.float32)
        np.maximum(out, 0, out=out)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _prnu(dn_dsnu, offset, gain, saturation_radiance):
        '''
        Radiance = max(0, DN_dsnu - Offset) / Gain,
        then DN_out = rint(Radiance / Saturation_Radiance x OUTPUT_DN_MAX), one band.
        '''
        offset = np.nan_to_num(offset.astype(np.float32))
        gain = gain.astype(np.float32)
        gain[~np.isfinite(gain) | (gain <= 0)] = np.inf   # -> radiance 0

        radiance = dn_dsnu - offset
        np.maximum(radiance, 0, out=radiance)
        radiance /= gain
        radiance *= config.OUTPUT_DN_MAX / saturation_radiance
        np.clip(radiance, 0, config.OUTPUT_DN_MAX, out=radiance)
        return np.rint(radiance).astype(config.OUTPUT_DTYPE)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_output_dtype(dn):
        '''
        Rounds a float DN block and clips it to the range of config.OUTPUT_DTYPE.
        '''
        return np.clip(np.rint(dn), 0, np.iinfo(config.OUTPUT_DTYPE).max).astype(config.OUTPUT_DTYPE)


def main():
    p = argparse.ArgumentParser(description="L0 RAW .h5 -> H5 Bayer extraction, demosaicing, "
                                            "DSNU / PRNU correction -> .tiff")
    p.add_argument("--input", required=True, help="Path to ONE input L0 RAW HDF5 product")
    args = p.parse_args()

    t0 = time.time()
    l1 = Level_1_Processing(input_hdf5_file=args.input)
    l1.apply_h5_bayer_extraction()
    l1.apply_demosaicing()
    l1.apply_dsnu_prnu_correction()

    result = {
        "satellite_name"    : l1.satellite_name,
        "bayer_tiff"        : l1.bayer_tiff_file,
        "dem_tiff"          : l1.dem_tiff_file,
        "dsnu_tiff"         : l1.dsnu_tiff_file,
        "prnu_tiff"         : l1.prnu_tiff_file,
        "seconds"           : round(time.time() - t0, 1),
    }
    logger.info("L1 processing complete: %s", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
