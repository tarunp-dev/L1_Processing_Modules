"""
Configuration for L1 Processing (H5 Bayer Extraction -> Demosaicing -> DSNU / PRNU).

Structure: one dict per payload (ALPHA / BRAVO), keyed by satellite_id in
SATELLITE_CONFIG, and by satellite_name in SATELLITE_CONFIG_BY_NAME for lookup
at runtime based on the <PAYLOAD> field of the input HDF5 file name.

Input HDF5 layout: "L0 HDF5 Product Structure: One File per Frame" (FormatSpecVersion 1.0).
"""

import os
from dotenv import load_dotenv
load_dotenv()

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

alpha_calibration_path = os.getenv("ALPHA_CALIBRATION_PATH", "/home/tarun/DC7A70CB7A70A3C4/flat_fields/alpha/@1562/config")
bravo_calibration_path = os.getenv("BRAVO_CALIBRATION_PATH", "/media/tarun/DC7A70CB7A70A3C4/flat_fields/bravo/@2000/config")

def _calibration_path(*parts):
    '''
    The function will join the base directory with the provided parts to create a full path to the calibration files.
    It uses the os.path.join method to concatenate the base directory with the provided parts, ensuring that the resulting path is valid and platform-independent.
    The function returns the full path to the calibration files, which can be used to access the calibration data for the L1 processing.
    '''
    return os.path.join(_BASE_DIR, *parts)

# ============================================================
# Global / shared parameters
# ============================================================

ADC_BIT_DEPTH = 14
OUTPUT_DN_MAX = (2 ** ADC_BIT_DEPTH) - 1
OUTPUT_DTYPE = "uint16"

# Row-block size for memory-safe DSNU / PRNU correction.
BLOCK_ROWS = 512

BAND_IDS = {"R": 0, "G": 1, "B": 2}
BANDS = ("R", "G", "B")

RADIANCE_UNIT = "mW/cm2/sr/um"


# ============================================================
# Input / output file names
# ============================================================

# Input (L0 spec): <SAT>_<PAYLOAD>_<SENSOR>_<YYYYMMDDTHHMMSS>_O<orbit>_S<session>_F<frame>_RAW.h5
PRODUCT_NAME_REGEX = (r"^(?P<sat>[A-Za-z0-9]+)_(?P<payload>[A-Za-z0-9]+)_(?P<sensor>[A-Za-z0-9]+)_"
                      r"(?P<datetime>\d{8}T\d{6})_O(?P<orbit>\d+)_S(?P<session>\d+)_F(?P<frame>\d+)_"
                      r"(?P<product>[A-Za-z0-9]+)$")
INPUT_PRODUCT_TYPE = "RAW"
OUTPUT_EXT = ".tiff"
PRODUCT_BAYER = "BAYER"
PRODUCT_DEM = "DEM"
PRODUCT_DSNU = "DSNU"
PRODUCT_PRNU = "PRNU"


# ============================================================
# H5 Bayer extraction
# ============================================================

H5_IMAGE_DATASET = "Payload/Image"
H5_PAYLOAD_GROUP = "Payload"
H5_ROOT_ATTRS = ("ProductID", "MissionName", "SatelliteName", "SatelliteId", "PayloadName",
                 "SensorType", "ProductLevel", "FormatSpecVersion", "GenerationDateTime",
                 "SoftwareName", "SoftwareVersion")

H5_PAYLOAD_ATTRS = ("FrameId", "NoOfBands", "BandNumbers", "ImageRows", "ImageColumns",
                    "BytesPerPixel", "BitsPerPixel", "BayerPattern", "ExposureStartTime",
                    "ExposureEndTime", "ExposureStartOnboardTime", "ExposureTime",
                    "DetectorTemperature", "FrameQualityFlag",
                    "B1_Lmin", "B2_Lmin", "B3_Lmin", "B1_Lmax", "B2_Lmax", "B3_Lmax")


# ============================================================
# Demosaicing
# ============================================================

BAYER_PATTERN = "RGGB"
DEMOSAIC_ALGORITHM = "ea"
DEMOSAIC_CV2_CODES = {
    "ea": {
        "RGGB"          : "COLOR_BayerBG2RGB_EA",
    },
}


# ============================================================
# ALPHA payload
# ============================================================
ALPHA = {
    "satellite_id"      : 123,
    "satellite_name"    : "ALPHA",
    "exposure_us"       : 1562,
    "sensor_width"      : 13468,
    "sensor_height"     : 9564,
    "usable_window"     : None,   # None => full sensor frame is usable

    "dark_frame_path"   : _calibration_path(alpha_calibration_path, "darkframes_average.tiff"),
    "gain_path": {
        "R"             : _calibration_path(alpha_calibration_path, "Channel_R_gain.tif"),
        "G"             : _calibration_path(alpha_calibration_path, "Channel_G_gain.tif"),
        "B"             : _calibration_path(alpha_calibration_path, "Channel_B_gain.tif"),
    },
    "offset_path": {
        "R"             : _calibration_path(alpha_calibration_path, "Channel_R_offset.tif"),
        "G"             : _calibration_path(alpha_calibration_path, "Channel_G_offset.tif"),
        "B"             : _calibration_path(alpha_calibration_path, "Channel_B_offset.tif"),
    },

    "saturation_radiance": {
        "R"             : 40.5088,
        "G"             : 13.9273,
        "B"             : 19.3128,
    },
}


# ============================================================
# BRAVO payload
# ============================================================
BRAVO = {
    "satellite_id"              : 456,
    "satellite_name"            : "BRAVO",
    "exposure_us"               : 2000,
    "sensor_width"              : 13468,
    "sensor_height"             : 9564,
    "usable_window"             : (3038, 10429, 1898, 7666),  # (xmin, xmax, ymin, ymax)

    "dark_frame_path"           : _calibration_path(bravo_calibration_path, "dark_frame_average.tiff"),
    "gain_path": {
        "R"                     : _calibration_path(bravo_calibration_path, "Channel_R_gain.tif"),
        "G"                     : _calibration_path(bravo_calibration_path, "Channel_G_gain.tif"),
        "B"                     : _calibration_path(bravo_calibration_path, "Channel_B_gain.tif"),
    },
    "offset_path": {
        "R"                     : _calibration_path(bravo_calibration_path, "Channel_R_offset.tif"),
        "G"                     : _calibration_path(bravo_calibration_path, "Channel_G_offset.tif"),
        "B"                     : _calibration_path(bravo_calibration_path, "Channel_B_offset.tif"),
    },

    "saturation_radiance": {
        "R"                     : 49.7373,
        "G"                     : 17.1794,
        "B"                     : 26.6689,
    },
}


SATELLITE_CONFIG = {
    ALPHA["satellite_id"]: ALPHA,
    BRAVO["satellite_id"]: BRAVO,
}

SATELLITE_CONFIG_BY_NAME = {
    ALPHA["satellite_name"]: ALPHA,
    BRAVO["satellite_name"]: BRAVO,
}


def get_config_by_satellite_id(satellite_id):
    if satellite_id not in SATELLITE_CONFIG:
        raise ValueError(f"No configuration found for satellite_id={satellite_id}")
    return SATELLITE_CONFIG[satellite_id]


def get_config_by_satellite_name(satellite_name):
    key = str(satellite_name).strip().upper()
    if key not in SATELLITE_CONFIG_BY_NAME:
        raise ValueError(f"No configuration found for satellite_name={satellite_name!r}")
    return SATELLITE_CONFIG_BY_NAME[key]
