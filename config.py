"""
Configuration for Demosaicing (demosiac.py) and DSNU / PRNU Correction (prnu_dsnu.py).
One dict per payload (ALPHA / BRAVO), looked up by satellite_name.
"""

# Global / shared parameters
ADC_BIT_DEPTH = 14
SATURATION_DN = 16383
OUTPUT_DN_MAX = (2 ** ADC_BIT_DEPTH) - 1   # 16383
OUTPUT_DTYPE = "uint16"

BLOCK_ROWS = 512                   # rows read / corrected at a time in prnu_dsnu.py

BAND_IDS = {"R": 0, "G": 1, "B": 2}
BANDS = ("R", "G", "B")

RADIANCE_UNIT = "mW/cm2/sr/um"

TIFF_EXT = (".tif", ".tiff")

# Demosaicing (demosiac.py)
BAYER_PATTERN = "RGGB"             # sensor CFA order of the top-left 2x2 (same sensor on both payloads)
DEMOSAIC_ALGORITHM = "ea"          # "ea" = OpenCV edge-aware, "bilinear" = plain bilinear
DEMOSAIC_BORDER_FIX = False        # replicate inner pixels into the outer DEMOSAIC_BORDER_PX ring
DEMOSAIC_BORDER_PX = 2
DEMOSAIC_FLOAT_SCALE = None        # float input x scale -> uint16; None => 65535 if max <= 1.0, else 1.0
DEMOSAIC_DIR_PREFIX = "demosaiced_rgb_cv2_"   # output: <input>/<prefix><algorithm>
CV2_INPUT_MAX = 65535              # cv2 Bayer conversion takes uint8 / uint16 only

# OpenCV names Bayer patterns from a different starting pixel than the usual
# sensor convention, so the names look "wrong": an RGGB sensor is BayerBG in OpenCV.
# Values are cv2 attribute names (cv2.COLOR_...).
DEMOSAIC_CV2_CODES = {
    "ea": {
        "RGGB"          : "COLOR_BayerBG2RGB_EA",
        "BGGR"          : "COLOR_BayerRG2RGB_EA",
        "GRBG"          : "COLOR_BayerGB2RGB_EA",
        "GBRG"          : "COLOR_BayerGR2RGB_EA",
    },
    "bilinear": {
        "RGGB"          : "COLOR_BayerBG2RGB",
        "BGGR"          : "COLOR_BayerRG2RGB",
        "GRBG"          : "COLOR_BayerGB2RGB",
        "GBRG"          : "COLOR_BayerGR2RGB",
    },
}
BAYER_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")

# ALPHA payload
ALPHA = {
    "satellite_name"    : "ALPHA",
    "exposure_us"       : 1562,
    "sensor_width"      : 13468,
    "sensor_height"     : 9564,
    "usable_window"     : None,   # None => full sensor frame

    "dark_frame_path"   : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/alpha/@1562/darkframes_average.tiff",

    "gain_path": {
        "R"             : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/alpha/@1562/A_Red/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_R_gain.tif",
        "G"             : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/alpha/@1562/A_Green/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_G_gain.tif",
        "B"             : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/alpha/@1562/A_Blue/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_B_gain.tif",
    },
    "offset_path": {
        "R"             : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/alpha/@1562/A_Red/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_R_offset.tif",
        "G"             : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/alpha/@1562/A_Green/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_G_offset.tif",
        "B"             : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/alpha/@1562/A_Blue/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_B_offset.tif",
    },

    "saturation_radiance": {
        "R"             : 40.5088,
        "G"             : 13.9273,
        "B"             : 19.3128,
    },
}

# BRAVO payload
BRAVO = {
    "satellite_name"            : "BRAVO",
    "exposure_us"               : 2000,
    "sensor_width"              : 13468,
    "sensor_height"             : 9564,
    "usable_window"             : (3038, 10429, 1898, 7666),  # (xmin, xmax, ymin, ymax)

    "dark_frame_path"           : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/bravo/@2000/dark_frame_average.tiff",

    "gain_path": {
        "R"                     : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/bravo/@2000/B_Red/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_R_gain.tif",
        "G"                     : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/bravo/@2000/B_Green/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_G_gain.tif",
        "B"                     : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/bravo/@2000/B_Blue/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_B_gain.tif",
    },
    "offset_path": {
        "R"                     : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/bravo/@2000/B_Red/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_R_offset.tif",
        "G"                     : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/bravo/@2000/B_Green/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_G_offset.tif",
        "B"                     : "/media/tarun/DC7A70CB7A70A3C4/flat_fields/bravo/@2000/B_Blue/Without_Filter_Flat_Fields/Flat_Field_Plots/Channel_B_offset.tif",
    },

    "saturation_radiance": {
        "R"                     : 49.7373,
        "G"                     : 17.1794,
        "B"                     : 26.6689,
    },
}


SATELLITE_CONFIG_BY_NAME = {
    ALPHA["satellite_name"]: ALPHA,
    BRAVO["satellite_name"]: BRAVO,
}


def get_config_by_satellite_name(satellite_name):
    key = str(satellite_name).strip().upper()
    if key not in SATELLITE_CONFIG_BY_NAME:
        raise ValueError(f"No configuration found for satellite_name={satellite_name!r}")
    return SATELLITE_CONFIG_BY_NAME[key]
