"""
Configuration for DSNU / PRNU Correction (prnu_dsnu.py).
One dict per payload (ALPHA / BRAVO), looked up by satellite_name.
"""

# Global / shared parameters
ADC_BIT_DEPTH = 14
SATURATION_DN = 16380
OUTPUT_DN_MAX = (2 ** ADC_BIT_DEPTH) - 1   # 16383
OUTPUT_DTYPE = "uint16"

BLOCK_ROWS = 32
WORKERS = 4
RAM_MARGIN_BYTES = 2 * 1024 ** 3

BAND_IDS = {"R": 0, "G": 1, "B": 2}
BANDS = ("R", "G", "B")

RADIANCE_UNIT = "mW/cm2/sr/um"

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
        "R"             : 152.34,
        "G"             : 51.70,
        "B"             : 71.19,
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
        "R"                     : 185.2881,
        "G"                     : 66.4324,
        "B"                     : 96.18,
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
