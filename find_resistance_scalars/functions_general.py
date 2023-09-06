# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 12/07/2023.

"""
Useful functions and lists file for all the scripts in this folder.
"""

import numpy as np
import time

resistance_scalars_names = ["X11A", "X12A", "X21A", "X22A",
                            "Y11A", "Y12A", "Y21A", "Y22A",
                            "Y11B", "Y12B", "Y21B", "Y22B",
                            "X11C", "X12C", "X21C", "X22C",
                            "Y11C", "Y12C", "Y21C", "Y22C",
                            "X11G", "X12G", "X21G", "X22G",
                            "Y11G", "Y12G", "Y21G", "Y22G",
                            "Y11H", "Y12H", "Y21H", "Y22H",
                            "X11M", "X12M", "X21M", "X22M",
                            "Y11M", "Y12M", "Y21M", "Y22M",
                            "Z11M", "Z12M", "Z21M", "Z22M"]
mobility_scalars_names = ["x11a", "x12a", "x21a", "x22a",
                          "y11a", "y12a", "y21a", "y22a",
                          "y11b", "y12b", "y21b", "y22b",
                          "x11c", "x12c", "x21c", "x22c",
                          "y11c", "y12c", "y21c", "y22c",
                          "x11g", "x12g", "x21g", "x22g",
                          "y11g", "y12g", "y21g", "y22g",
                          "y11h", "y12h", "y21h", "y22h",
                          "x11m", "x12m", "x21m", "x22m",
                          "y11m", "y12m", "y21m", "y22m",
                          "z11m", "z12m", "z21m", "z22m"]
general_resistance_scalars_names = ["XA", "YA", "YB", "XC", "YC",
                                    "XG", "YG", "YH",
                                    "XM", "YM", "ZM"]
general_mobility_scalars_names = ["xa", "ya", "yb", "xc", "yc",
                                  "xg", "yg", "yh",
                                  "xm", "ym", "zm"]


def save_human_table(filename, description, elapsed_time_hms, firstrow, data):
    """Save a table of data in a human-readable form."""
    with open(filename, 'a') as outputfile:
        heading = (description + ", generated "
                   + time.strftime("%d/%m/%Y %H:%M:%S")
                   + ". Time to generate " + elapsed_time_hms)
        np.savetxt(outputfile, np.array([heading]), fmt="%s")
        np.savetxt(outputfile,
                   firstrow,
                   newline=" ", fmt="%15s")
        outputfile.write("\n")
        np.savetxt(outputfile, data, newline="\n", fmt="% .8e")


def format_seconds(elapsed_time):
    """Return a number of seconds in the form 0h00m00s."""
    et_m, et_s = divmod(elapsed_time, 60)
    et_h, et_m = divmod(et_m, 60)
    return "%dh%02dm%02ds" % (et_h, et_m, et_s)
