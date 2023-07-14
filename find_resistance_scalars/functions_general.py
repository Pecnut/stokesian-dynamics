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
                          "x11m", "x12m", "x21m",  "x22m",
                          "y11m", "y12m", "y21m", "y22m",
                          "z11m", "z12m", "z21m", "z22m"]
general_resistance_scalars_names = ["XA", "YA", "YB", "XC", "YC",
                                    "XG", "YG", "YH",
                                    "XM", "YM", "ZM"]
general_mobility_scalars_names = ["xa", "ya", "yb", "xc", "yc",
                                  "xg", "yg", "yh",
                                  "xm", "ym", "zm"]


def save_human_table(filename, description, elapsed_time_hms, firstrow, data):
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
    et_m, et_s = divmod(elapsed_time, 60)
    et_h, et_m = divmod(et_m, 60)
    return "%dh%02dm%02ds" % (et_h, et_m, et_s)


def convert_mobility_to_resistance(mobility_scalars):
    [x11a, x12a, x21a, x22a,
     y11a, y12a, y21a, y22a,
     y11b, y12b, y21b, y22b,
     x11c, x12c, x21c, x22c,
     y11c, y12c, y21c, y22c,
     x11g, x12g, x21g, x22g,
     y11g, y12g, y21g, y22g,
     y11h, y12h, y21h, y22h,
     x11m, x12m, x21m, x22m,
     y11m, y12m, y21m, y22m,
     z11m, z12m, z21m, z22m] = mobility_scalars

    # Invert mobility scalars to get resistance scalars -----------------------
    matrix_xc = np.array([[x11c, x12c], [x21c, x22c]])
    matrix_xa = np.array([[x11a, x12a], [x21a, x22a]])
    matrix_xg = np.array([[x11g, x12g], [x21g, x22g]])
    matrix_xgt = np.array([[x11g, x21g], [x12g, x22g]])
    matrix_xm = np.array([[x11m, x12m], [x21m, x22m]])
    matrix_yabc = np.array([[y11a, y12a, y11b, y21b],
                            [y12a, y22a, y12b, y22b],
                            [y11b, y12b, y11c, y12c],
                            [y21b, y22b, y12c, y22c]])
    matrix_ygh = np.array([[y11g, y21g], [y12g, y22g],
                           [-y11h, -y21h], [-y12h, -y22h]])
    matrix_ygt = np.array([[y11g, y21g], [y12g, y22g]])
    matrix_yht = np.array([[y11h, y21h], [y12h, y22h]])
    matrix_ym = np.array([[y11m, y12m], [y21m, y22m]])

    matrix_XA = np.linalg.inv(matrix_xa)
    matrix_XC = np.linalg.inv(matrix_xc)
    matrix_XG = np.dot(matrix_xg, matrix_XA)
    matrix_YABC = np.linalg.inv(matrix_yabc)
    matrix_YGH = np.dot(matrix_YABC, matrix_ygh)

    X11A = matrix_XA[0, 0]
    X12A = matrix_XA[0, 1]
    X21A = matrix_XA[1, 0]
    X22A = matrix_XA[1, 1]

    X11C = matrix_XC[0, 0]
    X12C = matrix_XC[0, 1]
    X21C = matrix_XC[1, 0]
    X22C = matrix_XC[1, 1]

    X11G = matrix_XG[0, 0]
    X12G = matrix_XG[0, 1]
    X21G = matrix_XG[1, 0]
    X22G = matrix_XG[1, 1]

    Y11A = matrix_YABC[0, 0]
    Y12A = matrix_YABC[0, 1]
    Y21A = matrix_YABC[1, 0]
    Y22A = matrix_YABC[1, 1]
    Y11B = matrix_YABC[0, 2]
    Y21B = matrix_YABC[0, 3]
    Y12B = matrix_YABC[1, 2]
    Y22B = matrix_YABC[1, 3]
    Y11C = matrix_YABC[2, 2]
    Y12C = matrix_YABC[2, 3]
    Y21C = matrix_YABC[3, 2]
    Y22C = matrix_YABC[3, 3]

    Y11G = matrix_YGH[0, 0]
    Y21G = matrix_YGH[0, 1]
    Y12G = matrix_YGH[1, 0]
    Y22G = matrix_YGH[1, 1]
    Y11H = -matrix_YGH[2, 0]
    Y21H = -matrix_YGH[2, 1]
    Y12H = -matrix_YGH[3, 0]
    Y22H = -matrix_YGH[3, 1]

    matrix_YG = np.array([[Y11G, Y12G], [Y21G, Y22G]])
    matrix_YH = np.array([[Y11H, Y12H], [Y21H, Y22H]])

    matrix_XM = matrix_xm + (2/3)*np.dot(matrix_XG, matrix_xgt)
    matrix_YM = matrix_ym + 2*(np.dot(matrix_YG, matrix_ygt)
                               + np.dot(matrix_YH, matrix_yht))

    X11M = matrix_XM[0, 0]
    X12M = matrix_XM[0, 1]
    X21M = matrix_XM[1, 0]
    X22M = matrix_XM[1, 1]

    Y11M = matrix_YM[0, 0]
    Y12M = matrix_YM[0, 1]
    Y21M = matrix_YM[1, 0]
    Y22M = matrix_YM[1, 1]

    Z11M = z11m
    Z12M = z12m
    Z21M = z21m
    Z22M = z22m

    resistance_scalars = [X11A, X12A, X21A, X22A,
                          Y11A, Y12A, Y21A, Y22A,
                          Y11B, Y12B, Y21B, Y22B,
                          X11C, X12C, X21C, X22C,
                          Y11C, Y12C, Y21C, Y22C,
                          X11G, X12G, X21G, X22G,
                          Y11G, Y12G, Y21G, Y22G,
                          Y11H, Y12H, Y21H, Y22H,
                          X11M, X12M, X21M, X22M,
                          Y11M, Y12M, Y21M, Y22M,
                          Z11M, Z12M, Z21M, Z22M]
    return resistance_scalars
