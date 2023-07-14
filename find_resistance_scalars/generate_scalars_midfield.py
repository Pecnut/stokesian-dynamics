# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 31/03/2014.
"""
Standalone script to generate nondimensional midfield resistance scalars 
X11A, X12A, Y11B, ..., Z12M  for values of (s,lambda) read in from a text file.

Calls find_resistance_scalars(s,lam), which returns mobility scalars and
resistance scalars in an array.

Reads:
    values_of_s_dash_midfield.txt: List of midfield distances s'.
    values_of_lambda.txt: List of size ratios lambda.

Writes:
    scalars_pairs_resistance_midfield.npy  and  .txt.
    scalars_pairs_mobility_midfield.npy  and  .txt.
    scalars_general_resistance_midfield.npy  and  .txt.
    scalars_general_mobility_midfield.npy  and .txt.
"""
import subprocess
import numpy as np
from numpy import linalg, sqrt
from functions_general import (resistance_scalars_names,
                               general_resistance_scalars_names,
                               mobility_scalars_names,
                               general_mobility_scalars_names,
                               convert_mobility_to_resistance,
                               format_seconds, save_human_table)
import time
import sys


def run_fortran(s_dash, lam, case):
    """Run Helen's code directly.

    Args:
        s_dash: Scaled separation distance s'.
        lam: Size ratios lambda.
        case: "F11", "F12" etc.

    Returns:
        UOE as an array
    """
    s = 0.5 * (1 + lam) * s_dash
    print("  using s = " + str(s) + " and case = " + case)
    if sys.platform == "win32":
        fortran_code_directory = "helen_fortran\\"
        starter = ""
    else:
        fortran_code_directory = "helen_fortran/"
        starter = "./"
    command_to_run = (starter + "lamb.exe " + str(s) + " 1 " + str(lam) + " "
                      + case + " short")
    output_text = subprocess.check_output(
        fortran_code_directory + command_to_run, shell=True)
    return output_text.split()


def find_resistance_scalars(s_dash, lam):
    """Use Helen's code to find all resistance scalars.

    Args:
        s_dash: Scaled separation distance s'.
        lam: Size ratio lambda.

    Returns:
        (mobility_scalars, resistance_scalars) as arrays (X11A, X12A, ...) etc.
        and outputs to screen confirmation and timing of each computation for
        each (s,lambda) pair.
    """
    start_time = time.time()

    # Run Helen's Fortran code ------------------------------------------------
    UOS_output_text = np.array([run_fortran(s_dash, lam, "F11"),
                                run_fortran(s_dash, lam, "F12"),
                                run_fortran(s_dash, lam, "F21"),
                                run_fortran(s_dash, lam, "F22"),
                                run_fortran(s_dash, lam, "T11"),
                                run_fortran(s_dash, lam, "T12"),
                                run_fortran(s_dash, lam, "T21"),
                                run_fortran(s_dash, lam, "T22"),
                                run_fortran(s_dash, lam, "E11"),
                                run_fortran(s_dash, lam, "E13"),
                                run_fortran(s_dash, lam, "E14"),
                                run_fortran(s_dash, lam, "E21"),
                                run_fortran(s_dash, lam, "E23"),
                                run_fortran(s_dash, lam, "E24")])
    UOS_output = UOS_output_text.astype(float)

    # Reorder output ----------------------------------------------------------
    rp = 0.5 * (1 + np.sqrt(3))
    rm = 0.5 * (-1 + np.sqrt(3))
    r2 = np.sqrt(2)
    reorder = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, rp, rm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, r2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, rm, rp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, r2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, rp, rm, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, rm, rp, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r2]])

    reordered_UOS = np.dot(reorder, UOS_output.transpose()).transpose()

    minus1 = -1

    np.set_printoptions(linewidth=600)

    # Extract mobility scalars from output ------------------------------------

    x11a = reordered_UOS[0, 0]
    x12a = reordered_UOS[2, 0]
    x21a = reordered_UOS[0, 3]
    x22a = reordered_UOS[2, 3]

    y11a = reordered_UOS[1, 1]
    y12a = reordered_UOS[3, 1]
    y21a = reordered_UOS[1, 4]
    y22a = reordered_UOS[3, 4]

    y11b = reordered_UOS[1, 8] * -1
    y12b = reordered_UOS[3, 8] * -1
    y21b = reordered_UOS[1, 11] * -1
    y22b = reordered_UOS[3, 11] * -1

    x11c = reordered_UOS[4, 6]
    x12c = reordered_UOS[6, 6]
    x21c = reordered_UOS[4, 9]
    x22c = reordered_UOS[6, 9]

    y11c = reordered_UOS[5, 7]
    y12c = reordered_UOS[7, 7]
    y21c = reordered_UOS[5, 10]
    y22c = reordered_UOS[7, 10]

    x11g = reordered_UOS[0, 12] / ((3 + sqrt(3)) / 6) * minus1
    x12g = reordered_UOS[2, 12] / ((3 + sqrt(3)) / 6) * minus1
    x21g = reordered_UOS[0, 17] / ((3 + sqrt(3)) / 6) * minus1
    x22g = reordered_UOS[2, 17] / ((3 + sqrt(3)) / 6) * minus1

    y11g = reordered_UOS[1, 13] / (sqrt(2))
    y12g = reordered_UOS[3, 13] / (sqrt(2))
    y21g = reordered_UOS[1, 18] / (sqrt(2))
    y22g = reordered_UOS[3, 18] / (sqrt(2))

    y11h = reordered_UOS[5, 15] / (-sqrt(2))
    y12h = reordered_UOS[7, 15] / (-sqrt(2))
    y21h = reordered_UOS[5, 20] / (-sqrt(2))
    y22h = reordered_UOS[7, 20] / (-sqrt(2))

    x11m = reordered_UOS[8, 12] / ((3 + sqrt(3)) / 4)
    x12m = reordered_UOS[11, 12] / ((3 + sqrt(3)) / 4)
    x21m = reordered_UOS[8, 17] / ((3 + sqrt(3)) / 4)
    x22m = reordered_UOS[11, 17] / ((3 + sqrt(3)) / 4)

    y11m = reordered_UOS[10, 13] / sqrt(2)
    y12m = reordered_UOS[13, 13] / sqrt(2)
    y21m = reordered_UOS[10, 18] / sqrt(2)
    y22m = reordered_UOS[13, 18] / sqrt(2)

    z11m = reordered_UOS[9, 16] / sqrt(2)
    z12m = reordered_UOS[12, 16] / sqrt(2)
    z21m = reordered_UOS[9, 21] / sqrt(2)
    z22m = reordered_UOS[12, 21] / sqrt(2)

    # Scale to match the scaling on Minfinity
    # > scale a, b, c on 6pi,
    # > scale g, h    on 1,
    # > scale m       on 1/6pi.
    sc_a = sc_b = sc_c = 6 * np.pi
    sc_g = sc_h = 1
    sc_m = 1. / (6 * np.pi)
    #sc_a = sc_b = sc_c = sc_g = sc_h = sc_m = 1.0

    x11a = x11a * sc_a
    x12a = x12a * sc_a
    x21a = x21a * sc_a
    x22a = x22a * sc_a
    y11a = y11a * sc_a
    y12a = y12a * sc_a
    y21a = y21a * sc_a
    y22a = y22a * sc_a

    y11b = y11b * sc_b
    y12b = y12b * sc_b
    y21b = y21b * sc_b
    y22b = y22b * sc_b

    x11c = x11c * sc_c
    x12c = x12c * sc_c
    x21c = x21c * sc_c
    x22c = x22c * sc_c
    y11c = y11c * sc_c
    y12c = y12c * sc_c
    y21c = y21c * sc_c
    y22c = y22c * sc_c

    x11g = x11g * sc_g
    x12g = x12g * sc_g
    x21g = x21g * sc_g
    x22g = x22g * sc_g
    y11g = y11g * sc_g
    y12g = y12g * sc_g
    y21g = y21g * sc_g
    y22g = y22g * sc_g

    y11h = y11h * sc_h
    y12h = y12h * sc_h
    y21h = y21h * sc_h
    y22h = y22h * sc_h

    x11m = x11m * sc_m
    x12m = x12m * sc_m
    x21m = x21m * sc_m
    x22m = x22m * sc_m
    y11m = y11m * sc_m
    y12m = y12m * sc_m
    y21m = y21m * sc_m
    y22m = y22m * sc_m
    z11m = z11m * sc_m
    z12m = z12m * sc_m
    z21m = z21m * sc_m
    z22m = z22m * sc_m

    mobility_scalars = [x11a, x12a, x21a, x22a,
                        y11a, y12a, y21a, y22a,
                        y11b, y12b, y21b, y22b,
                        x11c, x12c, x21c, x22c,
                        y11c, y12c, y21c, y22c,
                        x11g, x12g, x21g, x22g,
                        y11g, y12g, y21g, y22g,
                        y11h, y12h, y21h, y22h,
                        x11m, x12m, x21m, x22m,
                        y11m, y12m, y21m, y22m,
                        z11m, z12m, z21m, z22m]

    resistance_scalars = convert_mobility_to_resistance(mobility_scalars)

    elapsed_time = time.time() - start_time
    print("[s'=" + str('{0:.2f}'.format(s_dash)) + "]"
          + "[lam=" + str('{0:.2f}'.format(lam)) + "]"
          + "[time=" + str('{0:.1f}'.format(elapsed_time)) + "s]")

    return np.array([mobility_scalars, resistance_scalars])


start_time = time.time()

# Initialise variables
s_dash_range = np.loadtxt('values_of_s_dash_midfield.txt', ndmin=1)
s_dash_length = s_dash_range.shape[0]
lam_range = np.loadtxt('values_of_lambda.txt', ndmin=1)
lam_length = lam_range.shape[0]

scalars_length = len(resistance_scalars_names)
general_scalars_length = len(general_resistance_scalars_names)
xyz_table = np.zeros((scalars_length, s_dash_length, lam_length))
XYZ_table = np.zeros((scalars_length, s_dash_length, lam_length))
xyz_human = np.zeros((s_dash_length * lam_length, scalars_length + 2))
XYZ_human = np.zeros((s_dash_length * lam_length, scalars_length + 2))

# Now run loop
for lam_index, lam in enumerate(lam_range):
    for s_dash_index, s_dash in enumerate(s_dash_range):
        print("Running s' = ", str(s_dash), ", lambda = " + str(lam) + " ...")
        # If lambda > 1, Helen's Fortran code breaks.
        mobility_scalars, resistance_scalars = find_resistance_scalars(s_dash, 
                                                                       lam)
        xyz_table[:, s_dash_index, lam_index] = mobility_scalars
        XYZ_table[:, s_dash_index, lam_index] = resistance_scalars

# The XYZ_table/xyz_table give us for (s,lam) the x11a x12a etc. But we need
# x1a, x2a etc.
lam_range_with_reciprocals = np.concatenate(
    (lam_range, [1/l for l in lam_range if 1/l not in lam_range]))
lam_range_with_reciprocals.sort()
lam_wr_length = lam_range_with_reciprocals.shape[0]
xyz_general_table = np.zeros(
    (general_scalars_length, 2, s_dash_length, lam_wr_length))
XYZ_general_table = np.zeros(
    (general_scalars_length, 2, s_dash_length, lam_wr_length))
xyz_general_human = np.zeros(
    (s_dash_length * lam_wr_length * 2, general_scalars_length + 3))
XYZ_general_human = np.zeros(
    (s_dash_length * lam_wr_length * 2, general_scalars_length + 3))
# minus_in_B_and_G: see the Note on notation in my writeup.
# This represents 'k', which is -1 for B and G, and 1 otherwise.
minus_in_B_and_G_one_line = np.array([1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1])
minus_in_B_and_G = np.tile(minus_in_B_and_G_one_line, (s_dash_length, 1)).T

for lam_wr_index, lam in enumerate(lam_range_with_reciprocals):
    if lam in lam_range:  # x11a, x12a etc
        lam_index = np.argwhere(lam_range == lam)[0, 0]
        xyz_general_table[:, 0, :, lam_wr_index] = xyz_table[
            (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40), :, lam_index]
        xyz_general_table[:, 1, :, lam_wr_index] = xyz_table[
            (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41), :, lam_index]
        XYZ_general_table[:, 0, :, lam_wr_index] = XYZ_table[
            (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40), :, lam_index]
        XYZ_general_table[:, 1, :, lam_wr_index] = XYZ_table[
            (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41), :, lam_index]
    else:  # x21a, x22a etc
        lam_index = np.argwhere(lam_range == (1 / lam))[0, 0]
        xyz_general_table[:, 0, :, lam_wr_index] = (xyz_table[
            (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43), :, lam_index]
            * minus_in_B_and_G)
        xyz_general_table[:, 1, :, lam_wr_index] = (xyz_table[
            (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42), :, lam_index]
            * minus_in_B_and_G)
        XYZ_general_table[:, 0, :, lam_wr_index] = (XYZ_table[
            (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43), :, lam_index]
            * minus_in_B_and_G)
        XYZ_general_table[:, 1, :, lam_wr_index] = (XYZ_table[
            (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42), :, lam_index]
            * minus_in_B_and_G)
# Time elapsed
elapsed_time = time.time() - start_time
elapsed_time_hms = format_seconds(elapsed_time)
print("Time elapsed " + elapsed_time_hms)

# Write XYZ_table and xyz_table to file (computer readable)
with open('scalars_pairs_resistance_midfield.npy', 'wb') as outputfile:
    np.save(outputfile, XYZ_table)
with open('scalars_pairs_mobility_midfield.npy', 'wb') as outputfile:
    np.save(outputfile, xyz_table)
with open('scalars_general_resistance_midfield.npy', 'wb') as outputfile:
    np.save(outputfile, XYZ_general_table)
with open('scalars_general_mobility_midfield.npy', 'wb') as outputfile:
    np.save(outputfile, xyz_general_table)

# Write XYZ_table and xyz_table to file (human readable)
for s_dash_index, s_dash in enumerate(s_dash_range):
    s_dash_length = s_dash_range.shape[0]
    lam_length = lam_range.shape[0]
    lam_wr_length = lam_range_with_reciprocals.shape[0]
    for lam_index, lam in enumerate(lam_range):
        xyz_outputline = np.append([s_dash, lam],
                                   xyz_table[:, s_dash_index, lam_index])
        xyz_human[lam_index*s_dash_length + s_dash_index, :] = xyz_outputline
        XYZ_outputline = np.append([s_dash, lam],
                                   XYZ_table[:, s_dash_index, lam_index])
        XYZ_human[lam_index*s_dash_length + s_dash_index, :] = XYZ_outputline
    for lam_wr_index, lam in enumerate(lam_range_with_reciprocals):
        for gam in range(2):
            i = (lam_wr_index*s_dash_length + s_dash_index)*2 + gam
            xyz_outputline = np.append(
                [s_dash, lam, gam],
                xyz_general_table[:, gam, s_dash_index, lam_wr_index])
            xyz_general_human[i, :] = xyz_outputline
            XYZ_outputline = np.append(
                [s_dash, lam, gam],
                XYZ_general_table[:, gam, s_dash_index, lam_wr_index])
            XYZ_general_human[i, :] = XYZ_outputline

save_human_table('scalars_pairs_resistance_midfield.txt', 
                 'Resistance scalars', 
                 elapsed_time_hms, 
                 np.append(["s'", "lambda"], resistance_scalars_names),
                 XYZ_human)

save_human_table('scalars_pairs_mobility_midfield.txt',
                 'Mobility scalars',
                 elapsed_time_hms,
                 np.append(["s'", "lambda"], mobility_scalars_names),
                 xyz_human)

save_human_table('scalars_general_resistance_midfield.txt',
                 'Resistance scalars',
                 elapsed_time_hms,
                 np.append(["s'", "lambda", "gamma"],
                           general_resistance_scalars_names),
                 XYZ_general_human)

save_human_table('scalars_general_mobility_midfield.txt',
                 'Mobility scalars',
                 elapsed_time_hms,
                 np.append(["s'", "lambda", "gamma"],
                           general_mobility_scalars_names),
                 xyz_general_human)
