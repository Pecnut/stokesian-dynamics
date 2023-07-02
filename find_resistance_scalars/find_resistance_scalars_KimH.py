# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 31/03/2014.
'''
Find resistance scalars for (s,lambda) as read in from a text file.

Inputs: None directly, but inputs values of s and lambda from
        'values_of_s_midfield.txt' and 'values_of_lambda.txt'.
Outputs: Calls find_resistance_scalars(s,lam), which returns mobility scalars and
         resistance scalars in an array. So this takes this info and writes it to
         'resistance_scalars_human_readable.txt',
         'resistance_scalars_computer_readable.txt',
         'mobility_scalars_human_readable.txt',
         'mobility_scalars_computer_readable.txt',
'''
import subprocess
import numpy as np
from numpy import linalg, sqrt
import time
import sys


def run_fortran(s_dash, lam, case):
    '''Run Helen's code directly.
    Inputs: (s', lambda, case [as "F11", "F12" etc])
    Outputs: UOE as an array'''
    s = 0.5 * (1 + lam) * s_dash
    print("  using s = " + str(s) + " and case = " + case)
    if sys.platform == "win32":
        fortran_code_directory = "helen_fortran\\"
        starter = ""
    else:
        fortran_code_directory = "helen_fortran/"
        starter = "./"
    command_to_run = starter + "output-mac.exe " + str(s) + " 1 " + str(lam) + " " + case + " short"
    output_text = subprocess.check_output(fortran_code_directory + command_to_run, shell=True)
    return output_text.split()


def find_resistance_scalars(s_dash, lam):
    '''Use Helen's code to find all resistance scalars.
    Inputs: (s', lambda)
    Outputs: Returns (mobility_scalars, resistance_scalars) as arrays (X11A, X12A, ...) etc
             and outputs to screen confirmation and timing of each computation for each (s,lambda) pair.
    '''
    start_time = time.time()

    # Run Helen's Fortran code ---------------------------------------------------
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

    # Reorder output -------------------------------------------------------------
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

    # Extract mobility scalars from output ---------------------------------------

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

    mobility_scalars = np.array([x11a, x12a, x21a, x22a,
                                 y11a, y12a, y21a, y22a,
                                 y11b, y12b, y21b, y22b,
                                 x11c, x12c, x21c, x22c,
                                 y11c, y12c, y21c, y22c,
                                 x11g, x12g, x21g, x22g,
                                 y11g, y12g, y21g, y22g,
                                 y11h, y12h, y21h, y22h,
                                 x11m, x12m, x21m, x22m,
                                 y11m, y12m, y21m, y22m,
                                 z11m, z12m, z21m, z22m])
    '''
    # Output to text file --------------------------------------------------------
    # This isn't read in again, it's just for human interest.

    outputline = np.append([s, lam], mobility_scalars)
    with open('mobility_scalars_human_readable.txt','a') as outputfile:
        np.savetxt(outputfile, outputline, newline=" ", fmt="% .8e")
        outputfile.write("\n")
    '''

    # Invert mobility scalars to get resistance scalars --------------------------
    matrix_xc = np.array([[x11c, x12c], [x21c, x22c]])
    matrix_xa = np.array([[x11a, x12a], [x21a, x22a]])
    matrix_xg = np.array([[x11g, x12g], [x21g, x22g]])
    matrix_xgt = np.array([[x11g, x21g], [x12g, x22g]])
    matrix_xm = np.array([[x11m, x12m], [x21m, x22m]])
    matrix_yabc = np.array([[y11a, y12a, y11b, y21b], [y12a, y22a, y12b, y22b], [y11b, y12b, y11c, y12c], [y21b, y22b, y12c, y22c]])
    matrix_ygh = np.array([[y11g, y21g], [y12g, y22g], [-y11h, -y21h], [-y12h, -y22h]])
    matrix_ygt = np.array([[y11g, y21g], [y12g, y22g]])
    matrix_yht = np.array([[y11h, y21h], [y12h, y22h]])
    matrix_ym = np.array([[y11m, y12m], [y21m, y22m]])

    matrix_XA = linalg.inv(matrix_xa)
    matrix_XC = linalg.inv(matrix_xc)
    matrix_XG = np.dot(matrix_xg, matrix_XA)
    matrix_YABC = linalg.inv(matrix_yabc)
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

    matrix_XM = matrix_xm + (2. / 3.) * np.dot(matrix_XG, matrix_xgt)
    matrix_YM = matrix_ym + 2. * (np.dot(matrix_YG, matrix_ygt) + np.dot(matrix_YH, matrix_yht))

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

    resistance_scalars = np.array([X11A, X12A, X21A, X22A,
                                   Y11A, Y12A, Y21A, Y22A,
                                   Y11B, Y12B, Y21B, Y22B,
                                   X11C, X12C, X21C, X22C,
                                   Y11C, Y12C, Y21C, Y22C,
                                   X11G, X12G, X21G, X22G,
                                   Y11G, Y12G, Y21G, Y22G,
                                   Y11H, Y12H, Y21H, Y22H,
                                   X11M, X12M, X21M, X22M,
                                   Y11M, Y12M, Y21M, Y22M,
                                   Z11M, Z12M, Z21M, Z22M])

    elapsed_time = time.time() - start_time
    print("[s'=" + str('{0:.2f}'.format(s_dash)) + "][lam=" + str('{0:.2f}'.format(lam)) + "][time=" + str('{0:.1f}'.format(elapsed_time)) + "s]")

    return np.array([mobility_scalars, resistance_scalars])


looper_start_time = time.time()

resistance_scalars_names = np.array(["X11A", "X12A", "X21A", "X22A",
                                     "Y11A", "Y12A", "Y21A", "Y22A",
                                     "Y11B", "Y12B", "Y21B", "Y22B",
                                     "X11C", "X12C", "X21C", "X22C",
                                     "Y11C", "Y12C", "Y21C", "Y22C",
                                     "X11G", "X12G", "X21G", "X22G",
                                     "Y11G", "Y12G", "Y21G", "Y22G",
                                     "Y11H", "Y12H", "Y21H", "Y22H",
                                     "X11M", "X12M", "X21M", "X22M",
                                     "Y11M", "Y12M", "Y21M", "Y22M",
                                     "Z11M", "Z12M", "Z21M", "Z22M"])
mobility_scalars_names = np.array(["x11a", "x12a", "x21a", "x22a",
                                   "y11a", "y12a", "y21a", "y22a",
                                   "y11b", "y12b", "y21b", "y22b",
                                   "x11c", "x12c", "x21c", "x22c",
                                   "y11c", "y12c", "y21c", "y22c",
                                   "x11g", "x12g", "x21g", "x22g",
                                   "y11g", "y12g", "y21g", "y22g",
                                   "y11h", "y12h", "y21h", "y22h",
                                   "x11m", "x12m", "x21m",  "x22m",
                                   "y11m", "y12m", "y21m", "y22m",
                                   "z11m", "z12m", "z21m", "z22m"])
general_resistance_scalars_names = np.array(["XA", "YA", "YB", "XC", "YC", "XG", "YG", "YH", "XM", "YM", "ZM"])
general_mobility_scalars_names = np.array(["xa", "ya", "yb", "xc", "yc", "xg", "yg", "yh", "xm", "ym", "zm"])

# Initialise variables
s_dash_range = np.loadtxt('values_of_s_dash_midfield.txt')
lam_range = np.loadtxt('values_of_lambda.txt')
if lam_range.shape == tuple():
    lam_length = 1
    lam_range = np.array([lam_range + 0])
else:
    lam_length = lam_range.shape[0]
if s_dash_range.shape == tuple():
    s_dash_length = 1
    s_dash_range = np.array([s_dash_range + 0])
else:
    s_dash_length = s_dash_range.shape[0]

scalars_length = resistance_scalars_names.shape[0]
general_scalars_length = general_resistance_scalars_names.shape[0]
xyz_table = np.zeros((scalars_length, s_dash_length, lam_length))
XYZ_table = np.zeros((scalars_length, s_dash_length, lam_length))
xyz_human = np.zeros((s_dash_length * lam_length, scalars_length + 2))
XYZ_human = np.zeros((s_dash_length * lam_length, scalars_length + 2))

# Now run loop
for lam in lam_range:
    for s_dash in s_dash_range:
        print("Running s' = ", str(s_dash), ", lambda = " + str(lam) + " ...")
        s_dash_index = np.argwhere(s_dash_range == s_dash)[0, 0]
        lam_index = np.argwhere(lam_range == lam)[0, 0]
        mob_res_output = find_resistance_scalars(s_dash, lam)    # It appears that if lambda > 1, Helen's Fortran code breaks.
        mobility_scalars = mob_res_output[0]
        resistance_scalars = mob_res_output[1]
        xyz_table[:, s_dash_index, lam_index] = mobility_scalars
        XYZ_table[:, s_dash_index, lam_index] = resistance_scalars

# Now we have XYZ_table, xyz_table, generate more useful XYZ_short_table/xyz_short_table
''' The XYZ_table/xyz_table give us for (s,lam) the x11a x12a etc. But we need
    x1a, x2a etc.'''
lam_range_with_reciprocals = lam_range
for lam in lam_range:
    if (1. / lam) not in lam_range_with_reciprocals:
        lam_range_with_reciprocals = np.append(lam_range_with_reciprocals, (1. / lam))
lam_range_with_reciprocals.sort()
lam_wr_length = lam_range_with_reciprocals.shape[0]
xyz_general_table = np.zeros((general_scalars_length, 2, s_dash_length, lam_wr_length))
XYZ_general_table = np.zeros((general_scalars_length, 2, s_dash_length, lam_wr_length))
xyz_general_human = np.zeros((s_dash_length * lam_wr_length * 2, general_scalars_length + 3))
XYZ_general_human = np.zeros((s_dash_length * lam_wr_length * 2, general_scalars_length + 3))
# minus_in_B_and_G: see the Note on notation in my writeup. This represents 'k', which is -1 for B and G, and 1 otherwise.
minus_in_B_and_G_one_line = np.array([1,  1, -1,  1,  1, -1, -1,  1,  1,  1,  1])
minus_in_B_and_G = np.tile(minus_in_B_and_G_one_line, (s_dash_length, 1)).transpose()

for lam in lam_range_with_reciprocals:
    lam_wr_index = np.argwhere(lam_range_with_reciprocals == lam)[0, 0]
    if lam in lam_range:  # x11a, x12a etc
        lam_index = np.argwhere(lam_range == lam)[0, 0]
        xyz_general_table[:, 0, :, lam_wr_index] = xyz_table[(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40), :, lam_index]
        xyz_general_table[:, 1, :, lam_wr_index] = xyz_table[(1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41), :, lam_index]
        XYZ_general_table[:, 0, :, lam_wr_index] = XYZ_table[(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40), :, lam_index]
        XYZ_general_table[:, 1, :, lam_wr_index] = XYZ_table[(1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41), :, lam_index]
    else:  # x21a, x22a etc
        lam_index = np.argwhere(lam_range == (1 / lam))[0, 0]
        xyz_general_table[:, 0, :, lam_wr_index] = xyz_table[(3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43), :, lam_index] * minus_in_B_and_G
        xyz_general_table[:, 1, :, lam_wr_index] = xyz_table[(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42), :, lam_index] * minus_in_B_and_G
        XYZ_general_table[:, 0, :, lam_wr_index] = XYZ_table[(3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43), :, lam_index] * minus_in_B_and_G
        XYZ_general_table[:, 1, :, lam_wr_index] = XYZ_table[(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42), :, lam_index] * minus_in_B_and_G
# Time elapsed
looper_elapsed_time = time.time() - looper_start_time
let_m, let_s = divmod(looper_elapsed_time, 60)
let_h, let_m = divmod(let_m, 60)
looper_elapsed_time_hms = "%dh%02dm%02ds" % (let_h, let_m, let_s)
print("Time elapsed " + looper_elapsed_time_hms)

# Write XYZ_table and xyz_table to file (computer readable)
with open('scalars_pairs_resistance_blob_midfield.txt', 'wb') as outputfile:
    np.save(outputfile, XYZ_table)
with open('scalars_pairs_mobility_blob_midfield.txt', 'wb') as outputfile:
    np.save(outputfile, xyz_table)
with open('scalars_general_resistance_blob_midfield.txt', 'wb') as outputfile:
    np.save(outputfile, XYZ_general_table)
with open('scalars_general_mobility_blob_midfield.txt', 'wb') as outputfile:
    np.save(outputfile, xyz_general_table)

# Write XYZ_table and xyz_table to file (human readable)
for s_dash in s_dash_range:
    s_dash_index = np.argwhere(s_dash_range == s_dash)[0, 0]
    s_dash_length = s_dash_range.shape[0]
    lam_length = lam_range.shape[0]
    lam_wr_length = lam_range_with_reciprocals.shape[0]
    for lam in lam_range:
        lam_index = np.argwhere(lam_range == lam)[0, 0]
        xyz_outputline = np.append([s_dash, lam], xyz_table[:, s_dash_index, lam_index])
        xyz_human[lam_index * s_dash_length + s_dash_index, :] = xyz_outputline
        XYZ_outputline = np.append([s_dash, lam], XYZ_table[:, s_dash_index, lam_index])
        XYZ_human[lam_index * s_dash_length + s_dash_index, :] = XYZ_outputline
    for lam in lam_range_with_reciprocals:
        lam_wr_index = np.argwhere(lam_range_with_reciprocals == lam)[0, 0]
        for gam in range(2):
            xyz_outputline = np.append([s_dash, lam, gam], xyz_general_table[:, gam, s_dash_index, lam_wr_index])
            xyz_general_human[(lam_wr_index * s_dash_length + s_dash_index) * 2 + gam, :] = xyz_outputline
            XYZ_outputline = np.append([s_dash, lam, gam], XYZ_general_table[:, gam, s_dash_index, lam_wr_index])
            XYZ_general_human[(lam_wr_index * s_dash_length + s_dash_index) * 2 + gam, :] = XYZ_outputline


with open('scalars_pairs_resistance_text_midfield.txt', 'a') as outputfile:
    heading = "Resistance scalars, generated " + time.strftime("%d/%m/%Y %H:%M:%S") + ". Time to generate " + looper_elapsed_time_hms
    np.savetxt(outputfile, np.array([heading]), fmt="%s")
    np.savetxt(outputfile, np.append(["s'", "lambda"], resistance_scalars_names), newline=" ", fmt="%15s")
    outputfile.write("\n")
    np.savetxt(outputfile, XYZ_human, newline="\n", fmt="% .8e")
with open('scalars_pairs_mobility_text_midfield.txt', 'a') as outputfile:
    heading = "Mobility scalars, generated " + time.strftime("%d/%m/%Y %H:%M:%S") + ". Time to generate " + looper_elapsed_time_hms
    np.savetxt(outputfile, np.array([heading]), fmt="%s")
    np.savetxt(outputfile, np.append(["s'", "lambda"], mobility_scalars_names), newline=" ", fmt="%15s")
    outputfile.write("\n")
    np.savetxt(outputfile, xyz_human, newline="\n", fmt="% .8e")
with open('scalars_general_resistance_text_midfield.txt', 'a') as outputfile:
    heading = "Resistance scalars, generated " + time.strftime("%d/%m/%Y %H:%M:%S") + ". Time to generate " + looper_elapsed_time_hms
    np.savetxt(outputfile, np.array([heading]), fmt="%s")
    np.savetxt(outputfile, np.append(["s'", "lambda", "gamma"], general_resistance_scalars_names), newline=" ", fmt="%15s")
    outputfile.write("\n")
    np.savetxt(outputfile, XYZ_general_human, newline="\n", fmt="% .8e")
with open('scalars_general_mobility_text_midfield.txt', 'a') as outputfile:
    heading = "Mobility scalars, generated " + time.strftime("%d/%m/%Y %H:%M:%S") + ". Time to generate " + looper_elapsed_time_hms
    np.savetxt(outputfile, np.array([heading]), fmt="%s")
    np.savetxt(outputfile, np.append(["s'", "lambda", "gamma"], general_mobility_scalars_names), newline=" ", fmt="%15s")
    outputfile.write("\n")
    np.savetxt(outputfile, xyz_general_human, newline="\n", fmt="% .8e")
    # newline=" " says use a space instead of a newline in between array elements
    # fmt="% .8e" says ( )  use either a space or a minus sign for good spacing
    #                  (.8) use 8 digits of precision
    #                  (e)  use exponential form
