# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 31/03/2014.
"""
Script to generate nondimensional midfield resistance scalars
X11A, X12A, Y11B, ..., Z12M  for values of (s,lambda) read in from a text file.

Requires Fortran files in /helen_fortran/ to be compiled first. See README.

References: Kim & Karilla, 2005. Microhydrodynamics. Section 7.2.
            Townsend, 2017. PhD thesis: The mechanics of suspensions.
                Sections 2.4.2-2.4.3 and A.1.2-A.1.3.

Reads:
    values_of_s_dash_midfield.txt: List of midfield distances s'.
    values_of_lambda.txt: List of size ratios lambda.

Writes:
    scalars_general_resistance_midfield.npy  and  .txt.
    scalars_general_mobility_midfield.npy  and .txt.
    and if desired:
        scalars_pairs_resistance_midfield.npy  and  .txt.
        scalars_pairs_mobility_midfield.npy  and  .txt.
"""
import subprocess
import numpy as np
from numpy import sqrt
from functions_general import (resistance_scalars_names,
                               general_resistance_scalars_names,
                               mobility_scalars_names,
                               general_mobility_scalars_names,
                               format_seconds, save_human_table)
import time
import sys


def run_fortran(s_dash, lam, case):
    """Run Helen's two-sphere Fortran code directly.

    The Fortran code solves the dimensional Kim & Karilla mobility problem,
        (U O S) = M (F T E),
    for given F, T and E, and sphere separation and size ratio. Works for
    lambda <=1  and  s_dash < ~2.014.

    Args:
        s_dash: Scaled separation distance s'.
        lam: Size ratios lambda.
        case: A code, "F11", "F12" etc., which is understood in `2sphere.f` to
            represent, e.g., "a force of 1 parallel to the separation vector".

    Returns:
        1D array (U O S) in dimensional form (i.e. U ~ 1/(6 pi a) F).
    """
    s = 0.5 * (1 + lam) * s_dash  # Dimensionalise the distance. Assumes a1=1.
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


def find_dim_kim_mobility_scalars(s_dash, lam):
    """Use the Fortran code in /helen_fortran/ to find all dimensional
    Kim & Karilla mobility scalars.

    The Fortran code solves the dimensional Kim & Karilla mobility problem,
        (U O S) = M (F T E),
    for given F, T and E, and sphere separation and size ratio.

    Sphere radii are assumed to be a_1 = 1, a_2 = lam. The Fortran code
    requires lam<=1.

    Args:
        s_dash: Scaled separation distance s'.
        lam: Size ratio lambda.

    Returns:
        mobility_scalars as array (x11a, x12a, ...) etc.
        and outputs to screen confirmation and timing of each computation for
        each (s,lambda) pair.
    """
    start_time = time.time()

    # Run Helen's Fortran code
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

    # Reorder output
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

    np.set_printoptions(linewidth=600)

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

    x11g = reordered_UOS[0, 12] / ((3 + sqrt(3)) / 6)
    x12g = reordered_UOS[2, 12] / ((3 + sqrt(3)) / 6)
    x21g = reordered_UOS[0, 17] / ((3 + sqrt(3)) / 6)
    x22g = reordered_UOS[2, 17] / ((3 + sqrt(3)) / 6)

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

    elapsed_time = time.time() - start_time
    print("[s'=" + str('{0:.2f}'.format(s_dash)) + "]"
          + "[lam=" + str('{0:.2f}'.format(lam)) + "]"
          + "[time=" + str('{0:.1f}'.format(elapsed_time)) + "s]")

    return np.array(mobility_scalars)


def convert_dim_kim_mobility_to_dim_resistance(mobility_scalars):
    """Convert a list of dimensional Kim & Karilla formulation mobility scalars
    to a list of dimensional resistance scalars.

    References: Kim & Karilla, 2005. Microhydrodynamics. Section 7.2.
                Townsend, 2017. PhD thesis: The mechanics of suspensions.
                    Sections A.1.2-A.1.3.

    The Kim & Karilla formulation of the mobility problem is:
        (U O S) = M (F T E)
    and the resistance problem is
        (F T S) = R (U O E).

    See PhD thesis section A.1.2 and A.1.3."""
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
    return np.array(resistance_scalars)


def nondimensionalise_scalars(scalars, scales):
    '''
    Convert dimensional resistance/mobility scalars into nondimensional
    resistance/mobility scalars.
    '''
    nd_scalars = scalars
    nd_scalars[:8] /= scales[0]  # A
    nd_scalars[8:12] /= scales[1]  # B
    nd_scalars[12:20] /= scales[2]  # C
    nd_scalars[20:28] /= scales[3]  # G
    nd_scalars[28:32] /= scales[4]  # H
    nd_scalars[32:] /= scales[5]  # M
    return nd_scalars


save_pairs_file_as_well = False
start_time = time.time()

# Initialise variables
s_dash_range = np.loadtxt('values_of_s_dash_midfield.txt', ndmin=1)
s_dash_length = s_dash_range.shape[0]
lam_range = np.loadtxt('values_of_lambda.txt', ndmin=1)
lam_length = lam_range.shape[0]

scalars_length = len(resistance_scalars_names)
general_scalars_length = len(general_resistance_scalars_names)
XYZ_table = np.zeros((scalars_length, s_dash_length, lam_length))
XYZ_human = np.zeros((s_dash_length * lam_length, scalars_length + 2))
xyz_table = np.zeros((scalars_length, s_dash_length, lam_length))
xyz_human = np.zeros((s_dash_length * lam_length, scalars_length + 2))

# Now run loop
# Nondimensionalising scales, where a1=1
resistance_scales = [6*np.pi, 4*np.pi, 8*np.pi, 4*np.pi, 8*np.pi, 20/3*np.pi]
mobility_scales = [1/(6*np.pi), 1/(4*np.pi), 1/(8*np.pi), 2, 1, 20/3*np.pi]

for lam_index, lam in enumerate(lam_range):
    for s_dash_index, s_dash in enumerate(s_dash_range):
        print("Running s' = ", str(s_dash), ", lambda = " + str(lam) + " ...")
        # If lambda > 1, Helen's Fortran code breaks.
        dim_mobility_scalars = find_dim_kim_mobility_scalars(s_dash, lam)
        dim_resistance_scalars = convert_dim_kim_mobility_to_dim_resistance(
            dim_mobility_scalars)
        nondim_mobility_scalars = nondimensionalise_scalars(
            dim_mobility_scalars, mobility_scales)
        nondim_resistance_scalars = nondimensionalise_scalars(
            dim_resistance_scalars, resistance_scales)
        XYZ_table[:, s_dash_index, lam_index] = nondim_resistance_scalars
        xyz_table[:, s_dash_index, lam_index] = nondim_mobility_scalars

# XYZ_table gives us for (s,lam) the values of X11A, X12A, X21A, etc. Since
#   X21A(1/lam) = X12A(lam),  (and similarly for other scalars modulo +/-),
# we only store _11_ and _12_ for different lams in XYZ_general_table, and call
# these _1_ and _2_ etc. See PhD thesis section 2.4.2 and A.2.1.

# The minus sign differences for 12/21 are only for B and G, and are
# represented by 'kappa' in the thesis, which is -1 for B and G, and 1
# otherwise. See Kim & Karilla, section 11.3, p. 278.
minus_in_B_and_G_one_line = np.array([1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1])
minus_in_B_and_G = np.tile(minus_in_B_and_G_one_line, (s_dash_length, 1)).T

lam_range_with_reciprocals = np.sort(np.concatenate(
    (lam_range, [1/l for l in lam_range if 1/l not in lam_range])))
lam_wr_length = lam_range_with_reciprocals.shape[0]
XYZ_general_table = np.zeros(
    (general_scalars_length, 2, s_dash_length, lam_wr_length))
XYZ_general_human = np.zeros(
    (s_dash_length * lam_wr_length * 2, general_scalars_length + 3))
xyz_general_table = np.zeros(
    (general_scalars_length, 2, s_dash_length, lam_wr_length))
xyz_general_human = np.zeros(
    (s_dash_length * lam_wr_length * 2, general_scalars_length + 3))

for lam_wr_index, lam in enumerate(lam_range_with_reciprocals):
    if lam in lam_range:  # X11A, X12A, etc.
        lam_index = np.argwhere(lam_range == lam)[0, 0]
        XYZ_general_table[:, 0, :, lam_wr_index] = XYZ_table[
            (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40), :, lam_index]
        XYZ_general_table[:, 1, :, lam_wr_index] = XYZ_table[
            (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41), :, lam_index]
        xyz_general_table[:, 0, :, lam_wr_index] = xyz_table[
            (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40), :, lam_index]
        xyz_general_table[:, 1, :, lam_wr_index] = xyz_table[
            (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41), :, lam_index]
    else:
        # Use X21A(1/lam), X22A(1/lam),... to generate X11A(lam), X12A(lam),...

        # When we generate X12A(lam) from X21A(1/lam), the values in XYZ_table
        # are nondimensionalised on a_2 from our perspective (a_1 from their
        # perspective, which is set at 1). But we want to nondim on a_1, so we
        # multiply by (a_2/a_1)^n, i.e. lam^n.
        lam_scales_one_line_r = np.array([lam, lam, lam**2, lam**3, lam**3,
                                          lam**2, lam**2, lam**3,
                                          lam**3, lam**3, lam**3])
        lam_scales_r = np.tile(lam_scales_one_line_r, (s_dash_length, 1)).T

        lam_scales_one_line_m = np.array([1/lam, 1/lam, 1/lam**2,
                                          1/lam**3, 1/lam**3,
                                          lam, lam, 1,
                                          lam**3, lam**3, lam**3])
        lam_scales_m = np.tile(lam_scales_one_line_m, (s_dash_length, 1)).T

        lam_index = np.argwhere(lam_range == (1 / lam))[0, 0]
        XYZ_general_table[:, 0, :, lam_wr_index] = (XYZ_table[
            (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43), :, lam_index]
            * minus_in_B_and_G * lam_scales_r)
        XYZ_general_table[:, 1, :, lam_wr_index] = (XYZ_table[
            (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42), :, lam_index]
            * minus_in_B_and_G * lam_scales_r)

        xyz_general_table[:, 0, :, lam_wr_index] = (xyz_table[
            (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43), :, lam_index]
            * minus_in_B_and_G * lam_scales_m)
        xyz_general_table[:, 1, :, lam_wr_index] = (xyz_table[
            (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42), :, lam_index]
            * minus_in_B_and_G * lam_scales_m)
# Time elapsed
elapsed_time = time.time() - start_time
elapsed_time_hms = format_seconds(elapsed_time)
print("Time elapsed " + elapsed_time_hms)

# Write XYZ_table to file (computer readable)
if save_pairs_file_as_well:
    with open('scalars_pairs_resistance_midfield.npy', 'wb') as outputfile:
        np.save(outputfile, XYZ_table)
with open('scalars_general_resistance_midfield.npy', 'wb') as outputfile:
    np.save(outputfile, XYZ_general_table)

# Write XYZ_table to file (human readable)
for s_dash_index, s_dash in enumerate(s_dash_range):
    s_dash_length = s_dash_range.shape[0]
    lam_length = lam_range.shape[0]
    lam_wr_length = lam_range_with_reciprocals.shape[0]
    for lam_index, lam in enumerate(lam_range):
        XYZ_outputline = np.append([s_dash, lam],
                                   XYZ_table[:, s_dash_index, lam_index])
        XYZ_human[lam_index*s_dash_length + s_dash_index, :] = XYZ_outputline
        xyz_outputline = np.append([s_dash, lam],
                                   xyz_table[:, s_dash_index, lam_index])
        xyz_human[lam_index*s_dash_length + s_dash_index, :] = xyz_outputline
    for lam_wr_index, lam in enumerate(lam_range_with_reciprocals):
        for gam in range(2):
            i = (lam_wr_index*s_dash_length + s_dash_index)*2 + gam
            XYZ_outputline = np.append(
                [s_dash, lam, gam],
                XYZ_general_table[:, gam, s_dash_index, lam_wr_index])
            XYZ_general_human[i, :] = XYZ_outputline
            xyz_outputline = np.append(
                [s_dash, lam, gam],
                xyz_general_table[:, gam, s_dash_index, lam_wr_index])
            xyz_general_human[i, :] = xyz_outputline

if save_pairs_file_as_well:
    save_human_table('scalars_pairs_resistance_midfield.txt',
                     'Nondimensionalised resistance scalars',
                     elapsed_time_hms,
                     np.append(["s'", "lambda"], resistance_scalars_names),
                     XYZ_human)
    save_human_table('scalars_pairs_mobility_midfield.txt',
                     'Nondimensionalised mobility scalars',
                     elapsed_time_hms,
                     np.append(["s'", "lambda"], mobility_scalars_names),
                     xyz_human)

save_human_table('scalars_general_resistance_midfield.txt',
                 'Nondimensionalised resistance scalars',
                 elapsed_time_hms,
                 np.append(["s'", "lambda", "gamma"],
                           general_resistance_scalars_names),
                 XYZ_general_human)
save_human_table('scalars_general_mobility_midfield.txt',
                 'Nondimensionalised mobility scalars',
                 elapsed_time_hms,
                 np.append(["s'", "lambda", "gamma"],
                           general_mobility_scalars_names),
                 xyz_general_human)
