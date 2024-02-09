# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 12/07/2023.
"""
Standalone script to generate nondimensional nearfield resistance scalars
X11A, X12A, Y11B, ..., Z12M  for values of (s,lambda) read in from a text file.

Reference: Townsend 2023, 'Generating, from scratch, the near-field asymptotic
forms of scalar resistance functions for two unequal rigid spheres in
low-Reynolds-number flow.' Physics of Fluids 35 (12), 127126.

Formulae are based on Jeffrey & Onishi 1984, Jeffrey 1992, and Ichiki et al.
2013 but have been corrected in the document referenced above.
Equation numbers relating to:
    Townsend 2018      are written (1),
    Jeffrey & Onishi      ,,       (J&O 1.1),
    Jeffrey               ,,       (J 1),
    Ichiki et al.         ,,       (I 1).

If you are running this as part of the Python Stokesian Dynamics
implementation: You can just run this file and read no further.

For two spheres of radius  a_1  and  a_2 , a distance  r  apart, the
nondimensional particle separation distance  s'  is defined as
    s' = 2r / (a_1 + a+2).

The size ratio is  lambda = a_2/a_1.

These scalars are NONDIMENSIONAL. To recover the dimensional forms, follow the
Kim & Karrila (sec 11.3) approach and multiply by
   6 pi a_1,  4 pi a_1^2,  8 pi a_1^3,  4 pi a_1^2,  8 pi a_1^3,  20/3 pi a_1^3
where the scalar has the superscript
   A,         B,           C,           G,           H,           M.

Reads:
    values_of_s_dash_nearfield.txt: List of nearfield distances s'.
    values_of_lambda.txt: List of size ratios lambda. Note that only values
        lambda<=1  are expected in this file. The scalars for the reciprocal
        values of  lambda  are generated automatically.

Writes:
    scalars_general_resistance_nearfield.npy:
        this file is for reading directly by Python in constructing a larger
        database of resistance scalars.
    scalars_general_resistance_nearfield.txt:
        this file is created to be human readable and is not used by any
        further script. This file lists scalars in the form:
            s' lambda 0 X11A Y11A Y11B X11C Y11C X11G Y11G Y11H X11M Y11M Z11M
            s' lambda 1 X12A Y12A Y12B X12C Y12C X12G Y12G Y12H X12M Y12M Z12M
"""

import numpy as np
import time
from functions_general import (resistance_scalars_names,
                               general_resistance_scalars_names,
                               format_seconds, save_human_table)
from functions_nearfield import (X11A, X12A, Y11A, Y12A, Y11B, Y12B,
                                 X11C, X12C, Y11C, Y12C,
                                 X11G, X12G, Y11G, Y12G, Y11H, Y12H,
                                 X11M, X12M, Y11M, Y12M, Z11M, Z12M)

start_time = time.time()

scalar_functions = (X11A, X12A, Y11A, Y12A, Y11B, Y12B,
                    X11C, X12C, Y11C, Y12C,
                    X11G, X12G, Y11G, Y12G, Y11H, Y12H,
                    X11M, X12M, Y11M, Y12M, Z11M, Z12M)

# Initialise variables
s_dash_range = np.loadtxt('values_of_s_dash_nearfield.txt', ndmin=1)
s_dash_length = s_dash_range.shape[0]
lam_range = np.loadtxt('values_of_lambda.txt', ndmin=1)
lam_range_with_reciprocals = np.sort(np.concatenate(
    (lam_range, [1/l for l in lam_range if 1/l not in lam_range])))
lam_wr_length = lam_range_with_reciprocals.shape[0]

scalars_length = len(resistance_scalars_names)
general_scalars_length = len(general_resistance_scalars_names)
XYZ_general_table = np.zeros(
    (general_scalars_length, 2, s_dash_length, lam_wr_length))
XYZ_general_human = np.zeros(
    (s_dash_length * lam_wr_length * 2, general_scalars_length + 3))

# Now run loop
for lam_index, lam in enumerate(lam_range_with_reciprocals):
    for s_dash_index, s_dash in enumerate(s_dash_range):
        print("Running s' = ", str(s_dash), ", lambda = " + str(lam) + " ...")
        xi = s_dash - 2
        for f_index, f in enumerate(scalar_functions):
            print("   ", f.__name__, "...")
            gamma = f_index % 2
            general_index = f_index // 2
            XYZ_general_table[general_index, gamma,
                              s_dash_index, lam_index] = f(xi, lam)

# Time elapsed
elapsed_time = time.time() - start_time
elapsed_time_hms = format_seconds(elapsed_time)
print("Time elapsed " + elapsed_time_hms)

# Save computer readable version
with open('scalars_general_resistance_nearfield.npy', 'wb') as outputfile:
    np.save(outputfile, XYZ_general_table)

# Save human readable version
for s_dash_index, s_dash in enumerate(s_dash_range):
    for lam_wr_index, lam in enumerate(lam_range_with_reciprocals):
        for gam in range(2):
            i = (lam_wr_index*s_dash_length + s_dash_index)*2 + gam
            XYZ_outputline = np.append(
                [s_dash, lam, gam],
                XYZ_general_table[:, gam, s_dash_index, lam_wr_index])
            XYZ_general_human[i, :] = XYZ_outputline

save_human_table('scalars_general_resistance_nearfield.txt',
                 'Nondimensionalised resistance scalars',
                 elapsed_time_hms,
                 np.append(["s'", "lambda", "gamma"],
                           general_resistance_scalars_names),
                 XYZ_general_human)
