# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com.
"""
Standalone script to combine nearfield and midfield resistance scalars.

Reads:
    values_of_s_midfield.txt: List of midfield distances s'.
    values_of_s_nearfield.txt: List of nearfield distances s'.
    values_of_lambda.txt: List of size ratios lambda.

    scalars_general_resistance_midfield.npy: 
        Midfield resistance scalars (from find_resistsance_scalars_KimH.py).
    scalars_general_resistance_nearfield_mathematica_for_python.txt:
        Nearfield resistance scalars (from Mathematica).

Writes:
    values_of_s_dash.txt: Combined text file of near- and midfield s'.
    scalars_general_resistance.npy: Combined resistance scalars in NPY format.
    scalars_general_resistance.txt: Human-readable version.
"""

import numpy as np
from functions_general import (general_resistance_scalars_names,
                               save_human_table)

values_of_s_dash_nearfield = np.loadtxt('values_of_s_dash_nearfield.txt', 
                                        ndmin=1)
values_of_s_dash_midfield = np.loadtxt('values_of_s_dash_midfield.txt', 
                                       ndmin=1)
values_of_s_dash = np.concatenate((values_of_s_dash_nearfield, 
                                   values_of_s_dash_midfield))
values_of_s_dash.sort()
np.savetxt('values_of_s_dash.txt', values_of_s_dash, fmt="% .8e")
s_dash_range = values_of_s_dash
s_dash_length = s_dash_range.shape[0]

lam_range = np.loadtxt('values_of_lambda.txt', ndmin=1)
lam_length = lam_range.shape[0]

with open('scalars_general_resistance_midfield.npy', 'rb') as inputfile:
    XYZ_mid_raw = np.load(inputfile)

with open('scalars_general_resistance_nearfield.npy', 'rb') as inputfile:
    XYZ_near_raw = np.load(inputfile)    

print(XYZ_mid_raw.shape)
print(XYZ_near_raw.shape)


XYZ_general_table = np.concatenate((XYZ_near_raw, XYZ_mid_raw), axis=2)
general_scalars_length = len(general_resistance_scalars_names)

with open('scalars_general_resistance.npy', 'wb') as outputfile:
    np.save(outputfile, XYZ_general_table)


# Write XYZ_table and xyz_table to file (human readable)
lam_range_with_reciprocals = np.concatenate(
    (lam_range, [1/l for l in lam_range if 1/l not in lam_range]))        
lam_range_with_reciprocals.sort()
lam_wr_length = lam_range_with_reciprocals.shape[0]
XYZ_general_human = np.zeros((s_dash_length*lam_wr_length*2, 
                              general_scalars_length + 3))
for s_dash_index, s_dash in enumerate(s_dash_range):
    s_dash_length = s_dash_range.shape[0]
    lam_length = lam_range.shape[0]
    lam_wr_length = lam_range_with_reciprocals.shape[0]
    for lam_wr_index, lam in enumerate(lam_range_with_reciprocals):
        for gam in range(2):
            XYZ_outputline = np.append(
                [s_dash, lam, gam], 
                XYZ_general_table[:, gam, s_dash_index, lam_wr_index])
            i = (lam_wr_index*s_dash_length + s_dash_index)*2 + gam
            XYZ_general_human[i, :] = XYZ_outputline

save_human_table('scalars_general_resistance.txt',
                 'Resistance scalars, combined',
                 "0s",
                 np.append(["s'", "lambda", "gamma"],
                           general_resistance_scalars_names),
                 XYZ_general_human)

print("Nearfield and midfield scalars combined successfully.")
