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
import time

values_of_s_dash_nearfield = np.loadtxt('values_of_s_dash_nearfield.txt', 
                                        ndmin=1)
values_of_s_dash_midfield = np.loadtxt('values_of_s_dash_midfield.txt', 
                                       ndmin=1)
values_of_s_dash = np.concatenate((values_of_s_dash_nearfield, 
                                   values_of_s_dash_midfield))
values_of_s_dash.sort()
np.savetxt('values_of_s_dash.txt', values_of_s_dash, fmt="% .8e")
lam_range = np.loadtxt('values_of_lambda.txt')
s_dash_range = values_of_s_dash
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

with open('scalars_general_resistance_midfield.npy', 'rb') as inputfile:
    XYZ_mid_raw = np.load(inputfile)

with open('scalars_general_resistance_nearfield_mathematica_for_python.txt', 
          'rb') as inputfile:
    mathematica = inputfile.read()
    square_brackets = mathematica
    # for pair in [["{", "["], ["}", "]"], ["*^", "E"], ["\r\n", ","]]: # Win
    for pair in [["{", "["], ["}", "]"], ["*^", "E"], ["\n", ","]]: # Unix/Mac
        square_brackets = square_brackets.replace(pair[0], pair[1])
    XYZ_near_raw = np.array(eval(square_brackets)).transpose()

print(XYZ_mid_raw.shape)
print(XYZ_near_raw.shape)


XYZ_general_table = np.concatenate((XYZ_near_raw, XYZ_mid_raw), axis=2)

general_resistance_scalars_names = np.array(["XA", "YA", "YB", "XC", "YC",
                                             "XG", "YG", "YH", "XM", "YM",
                                             "ZM"])
general_scalars_length = len(general_resistance_scalars_names)

with open('scalars_general_resistance.npy', 'wb') as outputfile:
    np.save(outputfile, XYZ_general_table)


# Write XYZ_table and xyz_table to file (human readable)
lam_range_with_reciprocals = lam_range
for lam in lam_range:
    if (1. / lam) not in lam_range_with_reciprocals:
        lam_range_with_reciprocals = np.append(lam_range_with_reciprocals, 
                                               1/lam)
lam_range_with_reciprocals.sort()
lam_wr_length = lam_range_with_reciprocals.shape[0]
XYZ_general_human = np.zeros((s_dash_length*lam_wr_length*2, 
                              general_scalars_length + 3))
for s_dash in s_dash_range:
    s_dash_index = np.argwhere(s_dash_range == s_dash)[0, 0]
    s_dash_length = s_dash_range.shape[0]
    lam_length = lam_range.shape[0]
    lam_wr_length = lam_range_with_reciprocals.shape[0]
    for lam in lam_range_with_reciprocals:
        lam_wr_index = np.argwhere(lam_range_with_reciprocals == lam)[0, 0]
        for gam in range(2):
            XYZ_outputline = np.append(
                [s_dash, lam, gam], 
                XYZ_general_table[:, gam, s_dash_index, lam_wr_index])
            i = (lam_wr_index*s_dash_length + s_dash_index)*2 + gam
            XYZ_general_human[i, :] = XYZ_outputline


with open('scalars_general_resistance.txt', 'a') as outputfile:
    heading = ("Resistance scalars, combined " 
               + time.strftime("%d/%m/%Y %H:%M:%S") + ".")
    np.savetxt(outputfile, np.array([heading]), fmt="%s")
    np.savetxt(outputfile, 
               np.append(["s'", "lambda", "gamma"], 
                         general_resistance_scalars_names), 
               newline=" ", fmt="%15s")
    outputfile.write("\n")
    np.savetxt(outputfile, XYZ_general_human, newline="\n", fmt="% .8e")