#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 24/08/2016

"""Reads in resistance scalars from scalars_general_resistance.npy and 
converts them to their 'd' form, i.e. with the R2binfinity term already 
subtracted away from it. Then it saves it to
scalars_general_resistance_dnominf.npy.

Note that the 'dnominf' form means R2binfinity is just the inverse of Minfinity
with diagonal entries only. This is good for when you want to "turn Minfinity
off", which means just make Minfinity a diagonal matrix (so self terms only).

It's in the main folder because otherwise it's a pain to import
generate_Minfinity and then use the scripts which it call which expect to be 
called from this folder.
"""

import numpy as np
from functions_shared import *

looper_start_time = time.time()

with open('find_resistance_scalars/scalars_general_resistance.npy', 'rb') as inputfile:
    XYZ_raw = np.load(inputfile)
XYZd_raw = np.copy(XYZ_raw)

s_dash_range = np.loadtxt('find_resistance_scalars/values_of_s_dash.txt')
lam_range = np.loadtxt('find_resistance_scalars/values_of_lambda.txt')
lam_range_with_reciprocals = np.copy(lam_range)
for lam in lam_range:
    if 1/lam not in lam_range_with_reciprocals:
        lam_range_with_reciprocals = np.append(lam_range_with_reciprocals, 1/lam)
lam_range_with_reciprocals.sort()

for lam in lam_range_with_reciprocals:
    lam_index = np.argwhere(lam_range_with_reciprocals == lam)[0, 0]
    for s_dash in s_dash_range:
        s_dash_index = np.argwhere(s_dash_range == s_dash)[0, 0]
        if lam <= 1:
            sphere_sizes = np.array([1, lam])
        else:
            sphere_sizes = np.array([1/lam, 1])
        a1 = sphere_sizes[0]
        a2 = sphere_sizes[1]
        # s = (a1+a2)/2 * s'
        sphere_positions = np.array([
            [0, 0, 0], [s_dash*(sphere_sizes[0]+sphere_sizes[1])*0.5, 0, 0]])
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])
        posdata = (sphere_sizes, sphere_positions, sphere_rotations,
                   dumbbell_sizes, dumbbell_positions, dumbbell_deltax)

        Rinfinity = np.diag([
            a1, a1, a1, a2, a2, a2,
            a1**3/0.75, a1**3/0.75, a1**3/0.75,
            a2**3/0.75, a2**3/0.75, a2**3/0.75,
            a1**3/0.9, a1**3/0.9, a1**3/0.9, a1**3/0.9, a1**3/0.9,
            a2**3/0.9, a2**3/0.9, a2**3/0.9, a2**3/0.9, a2**3/0.9])/6/np.pi

        R = (sqrt(3)+3)/6
        S = sqrt(2)
        XA0 = Rinfinity[0, 0]
        XA1 = Rinfinity[0, 3]
        YA0 = Rinfinity[1, 1]
        YA1 = Rinfinity[1, 4]
        YB0 = Rinfinity[2, 7]
        YB1 = Rinfinity[5, 7]
        XC0 = Rinfinity[6, 6]
        XC1 = Rinfinity[6, 9]
        YC0 = Rinfinity[7, 7]
        YC1 = Rinfinity[7, 10]
        XG0 = Rinfinity[0, 12]/R
        XG1 = Rinfinity[3, 12]/R
        YG0 = Rinfinity[1, 13]/S
        YG1 = Rinfinity[4, 13]/S
        YH0 = Rinfinity[7, 15]/(-S)
        YH1 = Rinfinity[10, 15]/(-S)
        YM0 = Rinfinity[13, 13]
        YM1 = Rinfinity[13, 18]
        ZM0 = Rinfinity[16, 16]
        ZM1 = Rinfinity[16, 21]
        XM0 = ZM0 - 4*Rinfinity[14, 12]
        XM1 = ZM1 - 4*Rinfinity[14, 17]

        XYZd_raw[0, 0, s_dash_index, lam_index] = XYZ_raw[0, 0, s_dash_index, lam_index] - XA0
        XYZd_raw[0, 1, s_dash_index, lam_index] = XYZ_raw[0, 1, s_dash_index, lam_index] - XA1
        XYZd_raw[1, 0, s_dash_index, lam_index] = XYZ_raw[1, 0, s_dash_index, lam_index] - YA0
        XYZd_raw[1, 1, s_dash_index, lam_index] = XYZ_raw[1, 1, s_dash_index, lam_index] - YA1
        XYZd_raw[2, 0, s_dash_index, lam_index] = XYZ_raw[2, 0, s_dash_index, lam_index] - YB0
        XYZd_raw[2, 1, s_dash_index, lam_index] = XYZ_raw[2, 1, s_dash_index, lam_index] - YB1
        XYZd_raw[3, 0, s_dash_index, lam_index] = XYZ_raw[3, 0, s_dash_index, lam_index] - XC0
        XYZd_raw[3, 1, s_dash_index, lam_index] = XYZ_raw[3, 1, s_dash_index, lam_index] - XC1
        XYZd_raw[4, 0, s_dash_index, lam_index] = XYZ_raw[4, 0, s_dash_index, lam_index] - YC0
        XYZd_raw[4, 1, s_dash_index, lam_index] = XYZ_raw[4, 1, s_dash_index, lam_index] - YC1
        XYZd_raw[5, 0, s_dash_index, lam_index] = XYZ_raw[5, 0, s_dash_index, lam_index] - XG0
        XYZd_raw[5, 1, s_dash_index, lam_index] = XYZ_raw[5, 1, s_dash_index, lam_index] - XG1
        XYZd_raw[6, 0, s_dash_index, lam_index] = XYZ_raw[6, 0, s_dash_index, lam_index] - YG0
        XYZd_raw[6, 1, s_dash_index, lam_index] = XYZ_raw[6, 1, s_dash_index, lam_index] - YG1
        XYZd_raw[7, 0, s_dash_index, lam_index] = XYZ_raw[7, 0, s_dash_index, lam_index] - YH0
        XYZd_raw[7, 1, s_dash_index, lam_index] = XYZ_raw[7, 1, s_dash_index, lam_index] - YH1
        XYZd_raw[8, 0, s_dash_index, lam_index] = XYZ_raw[8, 0, s_dash_index, lam_index] - XM0
        XYZd_raw[8, 1, s_dash_index, lam_index] = XYZ_raw[8, 1, s_dash_index, lam_index] - XM1
        XYZd_raw[9, 0, s_dash_index, lam_index] = XYZ_raw[9, 0, s_dash_index, lam_index] - YM0
        XYZd_raw[9, 1, s_dash_index, lam_index] = XYZ_raw[9, 1, s_dash_index, lam_index] - YM1
        XYZd_raw[10, 0, s_dash_index, lam_index] = XYZ_raw[10, 0, s_dash_index, lam_index] - ZM0
        XYZd_raw[10, 1, s_dash_index, lam_index] = XYZ_raw[10, 1, s_dash_index, lam_index] - ZM1

# computer readable
with open('find_resistance_scalars/scalars_general_resistance_dnominf.npy', 'wb') as outputfile:
    np.save(outputfile, XYZd_raw)

# human readable
general_resistance_scalars_names = np.array(["XA", "YA", "YB", "XC", "YC",
                                             "XG", "YG", "YH", "XM", "YM",
                                             "ZM"])
s_dash_length = s_dash_range.shape[0]
lam_length = lam_range.shape[0]
lam_wr_length = lam_range_with_reciprocals.shape[0]
general_scalars_length = general_resistance_scalars_names.shape[0]
XYZ_general_human = np.zeros((s_dash_length*lam_wr_length*2, 
                              general_scalars_length + 3))
for s_dash in s_dash_range:
    s_dash_index = np.argwhere(s_dash_range == s_dash)[0, 0]
    for lam in lam_range_with_reciprocals:
        lam_wr_index = np.argwhere(lam_range_with_reciprocals == lam)[0, 0]
        for gam in range(2):
            XYZ_outputline = np.append(
                [s_dash, lam, gam], XYZd_raw[:, gam, s_dash_index, lam_wr_index])
            XYZ_general_human[(lam_wr_index*s_dash_length + s_dash_index)*2 + gam, :] = XYZ_outputline
with open('find_resistance_scalars/scalars_general_resistance_dnominf.txt', 'a') as outputfile:
    heading = ("'D'-form resistance scalars, generated " 
               + time.strftime("%d/%m/%Y %H:%M:%S"))
    np.savetxt(outputfile, np.array([heading]), fmt="%s")
    np.savetxt(outputfile, 
               np.append(["s'", "lambda", "gamma"], 
                         general_resistance_scalars_names), 
               newline=" ", fmt="%15s")
    outputfile.write("\n")
    np.savetxt(outputfile, XYZ_general_human, newline="\n", fmt="% .8e")

# Time elapsed
looper_elapsed_time = time.time() - looper_start_time
let_m, let_s = divmod(looper_elapsed_time, 60)
let_h, let_m = divmod(let_m, 60)
looper_elapsed_time_hms = "%dh%02dm%02ds" % (let_h, let_m, let_s)
print("Time elapsed " + looper_elapsed_time_hms)
