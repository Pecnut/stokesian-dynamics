#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 28/12/2014

"""Reads in resistance scalars from
    find_resistance_scalars/scalars_general_resistance.npy
and converts them to their 'd' form, i.e. with the R2binfinity term already
subtracted away from it. Then it saves it to
    find_resistance_scalars/scalars_general_resistance_d.npy  and  .txt

It's in the main folder because otherwise it's a pain to import
generate_Minfinity and then use the scripts which it calls which expect to be
called from this folder.
"""

import numpy as np
import time
from functions.generate_Minfinity import generate_Minfinity
from functions.shared import add_sphere_rotations_to_positions, sqrt

looper_start_time = time.time()

with open('find_resistance_scalars/scalars_general_resistance.npy', 'rb') as inputfile:
    XYZ_raw = np.load(inputfile)
XYZd_raw = np.copy(XYZ_raw)

s_dash_range = np.loadtxt('find_resistance_scalars/values_of_s_dash.txt')
lam_range = np.loadtxt('find_resistance_scalars/values_of_lambda.txt')
lam_range_with_reciprocals = np.copy(lam_range)
for lam in lam_range:
    if 1/lam not in lam_range_with_reciprocals:
        lam_range_with_reciprocals = np.append(lam_range_with_reciprocals,
                                               1/lam)
lam_range_with_reciprocals.sort()

for lam_index, lam in enumerate(lam_range_with_reciprocals):
    for s_dash_index, s_dash in enumerate(s_dash_range):
        sphere_sizes = np.array([1, lam])
        s = s_dash*(sphere_sizes[0]+sphere_sizes[1])*0.5  # s = (a1+a2)/2 * s'
        sphere_positions = np.array([[0, 0, 0], [s, 0, 0]])
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])
        posdata = (sphere_sizes, sphere_positions, sphere_rotations,
                   dumbbell_sizes, dumbbell_positions, dumbbell_deltax)
        (Minfinity, heading) = generate_Minfinity(posdata)
        Rinfinity = np.linalg.inv(Minfinity)

        a1 = sphere_sizes[0]
        scale_A = 6*np.pi*a1
        scale_B = 4*np.pi*a1**2
        scale_C = 8*np.pi*a1**3
        scale_G = 4*np.pi*a1**2
        scale_H = 8*np.pi*a1**3
        scale_M = 20/3*np.pi*a1**3

        R = (sqrt(3)+3)/6
        S = sqrt(2)
        XA0 = Rinfinity[0, 0]/scale_A
        XA1 = Rinfinity[0, 3]/scale_A
        YA0 = Rinfinity[1, 1]/scale_A
        YA1 = Rinfinity[1, 4]/scale_A
        YB0 = Rinfinity[2, 7]/scale_B
        YB1 = Rinfinity[5, 7]/scale_B
        XC0 = Rinfinity[6, 6]/scale_C
        XC1 = Rinfinity[6, 9]/scale_C
        YC0 = Rinfinity[7, 7]/scale_C
        YC1 = Rinfinity[7, 10]/scale_C
        XG0 = Rinfinity[0, 12]/R/scale_G
        XG1 = Rinfinity[3, 12]/R/scale_G
        YG0 = Rinfinity[1, 13]/S/scale_G
        YG1 = Rinfinity[4, 13]/S/scale_G
        YH0 = Rinfinity[7, 15]/(-S)/scale_H
        YH1 = Rinfinity[10, 15]/(-S)/scale_H
        YM0 = Rinfinity[13, 13]/scale_M
        YM1 = Rinfinity[13, 18]/scale_M
        ZM0 = Rinfinity[16, 16]/scale_M
        ZM1 = Rinfinity[16, 21]/scale_M
        XM0 = ZM0 - 4*Rinfinity[14, 12]/scale_M
        XM1 = ZM1 - 4*Rinfinity[14, 17]/scale_M

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

        # if s_dash == 4.5 and lam == 100:
        #     print(XYZ_raw[0, 0, s_dash_index, lam_index], XA0, Minfinity[0,0]*scale_A)

# computer readable
with open('find_resistance_scalars/scalars_general_resistance_d.npy', 'wb+') as outputfile:
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
for s_dash_index, s_dash in enumerate(s_dash_range):
    for lam_wr_index, lam in enumerate(lam_range_with_reciprocals):
        for gam in range(2):
            XYZ_outputline = np.append([s_dash, lam, gam],
                                       XYZd_raw[:, gam, s_dash_index, lam_wr_index])
            XYZ_general_human[(lam_wr_index*s_dash_length + s_dash_index)*2 + gam, :] = XYZ_outputline
with open('find_resistance_scalars/scalars_general_resistance_d.txt', 'a') as outputfile:
    heading = ("'D'-form nondimensionalised resistance scalars, generated "
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
print("Resistance scalars successfully converted to 'd' form.")
print("Time elapsed " + looper_elapsed_time_hms)
