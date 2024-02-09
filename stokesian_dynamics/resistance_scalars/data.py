#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

import numpy as np
import os
from settings import use_drag_Minfinity, use_drag_R2Binfinity

resistance_scalars_folder = os.path.dirname(os.path.abspath(__file__))

# Note: the s_range is the same currently for all values of lambda. This may be
# a bad idea in the long run. Not sure.
s_dash_range = np.loadtxt(f'{resistance_scalars_folder}/values_of_s_dash.txt')
range_s_dash_range = range(s_dash_range.shape[0])
range_len_of_s_dash_range = range(s_dash_range.shape[0])
lam_range = np.loadtxt(f'{resistance_scalars_folder}/values_of_lambda.txt')
lam_range_with_reciprocals = lam_range
for lam in lam_range_with_reciprocals:
    if (1./lam) not in lam_range_with_reciprocals:
        lam_range_with_reciprocals = np.append(lam_range_with_reciprocals,
                                               (1/lam))
lam_range_with_reciprocals.sort()

# The resistance scalars already have R2Binfinity subtracted off. Decide here
# whether this is with the full R2Binfinity or the drag-only R2Binfinity.
if not use_drag_Minfinity or (use_drag_Minfinity and use_drag_R2Binfinity):
    filename = f'{resistance_scalars_folder}/scalars_general_resistance_d.npy'
else:
    filename = f'{resistance_scalars_folder}/scalars_general_resistance_d_drag.npy'

try:
    with open(filename, 'rb') as inputfile:
        XYZ_raw = np.load(inputfile)
except IOError:
    # Only likely to be in this position when generating the resistance scalars.
    XYZ_raw = []

filename = f'{resistance_scalars_folder}/scalars_general_resistance.npy'
with open(filename, 'rb') as inputfile:
    XYZ_raw_no_d = np.load(inputfile)
s_dash_range = np.loadtxt(f'{resistance_scalars_folder}/values_of_s_dash.txt')
