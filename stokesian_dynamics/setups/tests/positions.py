#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/02/2024

"""Numbered list of all positions data used in the tests in the /test/ 
folder."""

import numpy as np
from functions.shared import add_sphere_rotations_to_positions


def pos_setup_tests(n):
    """Return position data and description for position setup number n.

    Returns:
        posdata: Position position, size and count data list.
        desc: Description of the setup for use in filenames.
    """
    desc = "test"

    if n == -1:
        # Two large spheres, s'=2.1 apart
        sphere_sizes = np.array([1, 1])
        s_dash = 2.1
        s = s_dash/sum(sphere_sizes)*2
        sphere_positions = np.array([[-s/2, 0, 0], [s/2, 0, 0]])

    if n == -2:
        # Two large spheres, s'=3.9 apart
        sphere_sizes = np.array([1, 1])
        sphere_positions = np.array([[0, 0, 0], [3.9, 0, 0]])

    if n == -3:
        # One big one small sphere, s'=2.1 apart
        sphere_sizes = np.array([1, 0.1])
        sphere_positions = np.array([[0, 0, 0], [1.155, 0, 0]])

    if n == -4:
        # One big one small sphere, s'=3.9 apart
        sphere_sizes = np.array([1, 0.1])
        sphere_positions = np.array([[0, 0, 0], [2.145, 0, 0]])

    if n == -5:
        # One small one small big, s'=2.1 apart (Minus sign test cf. -3 and -4)
        sphere_sizes = np.array([0.1, 1])
        sphere_positions = np.array([[0, 0, 0], [1.155, 0, 0]])

    if n == -6:
        # One small one big sphere, s'=3.9 apart
        sphere_sizes = np.array([0.1, 1])
        sphere_positions = np.array([[0, 0, 0], [2.145, 0, 0]])

    if n == -7:
        # Two small spheres, s'=2.1 apart
        sphere_sizes = np.array([0.1, 0.1])
        sphere_positions = np.array([[0, 0, 0], [0.21, 0, 0]])

    if n == -8:
        # Two small spheres, s'=3.9 apart
        sphere_sizes = np.array([0.1, 0.1])
        sphere_positions = np.array([[0, 0, 0], [0.39, 0, 0]])

    try:
        sphere_sizes
    except NameError:
        print("ERROR: You have not inputted a valid position setup number.")

    sphere_rotations = add_sphere_rotations_to_positions(
        sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
    dumbbell_sizes = np.array([])
    dumbbell_positions = np.empty([0, 3])
    dumbbell_deltax = np.empty([0, 3])

    posdata = (sphere_sizes, sphere_positions, sphere_rotations,
               dumbbell_sizes, dumbbell_positions, dumbbell_deltax)
    return posdata, desc
