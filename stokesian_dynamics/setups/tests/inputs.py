#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/02/2024

"""Numbered list of all input forces/velocities used in the tests in the
/test/ folder."""

import numpy as np
from functions.shared import throw_error


def input_ftsuoe_tests(n, Fa_in, Ta_in, Ea_in):
    """Define all input forces/velocities.

    Args:
        n (int): Index which chooses one of the below input forces/velocities.

    Returns:
        Fa_in:  Forces on spheres
        Ta_in:  Torque on spheres
        Sa_in:  Stresslets on spheres
        Fb_in:  Forces on dumbbells (total force on dumbbell, F1+F2)
        DFb_in: Internal force on dumbbells (Delta F = F2-F1)
        Ua_in:  Velocity of spheres
        Oa_in:  Angular velocity of spheres
        Ea_in:  Rate of strain, E^infinity
        Ub_in:  Velocity of dumbbells
        HalfDUb_in: HALF the velocity difference of the dumbbells ((U2-U1)/2)
        desc: Human-readable description to be added to filename.
        U_infinity, O_infinity, centre_of_background_flow: Background flow.
        Ot_infinity, Et_infinity: Integral of U_infinity and O_infinity dt.
        box_bottom_left, box_top_right: Coordinates of periodic box if desired.
            Simulation is assumed non-periodic if these are equal.
        mu: Newtonian background fluid viscosity.
    """


    if n == -1:
        # Two-particle test case F11, used in pytest
        Fa_in[0] = [1, 0, 0]

    elif n == -2:
        # Two-particle test case F12, used in pytest
        Fa_in[0] = [0, 1, 0]

    elif n == -3:
        # Two-particle test case F21, used in pytest
        Fa_in[1] = [1, 0, 0]

    elif n == -4:
        # Two-particle test case F22, used in pytest
        Fa_in[1] = [0, 1, 0]

    elif n == -5:
        # Two-particle test case T11, used in pytest
        Ta_in[0] = [1, 0, 0]

    elif n == -6:
        # Two-particle test case T12, used in pytest
        Ta_in[0] = [0, 1, 0]

    elif n == -7:
        # Two-particle test case T21, used in pytest
        Ta_in[1] = [1, 0, 0]

    elif n == -8:
        # Two-particle test case T22, used in pytest
        Ta_in[1] = [0, 1, 0]

    elif n == -9:
        # Two-particle test case E11, used in pytest
        Ea_in[0] = [[1, 0, 0],[0, -0.5, 0],[0, 0, -0.5]]
        Ea_in[1] = [[1, 0, 0],[0, -0.5, 0],[0, 0, -0.5]]

    elif n == -10:
        # Two-particle test case E13, used in pytest
        Ea_in[0] = [[0, 0, 0],[0, 0, 1],[0, 1, 0]]
        Ea_in[1] = [[0, 0, 0],[0, 0, 1],[0, 1, 0]]

    elif n == -11:
        # Two-particle test case E14, used in pytest
        Ea_in[0] = [[0, 1, 0],[1, 0, 0],[0, 0, 0]]
        Ea_in[1] = [[0, 1, 0],[1, 0, 0],[0, 0, 0]]

    elif n == -12:
        # Two-particle test case F1, used in pytest
        Fa_in[0] = [1, 0, 0]
        Fa_in[1] = [-1, 0, 0]

    elif n == -13:
        # Two-particle test case T1, used in pytest
        Ta_in[0] = [1, 0, 0]
        Ta_in[1] = [-1, 0, 0]

    # elif n == -14:
    #     # Two-particle test case E24, used in pytest
    #     Ea_in[1] = [[0, 1, 0],[1, 0, 0],[0, 0, 0]]

    else:
        throw_error("The input setup number you have requested (" + str(n) +
                    ") is not listed in setups/tests/inputs.py.")

    return (Fa_in, Ta_in, Ea_in)
