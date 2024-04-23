#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 25/05/2017

import numpy as np
from functions.simulation_tools import empty_vectors
from functions.shared import posdata_data, throw_error
from setups.functions_inputs import (oscillatory_shear, constant_shear,
                                     repulsion_forces)
from setups.functions_positions import simple_cubic_8
from setups.tests.inputs import input_ftsuoe_tests


def input_ftsuoe(n, posdata, frameno, timestep, last_velocities,
                 input_form='undefined', skip_computation=False,
                 grand_resistance_matrix_fte=0):
    """Define all input forces/velocities.

    Args:
        n (int): Index which chooses one of the below input forces/velocities.
        posdata: Position data of all particles.
        frameno (int): Frame number of the simulation.
        timestep: Timestep size.
        last_velocities: Velocities of particles in previous timestep of
            simulation.
        input_form (str): Specifies which of F/T/S/U/Omega/E you are providing
            as inputs.
        skip_computation (bool): You might want to do some computation in this
            function if the forces depend on e.g. position in a nontrivial way.
            But you might be just calling this function to retrieve something
            simple like `box_bottom_left`. This flag states whether that
            computation is necessary (False) or whether the function is just
            being called to extract a constant (True). You can then use this
            flag to skip the computation in your function.
        grand resistance_matrix_fte: Some computation in this function might
            require the grand resistance matrix, e.g. when computing friction.
            If required, this is passed in here.

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

    # Initialise all vectors in the left and right-hand sides.
    # Then define num_spheres and num_dumbbells
    (Fa_in, Ta_in, Sa_in, Sa_c_in, Fb_in, DFb_in, Ua_in, Oa_in, Ea_in, Ea_c_in,
        Ub_in, HalfDUb_in) = empty_vectors(posdata)
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
        dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
        element_sizes, element_positions, element_deltax, num_elements,
        num_elements_array, element_type, uv_start, uv_size,
        element_start_count) = posdata_data(posdata)

    # Give values. You must give at least half the total number of U/O/E/F/T/S
    #   values.
    # If you are giving a mix of F and U values for spheres, you must label the
    #   spheres s.t. the fixed velocity spheres are numbered first.
    # If you are giving a mix of F and U values for dumbbells, you must label
    #   the dumbbells s.t. the fixed velocity dumbbells are numbered first.

    # Defaults
    mu = 1
    desc = ""
    box_bottom_left = np.array([0, 0, 0])
    box_top_right = np.array([0, 0, 0])
    # Background velocity is given by
    #   u^infinity = U^infinity + Omega^infinity cross x + E^infinity dot x.
    # U^infinity and O^infinity are reset here and are changed for each case if
    #   required.
    # E^infinity is input for each case as Ea_in, and is also reset here if
    #   you're using FTE form.
    U_infinity = np.array([0, 0, 0])
    O_infinity = np.array([0, 0, 0])
    if input_form == "fte":
        Fa_in[:] = [[0, 0, 0] for _ in range(num_spheres)]
        Ta_in[:] = [[0, 0, 0] for _ in range(num_spheres)]
        Ea_in[:] = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]] for _ in range(num_spheres)]
        Fb_in[:] = [[0, 0, 0] for _ in range(num_dumbbells)]
        DFb_in[:] = [[0, 0, 0] for _ in range(num_dumbbells)]
    elif input_form == "fts":
        Fa_in[:] = [[0, 0, 0] for _ in range(num_spheres)]
        Ta_in[:] = [[0, 0, 0] for _ in range(num_spheres)]
        Sa_in[:] = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]] for _ in range(num_spheres)]
        Fb_in[:] = [[0, 0, 0] for _ in range(num_dumbbells)]
        DFb_in[:] = [[0, 0, 0] for _ in range(num_dumbbells)]
    if input_form == "ufte":
        Ta_in[:] = [[0, 0, 0] for _ in range(num_spheres)]
        Ea_in[:] = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]] for _ in range(num_spheres)]
        Fb_in[:] = [[0, 0, 0] for _ in range(num_dumbbells)]
        DFb_in[:] = [[0, 0, 0] for _ in range(num_dumbbells)]
    centre_of_background_flow = np.array([0, 0, 0])
    Ot_infinity = np.array([0, 0, 0])
    Et_infinity = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    num_sphere_in_each_lid = 0

    if n < 0:
        # Tests for Pytest
        (Fa_in, Ta_in, Ea_in) = input_ftsuoe_tests(n, Fa_in, Ta_in, Ea_in)
        desc = "test"

    elif n == 1:
        # Gravity
        Fa_in[:] = [[0, 0, -1] for _ in range(num_spheres)]
        desc = "gravity"

    elif n == 2:
        # Gravity in periodic domain
        Fa_in[:] = [[0, 0, -1] for _ in range(num_spheres)]
        sphere_positions, box_bottom_left, box_top_right = simple_cubic_8(8)
        # sphere_positions is ignored inside input_ftsuoe, but to activate
        #   periodicity, you have to set box_bottom_left and box_top_right.
        desc = "gravity-periodic"

    elif n == 3:
        # Oscillatory background flow, about the point (2.25,0,2.25)
        natural_deltax = 2.
        spring_constant = -1
        DFb_in[:] = [list(spring_constant*(
            dumbbell_deltax[i]
            - natural_deltax*dumbbell_deltax[i]/np.linalg.norm(dumbbell_deltax[i])
        ))
            for i in range(num_dumbbells)]
        # Simple shear with speed gammadot
        # amplitude is amplitude at z = 1
        (Ea_in, U_infinity, O_infinity, centre_of_background_flow,
         Ot_infinity, Et_infinity) = oscillatory_shear(
            amplitude=1/3, period=10, frameno=frameno, timestep=timestep,
            phase=0, centre_of_background_flow=np.array([2.25, 0, 2.25]),
            num_spheres=num_spheres)
        desc = "oscillatory-background-flow"

    elif n == 4:
        # Repulsive force
        (Fa_in, Fb_in, DFb_in) = repulsion_forces(
            50, 20, num_spheres, num_dumbbells, sphere_positions,
            dumbbell_positions, dumbbell_deltax, sphere_sizes,
            dumbbell_sizes, num_sphere_in_each_lid, Fa_in, Fb_in, DFb_in)
        desc = "repulsion"

    elif n == 5:
        # Make half the spheres move to the left with a given velocity,
        #   and make the rest move to the right.
        Ua_in[:] = ([[-1, 0, 0] for _ in range(num_spheres//2)]
                    + [[1, 0, 0] for _ in range(num_spheres//2, num_spheres)])

    elif n == 6:
        # An example mixing prescribed forces and velocities.
        # Make half the wall spheres move to the left with a given velocity,
        #  and make the rest move to the right.
        # Give the remaining, non-wall spheres a force to the right.
        # Notice how Ua_in and Fa_in both have length 99. The software works
        #  out how many spheres have prescribed velocities, rather than forces,
        #  by counting how many of Ua_in have been filled in.
        # The fixed-velocity spheres must be numbered first.
        num_wall_spheres = 90
        num_other_spheres = 9
        Ua_in[:num_wall_spheres] = (
            [[-1, 0, 0] for _ in range(num_wall_spheres//2)]
            + [[1, 0, 0] for _ in range(num_wall_spheres//2,
                                        num_wall_spheres)])
        Fa_in[num_wall_spheres:] = [[10, 0, 0]
                                    for _ in range(num_other_spheres)]

    elif n == 7:
        # Constant shear
        (Ea_in, U_infinity, O_infinity, centre_of_background_flow,
         Ot_infinity, Et_infinity) = constant_shear(
            gammadot=1, frameno=frameno, timestep=timestep,
            num_spheres=num_spheres)
        desc = "constant-shear"

    else:
        throw_error("The input setup number you have requested (" + str(n) +
                    ") is not listed in setups/inputs.py.")

    return (Fa_in, Ta_in, Sa_in, Sa_c_in, Fb_in, DFb_in, Ua_in, Oa_in, Ea_in,
            Ea_c_in, Ub_in, HalfDUb_in, desc, U_infinity, O_infinity,
            centre_of_background_flow, Ot_infinity, Et_infinity,
            box_bottom_left, box_top_right, mu)
