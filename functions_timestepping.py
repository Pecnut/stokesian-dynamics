#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

import numpy as np
from numpy import linalg
from functions_generate_grand_resistance_matrix import (
    generate_grand_resistance_matrix, generate_grand_resistance_matrix_periodic)
from functions_shared import posdata_data, format_elapsed_time, throw_error
from functions_simulation_tools import (
    construct_force_vector_from_fts, deconstruct_velocity_vector_for_fts,
    fts_to_fte_matrix, fte_to_ufte_matrix, ufte_to_ufteu_matrix,
    fts_to_duf_matrix)
from input_setups import input_ftsuoe
import time


def euler_timestep(x, u, timestep):
    """Returns the next Euler timestep, `x + timestep * u`."""
    return (x + timestep * u).astype('float')


def ab2_timestep(x, u, u_previous, timestep):
    """Returns the next Adams-Bashforth 2 timestep."""
    return x + timestep * (1.5 * u - 0.5 * u_previous)


def did_something_go_wrong_with_dumbells(error, dumbbell_deltax,
                                         new_dumbbell_deltax,
                                         explosion_protection):
    """Check new dumbbell lengths for signs of numerical explosion.

    Specifically, see whether dumbbells have rotated or stretched too far in a timestep.

    Args:
        error (bool): Current error flag from any previous error checks.
        dumbbell_deltax: Existing dumbbell displacements.
        new_dumbbell_deltax: New dumbbell displacements after timestep.
        explosion_protection (bool): Flag whether to enact the length check.

    Returns:
        error: True if these checks flag up problems, else passes through inputted value."""

    for i in range(new_dumbbell_deltax.shape[0]):
        if np.arccos(np.round(np.dot(dumbbell_deltax[i], new_dumbbell_deltax[i]) / (np.linalg.norm(dumbbell_deltax[i]) * np.linalg.norm(new_dumbbell_deltax[i])), 4)) > np.pi / 2:
            print(" ")
            print(f"Code point H# reached on dumbbell {str(i)}")
            print(f"Old delta x: {str(dumbbell_deltax[i])}")
            print(f"New delta x: {str(new_dumbbell_deltax[i])}")
        if explosion_protection and np.linalg.norm(new_dumbbell_deltax[i]) > 5:
            print("ERROR")
            print(
                f"Dumbbell {str(i)} length ({str(np.linalg.norm(new_dumbbell_deltax[i]))}) has exceeded 5."
            )
            print("Something has probably gone wrong (normally your timestep is too large).")
            print("Code exiting gracefully.")
            error = True
    return error


def euler_timestep_rotation(sphere_positions, sphere_rotations,
                            new_sphere_positions, new_sphere_rotations,
                            Oa_out, timestep):
    """Returns new rotation vectors after an Euler timestep using Oa_out as the
    velocity.

    See comments inside the function for details."""

    for i in range(sphere_positions.shape[0]):
        R0 = sphere_positions[i]
        O = (Oa_out[i][0] ** 2 + Oa_out[i][1] ** 2 + Oa_out[i][2] ** 2) ** 0.5

        ''' To rotate from basis (x,y,z) to (X,Y,Z), where x,y,z,X,Y,Z are unit
        vectors, you just need to multiply by the matrix
        ( X_x   Y_x   Z_x )
        ( X_y   Y_y   Z_y ),
        ( X_z   Y_z   Z_z )
        where X_x means the x-component of X.
        Our Z is Omega = o_spheres[i], so we need to make it into a complete
        basis. To do that we pick a unit vector different to Omega (either zhat
        or xhat depending on Omega) and use 
        (Omega x zhat, Omega x (Omega x zhat), zhat) as our basis (X,Y,Z).
        That's it! [Only took me three days...]
        '''

        if np.array_equal(Oa_out[i], [0, 0, 0]):
            rot_matrix = np.identity(3)
        else:
            Otest = (abs(Oa_out[i] / O)).astype('float')
            perp1 = [0, 0, 1] if np.allclose(Otest, [1, 0, 0]) else [1, 0, 0]
            rot_matrix = np.array([np.cross(Oa_out[i], perp1) / O, np.cross(Oa_out[i], np.cross(Oa_out[i], perp1)) / O ** 2,  Oa_out[i] / O]).transpose()

        for j in range(2):
            ''' rb0 is the position ("r") of the endpoint of the pointy
            rotation vector in the external (x,y,z) frame ("b") at the 
            beginning of this process ("0") '''
            rb0 = sphere_rotations[i, j]

            ''' rbdashdash0_xyz is the position of the same endpoint in the
            frame of the rotating sphere ("b''"), which we set to have the 
            z-axis=Omega axis. It's in Cartesian coordinates. '''
            rbdashdash0_xyz = np.dot(linalg.inv(rot_matrix), (rb0 - R0))
            x0 = rbdashdash0_xyz[0]
            y0 = rbdashdash0_xyz[1]
            z0 = rbdashdash0_xyz[2]

            r0 = (x0 ** 2 + y0 ** 2 + z0 ** 2) ** 0.5
            t0 = np.arccos(z0 / r0)
            p0 = 0 if (x0 == 0 and y0 == 0) else np.arctan2(y0, x0)
            r = r0
            t = t0
            p = euler_timestep(p0, O, timestep)

            x = r * np.sin(t) * np.cos(p)
            y = r * np.sin(t) * np.sin(p)
            z = r * np.cos(t)
            rbdashdash_xyz = np.array([x, y, z])
            R = new_sphere_positions[i]
            rb = R + np.dot(rot_matrix, rbdashdash_xyz)
            new_sphere_rotations[i, j] = rb
    return new_sphere_rotations


def ab2_timestep_rotation(sphere_positions, sphere_rotations,
                          new_sphere_positions, new_sphere_rotations,
                          Oa_out, Oa_out_previous, timestep):
    """Returns new rotation vectors after an Adams-Bashforth 2 timestep using
    Oa_out as the velocity.

    AB2 = Euler (x_n + u_n * dt) but with u_n replaced by (1.5 u_n - 0.5 u_{n-1})
    """
    combined_Oa_for_ab2 = 1.5 * Oa_out - 0.5 * Oa_out_previous
    return euler_timestep_rotation(sphere_positions, sphere_rotations,
                                   new_sphere_positions, new_sphere_rotations,
                                   combined_Oa_for_ab2, timestep)


def orthogonal_proj(zfront, zback):
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, a, b],
                     [0, 0, -0.0001, zback]])


def do_we_have_all_size_ratios(error, element_sizes, lam_range, num_spheres):

    lambda_matrix = element_sizes / element_sizes[:, None]
    lam_range_including_reciprocals = np.concatenate([lam_range, 1 / lam_range])
    do_we_have_it_matrix = np.in1d(lambda_matrix, lam_range_including_reciprocals).reshape([len(element_sizes), len(element_sizes)])
    offending_elements = np.where(do_we_have_it_matrix == 0)
    if len(offending_elements[0]) > 0:
        offending_lambda_1 = lambda_matrix[offending_elements[0][0], offending_elements[0][1]]
        offending_element_str = np.empty(2, dtype='|S25')
        for i in (0, 1):
            offending_element_str[i] = (
                f"dumbbell {str(offending_elements[0][i] - num_spheres)}"
                if offending_elements[0][i] >= num_spheres
                else f"sphere {str(offending_elements[0][i])}"
            )
        print("ERROR")
        print(
            f"Element size ratio ({str(offending_lambda_1)} or {str(1 / offending_lambda_1)}) is not in our calculated list of size ratios"
        )
        print(
            f"Offending elements: {offending_element_str[0]} and {offending_element_str[1]} (counting from 0)"
        )
        return True
    return error


def are_some_of_the_particles_too_close(error, printout, s_dash_range,
                                        sphere_positions, dumbbell_positions,
                                        dumbbell_deltax, sphere_sizes,
                                        dumbbell_sizes, element_positions):
    """Customise me: A function you can adapt to check if particles are too close.

    By default, just returns the value of the `error` flag given to it."""

    # Error check 1 : are some of my particles too close together for R2Bexact?
    # min_s_dash_range = np.min(s_dash_range)  # This is the minimum s' we have calculated values for

    sphere_and_bead_positions = np.concatenate([sphere_positions,
                                                dumbbell_positions + 0.5 * dumbbell_deltax,
                                                dumbbell_positions - 0.5 * dumbbell_deltax])
    sphere_and_bead_sizes = np.concatenate([sphere_sizes, dumbbell_sizes, dumbbell_sizes])
    sphere_and_bead_positions = sphere_and_bead_positions.astype('float')
    distance_matrix = np.linalg.norm(sphere_and_bead_positions - sphere_and_bead_positions[:, None], axis=2)
    average_size = 0.5 * (sphere_and_bead_sizes + sphere_and_bead_sizes[:, None])
    distance_over_average_size = distance_matrix / average_size  # Matrix of s'

    # min_element_distance = np.min(distance_over_average_size[np.nonzero(distance_over_average_size)])
    # two_closest_elements = np.where(distance_over_average_size == min_element_distance)

    if printout > 0:
        print("")
        print("Positions")
        print(np.array_str(element_positions, max_line_width=100000, precision=5))
        print("Dumbbell Delta x")
        print(np.array_str(dumbbell_deltax, max_line_width=100000, precision=5))
        print("Separation distances (s)")
        print(np.array_str(distance_matrix, max_line_width=100000, precision=3))
        print("Scaled Separation distances (s')")
        print(np.array_str(distance_over_average_size, max_line_width=100000, precision=3))

    return error


def generate_output_FTSUOE(
        posdata, frameno, timestep, input_number,
        last_generated_Minfinity_inverse, regenerate_Minfinity, input_form,
        cutoff_factor, printout, use_drag_Minfinity, use_Minfinity_only,
        extract_force_on_wall_due_to_dumbbells, last_velocities,
        last_velocity_vector, checkpoint_start_from_frame,
        box_bottom_left, box_top_right, feed_every_n_timesteps=0):
    """Solve the grand mobility problem: for given force/velocity inputs, 
    return all computed velocities/forces.

    Args (selected):
        posdata: Contains position, size and count data for all particles.
        input_number: Index of the force/velocity inputs, listed in 
            `input_setups.py`.

    Returns:
        All force and velocity data, including both that given as inputs and 
        that computed by solving the grand mobility problem.
    """

    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
     element_sizes, element_positions, element_deltax, num_elements,
     num_elements_array, element_type, uv_start, uv_size,
     element_start_count) = posdata_data(posdata)
    # Get inputs first time in "skip_computation" mode, i.e. no complex
    # calculations for Fa_in, etc. This is really just to get the values of
    # box_bottom_left and box_top_right.
    (Fa_in, Ta_in, Sa_in, Sa_c_in, Fb_in, DFb_in,
     Ua_in, Oa_in, Ea_in, Ea_c_in, Ub_in, HalfDUb_in, input_description,
     U_infinity, O_infinity, centre_of_background_flow, amplitude, frequency,
     box_bottom_left, box_top_right, mu) = input_ftsuoe(
        input_number, posdata, frameno, timestep, last_velocities,
        input_form=input_form, skip_computation=True)

    force_on_wall_due_to_dumbbells = 0

    if input_form == "stokes_drag_dumbbells_only":
        solve_time_start = time.time()
        Fbeads = 0.5*np.concatenate([np.array(Fb_in) + np.array(DFb_in),
                                     np.array(Fb_in) - np.array(DFb_in)])
        a = dumbbell_sizes[0]
        drag_coeff = mu*a
        Ubeads = Fbeads/drag_coeff
        Nbeads = len(Fbeads)
        (Fa_out, Ta_out, Sa_out, Fb_out, DFb_out) = (
            Fa_in[:], Ta_in[:], Sa_in[:], Fb_in[:], DFb_in[:])
        (Ua_out, Oa_out, Ea_out) = (Fa_in[:], Fa_in[:], Ea_in[:])
        Ub_out = 0.5*(Ubeads[:Nbeads/2] + Ubeads[Nbeads/2:])
        HalfDUb_out = 0.5*(Ubeads[:Nbeads/2] - Ubeads[Nbeads/2:])
        gen_times = [0, 0, 0]

    else:
        if not np.array_equal(box_bottom_left - box_top_right, np.array([0, 0, 0])):
            # periodic
            (grand_resistance_matrix, heading, last_generated_Minfinity_inverse,
                gen_times) = generate_grand_resistance_matrix_periodic(
                posdata,
                last_generated_Minfinity_inverse,
                box_bottom_left, box_top_right,
                regenerate_Minfinity=regenerate_Minfinity,
                cutoff_factor=cutoff_factor, printout=printout,
                use_drag_Minfinity=use_drag_Minfinity,
                use_Minfinity_only=use_Minfinity_only, frameno=frameno, mu=mu,
                O_infinity=O_infinity, E_infinity=Ea_in[0],
                timestep=timestep,
                centre_of_background_flow=centre_of_background_flow,
                amplitude=amplitude, frequency=frequency)
        else:
            # non-periodic
            (grand_resistance_matrix, heading, last_generated_Minfinity_inverse,
                gen_times) = generate_grand_resistance_matrix(
                posdata,
                last_generated_Minfinity_inverse,
                regenerate_Minfinity=regenerate_Minfinity,
                cutoff_factor=cutoff_factor, printout=printout,
                use_drag_Minfinity=use_drag_Minfinity,
                use_Minfinity_only=use_Minfinity_only, frameno=frameno, mu=mu)

        solve_time_start = time.time()

        num_spheres = len(Ua_in)
        num_dumbbells = len(Ub_in)
        if input_form == 'fts':
            try:
                force_vector = construct_force_vector_from_fts(posdata, Fa_in, Ta_in, Sa_in, Fb_in, DFb_in)
            except:
                throw_error("FTS mode has been selected but not all values of F, T and S have been provided.")
            velocity_vector = np.linalg.solve(grand_resistance_matrix, force_vector)
            (Ua_out, Oa_out, Ea_out, Ub_out, HalfDUb_out) = deconstruct_velocity_vector_for_fts(posdata, velocity_vector)
            (Fa_out, Ta_out, Sa_out, Fb_out, DFb_out) = (Fa_in[:], Ta_in[:], Sa_in[:], Fb_in[:], DFb_in[:])
            if num_spheres == 0:
                Ea_out = Ea_in

        elif input_form == 'fte':
            # Call this the same name to reduce memory requirements (no need to reproduce)
            grand_resistance_matrix = fts_to_fte_matrix(posdata, grand_resistance_matrix)

            # Get inputs a second time not in "skip_computation" mode, putting
            # in the grand resistance matrix which is needed for some
            # calculations with friction.
            (Fa_in, Ta_in, Sa_in, Sa_c_in, Fb_in, DFb_in,
             Ua_in, Oa_in, Ea_in, Ea_c_in, Ub_in, HalfDUb_in,
             input_description, U_infinity, O_infinity,
             centre_of_background_flow, amplitude, frequency, box_bottom_left,
             box_top_right, mu) = input_ftsuoe(
                input_number, posdata, frameno, timestep, last_velocities,
                input_form=input_form,
                grand_resistance_matrix_fte=grand_resistance_matrix)
            try:
                force_vector = construct_force_vector_from_fts(
                    posdata, Fa_in, Ta_in, Ea_in, Fb_in, DFb_in)
            except:
                throw_error("FTE mode has been selected but not all values of F, T and E have been provided.")
            velocity_vector = np.linalg.solve(grand_resistance_matrix,
                                              force_vector)
            (Ua_out, Oa_out, Sa_out, Ub_out, HalfDUb_out) = deconstruct_velocity_vector_for_fts(
                posdata, velocity_vector)
            (Fa_out, Ta_out, Ea_out, Fb_out, DFb_out) = (
                Fa_in[:], Ta_in[:], Ea_in[:], Fb_in[:], DFb_in[:])
            if num_spheres == 0:
                Ea_out = Ea_in

        elif input_form == 'ufte':
            num_fixed_velocity_spheres = num_spheres - Ua_in.count(['pippa', 'pippa', 'pippa'])
            try:
                force_vector = construct_force_vector_from_fts(
                    posdata,
                    Ua_in[0:num_fixed_velocity_spheres] + Fa_in[num_fixed_velocity_spheres:num_spheres],
                    Ta_in, Ea_in, Fb_in, DFb_in)
            except:
                throw_error("UFTE mode has been selected but not enough values of U, F, T and E have been provided. At a guess, not all your spheres have either a U or an F.")
            force_vector = np.array(force_vector, float)
            grand_resistance_matrix_fte = fts_to_fte_matrix(
                posdata, grand_resistance_matrix)
            grand_resistance_matrix_ufte = fte_to_ufte_matrix(
                num_fixed_velocity_spheres, posdata, grand_resistance_matrix_fte)
            if extract_force_on_wall_due_to_dumbbells:
                grand_mobility_matrix_ufte = np.linalg.inv(grand_resistance_matrix_ufte)
                velocity_vector = np.dot(grand_mobility_matrix_ufte, force_vector)
            else:
                velocity_vector = np.linalg.solve(grand_resistance_matrix_ufte, force_vector)
                (FUa_out, Oa_out, Sa_out, Ub_out, HalfDUb_out) = deconstruct_velocity_vector_for_fts(posdata, velocity_vector)
                Fa_out = [['chen', 'chen', 'chen'] for i in range(num_spheres)]
                Ua_out = [['chen', 'chen', 'chen'] for i in range(num_spheres)]
                Fa_out[0:num_fixed_velocity_spheres] = FUa_out[0:num_fixed_velocity_spheres]
                Fa_out[num_fixed_velocity_spheres:num_spheres] = Fa_in[num_fixed_velocity_spheres:num_spheres]
                Ua_out[0:num_fixed_velocity_spheres] = Ua_in[0:num_fixed_velocity_spheres]
                Ua_out[num_fixed_velocity_spheres:num_spheres] = FUa_out[num_fixed_velocity_spheres:num_spheres]
                (Ta_out, Ea_out, Fb_out, DFb_out) = (Ta_in[:], Ea_in[:], Fb_in[:], DFb_in[:])

            if extract_force_on_wall_due_to_dumbbells:
                # For finding effect of the dumbbells on the measured Force on the walls.
                # Since   Fafixed = ()Uafixed + ()Fafree + ()Ta + ()E +   ()Fb  + ()DFb       ,
                #                                                       | want this bit |
                force_on_wall_due_to_dumbbells_matrix = grand_mobility_matrix_ufte[:num_fixed_velocity_spheres*3, 11*num_spheres:]
                dumbbell_forces = force_vector[11*num_spheres:]
                force_on_wall_due_to_dumbbells_flat = np.dot(force_on_wall_due_to_dumbbells_matrix, dumbbell_forces)
                force_on_wall_due_to_dumbbells = force_on_wall_due_to_dumbbells_flat.reshape(len(force_on_wall_due_to_dumbbells_flat)/3, 3)

        elif input_form == 'ufteu':
            num_fixed_velocity_spheres = num_spheres - Ua_in.count(['pippa', 'pippa', 'pippa'])
            num_fixed_velocity_dumbbells = num_dumbbells - Ub_in.count(['pippa', 'pippa', 'pippa'])
            try:
                force_vector = construct_force_vector_from_fts(
                    posdata,
                    Ua_in[0:num_fixed_velocity_spheres] + Fa_in[num_fixed_velocity_spheres:num_spheres],
                    Ta_in, Ea_in,
                    Ub_in[0:num_fixed_velocity_dumbbells] + Fb_in[num_fixed_velocity_dumbbells:num_dumbbells],
                    HalfDUb_in[0:num_fixed_velocity_dumbbells] + DFb_in[num_fixed_velocity_dumbbells:num_dumbbells])
            except:
                throw_error("UFTEU mode has been selected but not enough values of U, F, T and E and U(dumbbell) have been provided. At a guess, not all your spheres/dumbbells have either a U or an F.")

            force_vector = np.array(force_vector, float)
            grand_resistance_matrix_fte = fts_to_fte_matrix(
                posdata, grand_resistance_matrix)
            grand_resistance_matrix_ufte = fte_to_ufte_matrix(
                num_fixed_velocity_spheres, posdata, grand_resistance_matrix_fte)
            grand_resistance_matrix_ufteu = ufte_to_ufteu_matrix(
                num_fixed_velocity_dumbbells, num_fixed_velocity_spheres,
                posdata, grand_resistance_matrix_ufte)
            velocity_vector = np.linalg.solve(grand_resistance_matrix_ufteu,
                                              force_vector)

            (FUa_out, Oa_out, Sa_out, FUb_out, DFUb_out) = deconstruct_velocity_vector_for_fts(posdata, velocity_vector)
            Fa_out = [['chen', 'chen', 'chen'] for i in range(num_spheres)]
            Ua_out = [['chen', 'chen', 'chen'] for i in range(num_spheres)]
            Fa_out[0:num_fixed_velocity_spheres] = FUa_out[0:num_fixed_velocity_spheres]
            Fa_out[num_fixed_velocity_spheres:num_spheres] = Fa_in[num_fixed_velocity_spheres:num_spheres]
            Ua_out[0:num_fixed_velocity_spheres] = Ua_in[0:num_fixed_velocity_spheres]
            Ua_out[num_fixed_velocity_spheres:num_spheres] = FUa_out[num_fixed_velocity_spheres:num_spheres]
            (Ta_out, Ea_out) = (Ta_in[:], Ea_in[:])
            Fb_out = [['chen', 'chen', 'chen'] for i in range(num_dumbbells)]
            Ub_out = [['chen', 'chen', 'chen'] for i in range(num_dumbbells)]
            Fb_out[0:num_fixed_velocity_dumbbells] = FUb_out[0:num_fixed_velocity_dumbbells]
            Fb_out[num_fixed_velocity_dumbbells:num_dumbbells] = Fb_in[num_fixed_velocity_dumbbells:num_dumbbells]
            Ub_out[0:num_fixed_velocity_dumbbells] = Ub_in[0:num_fixed_velocity_dumbbells]
            Ub_out[num_fixed_velocity_dumbbells:num_dumbbells] = FUb_out[num_fixed_velocity_dumbbells:num_dumbbells]
            DFb_out = [['chen', 'chen', 'chen'] for i in range(num_dumbbells)]
            HalfDUb_out = [['chen', 'chen', 'chen'] for i in range(num_dumbbells)]
            DFb_out[0:num_fixed_velocity_dumbbells] = DFUb_out[0:num_fixed_velocity_dumbbells]
            DFb_out[num_fixed_velocity_dumbbells:num_dumbbells] = DFb_in[num_fixed_velocity_dumbbells:num_dumbbells]
            HalfDUb_out[0:num_fixed_velocity_dumbbells] = HalfDUb_in[0:num_fixed_velocity_dumbbells]
            HalfDUb_out[num_fixed_velocity_dumbbells:num_dumbbells] = DFUb_out[num_fixed_velocity_dumbbells:num_dumbbells]
            if extract_force_on_wall_due_to_dumbbells:
                print("WARNING: Cannot extract force on wall due to dumbbells in UFTEU mode. Use UFTE mode instead.")

        elif input_form == 'duf':  # Dumbbells only, some imposed velocities
            num_fixed_velocity_dumbbells = num_dumbbells - Ub_in.count(['pippa', 'pippa', 'pippa'])
            try:
                force_vector = construct_force_vector_from_fts(
                    posdata, Fa_in, Ta_in, Ea_in,
                    Ub_in[0:num_fixed_velocity_dumbbells] + Fb_in[num_fixed_velocity_dumbbells:num_dumbbells],
                    HalfDUb_in[0:num_fixed_velocity_dumbbells] + DFb_in[num_fixed_velocity_dumbbells:num_dumbbells])
            except:
                throw_error("DUF mode has been selected but not enough values of U (dumbbell) and F (dumbbell) have been provided. At a guess, not all your dumbbells have either a U or an F.")
            force_vector = np.array(force_vector, float)
            grand_resistance_matrix_duf = fts_to_duf_matrix(num_fixed_velocity_dumbbells,
                                                            posdata, grand_resistance_matrix)
            velocity_vector = np.linalg.solve(grand_resistance_matrix_duf, force_vector)
            (Fa_out, Oa_out, Sa_out, FUb_out, DFUb_out) = deconstruct_velocity_vector_for_fts(posdata, velocity_vector)
            Fb_out = [['chen', 'chen', 'chen'] for i in range(num_dumbbells)]
            Ub_out = [['chen', 'chen', 'chen'] for i in range(num_dumbbells)]
            DFb_out = [['chen', 'chen', 'chen'] for i in range(num_dumbbells)]
            HalfDUb_out = [['chen', 'chen', 'chen'] for i in range(num_dumbbells)]
            Fb_out[0:num_fixed_velocity_dumbbells] = FUb_out[0:num_fixed_velocity_dumbbells]
            Fb_out[num_fixed_velocity_dumbbells:num_dumbbells] = Fb_in[num_fixed_velocity_dumbbells:num_dumbbells]
            Ub_out[0:num_fixed_velocity_dumbbells] = Ub_in[0:num_fixed_velocity_dumbbells]
            Ub_out[num_fixed_velocity_dumbbells:num_dumbbells] = FUb_out[num_fixed_velocity_dumbbells:num_dumbbells]
            DFb_out[0:num_fixed_velocity_dumbbells] = DFUb_out[0:num_fixed_velocity_dumbbells]
            DFb_out[num_fixed_velocity_dumbbells:num_dumbbells] = DFb_in[num_fixed_velocity_dumbbells:num_dumbbells]
            HalfDUb_out[0:num_fixed_velocity_dumbbells] = HalfDUb_in[0:num_fixed_velocity_dumbbells]
            HalfDUb_out[num_fixed_velocity_dumbbells:num_dumbbells] = DFUb_out[num_fixed_velocity_dumbbells:num_dumbbells]
            (Fa_out, Ta_out, Ea_out) = (Fa_in[:], Ta_in[:], Ea_in[:])
            Ua_out, Oa_out = np.array([]), np.array([])

        Fa_out = np.asarray(Fa_out, float)
        Ta_out = np.asarray(Ta_out, float)
        Ea_out = np.asarray(Ea_out, float)
        Ua_out = np.asarray(Ua_out, float)
        Oa_out = np.asarray(Oa_out, float)
        Ub_out = np.asarray(Ub_out, float)
        HalfDUb_out = np.asarray(HalfDUb_out, float)

    elapsed_solve_time = time.time() - solve_time_start
    gen_times.append(elapsed_solve_time)

    if (printout > 0):
        print("Velocities on particles 0-9")
        print(np.asarray(Ua_out[0:10]))
        print(np.asarray(Ub_out[0:10]))
        print("Half Delta U velocity 0-9")
        print(np.asarray(HalfDUb_out[0:10]))
        print("Omegas on particles 0-9")
        print(np.asarray(Oa_out[0:10]))
        print("Forces 0-9 (F)")
        print(np.asarray(Fa_out[0:10]))
        print(np.asarray(Fb_out[0:10]))
        print("Delta F forces 0-9 (DF)")
        print(np.asarray(DFb_out[0:10]))
        print("Torques 0-9 (T)")
        print(np.asarray(Ta_out[0:10]))
        print("Strain rate")
        print(np.asarray(Ea_out))
        print("Stresslets 0-9 (S)")
        print(np.asarray(Sa_out[0:10]))

    return (Fa_out, Ta_out, Sa_out, Fb_out, DFb_out,
            Ua_out, Oa_out, Ea_out, Ub_out, HalfDUb_out,
            last_generated_Minfinity_inverse, gen_times,
            U_infinity, O_infinity, centre_of_background_flow,
            force_on_wall_due_to_dumbbells, last_velocity_vector)
