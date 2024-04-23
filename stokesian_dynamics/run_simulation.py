#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com

"""Main Stokesian Dynamics simulation script.

Run this Python script to perform a Stokesian Dynamics simulation.
See README.md for setup instructions and how to run your first simulation.

You can override the inputs:
    setup_number, input_number, timestep, num_frames
by passing them in as arguments from the command line."""

import numpy as np
import time
import sys
import os
import socket
import datetime
import psutil
import numba
from functions.email import send_email
from functions.shared import (posdata_data, format_elapsed_time, throw_error,
                              throw_warning, feed_particles_from_bottom,
                              sizeof_fmt)
from functions.timestepping import (
    euler_timestep, ab2_timestep, did_something_go_wrong_with_dumbells,
    euler_timestep_rotation, ab2_timestep_rotation, do_we_have_all_size_ratios,
    generate_output_FTSUOE, are_some_of_the_particles_too_close,
    calculate_time_left, format_time_left, format_finish_time,
    wrap_around, add_background_flow_spheres, add_background_flow_dumbbells)
from setups.inputs import input_ftsuoe
from setups.positions import (posdata, setup_description, checkpoint_filename,
                              checkpoint_start_from_frame)
from settings import (
    cutoff_factor, num_frames, view_graphics, viewbox_bottomleft_topright,
    printout, setup_number,
    view_labels, viewing_angle, timestep,
    trace_paths, two_d_plot, email_on_completion,
    save_positions_every_n_timesteps, save_forces_every_n_timesteps,
    save_forces_and_positions_to_temp_file_as_well,
    save_to_temp_file_every_n_timesteps, use_drag_Minfinity,
    use_Minfinity_only, input_form, invert_m_every, explosion_protection,
    input_number, extract_force_on_wall_due_to_dumbbells,
    feed_every_n_timesteps, feed_from_file,
    timestepping_scheme, bead_bead_interactions, fully_2d_problem,
    start_saving_after_first_n_timesteps, rk4_generate_minfinity_for_each_stage)
from resistance_scalars.data import s_dash_range, lam_range_with_reciprocals


# Input description of simulation
args = sys.argv[1:]
number_of_args = len(args)
if number_of_args == 0:
    desc = input('Optional brief description to append to filename: ')
else:
    desc = ""

total_time_start = time.time()


if view_graphics:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib import rcParams
    from functions.graphics import (plot_all_spheres, plot_all_dumbbells,
                                    plot_all_force_lines)


# Initialise
np.set_printoptions(threshold=10000)
grand_mobility_matrix = 0
error = False
previous_step_posdata = posdata
previous_timestamp = time.time()
saved_element_positions = np.array([])
times = [0 for _ in range(num_frames)]
output_folder = "output"
legion_random_id = ""

# Pictures initialise
if view_graphics:
    rcParams.update({'font.size': 11})
    rcParams.update({'figure.dpi': 120, 'figure.figsize': [6, 6],
                     'savefig.dpi': 140})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(viewing_angle[0], viewing_angle[1])
    spheres = list()
    dumbbell_lines = list()
    dumbbell_spheres = list()
    force_lines = list()
    force_text = list()
    torque_lines = list()
    velocity_lines = list()
    velocity_text = list()
    sphere_labels = list()
    angular_velocity_lines = list()
    sphere_lines = list()
    sphere_trace_lines = list()
    dumbbell_trace_lines = list()
    v = viewbox_bottomleft_topright.transpose()
    ax.auto_scale_xyz(v[0], v[1], v[2])
    ax.set_xlim3d(v[0, 0], v[0, 1])
    ax.set_ylim3d(v[1, 0], v[1, 1])
    ax.set_zlim3d(v[2, 0], v[2, 1])
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    if two_d_plot:
        ax.set_proj_type('ortho')
        ax.set_yticks([])
    else:
        ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    ax.set_zlabel("$z$")
    fig.tight_layout()


def initialise_frame():
    spheres = list()
    dumbbell_lines = list()
    dumbbell_spheres = list()
    force_lines = list()
    force_text = list()
    torque_lines = list()
    velocity_lines = list()
    velocity_text = list()
    sphere_labels = list()
    angular_velocity_lines = list()
    sphere_lines = list()
    sphere_trace_lines = list()
    dumbbell_trace_lines = list()
    previous_step_posdata = posdata
    return (spheres, dumbbell_lines, dumbbell_spheres, force_lines, force_text,
            torque_lines, velocity_lines, velocity_text, sphere_labels,
            angular_velocity_lines, sphere_lines, sphere_trace_lines,
            dumbbell_trace_lines, previous_step_posdata)


def generate_frame(frameno, grand_mobility_matrix, view_graphics=True,
                   cutoff_factor=2, viewbox_bottomleft_topright=np.array([]),
                   printout=0, view_labels=False, timestep=0.1, trace_paths=0,
                   input_form='general', filename='', output_folder='output',
                   legion_random_id='', box_bottom_left=np.array([0, 0, 0]),
                   box_top_right=np.array([0, 0, 0])):
    """Perform one timestep of the Stokesian Dynamics simulation."""
    global posdata, previous_step_posdata, times
    global spheres, dumbbell_lines, dumbbell_spheres, sphere_lines
    global sphere_trace_lines, dumbbell_trace_lines
    global force_lines, force_text, torque_lines, velocity_lines, velocity_text
    global angular_velocity_lines, sphere_labels
    global error, previous_timestamp, saved_element_positions, saved_deltax
    global saved_Fa_out, saved_Fb_out, saved_DFb_out, saved_Ea_out
    global saved_Sa_out, saved_force_on_wall_due_to_dumbbells
    global saved_sphere_rotations
    global last_generated_Minfinity_inverse, input_description
    global extract_force_on_wall_due_to_dumbbells
    global last_velocities, last_velocity_vector, checkpoint_start_from_frame
    global feed_every_n_timesteps
    if not (error):
        time_start = time.time()
        if frameno % 20 == 0 and frameno > 0:
            print("[Generating " + filename + "]")
        processing_message = (
            "Processing frame "
            + ("{:" + str(len(str(num_frames))) + ".0f}").format(frameno + 1)
            + "/" + str(num_frames) + "...")
        print(processing_message, end=" ")
        if frameno - checkpoint_start_from_frame == 0 and not numba.config.DISABLE_JIT:
            print("(First timestep will be slower as Numba compiles functions)")
            print(" "*len(processing_message), end=" ")

        '''
        # If we're feeding particles in from the bottom as it falls,
        # this is the function that does that. Otherwise it just passes through.
        posdata = feed_particles_from_bottom(posdata, feed_every_n_timesteps,
                                             feed_from_file, frameno)
        if feed_every_n_timesteps > 0 and printout>0:
            print("num of dumbbells",posdata[3].shape)
        '''

        if not np.array_equal(box_bottom_left-box_top_right, np.array([0, 0, 0])):
            periodic = True
        else:
            periodic = False

        # Input the positions of the particles
        (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
         dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
         element_sizes, element_positions, element_deltax, num_elements,
         num_elements_array, element_type, uv_start, uv_size,
         element_start_count) = posdata_data(posdata)

        '''
        if feed_every_n_timesteps > 0 and printout>0:
            print("number of dumbbells", dumbbell_sizes.shape)
        '''
        # Error checking
        # Commented out for speed but you should put it back in if you're not
        # doing a massive system.
        '''
        if num_elements > 1:
            error = are_some_of_the_particles_too_close(
                error, printout, s_dash_range, sphere_positions,
                dumbbell_positions, dumbbell_deltax, sphere_sizes,
                dumbbell_sizes, element_positions)
            error = do_we_have_all_size_ratios(error, element_sizes,
                                               lam_range, num_spheres)
        '''

    if not (error):
        if frameno % invert_m_every == 0 or frameno == checkpoint_start_from_frame:
            # Generate Minfinity this turn
            regenerate_Minfinity = True
            last_generated_Minfinity_inverse = np.array([0])
        else:
            regenerate_Minfinity = False

        if trace_paths > 0 and frameno % trace_paths == 0:
            previous_step_posdata = posdata

        if frameno == 0 or frameno == checkpoint_start_from_frame:
            Ua_blank = np.array([[0, 0, 0] for _ in range(num_spheres)])
            Ub_blank = np.array([[0, 0, 0] for _ in range(num_dumbbells)])
            DUb_blank = np.array([[0, 0, 0] for _ in range(num_dumbbells)])
            last_velocities = [Ua_blank, Ub_blank, DUb_blank, Ua_blank]  # Oa_blank at the end there
            last_velocity_vector = [0 for _ in range(11*num_spheres + 6*num_dumbbells)]

        if num_spheres == 0:
            new_sphere_positions = np.copy(sphere_positions)
        if num_dumbbells == 0:
            new_dumbbell_positions = np.copy(dumbbell_positions)
            new_dumbbell_deltax = np.copy(dumbbell_deltax)

        if timestepping_scheme in ['euler', 'ab2']:
            # EULER OR TWO-STEP ADAMS-BASHFORTH TIMESTEP
            new_sphere_rotations = (np.copy(sphere_rotations)).astype('float')

            # K1
            (Fa_out_k1, Ta_out_k1, Sa_out_k1, Fb_out_k1, DFb_out_k1,
             Ua_out_k1, Oa_out_k1, Ea_out_k1, Ub_out_k1, HalfDUb_out_k1,
             last_generated_Minfinity_inverse, gen_times,
             U_infinity_k1, O_infinity_k1, centre_of_background_flow,
             Ot_infinity_k1, Et_infinity_k1,
             force_on_wall_due_to_dumbbells_k1,
             last_velocity_vector) = generate_output_FTSUOE(
                posdata, frameno, timestep, input_number,
                last_generated_Minfinity_inverse, regenerate_Minfinity,
                input_form, cutoff_factor, printout, use_drag_Minfinity,
                use_Minfinity_only, extract_force_on_wall_due_to_dumbbells,
                last_velocities, last_velocity_vector)
            # Euler/AB2 timestepping k1
            if (num_spheres > 0):
                Ua_out_plus_infinities_k1, Oa_out_plus_infinities_k1 = add_background_flow_spheres(
                    Ua_out_k1, Oa_out_k1, Ea_out_k1, U_infinity_k1, O_infinity_k1,
                    sphere_positions, centre_of_background_flow)
                if fully_2d_problem:
                    Ua_out_plus_infinities_k1[:, 1] = 0
                if timestepping_scheme == "euler" or frameno - checkpoint_start_from_frame == 0:
                    new_sphere_positions = euler_timestep(
                        sphere_positions, Ua_out_plus_infinities_k1, timestep)
                    new_sphere_rotations = euler_timestep_rotation(
                        sphere_positions, sphere_rotations,
                        new_sphere_positions, new_sphere_rotations,
                        Oa_out_plus_infinities_k1, timestep)
                else:
                    new_sphere_positions = ab2_timestep(
                        sphere_positions, Ua_out_plus_infinities_k1, last_velocities[0], timestep)
                    new_sphere_rotations = ab2_timestep_rotation(
                        sphere_positions, sphere_rotations,
                        new_sphere_positions, new_sphere_rotations,
                        Oa_out_plus_infinities_k1, last_velocities[3],
                        timestep)
                if periodic:
                    new_sphere_positions = wrap_around(
                        new_sphere_positions, box_bottom_left, box_top_right,
                        Ot_infinity_k1, Et_infinity_k1)

            if (num_dumbbells > 0):
                Ub_out_plus_infinities_k1, HalfDUb_out_plus_infinities_k1 = add_background_flow_dumbbells(
                    Ub_out_k1, HalfDUb_out_k1, Ea_out_k1, U_infinity_k1, O_infinity_k1,
                    dumbbell_positions, dumbbell_deltax, centre_of_background_flow)
                if timestepping_scheme == "euler" or frameno - checkpoint_filename == 0:
                    new_dumbbell_positions = euler_timestep(
                        dumbbell_positions, Ub_out_plus_infinities_k1, timestep)
                    new_dumbbell_deltax = euler_timestep(
                        dumbbell_deltax, 2*HalfDUb_out_plus_infinities_k1, timestep)
                else:
                    new_dumbbell_positions = ab2_timestep(
                        dumbbell_positions, Ub_out_plus_infinities_k1, last_velocities[1], timestep)
                    new_dumbbell_deltax = ab2_timestep(
                        dumbbell_deltax, 2*HalfDUb_out_plus_infinities_k1, 2*last_velocities[2], timestep)
                error = did_something_go_wrong_with_dumbells(
                    error, dumbbell_deltax, new_dumbbell_deltax, explosion_protection)
                if periodic:
                    new_dumbbell_positions = wrap_around(
                        new_dumbbell_positions, box_bottom_left, box_top_right,
                        Ot_infinity_k1, Et_infinity_k1)

            if num_spheres > 0 and num_dumbbells == 0:
                last_velocities = [Ua_out_plus_infinities_k1, 0, 0, Oa_out_plus_infinities_k1]
            if num_spheres == 0 and num_dumbbells > 0:
                last_velocities = [0, Ub_out_plus_infinities_k1, HalfDUb_out_plus_infinities_k1, 0]
            if num_spheres > 0 and num_dumbbells > 0:
                last_velocities = [Ua_out_plus_infinities_k1,
                                   Ub_out_plus_infinities_k1,
                                   HalfDUb_out_plus_infinities_k1,
                                   Oa_out_plus_infinities_k1]

            Fa_out = np.asarray(Fa_out_k1)
            Fb_out = np.asarray(Fb_out_k1)
            DFb_out = np.asarray(DFb_out_k1)
            Sa_out = np.asarray(Sa_out_k1)
            force_on_wall_due_to_dumbbells = np.asarray(force_on_wall_due_to_dumbbells_k1)

        elif timestepping_scheme == "rk4":
            # RK4
            new_sphere_rotations = np.copy(sphere_rotations).astype('float')
            sphere_rotations_k1 = np.copy(sphere_rotations).astype('float')
            sphere_rotations_k2 = np.copy(sphere_rotations).astype('float')
            sphere_rotations_k3 = np.copy(sphere_rotations).astype('float')
            # Below lines are required for posdata calls if there are no spheres or dumbbells
            if num_spheres == 0:
                sphere_positions_k1 = np.copy(sphere_positions)
                sphere_positions_k2 = np.copy(sphere_positions)
                sphere_positions_k3 = np.copy(sphere_positions)
            if num_dumbbells == 0:
                dumbbell_positions_k1 = np.copy(dumbbell_positions)
                dumbbell_positions_k2 = np.copy(dumbbell_positions)
                dumbbell_positions_k3 = np.copy(dumbbell_positions)
                dumbbell_deltax_k1 = np.copy(dumbbell_deltax)
                dumbbell_deltax_k2 = np.copy(dumbbell_deltax)
                dumbbell_deltax_k3 = np.copy(dumbbell_deltax)

            # K1
            (Fa_out_k1, Ta_out_k1, Sa_out_k1, Fb_out_k1, DFb_out_k1,
             Ua_out_k1, Oa_out_k1, Ea_out_k1, Ub_out_k1, HalfDUb_out_k1,
             last_generated_Minfinity_inverse, gen_times,
             U_infinity_k1, O_infinity_k1, centre_of_background_flow,
             Ot_infinity_k1, Et_infinity_k1,
             force_on_wall_due_to_dumbbells_k1,
             last_velocity_vector) = generate_output_FTSUOE(
                posdata, frameno, timestep, input_number,
                last_generated_Minfinity_inverse, regenerate_Minfinity,
                input_form, cutoff_factor, printout, use_drag_Minfinity,
                use_Minfinity_only, extract_force_on_wall_due_to_dumbbells,
                last_velocities, last_velocity_vector)
            # Euler timestepping k1
            if (num_spheres > 0):
                Ua_out_plus_infinities_k1, Oa_out_plus_infinities_k1 = add_background_flow_spheres(
                    Ua_out_k1, Oa_out_k1, Ea_out_k1, U_infinity_k1, O_infinity_k1,
                    sphere_positions, centre_of_background_flow)
                sphere_positions_k1 = euler_timestep(
                    sphere_positions, Ua_out_plus_infinities_k1, timestep/2)
                sphere_rotations_k1 = euler_timestep_rotation(
                    sphere_positions, sphere_rotations, sphere_positions_k1,
                    sphere_rotations_k1, Oa_out_plus_infinities_k1, timestep/2)
                if periodic:
                    sphere_positions_k1 = wrap_around(
                        sphere_positions_k1, box_bottom_left, box_top_right,
                        Ot_infinity_k1, Et_infinity_k1)

            if (num_dumbbells > 0):
                Ub_out_plus_infinities_k1, HalfDUb_out_plus_infinities_k1 = add_background_flow_dumbbells(
                    Ub_out_k1, HalfDUb_out_k1, Ea_out_k1, U_infinity_k1, O_infinity_k1,
                    dumbbell_positions, dumbbell_deltax, centre_of_background_flow)
                dumbbell_positions_k1 = euler_timestep(
                    dumbbell_positions, Ub_out_plus_infinities_k1, timestep/2)
                dumbbell_deltax_k1 = euler_timestep(
                    dumbbell_deltax, 2*HalfDUb_out_plus_infinities_k1, timestep/2)
                error = did_something_go_wrong_with_dumbells(
                    error, dumbbell_deltax, dumbbell_deltax_k1, explosion_protection)
                if periodic:
                    dumbbell_positions_k1 = wrap_around(
                        dumbbell_positions_k1, box_bottom_left, box_top_right,
                        Ot_infinity_k1, Et_infinity_k1)

            posdata_k1 = (sphere_sizes, sphere_positions_k1,
                          sphere_rotations_k1, dumbbell_sizes,
                          dumbbell_positions_k1, dumbbell_deltax_k1)
            # See note in inputs, but essentially, might not want to
            # regenerate M_infinity_inverse for frame 1's K2, K3, K4,
            # when if invert_m_every>1, we use frame 1's K1 for everything.
            if not rk4_generate_minfinity_for_each_stage:
                regenerate_Minfinity = False

            # K2
            (Fa_out_k2, Ta_out_k2, Sa_out_k2, Fb_out_k2, DFb_out_k2,
             Ua_out_k2, Oa_out_k2, Ea_out_k2, Ub_out_k2, HalfDUb_out_k2,
             last_generated_Minfinity_inverse_ignore, gen_times_2,
             U_infinity_k2, O_infinity_k2, centre_of_background_flow,
             Ot_infinity_k2, Et_infinity_k2,
             force_on_wall_due_to_dumbbells_k2,
             last_velocity_vector) = generate_output_FTSUOE(
                posdata_k1, frameno + 0.5, timestep, input_number,
                last_generated_Minfinity_inverse, regenerate_Minfinity,
                input_form, cutoff_factor, printout, use_drag_Minfinity,
                use_Minfinity_only, extract_force_on_wall_due_to_dumbbells,
                last_velocities, last_velocity_vector)
            # Euler timestepping k2
            if (num_spheres > 0):
                Ua_out_plus_infinities_k2, Oa_out_plus_infinities_k2 = add_background_flow_spheres(
                    Ua_out_k2, Oa_out_k2, Ea_out_k2, U_infinity_k2, O_infinity_k2,
                    sphere_positions, centre_of_background_flow)
                sphere_positions_k2 = euler_timestep(
                    sphere_positions, Ua_out_plus_infinities_k2, timestep/2)
                sphere_rotations_k2 = euler_timestep_rotation(
                    sphere_positions, sphere_rotations, sphere_positions_k2,
                    sphere_rotations_k2, Oa_out_plus_infinities_k2, timestep/2)
                if periodic:
                    sphere_positions_k2 = wrap_around(
                        sphere_positions_k2, box_bottom_left, box_top_right,
                        Ot_infinity_k2, Et_infinity_k2)

            if (num_dumbbells > 0):
                Ub_out_plus_infinities_k2, HalfDUb_out_plus_infinities_k2 = add_background_flow_dumbbells(
                    Ub_out_k2, HalfDUb_out_k2, Ea_out_k1, U_infinity_k2, O_infinity_k2,
                    dumbbell_positions, dumbbell_deltax, centre_of_background_flow)
                dumbbell_positions_k2 = euler_timestep(
                    dumbbell_positions, Ub_out_plus_infinities_k2, timestep/2)
                dumbbell_deltax_k2 = euler_timestep(
                    dumbbell_deltax, 2*HalfDUb_out_plus_infinities_k2, timestep/2)
                error = did_something_go_wrong_with_dumbells(
                    error, dumbbell_deltax, dumbbell_deltax_k2, explosion_protection)
                if periodic:
                    dumbbell_positions_k2 = wrap_around(
                        dumbbell_positions_k2, box_bottom_left, box_top_right,
                        Ot_infinity_k2, Et_infinity_k2)

            posdata_k2 = (sphere_sizes, sphere_positions_k2,
                          sphere_rotations_k2, dumbbell_sizes,
                          dumbbell_positions_k2, dumbbell_deltax_k2)
            for i in range(len(gen_times)):
                gen_times[i] = gen_times[i] + gen_times_2[i]

            # K3
            (Fa_out_k3, Ta_out_k3, Sa_out_k3, Fb_out_k3, DFb_out_k3,
             Ua_out_k3, Oa_out_k3, Ea_out_k3, Ub_out_k3, HalfDUb_out_k3,
             last_generated_Minfinity_inverse_ignore, gen_times_2,
             U_infinity_k3, O_infinity_k3, centre_of_background_flow,
             Ot_infinity_k3, Et_infinity_k3,
             force_on_wall_due_to_dumbbells_k3,
             last_velocity_vector) = generate_output_FTSUOE(
                posdata_k2, frameno + 0.5, timestep, input_number,
                last_generated_Minfinity_inverse, regenerate_Minfinity,
                input_form, cutoff_factor, printout, use_drag_Minfinity,
                use_Minfinity_only, extract_force_on_wall_due_to_dumbbells,
                last_velocities, last_velocity_vector)
            # Euler timestepping k3
            if (num_spheres > 0):
                Ua_out_plus_infinities_k3, Oa_out_plus_infinities_k3 = add_background_flow_spheres(
                    Ua_out_k3, Oa_out_k3, Ea_out_k3, U_infinity_k3, O_infinity_k3,
                    sphere_positions, centre_of_background_flow)
                sphere_positions_k3 = euler_timestep(
                    sphere_positions, Ua_out_plus_infinities_k3, timestep)
                sphere_rotations_k3 = euler_timestep_rotation(
                    sphere_positions, sphere_rotations, sphere_positions_k3,
                    sphere_rotations_k3, Oa_out_plus_infinities_k3, timestep)
                if periodic:
                    sphere_positions_k3 = wrap_around(
                        sphere_positions_k3, box_bottom_left, box_top_right,
                        Ot_infinity_k3, Et_infinity_k3)

            if (num_dumbbells > 0):
                Ub_out_plus_infinities_k3, HalfDUb_out_plus_infinities_k3 = add_background_flow_dumbbells(
                    Ub_out_k3, HalfDUb_out_k3, Ea_out_k3, U_infinity_k3, O_infinity_k3,
                    dumbbell_positions, dumbbell_deltax, centre_of_background_flow)
                dumbbell_positions_k3 = euler_timestep(
                    dumbbell_positions, Ub_out_plus_infinities_k3, timestep)
                dumbbell_deltax_k3 = euler_timestep(
                    dumbbell_deltax, 2*HalfDUb_out_plus_infinities_k3, timestep)
                error = did_something_go_wrong_with_dumbells(
                    error, dumbbell_deltax, dumbbell_deltax_k3, explosion_protection)
                if periodic:
                    dumbbell_positions_k3 = wrap_around(
                        dumbbell_positions_k3, box_bottom_left, box_top_right,
                        Ot_infinity_k3, Et_infinity_k3)

            posdata_k3 = (sphere_sizes, sphere_positions_k3,
                          sphere_rotations_k3, dumbbell_sizes,
                          dumbbell_positions_k3, dumbbell_deltax_k3)
            for i in range(len(gen_times)):
                gen_times[i] = gen_times[i] + gen_times_2[i]

            # K4
            (Fa_out_k4, Ta_out_k4, Sa_out_k4, Fb_out_k4, DFb_out_k4,
             Ua_out_k4, Oa_out_k4, Ea_out_k4, Ub_out_k4, HalfDUb_out_k4,
             last_generated_Minfinity_inverse_ignore, gen_times_2,
             U_infinity_k4, O_infinity_k4, centre_of_background_flow,
             Ot_infinity_k4, Et_infinity_k4,
             force_on_wall_due_to_dumbbells_k4,
             last_velocity_vector) = generate_output_FTSUOE(
                posdata_k3, frameno + 1, timestep, input_number,
                last_generated_Minfinity_inverse, regenerate_Minfinity,
                input_form, cutoff_factor, printout, use_drag_Minfinity,
                use_Minfinity_only, extract_force_on_wall_due_to_dumbbells,
                last_velocities, last_velocity_vector)
            if (num_spheres > 0):
                Ua_out_plus_infinities_k4, Oa_out_plus_infinities_k4 = add_background_flow_spheres(
                    Ua_out_k4, Oa_out_k4, Ea_out_k4, U_infinity_k4, O_infinity_k4,
                    sphere_positions, centre_of_background_flow)
            if (num_dumbbells > 0):
                Ub_out_plus_infinities_k4, HalfDUb_out_plus_infinities_k4 = add_background_flow_dumbbells(
                    Ub_out_k4, HalfDUb_out_k4, Ea_out_k4, U_infinity_k4, O_infinity_k4,
                    dumbbell_positions, dumbbell_deltax, centre_of_background_flow)
            for i in range(len(gen_times)):
                gen_times[i] = gen_times[i] + gen_times_2[i]

            # Euler timestepping whole thing
            if (num_spheres > 0):
                Ua_out_blended = 1/6 * (Ua_out_plus_infinities_k1
                                        + 2*Ua_out_plus_infinities_k2
                                        + 2*Ua_out_plus_infinities_k3
                                        + Ua_out_plus_infinities_k4)
                Oa_out_blended = 1/6 * (Oa_out_plus_infinities_k1
                                        + 2*Oa_out_plus_infinities_k2
                                        + 2*Oa_out_plus_infinities_k3
                                        + Oa_out_plus_infinities_k4)
                # Ea_out_blended = 1/6 * (Ea_out_k1
                #                         + 2*Ea_out_k2
                #                         + 2*Ea_out_k3
                #                         + Ea_out_k4)

                if fully_2d_problem:
                    Ua_out_blended[:, 1] = 0

                new_sphere_positions = euler_timestep(sphere_positions, Ua_out_blended, timestep)
                if periodic:
                    new_sphere_positions = wrap_around(
                        new_sphere_positions, box_bottom_left, box_top_right,
                        Ot_infinity_k1, Et_infinity_k1)
                new_sphere_rotations = euler_timestep_rotation(
                    sphere_positions, sphere_rotations, new_sphere_positions,
                    new_sphere_rotations, Oa_out_blended, timestep)

            if (num_dumbbells > 0):
                Ub_out_blended = 1/6 * (Ub_out_plus_infinities_k1
                                        + 2*Ub_out_plus_infinities_k2
                                        + 2*Ub_out_plus_infinities_k3
                                        + Ub_out_plus_infinities_k4)
                DUb_out_blended = 2/6 * (HalfDUb_out_plus_infinities_k1
                                         + 2*HalfDUb_out_plus_infinities_k2
                                         + 2*HalfDUb_out_plus_infinities_k3
                                         + HalfDUb_out_plus_infinities_k4)
                new_dumbbell_positions = euler_timestep(
                    dumbbell_positions, Ub_out_blended, timestep)
                new_dumbbell_deltax = euler_timestep(
                    dumbbell_deltax, DUb_out_blended, timestep)
                error = did_something_go_wrong_with_dumbells(
                    error, dumbbell_deltax, new_dumbbell_deltax, explosion_protection)
                if periodic:
                    new_dumbbell_positions = wrap_around(
                        new_dumbbell_positions, box_bottom_left, box_top_right,
                        Ot_infinity_k1, Et_infinity_k1)

            if num_spheres > 0 and num_dumbbells == 0:
                last_velocities = [Ua_out_blended, 0, 0, Oa_out_blended]
            if num_spheres == 0 and num_dumbbells > 0:
                last_velocities = [0, Ub_out_blended, DUb_out_blended, 0]
            if num_spheres > 0 and num_dumbbells > 0:
                last_velocities = [Ua_out_blended, Ub_out_blended,
                                   DUb_out_blended, Oa_out_blended]

            Fa_out = 1/6 * (np.asarray(Fa_out_k1)
                            + 2*np.asarray(Fa_out_k2)
                            + 2*np.asarray(Fa_out_k3)
                            + np.asarray(Fa_out_k4))
            Fb_out = 1/6 * (np.asarray(Fb_out_k1)
                            + 2*np.asarray(Fb_out_k2)
                            + 2*np.asarray(Fb_out_k3)
                            + np.asarray(Fb_out_k4))
            DFb_out = 1/6 * (np.asarray(DFb_out_k1)
                             + 2*np.asarray(DFb_out_k2)
                             + 2*np.asarray(DFb_out_k3)
                             + np.asarray(DFb_out_k4))
            Sa_out = 1/6 * (np.asarray(Sa_out_k1)
                            + 2*np.asarray(Sa_out_k2)
                            + 2*np.asarray(Sa_out_k3)
                            + np.asarray(Sa_out_k4))
            force_on_wall_due_to_dumbbells = 1/6 * (
                np.asarray(force_on_wall_due_to_dumbbells_k1)
                + 2*np.asarray(force_on_wall_due_to_dumbbells_k2)
                + 2*np.asarray(force_on_wall_due_to_dumbbells_k3)
                + np.asarray(force_on_wall_due_to_dumbbells_k4))

        else:
            throw_error("Timestepping scheme not recognised. Choose 'euler', 'ab2' or 'rk4'.")

        for i in gen_times:
            print("[" + format_elapsed_time(i) + "]", end=" ")

        posdata_final = (sphere_sizes, new_sphere_positions,
                         new_sphere_rotations, dumbbell_sizes,
                         new_dumbbell_positions, new_dumbbell_deltax)

        # Save by appending current positions (etc.) to the
        # `saved_element_positions` (etc.) array, which gets bigger and bigger
        # and is saved at the end.
        # In the meantime it is saved as a temp file every
        # `save_to_temp_file_every_n_timesteps` timesteps.
        save_time_start = time.time()
        if (
            save_positions_every_n_timesteps > 0
            and frameno % save_positions_every_n_timesteps == 0
            and frameno >= start_saving_after_first_n_timesteps
        ):
            if frameno == start_saving_after_first_n_timesteps:  # usually 0
                saved_element_positions = np.array([element_positions])
                saved_sphere_rotations = np.array([sphere_rotations])
                saved_deltax = np.array([dumbbell_deltax])
            elif frameno != checkpoint_start_from_frame:
                saved_element_positions = np.append(np.copy(saved_element_positions),
                                                    np.array([element_positions]), 0)
                saved_sphere_rotations = np.append(np.copy(saved_sphere_rotations),
                                                   np.array([sphere_rotations]), 0)
                saved_deltax = np.append(np.copy(saved_deltax),
                                         np.array([dumbbell_deltax]), 0)
            if frameno == num_frames - 1:  # if final frame, add the final position as well
                new_element_positions = np.append(new_sphere_positions,
                                                  new_dumbbell_positions, 0)
                saved_element_positions = np.append(np.copy(saved_element_positions),
                                                    np.array([new_element_positions]), 0)
                saved_sphere_rotations = np.append(np.copy(saved_sphere_rotations),
                                                   np.array([new_sphere_rotations]), 0)
                saved_deltax = np.append(np.copy(saved_deltax),
                                         np.array([new_dumbbell_deltax]), 0)
        if (
            save_forces_every_n_timesteps > 0
            and frameno % save_forces_every_n_timesteps == 0
            and frameno >= start_saving_after_first_n_timesteps
        ):
            if frameno == start_saving_after_first_n_timesteps:  # usually 0
                saved_Fa_out = np.array([Fa_out])
                saved_Fb_out = np.array([Fb_out])
                saved_DFb_out = np.array([DFb_out])
                saved_Sa_out = np.array([Sa_out])
                saved_force_on_wall_due_to_dumbbells = np.array([force_on_wall_due_to_dumbbells])
            elif frameno != checkpoint_start_from_frame:
                saved_Fa_out = np.append(np.copy(saved_Fa_out),
                                         np.array([Fa_out]), 0)
                saved_Fb_out = np.append(np.copy(saved_Fb_out),
                                         np.array([Fb_out]), 0)
                saved_DFb_out = np.append(np.copy(saved_DFb_out),
                                          np.array([DFb_out]), 0)
                saved_Sa_out = np.append(np.copy(saved_Sa_out),
                                         np.array([Sa_out]), 0)
                saved_force_on_wall_due_to_dumbbells = np.append(np.copy(saved_force_on_wall_due_to_dumbbells),
                                                                 np.array([force_on_wall_due_to_dumbbells]), 0)

        # Backup file
        if (save_positions_every_n_timesteps > 0 and
                frameno % save_to_temp_file_every_n_timesteps == 0 and
                save_forces_every_n_timesteps > 0 and
                save_forces_and_positions_to_temp_file_as_well and
                frameno >= start_saving_after_first_n_timesteps):
            np.savez_compressed(output_folder + '/' + filename + legion_random_id + '_TEMP',
                                Fa=saved_Fa_out, Fb=saved_Fb_out, DFb=saved_DFb_out, Sa=saved_Sa_out,
                                centres=saved_element_positions, deltax=saved_deltax,
                                force_on_wall_due_to_dumbbells=saved_force_on_wall_due_to_dumbbells,
                                sphere_rotations=saved_sphere_rotations)
        save_elapsed_time = time.time() - save_time_start
        print("[" + format_elapsed_time(save_elapsed_time) + "]", end=" ")

        # Pictures
        if view_graphics:
            # Remove old spheres
            for q in (spheres + force_lines + torque_lines + velocity_lines
                      + angular_velocity_lines + sphere_lines + dumbbell_lines
                      + dumbbell_spheres):
                q.remove()
            if num_spheres > 0:
                (spheres, sphere_lines, sphere_trace_lines) = plot_all_spheres(
                    ax, frameno, posdata_final, previous_step_posdata,
                    trace_paths, sphere_trace_lines, Fa_out)

            no_line = True
            if num_dumbbells > 0:
                (dumbbell_spheres, dumbbell_lines,
                 dumbbell_trace_lines) = plot_all_dumbbells(
                    ax, posdata_final, trace_paths, dumbbell_trace_lines,
                    Fb_out, DFb_out, no_line=no_line)
            if view_labels:
                (force_lines, force_text) = plot_all_force_lines(
                    ax, posdata_final, Fa_out)
                # torque_lines = plot_all_torque_lines(
                #     ax, posdata_final, Ta_out)
                # (velocity_lines, velocity_text,
                #  sphere_labels) = plot_all_velocity_lines(
                #     ax, posdata_final, Ua_out)
                # Velocity in green
                # angular_velocity_lines = plot_all_angular_velocity_lines(
                #     ax, posdata_final, Oa_out)
                # Ang vel in white with green edging

            ax.set_title("  frame "
                         + ("{:" + str(len(str(num_frames))) + ".0f}").format(frameno + 1)
                         + "/" + str(num_frames),
                         loc='left', y=0.97, fontsize=11)
            ax.set_title("$t$ = "
                         + ("{:" + str(len(str(num_frames * timestep))) + ".2f}").format((frameno+1) * timestep)
                         + "/" + "{:.2f}".format((num_frames) * timestep)
                         + "  ",
                         loc='right', y=0.97, fontsize=11)
            ax.set_title("setup " + str(setup_number)
                         + ", input " + str(input_number),
                         loc='center', y=1.055, fontsize=11)
        posdata = posdata_final

        # Maximum memory the program has used since starting.
        memory_used_bytes = psutil.Process(os.getpid()).memory_info().rss
        print("[ " + sizeof_fmt(memory_used_bytes) + "]", end=" ")

        elapsed_time = time.time() - time_start
        print("[[" + "\033[1m" + format_elapsed_time(elapsed_time) + "\033[0m" + "]]", end=" ")

        # Work out how long left
        times[frameno] = elapsed_time

        time_left, flags = calculate_time_left(times, frameno, num_frames,
                                               invert_m_every,
                                               checkpoint_start_from_frame)

        print("[" + format_time_left(time_left, flags) + "]", end=" ")
        print("[" + format_finish_time(time_left, flags) + "]")

    if error:
        print("No frame " + str(frameno + 1))

    if view_graphics:
        return (spheres, dumbbell_lines, dumbbell_spheres, force_lines,
                force_text, torque_lines, velocity_lines, velocity_text,
                sphere_labels, angular_velocity_lines, sphere_lines,
                sphere_trace_lines, dumbbell_trace_lines)


# --- A little error checking on the inputs.
error = 0

#     1. If there are only dumbbells, you should be inverting using FTS.
(sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
 dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
 element_sizes, element_positions, element_deltax, num_elements,
 num_elements_array, element_type, uv_start, uv_size,
 element_start_count) = posdata_data(posdata)
if num_spheres == 0 and input_form not in ["fts", "duf", "RPY_dumbbells_only",
                                           "stokes_drag_dumbbells_only"]:
    error = throw_error("If you have only dumbbells, you need to use the FTS, "
                        + "DUF, RPY or Stokes Drag input forms.")
if num_spheres > 0 and input_form in ["duf", "RPY_dumbbells_only",
                                      "stokes_drag_dumbbells_only"]:
    error = throw_error("These input forms are for dumbbell-only simulations.")
if num_spheres > 0 and input_form in ["fts"]:
    throw_warning("If you have spheres, it is more likely that you want to "
                  + "use FTE, UFTE or UFTEU form, rather than FTS.")

(Fa_in, Ta_in, Sa_in, Sa_c_in, Fb_in, DFb_in,
 Ua_in, Oa_in, Ea_in, Ea_c_in, Ub_in, HalfDUb_in, input_description,
 U_infinity, O_infinity, centre_of_background_flow, Ot_infinity, Et_infinity,
 box_bottom_left, box_top_right, mu) = input_ftsuoe(
     input_number, posdata, 0, 0.1, [[], [], [], []],
     skip_computation=True, input_form=input_form)

# Are all size ratios in the XYZ table for R2Bexact?
sphere_sizes_unique = list(set(sphere_sizes))
size_ratios = list(set([i/j
                        for i in sphere_sizes_unique
                        for j in sphere_sizes_unique]))
for size_ratio in size_ratios:
    if not np.any(np.isclose(size_ratio, lam_range_with_reciprocals)):
        throw_error("Particle size ratio " + str(size_ratio) + " not in the "
                    + "table of calculated values")

if error == 0:
    # --- End error checking

    timestamp = time.strftime("%y%m%d%H%M")
    setup_descdash = ["-", ""][setup_description == ""]
    input_descdash = ["-", ""][input_description == ""]
    descdash = ["-", ""][desc == ""]

    if checkpoint_filename == "":
        filename = (timestamp
                    + "-s" + str(setup_number)
                    + "-i" + str(input_number)
                    + "-" + str(num_frames) + "fr"
                    + "-t" + str(timestep).replace('.', 'p')
                    + "-M" + str(invert_m_every)
                    + setup_descdash + str(setup_description)
                    + input_descdash + str(input_description)
                    + descdash + str(desc))
    else:
        filename = checkpoint_filename.replace("output\\", "").replace("output/", "").replace("_TEMP.npz", "")
        with np.load(checkpoint_filename) as saved_data:
            saved_Fa_out = saved_data['Fa']
            saved_Fb_out = saved_data['Fb']
            saved_DFb_out = saved_data['DFb']
            saved_Sa_out = saved_data['Sa']
            saved_element_positions = saved_data['centres']
            saved_sphere_rotations = saved_data['sphere_rotations']
            saved_deltax = saved_data['deltax']
            saved_force_on_wall_due_to_dumbbells = saved_data[
                'force_on_wall_due_to_dumbbells']

    warning_formatting_start = "\033[43m\033[30m"
    warning_formatting_end = "\033[39m\033[49m"

    # invert for finding dumbbell contribution to sphere-wall force
    invert_type = ["Fast R\\F", "Slow"][extract_force_on_wall_due_to_dumbbells]
    minf_status = ["ON", "OFF"][use_drag_Minfinity]
    r2b_status = ["ON", "OFF"][use_Minfinity_only]
    bead_bead_status = ["OFF", "ON"][bead_bead_interactions]
    periodic_status = ["ON", "OFF"][np.array_equal(
        box_bottom_left - box_top_right, np.array([0, 0, 0]))]
    if timestepping_scheme == "euler":
        timestep_method = "Euler"
    elif timestepping_scheme == "ab2":
        timestep_method = "AB2"
    elif timestepping_scheme == "rk4":
        timestep_method = "RK4"

    matrix_size = sizeof_fmt(48 * (11 * num_spheres + 6 * num_dumbbells)**2)

    info_box = u'''
+--------------------+-------------------+-----------------------+--------------------+
| Setup:    XXXXXX01 | Minfinity: XXXX05 | Matrix form: XXXXXXXXXXXXXXXXXXXXXXXXXXX09 |
| Input:    XXXXXX02 | R2Bexact:  XXXX06 | Solve using: XXXXXX10 | Numba:    XXXXXX18 |
| Frames:   XXXXXX03 | Bead-bead: XXXX07 | Inv M every: XXXXXX11 | Video:    XXXXXX13 |
| Timestep: XXXXXX04 | Timestep:  XXXX08 | Periodic:    XXXXXX12 | Memory:  ~XXXXXX14 |
+--------------------+-------------------+--------------------------------------------+
| Save every: XXXX16 | Save after: XXX17 | Machine: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX15 |
+--------------------+-------------------+--------------------------------------------+'''

    dic = {"XXXXXX01": str(setup_number),
           "XXXXXX02": str(input_number),
           "XXXXXX03": str(num_frames),
           "XXXXXX04": str(timestep),
           "XXXX05": minf_status,
           "XXXX06": r2b_status,
           "XXXX07": bead_bead_status,
           "XXXX08": timestep_method,
           "XXXXXXXXXXXXXXXXXXXXXXXXXXX09": input_form.upper(),
           "XXXXXX10": invert_type,
           "XXXXXX11": str(invert_m_every),
           "XXXXXX12": periodic_status,
           "XXXXXX13": ["OFF", "ON"][view_graphics],
           "XXXXXX14": matrix_size,
           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX15": socket.gethostname(),
           "XXXX16": str(save_positions_every_n_timesteps),
           "XXX17": str(start_saving_after_first_n_timesteps),
           "XXXXXX18": ["ON", "OFF"][numba.config.DISABLE_JIT]
           }
    warnings = {"XXXXXX01": 0,
                "XXXXXX02": 0,
                "XXXXXX03": 0,
                "XXXXXX04": 0,
                "XXXX05": int(minf_status == "OFF"),
                "XXXX06": int(r2b_status == "OFF"),
                "XXXX07": int(bead_bead_status == "OFF"),
                "XXXX08": int(timestep_method == "Euler"),
                "XXXXXXXXXXXXXXXXXXXXXXXXXXX09": 0,
                "XXXXXX10": 0,
                "XXXXXX11": 0,
                "XXXXXX12": 0,
                "XXXXXX13": 0,
                "XXXXXX14": 0,
                "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX15": 0,
                "XXXX16": 0,
                "XXX17": int(start_saving_after_first_n_timesteps != 0),
                "XXXXXX18": 0
                }
    bold_start = "\033[1m"
    bold_end = "\033[0m"
    for i, j in dic.items():
        if warnings[i] == 1:
            info_box = info_box.replace(i, warning_formatting_start + j.ljust(len(i)) + warning_formatting_end)
        else:
            info_box = info_box.replace(i, bold_start + j.ljust(len(i)) + bold_end)
    print(info_box)

    print("[Generating " + filename + "]")
    print("                      " + " " * 2 * len(str(num_frames))
          + "[ Minfy  ] [invMinfy] [R2Bex'd'] [ U=R\\F  ] [ Saving ] [MaxMemry] [[ Total  ]] [TimeLeft] [ ETA ]")

    if view_graphics and num_frames > 1:
        generate_frame_args = [grand_mobility_matrix, view_graphics,
                               cutoff_factor, viewbox_bottomleft_topright,
                               printout, view_labels, timestep, trace_paths,
                               input_form, filename, output_folder,
                               legion_random_id, box_bottom_left, box_top_right]
        ani = animation.FuncAnimation(fig, generate_frame,
                                      init_func=initialise_frame,
                                      frames=range(checkpoint_start_from_frame, num_frames),
                                      fargs=generate_frame_args, repeat=False,
                                      interval=200)
        plt.show()
    else:
        for frameno in range(checkpoint_start_from_frame, num_frames):
            generate_frame(frameno, grand_mobility_matrix, view_graphics,
                           cutoff_factor, viewbox_bottomleft_topright,
                           printout, view_labels, timestep, trace_paths,
                           input_form, filename, output_folder,
                           legion_random_id, box_bottom_left, box_top_right)

    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
     element_sizes, element_positions, element_deltax, num_elements,
     num_elements_array, element_type, uv_start, uv_size,
     element_start_count) = posdata_data(posdata)

    # Final save
    if save_forces_every_n_timesteps > 0 or save_positions_every_n_timesteps > 0:
        np.savez_compressed(output_folder + '/' + filename + legion_random_id + '',
                            Fa=saved_Fa_out, Fb=saved_Fb_out, DFb=saved_DFb_out, Sa=saved_Sa_out,
                            centres=saved_element_positions, deltax=saved_deltax,
                            force_on_wall_due_to_dumbbells=saved_force_on_wall_due_to_dumbbells,
                            sphere_rotations=saved_sphere_rotations)
        # Remove backup file
        backup_file = output_folder + '/' + filename + legion_random_id + '_TEMP.npz'
        if os.path.exists(backup_file):
            os.remove(backup_file)

    total_elapsed_time = time.time() - total_time_start
    print("[Total time to run " + format_elapsed_time(total_elapsed_time) + "]")
    print("[Complete: " + filename + "]")
    if email_on_completion:
        send_email("SD job on " + socket.gethostname() + " complete",
                   ("The job on " + socket.gethostname() + ", with filename\n\n"
                   + filename + ",\n\n which started on\n\n" +
                   datetime.datetime.fromtimestamp(total_time_start).strftime('%A %d %B %Y, %H:%M')
                   + ",\n\nis now complete. It took "
                   + format_elapsed_time(total_elapsed_time) + "."))
    print("")

    if sys.platform == "win32" and view_graphics:
        print("")
        print("Displaying")
    if view_graphics:
        plt.show()
