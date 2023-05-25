#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 12/05/2014

from functions_generate_Minfinity import generate_Minfinity
from functions_generate_R2Bexact import generate_R2Bexact
from functions_generate_R2Bexact_periodic import generate_R2Bexact_periodic
from functions_generate_Minfinity_periodic import generate_Minfinity_periodic
from functions_shared import save_matrix, tick, tock, posdata_data
import time
from numpy import linalg
import numpy as np
from inputs import cutoff_factor, num_frames, text_only, viewbox_bottomleft_topright, printout, setup_number, posdata, s_dash_range, lam_range, lam_range_with_reciprocals, XYZ_raw, view_labels, fps, viewing_angle, timestep, trace_paths, two_d_plot, save_positions_every_n_timesteps, save_forces_every_n_timesteps, XYZf, use_XYZd_values, input_form, invert_m_every


def generate_grand_resistance_matrix(posdata, last_generated_Minfinity_inverse, regenerate_Minfinity=False, cutoff_factor=2, printout=0, use_XYZd_values=True, use_drag_Minfinity=False, use_Minfinity_only=False, frameno=0, checkpoint_start_from_frame=0, feed_every_n_timesteps=0, mu=1):
    '''
    if use_XYZd_values:
        d = "-d"
    else:
        d = ""
    '''
    d = "-d"

    Minfinity_start_time = time.time()
    if not(use_drag_Minfinity):
        if regenerate_Minfinity:
            (Minfinity, headingM) = generate_Minfinity(posdata, printout, frameno=frameno, mu=mu)
            Minfinity_elapsed_time = time.time() - Minfinity_start_time
            Minfinity_inverse_start_time = time.time()
            Minfinity_inverse = linalg.inv(Minfinity)
            if printout > 0:
                print("Minfinity[0:12]")
                print(np.array_str(Minfinity[0:12, 0:12], max_line_width=100000))
        else:
            Minfinity_elapsed_time = time.time() - Minfinity_start_time
            Minfinity_inverse_start_time = time.time()
            Minfinity_inverse = last_generated_Minfinity_inverse
        Minfinity_inverse_elapsed_time = time.time() - Minfinity_inverse_start_time
    else:
        (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)
        Minfinity_inverse = mu * np.diag([sphere_sizes[i / 3] for i in xrange(3 * num_spheres)] + [1 / 0.75 * sphere_sizes[i / 3]**3 for i in xrange(3 * num_spheres)] + [1 / 0.9 * sphere_sizes[i / 5]**3 for i in xrange(5 * num_spheres)] + [2 * dumbbell_sizes[i / 3] for i in xrange(3 * num_dumbbells)] + [2 * dumbbell_sizes[i / 3] for i in xrange(3 * num_dumbbells)])
        Minfinity_elapsed_time = 0
        Minfinity_inverse_elapsed_time = 0

    if printout > 0:
        print("Minfinity_inverse")
        print(np.array_str(Minfinity_inverse, max_line_width=100000))

    if not use_Minfinity_only:
        R2Bexact_start_time = time.time()
        # Whether we use the d values or not is selected in inputs.py where we read in XYZ_raw.
        if printout > 1:
            print("cutoff_factor is ", cutoff_factor)
        (R2Bexact, heading) = generate_R2Bexact(posdata, printout, cutoff_factor=cutoff_factor, frameno=frameno, checkpoint_start_from_frame=checkpoint_start_from_frame, feed_every_n_timesteps=feed_every_n_timesteps, mu=mu)
        R2Bexact_elapsed_time = time.time() - R2Bexact_start_time
        if printout > 0:
            print("R2Bexact")
            print(np.array_str(R2Bexact.toarray(), max_line_width=100000))

        grand_resistance_matrix = Minfinity_inverse + R2Bexact.toarray()
        if printout > 0:
            print("grand R")
            print(np.array_str(grand_resistance_matrix, max_line_width=100000))

    else:
        grand_resistance_matrix = Minfinity_inverse
        R2Bexact = 0
        heading = ""
        R2Bexact_elapsed_time = 0

    if (printout > 1):
        print("M infinity")
        print(np.array_str(Minfinity, max_line_width=100000))
        print("\n\nR2Bexact")
        print(np.array_str(R2Bexact.toarray(), max_line_width=100000))
        print("\n\nGrand resistance matrix")
        print(np.array_str(grand_resistance_matrix, max_line_width=100000))
        save_matrix(grand_resistance_matrix, "Grand Resistance Matrix", "R-" + str(frameno) + ".txt")

    gen_times = [Minfinity_elapsed_time, Minfinity_inverse_elapsed_time, R2Bexact_elapsed_time]
    return (grand_resistance_matrix, heading, Minfinity_inverse, gen_times)


def generate_grand_resistance_matrix_periodic(posdata, last_generated_Minfinity_inverse,  box_bottom_left, box_top_right, regenerate_Minfinity=False, cutoff_factor=2, printout=0, use_XYZd_values=True, use_drag_Minfinity=False, use_Minfinity_only=False, frameno=0, checkpoint_start_from_frame=0, feed_every_n_timesteps=0, O_infinity=np.array([0, 0, 0]), E_infinity=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), timestep=0.1, centre_of_background_flow=np.array([0, 0, 0]), mu=1, amplitude=1, frequency=1):
    d = "-d"

    if not(use_drag_Minfinity):  # i.e. if we should do Minfinity properly:
        Minfinity_start_time = time.time()
        if regenerate_Minfinity:
            (Minfinity, headingM) = generate_Minfinity_periodic(posdata, box_bottom_left, box_top_right, printout, frameno=frameno, O_infinity=O_infinity, E_infinity=E_infinity, timestep=timestep, centre_of_background_flow=centre_of_background_flow, mu=mu, frequency=frequency, amplitude=amplitude)
            Minfinity_elapsed_time = time.time() - Minfinity_start_time
            Minfinity_inverse_start_time = time.time()
            Minfinity_inverse = linalg.inv(Minfinity)
        else:
            Minfinity_elapsed_time = time.time() - Minfinity_start_time
            Minfinity_inverse_start_time = time.time()
            Minfinity_inverse = last_generated_Minfinity_inverse
        Minfinity_inverse_elapsed_time = time.time() - Minfinity_inverse_start_time
    else:
        (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)
        Minfinity_inverse = mu * np.diag([sphere_sizes[i / 3] for i in xrange(3 * num_spheres)] + [1 / 0.75 * sphere_sizes[i / 3]**3 for i in xrange(3 * num_spheres)] + [1 / 0.9 * sphere_sizes[i / 5]**3 for i in xrange(5 * num_spheres)] + [2 * dumbbell_sizes[i / 3] for i in xrange(3 * num_dumbbells)] + [2 * dumbbell_sizes[i / 3] for i in xrange(3 * num_dumbbells)])
        Minfinity_elapsed_time = 0
        Minfinity_inverse_elapsed_time = 0

    if not use_Minfinity_only:
        R2Bexact_start_time = time.time()
        (R2Bexact, heading) = generate_R2Bexact_periodic(posdata, box_bottom_left, box_top_right, printout, cutoff_factor=cutoff_factor, frameno=frameno, checkpoint_start_from_frame=checkpoint_start_from_frame,
                                                         feed_every_n_timesteps=feed_every_n_timesteps, O_infinity=O_infinity, E_infinity=E_infinity, timestep=timestep, centre_of_background_flow=centre_of_background_flow, mu=mu, frequency=frequency, amplitude=amplitude)
        R2Bexact_elapsed_time = time.time() - R2Bexact_start_time
        grand_resistance_matrix = Minfinity_inverse + R2Bexact.toarray()
    else:
        grand_resistance_matrix = Minfinity_inverse
        R2Bexact = 0
        heading = ""
        R2Bexact_elapsed_time = 0

    gen_times = [Minfinity_elapsed_time, Minfinity_inverse_elapsed_time, R2Bexact_elapsed_time]
    return (grand_resistance_matrix, heading, Minfinity_inverse, gen_times)
