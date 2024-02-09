#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 12/05/2014

import numpy as np
import time
from functions.generate_Minfinity import generate_Minfinity
from functions.generate_R2Bexact import generate_R2Bexact
from functions.generate_Minfinity_periodic import generate_Minfinity_periodic
from functions.shared import save_matrix, posdata_data


def generate_grand_resistance_matrix(
        posdata, last_generated_Minfinity_inverse,
        regenerate_Minfinity=False, cutoff_factor=2, printout=0,
        use_drag_Minfinity=False, use_Minfinity_only=False, frameno=0, mu=1):
    """Return the grand resistance matrix for a non-periodic domain.

    Args:
        posdata: Particle position, size and count data
        last_generated_Minfinity_inverse: Precomputed Minfinity_inverse matrix
            from a previous timestep or timestep stage which may want to be
            used here.
        regenerate_Minfinity (bool): Instruction to either use the precomputed
            Minfinity_inverse (False) or to generate a new one (True)
        cutoff_factor: Cutoff factor for separation when R2Bexact applies
        printout: Flag you can use to toggle debug information
        use_drag_Minfinity (bool): Use a drag-based approximation to Minfinity
        use_Minfinity_only (bool): Don't use R2Bexact, just use Minfinity
        frameno (int): Frame number
        mu: Viscosity

    Returns:
        grand_resistance_matrix: Grand resistance matrix
        heading: Human readable name of the matrix
        Minfinity_inverse: Minfinity_inverse matrix
        gen_times: List of runtimes for various parts of the code
    """

    Minfinity_start_time = time.time()
    if not use_drag_Minfinity:
        if regenerate_Minfinity:
            (Minfinity, headingM) = generate_Minfinity(posdata, printout,
                                                       frameno=frameno, mu=mu)
            Minfinity_elapsed_time = time.time() - Minfinity_start_time
            Minfinity_inverse_start_time = time.time()
            Minfinity_inverse = np.linalg.inv(Minfinity)
            if printout > 0:
                print("Minfinity[0:12]")
                print(np.array_str(Minfinity[0:12, 0:12], max_line_width=100000))
        else:
            Minfinity_elapsed_time = time.time() - Minfinity_start_time
            Minfinity_inverse_start_time = time.time()
            Minfinity_inverse = last_generated_Minfinity_inverse
        Minfinity_inverse_elapsed_time = time.time() - Minfinity_inverse_start_time
    else:
        (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
            dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
            element_sizes, element_positions, element_deltax, num_elements,
            num_elements_array, element_type, uv_start, uv_size,
            element_start_count) = posdata_data(posdata)
        Minfinity_inverse = mu*np.diag(
            [sphere_sizes[i/3] for i in range(3*num_spheres)]
            + [1/0.75 * sphere_sizes[i/3]**3 for i in range(3*num_spheres)]
            + [1/0.9 * sphere_sizes[i/5]**3 for i in range(5*num_spheres)]
            + [2 * dumbbell_sizes[i/3] for i in range(3*num_dumbbells)]
            + [2 * dumbbell_sizes[i/3] for i in range(3*num_dumbbells)])
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
        (R2Bexact, heading) = generate_R2Bexact(posdata, printout=printout,
                                                cutoff_factor=cutoff_factor,
                                                frameno=frameno, mu=mu)
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
        save_matrix(grand_resistance_matrix, "Grand Resistance Matrix",
                    "R-" + str(frameno) + ".txt")

    gen_times = [Minfinity_elapsed_time, Minfinity_inverse_elapsed_time,
                 R2Bexact_elapsed_time]
    return (grand_resistance_matrix, heading, Minfinity_inverse, gen_times)


def generate_grand_resistance_matrix_periodic(
        posdata, last_generated_Minfinity_inverse, box_bottom_left, box_top_right,
        regenerate_Minfinity=False, cutoff_factor=2, printout=0,
        use_drag_Minfinity=False, use_Minfinity_only=False, frameno=0, mu=1,
        O_infinity=np.array([0, 0, 0]),
        E_infinity=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        timestep=0.1, centre_of_background_flow=np.array([0, 0, 0]),
        Ot_infinity=np.array([0, 0, 0]),
        Et_infinity=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])):
    """Return the grand resistance matrix for a periodic domain.

    Args:
        posdata: Particle position, size and count data
        last_generated_Minfinity_inverse: Precomputed Minfinity_inverse matrix
            from a previous timestep or timestep stage which may want to be
            used here.
        box_bottom_left, box_top_right: Coordinates of the periodic box.
        regenerate_Minfinity (bool): Instruction to either use the precomputed
            Minfinity_inverse (False) or to generate a new one (True).
        cutoff_factor: Cutoff factor for separation when R2Bexact applies.
        printout: Flag you can use to toggle debug information.
        use_drag_Minfinity (bool): Use a drag-based approximation to Minfinity.
        use_Minfinity_only (bool): Don't use R2Bexact, just use Minfinity.
        frameno (int): Frame number.
        mu: Viscosity.
        O_infinity, E_infinity: Background flow.
        timestep: Timestep size.
        centre_of_background_flow: Centre of any applied shear.
        Ot_infinity, Et_infinity: Integrals of O_infinity and E_infinity dt.

    Returns:
        grand_resistance_matrix: Grand resistance matrix
        heading: Human readable name of the matrix
        Minfinity_inverse: Minfinity_inverse matrix
        gen_times: List of runtimes for various parts of the code
    """

    if not use_drag_Minfinity:  # i.e. if we should do Minfinity properly:
        Minfinity_start_time = time.time()
        if regenerate_Minfinity:
            (Minfinity, headingM) = generate_Minfinity_periodic(
                posdata, box_bottom_left, box_top_right, printout,
                frameno=frameno, mu=mu,
                Ot_infinity=Ot_infinity, Et_infinity=Et_infinity)
            Minfinity_elapsed_time = time.time() - Minfinity_start_time
            Minfinity_inverse_start_time = time.time()
            Minfinity_inverse = np.linalg.inv(Minfinity)
        else:
            Minfinity_elapsed_time = time.time() - Minfinity_start_time
            Minfinity_inverse_start_time = time.time()
            Minfinity_inverse = last_generated_Minfinity_inverse
        Minfinity_inverse_elapsed_time = time.time() - Minfinity_inverse_start_time
    else:
        (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
            dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
            element_sizes, element_positions, element_deltax, num_elements,
            num_elements_array, element_type, uv_start, uv_size,
            element_start_count) = posdata_data(posdata)
        Minfinity_inverse = mu * np.diag(
            [sphere_sizes[i/3] for i in range(3*num_spheres)]
            + [1/0.75 * sphere_sizes[i/3]**3 for i in range(3*num_spheres)]
            + [1/0.9 * sphere_sizes[i/5]**3 for i in range(5*num_spheres)]
            + [2 * dumbbell_sizes[i/3] for i in range(3*num_dumbbells)]
            + [2 * dumbbell_sizes[i/3] for i in range(3*num_dumbbells)])
        Minfinity_elapsed_time = 0
        Minfinity_inverse_elapsed_time = 0

    if not use_Minfinity_only:
        R2Bexact_start_time = time.time()
        (R2Bexact, heading) = generate_R2Bexact(
            posdata, printout=printout, cutoff_factor=cutoff_factor,
            frameno=frameno, mu=mu,
            box_bottom_left=box_bottom_left, box_top_right=box_top_right,
            Ot_infinity=Ot_infinity, Et_infinity=Et_infinity)
        R2Bexact_elapsed_time = time.time() - R2Bexact_start_time
        grand_resistance_matrix = Minfinity_inverse + R2Bexact.toarray()
    else:
        grand_resistance_matrix = Minfinity_inverse
        R2Bexact = 0
        heading = ""
        R2Bexact_elapsed_time = 0

    gen_times = [Minfinity_elapsed_time, Minfinity_inverse_elapsed_time,
                 R2Bexact_elapsed_time]
    return (grand_resistance_matrix, heading, Minfinity_inverse, gen_times)
