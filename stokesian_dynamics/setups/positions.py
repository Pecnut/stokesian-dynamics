#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 17/10/2014

"""Position setup parameters for simulations, plus some useful distribution
functions.

Create a new setup by adding one to the list of 'if' statements in pos_setup.
"""

import numpy as np
from functions.shared import add_sphere_rotations_to_positions, same_setup_as
import glob
from settings import setup_number, input_number, num_frames, timestep
from setups.tests.positions import pos_setup_tests
from setups.functions_positions import (simple_cubic_8, randomise_spheres,
                                        randomise_dumbbells)


def pos_setup(n):
    """Return position data and description for position setup number n.

    Returns:
        posdata: Position position, size and count data list.
        desc: Description of the setup for use in filenames.
    """
    desc = ""

    if n < 0:
        # Tests for Pytest
        posdata, desc = pos_setup_tests(n)
        return posdata, desc

    if n == 1:
        # Example (a)
        # Durlofsky, Brady & Bossis, 1987. Dynamic simulation of hydro-
        # dynamically interacting particles. Figure 1. This test case looks at
        # horizontal chains of 5, 9 and 15 spheres sedimenting vertically.
        # The instantaneous drag coefficient, F/(6*pi*mu*a*U), is measured for
        # each sphere in the chain, in each case, i.e. it runs for 1 timestep.
        # Here we set up the chain of length 15.
        num_spheres = 15
        sphere_sizes = np.array([1 for _ in range(num_spheres)])
        sphere_positions = np.array([[4*i, 0, 0] for i in range(num_spheres)])
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])

    elif n == 2:
        # Example (b)
        # Durlofsky, Brady & Bossis, 1987. Dynamic simulation of hydro-
        # dynamically interacting particles. Figure 5. This test case considers
        # three particles sedimenting vertically, and looks at their
        # interesting paths over a large number of timesteps.
        sphere_sizes = np.array([1, 1, 1])
        sphere_positions = np.array([[-5, 0, 0], [0, 0, 0], [7, 0, 0]])
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])

    elif n == 3:
        # Example (c)
        # Brady, Phillips, Jester, Bossis 1988. Dynamic simulation of hydro-
        # dynamically interacting suspensions. Figure 1. Figure corrected by:
        # Sierou & Brady 2001. Accelerated Stokesian Dynamics simulations.
        # Figure 9. This test case is periodic and measures the velocity of a
        # sedimenting, simple cubic array for different particle concentrations.
        num_spheres = 8
        cube_side_length = 8
        sphere_sizes = np.array([1 for _ in range(num_spheres)])
        (sphere_positions, box_bottom_left,
            box_top_right) = simple_cubic_8(cube_side_length)
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])

    elif n == 4:
        # Example (d)
        # Two spheres, two dumbbells
        sphere_sizes = np.array([1, 1])
        sphere_positions = np.array([[0, 0, 0], [4.5, 0, 4.5]])
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([0.1, 0.1])
        dumbbell_positions = np.array([[4.5, 0, 0], [0, 0, 4.5]])
        dumbbell_deltax = np.array([[np.sqrt(2), 0, np.sqrt(2)],
                                    [np.sqrt(2), 0, np.sqrt(2)]])

    elif n == 5:
        # Example (e)
        # Randomly arranged spheres
        num_spheres = 41
        sphere_sizes = np.array([1 for _ in range(num_spheres)])
        # L is how wide you want to box for all the particles to fit inside
        # (not just putting the centres inside this box)
        L = 16.2
        # This will put the centres in a given box size
        sphere_positions = randomise_spheres(
            [-L/2+1, 0, -L/2+1], [L/2-1, 0, L/2-1], sphere_sizes,
            np.array([]), np.empty([0, 3]))
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])

    elif n == 6:
        # Example (f)
        # Two walls of spheres with dumbbells randomly distributed between them.
        num_spheres_per_wall = 45
        num_random_dumbbells = 100*2

        sphere_sizes = np.array([1 for _ in range(num_spheres_per_wall*2)])
        sep = 2.00001
        sphere_positions = np.array(
            [[sep*i-(num_spheres_per_wall//2)*sep, 0, 0]
             for i in range(num_spheres_per_wall)]
            + [[sep*i-(num_spheres_per_wall//2)*sep, 0, 11]
               for i in range(num_spheres_per_wall)])
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([0.1 for _ in range(num_random_dumbbells)])

        random_box_bottom_left = [-17, 0, 1+2*dumbbell_sizes[0]]
        random_box_top_right = [17, 0, 10-2*dumbbell_sizes[0]]
        (dumbbell_positions, dumbbell_deltax) = randomise_dumbbells(
            random_box_bottom_left, random_box_top_right, dumbbell_sizes, dx=2,
            phi=0)

    elif n == 7:
        # Example (g)
        # Two walls of spheres with an array of spheres between them.
        # Spheres which will be prescribed velocities, rather than forces,
        # must come first in sphere_sizes, sphere_positions and
        # sphere_rotations.
        num_spheres_per_wall = 45
        num_middle_spheres = 9
        num_spheres = num_spheres_per_wall*2 + num_middle_spheres

        sphere_sizes = np.array([1 for _ in range(num_spheres)])

        sep = 2.00001
        wall_sphere_positions = np.array(
            [[sep*i-(num_spheres_per_wall//2)*sep, 0, -5]
             for i in range(num_spheres_per_wall)]
            + [[sep*i-(num_spheres_per_wall//2)*sep, 0, 5]
               for i in range(num_spheres_per_wall)])

        middle_sphere_positions = np.array([
            [i,0,j] for i in [-4,0,4] for j in [-2.5,0,2.5]])

        sphere_positions = np.concatenate(
            (wall_sphere_positions, middle_sphere_positions))
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))

        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])

    elif n == 8:
        # Example (h)
        # To replicate setup of an existing output file
        (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
         dumbbell_positions, dumbbell_deltax) = same_setup_as('FILENAME',
                                                              frameno=0)


    try:
        sphere_sizes
    except NameError:
        print("ERROR: You have not inputted a valid position setup number.")

    posdata = (sphere_sizes, sphere_positions, sphere_rotations,
               dumbbell_sizes, dumbbell_positions, dumbbell_deltax)
    return posdata, desc


posdata, setup_description = pos_setup(setup_number)

# Checkpointing ---------|

# Checkpoint override. If a file exists with the same input number, position
# number, number of frames and timestep, AND it ends in _TEMP, try to
# continue it.

# Initialise variables
checkpoint_filename = ''
checkpoint_start_from_frame = 0

# NOTE: comment this section to disable Checkpointing
# Otherwise read in list of temporary files
checkpoint_start_from_frame = 0
for filename in sorted(glob.glob("output/*_TEMP.npz"), reverse=True):
    checkpoint_setup_number = int(filename.split("-")[1][1:])
    checkpoint_input_number = int(filename.split("-")[2][1:])
    checkpoint_num_frames = int(filename.split("-")[3][:-2])
    if filename.split("-")[4][-1] == "e":
        raw_timestep = filename.split("-")[4][1:] + "-" + filename.split("-")[5]
    else:
        raw_timestep = filename.split("-")[4][1:]
    checkpoint_timestep = float(raw_timestep.replace("p", "."))
    if (checkpoint_setup_number == setup_number
        and checkpoint_input_number == input_number
        and checkpoint_num_frames == num_frames
            and checkpoint_timestep == timestep):
        ignored_posdata, setup_description = pos_setup(checkpoint_setup_number)
        try:
            with np.load(filename) as data1:
                positions_centres = data1['centres']
                positions_deltax = data1['deltax']
            num_particles = positions_centres.shape[1]
            num_dumbbells = positions_deltax.shape[1]
            num_spheres = num_particles - num_dumbbells
            sphere_positions = positions_centres[-1, 0:num_spheres, :]
            dumbbell_positions = positions_centres[
                -1, num_spheres:num_particles, :]
            dumbbell_deltax = positions_deltax[-1, :, :]
            sphere_sizes = ignored_posdata[0]
            dumbbell_sizes = ignored_posdata[3]
            sphere_rotations = add_sphere_rotations_to_positions(
                sphere_positions, sphere_sizes, np.array([[1,0,0],[0,0,1]]))
            posdata = (sphere_sizes, sphere_positions, sphere_rotations,
                       dumbbell_sizes, dumbbell_positions, dumbbell_deltax)

            checkpoint_filename = filename
            checkpoint_start_from_frame = positions_centres.shape[0]

            word_checkpoint = "\033[42m\033[01m CHECKPOINT FOUND \033[0m\033[49m "
            print(
                f"{word_checkpoint}Continuing file '{checkpoint_filename}' "
                + f"from frame {str(checkpoint_start_from_frame + 1)}/{num_frames}"
            )
        except Exception:
            print("Checkpoint possibly found but the file was corrupt "
                  + "(normally because power was lost midway through saving). "
                  + "Starting from beginning.")

# End checkpointing -----|
