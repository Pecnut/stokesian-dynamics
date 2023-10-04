#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 17/10/2014

"""Position setup parameters for simulations, plus some useful distribution
functions.

Create a new setup by adding one to the list of 'if' statements in pos_setup.
"""

import numpy as np
from functions_shared import add_sphere_rotations_to_positions, same_setup_as
from os import sys
from test.position_setups import pos_setup_tests


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
        # Example case 1
        # Durlofsky, Brady & Bossis, 1987. Dynamic simulation of hydro-
        # dynamically interacting particles. Figure 1. This test case looks at
        # horizontal chains of 5, 9 and 15 spheres sedimenting vertically.
        # The instantaneous drag coefficient, F/(6*pi*mu*a*U), is measured for
        # each sphere in the chain, in each case, i.e. it runs for 1 timestep.
        # Here we set up the chain of length 15.
        num_spheres = 15
        sphere_sizes = np.array([1 for i in range(num_spheres)])
        sphere_positions = np.array([[4*i, 0, 0] for i in range(num_spheres)])
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])

    elif n == 2:
        # Example case 2
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
        # Example case 3
        # Brady, Phillips, Jester, Bossis 1988. Dynamic simulation of hydro-
        # dynamically interacting suspensions. Figure 1. Figure corrected by:
        # Sierou & Brady 2001. Accelerated Stokesian Dynamics simulations.
        # Figure 9. This test case is periodic and measures the velocity of a
        # sedimenting, simple cubic array for different particle concentrations.
        num_spheres = 8
        cube_side_length = 8
        sphere_sizes = np.array([1 for i in range(num_spheres)])
        (sphere_positions, box_bottom_left,
            box_top_right) = simple_cubic_8(cube_side_length)
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])

    elif n == 4:
        # Example case 4
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
        # Example case 5
        # Randomly arranged spheres
        num_spheres = 40
        sphere_sizes = np.array([1 for i in range(num_spheres)])
        # L is how wide you want to box for all the particles to fit inside
        # (not just putting the centres inside this box)
        L = 17
        # This will put the centres in a given box size
        sphere_positions = randomise_spheres(
            [-L/2.+1, 0, -L/2.+1], [L/2.-1, 0, L/2.-1], sphere_sizes,
            np.array([]), np.empty([0, 3]))
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])

    elif n == 6:
        # Example case 5
        # Two walls of spheres with dumbbells randomly distributed between them.
        num_lid_particles_each_lid = 45
        num_random_dumbbells = 100*2

        sphere_sizes = np.array([1 for _ in range(num_lid_particles_each_lid*2)])
        sep = 2.00001
        sphere_positions = np.array(
            [[sep*i-(num_lid_particles_each_lid//2)*sep, 0, 0]
             for i in range(num_lid_particles_each_lid)]
            + [[sep*i-(num_lid_particles_each_lid//2)*sep, 0, 11]
               for i in range(num_lid_particles_each_lid)])
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([0.1 for n in range(num_random_dumbbells)])

        random_box_bottom_left = [-17, 0, 1+2*dumbbell_sizes[0]]
        random_box_top_right = [17, 0, 10-2*dumbbell_sizes[0]]
        (dumbbell_positions, dumbbell_deltax) = randomise_dumbbells(
            random_box_bottom_left, random_box_top_right, dumbbell_sizes, dx=2,
            phi=0)

    elif n == 7:
        # To replicate setup of an existing output file
        (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
         dumbbell_positions, dumbbell_deltax) = same_setup_as('FILENAME',
                                                              frameno=0)

    elif n == 8:
        # Two sphere
        sphere_sizes = np.array([1, 1])
        sphere_positions = np.array([[-1.1, 0, 0], [1.1, 0, 0]])
        sphere_rotations = add_sphere_rotations_to_positions(
            sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        dumbbell_sizes = np.array([])
        dumbbell_positions = np.empty([0, 3])
        dumbbell_deltax = np.empty([0, 3])

    try:
        sphere_sizes
    except NameError:
        print("ERROR: You have not inputted a valid position setup number.")

    posdata = (sphere_sizes, sphere_positions, sphere_rotations,
               dumbbell_sizes, dumbbell_positions, dumbbell_deltax)
    return posdata, desc


def random_point_in_box(random_box_bottom_left, random_box_top_right):
    """Return random coordinate inside a box."""
    dimensions = (np.array(random_box_top_right)
                  - np.array(random_box_bottom_left))
    return random_box_bottom_left + np.random.rand(3)*dimensions


def overlaps(pos1, pos2, radius1, radius2):
    """Do objects 1 and 2 overlap each other?"""
    return np.linalg.norm(pos1-pos2) < (radius1+radius2)


def overlaps_periodic(pos1, pos2, radius1, radius2, dxy):
    """Do objects 1 and 2 overlap each other, when separated by dxy?"""
    return np.linalg.norm(pos1-pos2-dxy) < (radius1+radius2)


def randomise_spheres(random_box_bottom_left, random_box_top_right,
                      random_sphere_sizes, current_sphere_sizes,
                      current_sphere_positions):
    """Place spheres randomly inside a box, avoiding currently
    placed spheres.
    """
    sphere_positions_out = current_sphere_positions
    num_current_spheres = current_sphere_sizes.shape[0]
    all_sphere_sizes = np.concatenate([current_sphere_sizes,
                                       random_sphere_sizes])
    print("Randomly distributing " + str(len(random_sphere_sizes))
          + " spheres (using a naive method in Python)... ", end=" ")
    for i in range(len(random_sphere_sizes)):
        while 1 == 1:
            proposed_sphere_position = random_point_in_box(
                random_box_bottom_left, random_box_top_right)
            too_close = 0
            for j in range(num_current_spheres+i):
                if overlaps(proposed_sphere_position, sphere_positions_out[j],
                            random_sphere_sizes[i], all_sphere_sizes[j]):
                    too_close = too_close + 1
            if too_close == 0:
                symbol = '|'
                if (i+1) % 10 == 0:
                    symbol = 'X'
                if (i+1) % 100 == 0:
                    symbol = 'C'
                sys.stdout.write(symbol)
                sys.stdout.flush()
                break
        sphere_positions_out = np.append(sphere_positions_out,
                                         [proposed_sphere_position], axis=0)
    print(" succeeded.")

    random_sphere_positions = sphere_positions_out[
        current_sphere_positions.shape[0]:all_sphere_sizes.shape[0]]
    distance_matrix = np.linalg.norm(
        random_sphere_positions-random_sphere_positions[:, None], axis=2)
    min_element_distance = np.min(distance_matrix[np.nonzero(distance_matrix)])
    two_closest_elements = np.where(distance_matrix == min_element_distance)
    print("Min added sphere s': " + str(min_element_distance/random_sphere_sizes[0]))
    print("Closest two spheres: " + str(two_closest_elements[0]))

    box_dimensions = abs(np.asarray(random_box_top_right)
                         - np.asarray(random_box_bottom_left))

    if box_dimensions[1] == 0:  # 2D
        box_volume = box_dimensions[0]*box_dimensions[2]
        sphere_volumes = np.pi*np.dot(random_sphere_sizes, random_sphere_sizes)
    else:  # 3D
        box_volume = box_dimensions[0]*box_dimensions[1]*box_dimensions[2]
        sphere_volumes = 4/3*np.pi*np.sum(np.asarray(random_sphere_sizes)**3)
    volume_fraction = sphere_volumes/box_volume
    print("Volume fraction: " + str("{:.1f}".format(volume_fraction * 100)) + "%")

    return random_sphere_positions


def randomise_dumbbells(random_box_bottom_left, random_box_top_right,
                        dumbbell_sizes, dx=1, theta='r', phi='r',
                        current_sphere_sizes=np.array([]),
                        current_sphere_positions=np.empty([0, 3])):
    """Place dumbbell beads randomly inside a box, avoiding currently
    placed spheres.
    """
    dumbbell_positions = np.zeros([len(dumbbell_sizes), 3])
    dumbbell_deltax = np.zeros([len(dumbbell_sizes), 3])
    bead_positions = np.zeros([2*len(dumbbell_sizes), 3])
    random_theta = theta == 'r'
    random_phi = phi == 'r'

    print("Randomly distributing " + str(len(dumbbell_sizes))
          + " dumbbells... ", end=" ")

    for i in range(len(dumbbell_sizes)):
        while 1 == 1:
            proposed_bead1_position = random_point_in_box(
                random_box_bottom_left, random_box_top_right)
            too_close = 0
            for j in range(len(current_sphere_sizes)):
                if overlaps(proposed_bead1_position, current_sphere_positions[j],
                            dumbbell_sizes[i], current_sphere_sizes[j]):
                    too_close = too_close + 1
            if too_close == 0:
                for j in range(2*i):
                    if overlaps(proposed_bead1_position, bead_positions[j],
                                dumbbell_sizes[i], dumbbell_sizes[j//2]):
                        too_close = too_close + 1
            if too_close == 0:
                bingo = 0
                for tries in range(100):
                    if random_theta:
                        theta = np.random.rand()*np.pi*2
                    if random_phi:
                        phi = np.random.rand()*np.pi
                    proposed_bead2_position = (proposed_bead1_position
                                               + dx*np.array([np.sin(theta)*np.cos(phi),
                                                              np.sin(theta)*np.sin(phi),
                                                              np.cos(theta)]))
                    if ((proposed_bead2_position >= random_box_bottom_left).all()
                            and (proposed_bead2_position <= random_box_top_right).all()):
                        for j in range(len(current_sphere_sizes)):
                            if overlaps(proposed_bead2_position, current_sphere_positions[j],
                                        dumbbell_sizes[i], current_sphere_sizes[j]):
                                too_close = too_close + 1
                        if too_close == 0:
                            for j in range(2*i):
                                if overlaps(proposed_bead2_position, bead_positions[j],
                                            dumbbell_sizes[i], dumbbell_sizes[j//2]):
                                    too_close = too_close + 1
                        if too_close == 0:
                            bingo = 1
                            break
                if bingo == 1:
                    break

        sys.stdout.write('|')
        sys.stdout.flush()
        bead_positions[2*i] = proposed_bead1_position
        bead_positions[2*i+1] = proposed_bead2_position
        dumbbell_positions[i] = 0.5*(proposed_bead1_position+proposed_bead2_position)
        dumbbell_deltax[i] = (proposed_bead2_position-proposed_bead1_position)
    print(" succeeded.")

    bead_positions = np.concatenate((dumbbell_positions + 0.5*dumbbell_deltax,
                                     dumbbell_positions - 0.5*dumbbell_deltax), axis=0)
    distance_matrix = np.linalg.norm(bead_positions-bead_positions[:, None], axis=2)
    min_element_distance = np.min(distance_matrix[np.nonzero(distance_matrix)])
    two_closest_elements = np.where(distance_matrix == min_element_distance)
    print("Min dumbbell s': " + str(min_element_distance/dumbbell_sizes[0]))
    print("Closest two elements: " + str(two_closest_elements[0]))

    box_dimensions = abs(np.asarray(random_box_top_right) - np.asarray(random_box_bottom_left))

    if box_dimensions[1] == 0:  # 2D
        box_area = box_dimensions[0]*box_dimensions[2]
        box_volume = box_dimensions[0]*box_dimensions[2]*dumbbell_sizes[0]*2
        sphere_areas = np.pi*np.dot(dumbbell_sizes, dumbbell_sizes)*2
        sphere_volumes = 4./3. * np.pi * np.sum(np.asarray(dumbbell_sizes)**3)*2
        area_fraction = sphere_areas/box_area
        volume_fraction = sphere_volumes/box_volume
        print("Area fraction: " + str("{:.1f}".format(area_fraction * 100)) + "%")
        print("Effective volume fraction: " + str("{:.1f}".format(volume_fraction * 100)) + "%")

    else:  # 3D
        box_volume = box_dimensions[0]*box_dimensions[1]*box_dimensions[2]
        sphere_volumes = 4./3. * np.pi * np.sum(np.asarray(dumbbell_sizes)**3)*2
        volume_fraction = sphere_volumes/box_volume
        print("Volume fraction: " + str("{:.1f}".format(volume_fraction * 100)) + "%")

    return dumbbell_positions, dumbbell_deltax


def not_in_spheres(position, current_spheres_positions, current_sphere_size,
                   dumbbell_size):
    """Return bool saying whether proposed dumbbell bead position overlaps with
    an existing sphere."""
    for centre in current_spheres_positions:
        if np.linalg.norm(position-centre) < current_sphere_size+dumbbell_size:
            return False
    return True


def randomise_dumbbells_periodic(random_box_bottom_left, random_box_top_right,
                                 dumbbell_sizes, dx=1, theta='r', phi='r',
                                 current_sphere_sizes=np.array([]),
                                 current_sphere_positions=np.empty([0, 3])):
    """Place dumbbell beads randomly inside a periodic box, avoiding currently
    placed spheres.
    """
    dumbbell_positions = np.zeros([len(dumbbell_sizes), 3])
    dumbbell_deltax = np.zeros([len(dumbbell_sizes), 3])
    bead_positions = np.zeros([2*len(dumbbell_sizes), 3])
    random_theta = theta == 'r'
    random_phi = phi == 'r'

    Dx, Dy, Dz = np.array(random_box_top_right) - np.array(random_box_bottom_left)

    print("Randomly distributing " + str(len(dumbbell_sizes))
          + " dumbbells... ", end=" ")

    for i in range(len(dumbbell_sizes)):
        while 1 == 1:
            proposed_bead1_position = random_point_in_box(
                random_box_bottom_left, random_box_top_right)
            too_close = 0
            for m in [-1, 0, 1]:
                for n in [-1, 0, 1]:
                    dxy = np.array([m*Dx, 0, n*Dz])
                    for j in range(len(current_sphere_sizes)):
                        if overlaps_periodic(proposed_bead1_position,
                                             current_sphere_positions[j],
                                             dumbbell_sizes[i],
                                             current_sphere_sizes[j], dxy):
                            too_close = too_close + 1
                    if too_close == 0:
                        for j in range(2*i):
                            if overlaps_periodic(proposed_bead1_position,
                                                 bead_positions[j],
                                                 dumbbell_sizes[i],
                                                 dumbbell_sizes[j//2], dxy):
                                too_close = too_close + 1
            if too_close == 0:
                bingo = 0
                for tries in range(100):
                    if random_theta:
                        theta = np.random.rand()*np.pi*2
                    if random_phi:
                        phi = np.random.rand()*np.pi
                    proposed_bead2_position = (proposed_bead1_position
                                               + dx*np.array([np.sin(theta)*np.cos(phi),
                                                              np.sin(theta)*np.sin(phi),
                                                              np.cos(theta)]))
                    # NOTE: I have turned off checking whether the second bead is in the box.
                    for m in [-1, 0, 1]:
                        for n in [-1, 0, 1]:
                            dxy = np.array([m*Dx, 0, n*Dz])
                            for j in range(len(current_sphere_sizes)):
                                if overlaps_periodic(proposed_bead2_position,
                                                     current_sphere_positions[j],
                                                     dumbbell_sizes[i],
                                                     current_sphere_sizes[j],
                                                     dxy):
                                    too_close = too_close + 1
                            if too_close == 0:
                                for j in range(2*i):
                                    if overlaps_periodic(proposed_bead2_position,
                                                         bead_positions[j],
                                                         dumbbell_sizes[i],
                                                         dumbbell_sizes[j//2],
                                                         dxy):
                                        too_close = too_close + 1
                    if too_close == 0:
                        bingo = 1
                        break
                if bingo == 1:
                    break

        q = 1000
        qq = 1000
        for m in [-1, 0, 1]:
            for n in [-1, 0, 1]:
                dxy = np.array([m*Dx, 0, n*Dz])
                for p in bead_positions[:i]:
                    q = min(q, np.linalg.norm(proposed_bead2_position - (p+dxy)))
                    qq = min(qq, np.linalg.norm(proposed_bead1_position - (p+dxy)))
        if min(q, qq) < 0.2:
            print(min(q, qq))
            from IPython import embed
            embed()

        sys.stdout.write('|')
        sys.stdout.flush()
        bead_positions[2*i] = proposed_bead1_position
        bead_positions[2*i+1] = proposed_bead2_position
        dumbbell_positions[i] = 0.5*(proposed_bead1_position+proposed_bead2_position)
        dumbbell_deltax[i] = (proposed_bead2_position-proposed_bead1_position)

        # Move the dumbbell_positions (centre) to inside the periodic box
        dumbbell_positions[i] = (np.mod(
            dumbbell_positions[i]-random_box_bottom_left, [Dx, 1e6, Dz]
        ) + random_box_bottom_left)

    print(" succeeded.")

    bead_positions = np.concatenate((dumbbell_positions + 0.5*dumbbell_deltax,
                                     dumbbell_positions - 0.5*dumbbell_deltax), axis=0)
    distance_matrix = np.linalg.norm(bead_positions-bead_positions[:, None], axis=2)
    min_element_distance = np.min(distance_matrix[np.nonzero(distance_matrix)])
    two_closest_elements = np.where(distance_matrix == min_element_distance)
    print("Min dumbbell s': " + str(min_element_distance/dumbbell_sizes[0]))
    print("Closest two elements: " + str(two_closest_elements[0]))
    print("Mean dumbbell pitch: " + "%3.1f" % (
        np.mean(np.arccos(np.abs(np.dot(
            dumbbell_deltax/np.linalg.norm(dumbbell_deltax, axis=1)[:, None],
            np.array([1, 0, 0]))))*180/np.pi, axis=0)
    ) + "°")

    box_dimensions = abs(np.asarray(random_box_top_right)
                         - np.asarray(random_box_bottom_left))

    if box_dimensions[1] == 0:  # 2D
        box_area = box_dimensions[0]*box_dimensions[2]
        box_volume = box_dimensions[0]*box_dimensions[2]*dumbbell_sizes[0]*2
        sphere_areas = np.pi*np.dot(dumbbell_sizes, dumbbell_sizes)*2
        sphere_volumes = 4./3. * np.pi * np.sum(np.asarray(dumbbell_sizes)**3)*2
        area_fraction = sphere_areas/box_area
        volume_fraction = sphere_volumes/box_volume
        print("Area fraction: " + str("{:.1f}".format(area_fraction*100)) + "%")
        print("Effective volume fraction: " + str("{:.1f}".format(volume_fraction*100)) + "%")

    else:  # 3D
        box_volume = box_dimensions[0]*box_dimensions[1]*box_dimensions[2]
        sphere_volumes = 4./3. * np.pi * np.sum(np.asarray(dumbbell_sizes)**3)*2
        volume_fraction = sphere_volumes/box_volume
        print("Volume fraction: " + str("{:.1f}".format(volume_fraction * 100)) + "%")

    return dumbbell_positions, dumbbell_deltax


def randomise_beads_inside_quadrilateral(quadrilateral, dumbbell_sizes,
                                         current_sphere_positions,
                                         current_sphere_size):
    """Place dumbbell beads randomly inside a quadrilateral.

    Naive algorithm so works best for low densities.
    """
    print("Randomly distributing dumbbells... ")
    num_dumbbells = len(dumbbell_sizes)
    num_current_spheres = len(current_sphere_positions)
    random_box_bottom_left = np.array(
        [np.min(quadrilateral[:, 0]), 0, np.min(quadrilateral[:, 1])])
    random_box_top_right = np.array(
        [np.max(quadrilateral[:, 0]), 0, np.max(quadrilateral[:, 1])])
    bead_positions = np.zeros([2*num_dumbbells, 3])

    for i in range(num_dumbbells*2):
        fail = True
        while fail:
            fail = False
            proposed_bead1_position = random_point_in_box(
                random_box_bottom_left, random_box_top_right)
            if not point_inside_polygon(proposed_bead1_position[0],
                                        proposed_bead1_position[2],
                                        quadrilateral):
                fail = True
            if not fail:
                for j in range(num_current_spheres):
                    if not fail and overlaps(proposed_bead1_position,
                                             current_sphere_positions[j],
                                             dumbbell_sizes[i//2],
                                             current_sphere_size):
                        fail = True
            if not fail:
                for j in range(i):
                    if not fail and overlaps(proposed_bead1_position,
                                             bead_positions[j],
                                             dumbbell_sizes[i//2],
                                             dumbbell_sizes[j//2]):
                        fail = True

        sys.stdout.write('|')
        sys.stdout.flush()
        bead_positions[i] = proposed_bead1_position

    pos3 = bead_positions.reshape([2, bead_positions.shape[0]/2, 3])
    pos3_dumbbell = 0.5*(pos3[0] + pos3[1])
    pos3_deltax = 0.5*(pos3[1] - pos3[0])

    print("... succeeded.")

    return pos3_dumbbell, pos3_deltax


def point_inside_polygon(x, y, poly):
    """Given a convex polygon with coords poly, state whether (x,y) is inside.
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def simple_cubic_8(side_length):
    """Return simple cubic array of 8 spheres with given cube side length."""
    s = side_length
    sphere_positions = np.array([[float(i), float(j), float(k)]
                                 for i in [-s/4, s/4]
                                 for j in [-s/4, s/4]
                                 for k in [-s/4, s/4]])
    box_bottom_left = np.array([-s/2, -s/2, -s/2])
    box_top_right = np.array([s/2, s/2, s/2])
    return sphere_positions, box_bottom_left, box_top_right
