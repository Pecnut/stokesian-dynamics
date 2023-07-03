#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 12/05/2014

import numpy as np
from numpy import sqrt
import time
import sys
from numba import njit


def throw_error(message):
    """Quit immediately with a given error message."""
    word_error = "\033[41m\033[01m ERROR \033[0m "
    sys.exit(word_error + message)
    return 1


def throw_warning(message):
    """Print a warning message."""
    word_error = "\033[43m\033[30m WARNING \033[0m "
    print(word_error + message)


@njit
def norm(x):
    """Returns Euclidean norm of a vector x."""
    return (x[0]**2 + x[1]**2 + x[2]**2)**0.5


def x(posdata, a1_index, a2_index):
    """Returns displacement between particles a1 and a2. Sign convention is a2 - a1."""
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)
    return element_positions[a2_index]-element_positions[a1_index]


def lam(posdata, a1_index, a2_index):
    """Returns size of particle a2 divided by size of particle 1."""
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)
    return element_sizes[a2_index]/element_sizes[a1_index]


def kron(i, j):
    """Kronecker delta:  delta_ij"""
    return float(i == j)


def kronkron(i, j, k, l):
    """Product of two Kronecker deltas:  delta_ij * delta_kl"""
    return float(i == j & k == l)


@njit
def levi(i, j, k):
    """Levi-Civita symbol:  epsilon_ijk"""
    if i == j or j == k or k == i:
        return 0
    elif [i, j, k] in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        return 1
    else:
        return -1


def contraction(i, j, k):
    """Element of contraction matrix \mathcal{E}_ijk.
    
    Use for converting 3x3 symmetric traceless matrix to 5-element vector."""
    return np.array([[[0.5*(sqrt(3)+1), 0, 0], [0, 0.5*(sqrt(3)-1), 0], [0, 0, 0]], [[0, sqrt(2), 0], [0, 0, 0], [0, 0, 0]], [[0.5*(sqrt(3)-1), 0, 0], [0, 0.5*(sqrt(3)+1), 0], [0, 0, 0]], [[0, 0, sqrt(2)], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, sqrt(2)], [0, 0, 0]]])[i, j, k]


def symmetrise(a):
    """Symmetrise a matrix, `a`"""
    return a + a.T - np.diag(a.diagonal())


def save_matrix(matrix, heading, filename):
    """Save a matrix to a text file with a given heading as the first line in the file."""
    with open(filename, 'a') as outputfile:
        np.savetxt(outputfile, np.array([heading]), fmt="%s")
        np.savetxt(outputfile, matrix, newline="\n", fmt="% .8e")


def posdata_data(posdata):
    """Return useful particle position, size and count information from a single `posdata` list."""
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes, dumbbell_positions, dumbbell_deltax) = posdata

    num_spheres = sphere_sizes.shape[0]
    num_dumbbells = dumbbell_sizes.shape[0]

    element_sizes = np.append(sphere_sizes, dumbbell_sizes)
    element_positions = np.append(sphere_positions, dumbbell_positions, 0)
    element_deltax = np.append(np.zeros([num_spheres, 3]), dumbbell_deltax, 0)
    num_elements = num_spheres + num_dumbbells

    num_elements_array = [num_spheres, num_dumbbells]
    element_type = [0, 0, 0, 1, 1]
    uv_start = [0, 3*num_spheres, 6*num_spheres, 11*num_spheres, 11*num_spheres + 3*num_dumbbells]
    uv_size = [3, 3, 5, 3, 3]
    element_start_count = [0, 0, 0, num_spheres, num_spheres]

    return (sphere_sizes, np.asarray(sphere_positions), sphere_rotations,  dumbbell_sizes, np.asarray(dumbbell_positions), np.asarray(dumbbell_deltax), num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count)


def add_sphere_rotations_to_positions(sphere_positions, sphere_sizes, sphere_rotations):
    """Take a pair of vectors which start at the origin and use them as particle rotation vectors.
    
    Args:
        sphere_positions: Array of sphere positions
        sphere_sizes: Array of sphere sizes
        sphere_rotations: A pair of unit vectors which represent 1 unit 'horizontal'
            and 'vertical'.
    
    Returns:
        b: Array of pairs of 'rotation vectors' for each particle which start at the 
            particle centre and extend in the 'horizontal' and 'vertical' direction
            specified. These can now be used in `sphere_rotations` definitions.
    """
    b = np.zeros([sphere_positions.shape[0], 2, sphere_positions.shape[1]])
    num_spheres = sphere_sizes.shape[0]
    addrot1 = (sphere_sizes * np.tile(sphere_rotations[0, :], (num_spheres, 1)).transpose()).transpose()
    addrot2 = (sphere_sizes * np.tile(sphere_rotations[1, :], (num_spheres, 1)).transpose()).transpose()
    b[:, 0, :] = sphere_positions + addrot1
    b[:, 1, :] = sphere_positions + addrot2
    return b


def is_dumbbell(posdata, a_index):
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)
    return a_index >= num_spheres


def is_sphere(posdata, a_index):
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)
    return a_index < num_spheres


def tick(clock, num):
    clock[num] = time.time()


def tock(clock, num):
    print(f"[{format_elapsed_time(time.time() - clock[num])}]", end=" ")


def format_elapsed_time(elapsed_time):
    if elapsed_time >= 86400:
        tr2b_m, tr2b_s = divmod(elapsed_time, 60)
        tr2b_h, tr2b_m = divmod(tr2b_m, 60)
        tr2b_d, tr2b_h = divmod(tr2b_h, 24)
        return "%dd%2d:%02d:%02d" % (tr2b_d, tr2b_h, tr2b_m, tr2b_s)
    elif elapsed_time >= 3600:
        tr2b_m, tr2b_s = divmod(elapsed_time, 60)
        tr2b_h, tr2b_m = divmod(tr2b_m, 60)
        return "%2d:%02d:%02d" % (tr2b_h, tr2b_m, tr2b_s)
    elif elapsed_time >= 60:
        tr2b_m, tr2b_s = divmod(elapsed_time, 60)
        tr2b_h, tr2b_m = divmod(tr2b_m, 60)
        return "   %2d:%02d" % (tr2b_m, tr2b_s)
    else:
        return '{:7.1f}'.format(elapsed_time) + "s"


def same_setup_as(filename, frameno=0, sphere_size=1, dumbbell_size=0.1, local=True):
    """Load posdata list from a saved file at a given frame number."""
    
    # Allows you to choose between different locations for your storage file.
    if local:
        data1 = np.load("output/" + filename + ".npz")
    else:
        data1 = np.load("output/" + filename + ".npz")
    positions_centres = data1['centres']
    positions_deltax = data1['deltax']
    num_particles = positions_centres.shape[1]
    num_dumbbells = positions_deltax.shape[1]
    num_spheres = num_particles - num_dumbbells
    sphere_positions = positions_centres[frameno, 0:num_spheres, :]
    dumbbell_positions = positions_centres[frameno, num_spheres:num_particles, :]
    dumbbell_deltax = positions_deltax[frameno, :, :]
    sphere_sizes = np.array([sphere_size for _ in range(num_spheres)])
    dumbbell_sizes = np.array([dumbbell_size for _ in range(num_dumbbells)])
    sphere_rotations = add_sphere_rotations_to_positions(sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
    return (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax)


def feed_particles_from_bottom(posdata, feed_every_n_timesteps, feed_from_file, frameno, reference_particle=0):
    """For sedimentation sims: Delete particles that are too far above the reference 
        particle and add new ones beneath it. Reads new particle positions in from a
        previously saved simulation file at a given frame number.        
    """

    if feed_every_n_timesteps == 0 or frameno % feed_every_n_timesteps != 0 or frameno == 0:
        return posdata
    # For simplicity's sake, I am going to assume :
    # 1. That the reference_particle is a sphere, and that the other particles are all dumbbell beads.
    # 2. That all dumbbell beads are the same size (0.1).
    # This may not always be the case (e.g. walls) but for now it is easier to implement.
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax) = posdata
    num_beads_before = dumbbell_sizes.shape[0]*2
    bead_positions = np.concatenate((dumbbell_positions + 0.5*dumbbell_deltax, dumbbell_positions - 0.5*dumbbell_deltax), axis=0)
    height_from_reference_particle_to_top_bead = np.max(bead_positions[:, 2]-sphere_positions[reference_particle, 2])
    height_from_reference_particle_to_bottom_bead = -np.min(bead_positions[:, 2]-sphere_positions[reference_particle, 2])
    height_needed_below = 0.5*(height_from_reference_particle_to_top_bead - height_from_reference_particle_to_bottom_bead)
    print("sphere position", sphere_positions)
    print("height_needed_below", height_needed_below)
    # chop off top
    keep_bead_positions = bead_positions[bead_positions[:, 2] <= sphere_positions[reference_particle, 2] + height_from_reference_particle_to_bottom_bead + height_needed_below]
    num_deleted_beads_from_top = num_beads_before - keep_bead_positions.shape[0]
    # Prepare beads to add below. Read in a full block of beads.
    (add_sphere_sizes, add_sphere_positions, add_sphere_rotations, add_dumbbell_sizes, add_dumbbell_positions, add_dumbbell_deltax) = same_setup_as(feed_from_file, frameno=0, sphere_size=sphere_sizes[0], dumbbell_size=dumbbell_sizes[0])  # load file
    add_bead_positions = np.concatenate((add_dumbbell_positions + 0.5*add_dumbbell_deltax, add_dumbbell_positions - 0.5*add_dumbbell_deltax), axis=0)
    add_bead_positions = add_bead_positions[add_bead_positions[:, 2].argsort()]  # Sort the beads in height order
    add_bead_positions_to_add = add_bead_positions[:num_deleted_beads_from_top]
    top_new_bead_position = np.max(add_bead_positions_to_add[:, 2])
    add_bead_positions_to_add = add_bead_positions_to_add - [0, 0, top_new_bead_position - np.min(bead_positions[:, 2])]  # Shift down to the bottom of the box
    print("add_bead_positions_to_add (size of)", add_bead_positions_to_add.shape)
    print("top of the add_bead_positions_to_add", np.max(add_bead_positions_to_add[:, 2]))
    print("bottom of the add_bead_positions_to_add", np.min(add_bead_positions_to_add[:, 2]))
    print("bottom of the beads that are already there", np.min(keep_bead_positions[:, 2]))
    # check for overlaps
    add_bead_positions_to_add_really = np.empty([0, 3])
    for s in add_bead_positions_to_add:
        overlap = 0
        for t in keep_bead_positions:
            if np.linalg.norm(s-t) <= 2*dumbbell_sizes[0]:
                overlap = 1
        if overlap == 0:
            add_bead_positions_to_add_really = np.append(add_bead_positions_to_add_really, np.array([s]), axis=0)
    # stitch together
    ("add_bead_positions_to_add_really", add_bead_positions_to_add_really)
    new_bead_positions = np.concatenate((keep_bead_positions, add_bead_positions_to_add_really), axis=0)
        # There is a problem if the overall number of dumbbells changes when saving. To fix this:
        # If the number of beads has increased, delete the last ones
    if new_bead_positions.shape[0] > num_beads_before:
        new_bead_positions = new_bead_positions[:num_beads_before]
        # If the number of beads is not enough, add some of the old ones back in
    if new_bead_positions.shape[0] < num_beads_before:
        # Not checking for overlap with the deleted ones. In theory this shouldn't happen...
        num_beads_needed = num_beads_before - new_bead_positions.shape[0]
        deleted_beads = bead_positions[bead_positions[:, 2] > sphere_positions[reference_particle, 2] + height_from_reference_particle_to_bottom_bead + height_needed_below]
        readd_beads = deleted_beads[:num_beads_needed]
        new_bead_positions = np.concatenate([new_bead_positions, readd_beads], axis=0)

    # form back into dumbbells
    beads1 = new_bead_positions[:new_bead_positions.shape[0]/2]
    beads2 = new_bead_positions[new_bead_positions.shape[0]/2:(new_bead_positions.shape[0]/2)*2]
    new_dumbbell_positions = 0.5*(beads1+beads2)
    new_dumbbell_deltax = beads2-beads1
    new_dumbbell_sizes = np.array(
        [dumbbell_sizes[0] for _ in range(new_dumbbell_positions.shape[0])]
    )

    return (sphere_sizes, sphere_positions, sphere_rotations, new_dumbbell_sizes, new_dumbbell_positions, new_dumbbell_deltax)


def close_particles(bead_positions, bead_sizes, cutoff_factor, 
                    box_bottom_left=np.array([0, 0, 0]), 
                    box_top_right=np.array([0, 0, 0]), 
                    O_infinity=np.array([0, 0, 0]), 
                    E_infinity=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), 
                    frameno=0, timestep=0.1, amplitude=1, frequency=1):
    """Find particles that are closer than a given cutoff."""
        
    from scipy.spatial.distance import pdist, squareform
    cutoff = 2*cutoff_factor
    middle_of_box = 0.5*(box_bottom_left+box_top_right)-box_bottom_left
    box_dimensions = box_top_right - box_bottom_left
    periodic = not np.array_equal(box_dimensions, np.array([0, 0, 0]))
    if periodic:
        # unshear box,
        basis_canonical = np.diag(box_dimensions)  # which equals np.array([[Lx,0,0],[0,Ly,0],[0,0,Lz]])

        # NOTE: For CONTINUOUS shear, set the following
        #time_t = frameno*timestep
        # sheared_basis_vectors_add_on = (np.cross(np.array(O_infinity)*time_t,basis_canonical).transpose() + np.dot(np.array(E_infinity)*time_t,(basis_canonical).transpose())).transpose()# + basis_canonical
        # NOTE: For OSCILLATORY shear, set the following (basically there isn't a way to find out shear given E)
        time_t = frameno*timestep
        gamma = amplitude*np.sin(time_t*frequency)
        Ot_infinity = np.array([0, 0.5*gamma, 0])
        Et_infinity = [[0, 0, 0.5*gamma], [0, 0, 0], [0.5*gamma, 0, 0]]
        sheared_basis_vectors_add_on = (np.cross(Ot_infinity, basis_canonical).transpose() + np.dot(Et_infinity, (basis_canonical).transpose())).transpose()

        sheared_basis_vectors_add_on_mod = np.mod(sheared_basis_vectors_add_on, box_dimensions)
        sheared_basis_vectors = basis_canonical + sheared_basis_vectors_add_on_mod
        # Hence
        unsheared_positions = np.dot(np.dot(bead_positions, np.linalg.inv(sheared_basis_vectors)), basis_canonical)
        closer_than_cutoff_pairs = []
        displacements_pairs = np.empty([0, 3])
        # This lists pairs that are closer than 4 units, regardless of size ratio.
        for i in range(len(unsheared_positions)):
            sphere_position = unsheared_positions[i]
            sheared_centred_box = np.dot(np.mod(np.dot(unsheared_positions-sphere_position, np.linalg.inv(basis_canonical)) + 0.5, [1, 1, 1]) - 0.5, sheared_basis_vectors)
            # Shear again
            particles_in_cube = np.where(np.max(np.abs(sheared_centred_box), axis=1) <= cutoff)  # Take particles only within a cube of the correct radius
            particles_in_radius = np.where(np.linalg.norm(sheared_centred_box[particles_in_cube], axis=1) <= cutoff)  # Take particles only within a sphere of the correct radius
            particles_in_radius_ids = particles_in_cube[0][particles_in_radius][particles_in_cube[0][particles_in_radius] >= i]
            closer_than_cutoff_pairs += [(i, j) for j in particles_in_radius_ids]
            displacements_pairs = np.concatenate((displacements_pairs, sheared_centred_box[particles_in_radius_ids]), axis=0)
        distances_pairs = np.linalg.norm(displacements_pairs, axis=1)
        # Now take them and see if they are still closer when you divide by their average size
        jj = 0
        closer_than_cutoff_pairs_scaled = []
        displacements_pairs_scaled = np.empty([0, 3])
        distances_pairs_scaled = []
        size_ratios = []
        for (a1_index, a2_index) in closer_than_cutoff_pairs:
            if 2*distances_pairs[jj]/(bead_sizes[a1_index]+bead_sizes[a2_index]) <= cutoff:
                closer_than_cutoff_pairs_scaled.append((a1_index, a2_index))
                displacements_pairs_scaled = np.concatenate((displacements_pairs_scaled, np.array([2*displacements_pairs[jj]/(bead_sizes[a1_index]+bead_sizes[a2_index])])), axis=0)
                distances_pairs_scaled.append(2*distances_pairs[jj]/(bead_sizes[a1_index]+bead_sizes[a2_index]))
                size_ratios.append(bead_sizes[a2_index]/bead_sizes[a1_index])
            jj = jj + 1

        closer_than_cutoff = closer_than_cutoff_pairs_scaled

    else:
        closer_than_cutoff_pairs = []
        displacements_pairs = np.empty([0, 3])

        distance_matrix = squareform(pdist(bead_positions))
        average_size = 0.5 * (bead_sizes + bead_sizes[:, None])
        distance_over_average_size = distance_matrix / average_size  # Matrix of s'
        closer_than_cutoff = np.where(distance_over_average_size < cutoff)
        closer_than_cutoff = zip(closer_than_cutoff[0], closer_than_cutoff[1])
        closer_than_cutoff = [f for f in closer_than_cutoff if f[0] <= f[1]]

        size_ratios = [bead_sizes[b]/bead_sizes[a] for (a, b) in closer_than_cutoff]
        displacements_pairs_scaled = np.array([2*(bead_positions[b] - bead_positions[a])/(bead_sizes[a]+bead_sizes[b]) for (a, b) in closer_than_cutoff])
        distances_pairs_scaled = [distance_over_average_size[a, b] for (a, b) in closer_than_cutoff]

    return closer_than_cutoff, displacements_pairs_scaled, distances_pairs_scaled, size_ratios
