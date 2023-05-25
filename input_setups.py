#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 25/05/2017

import numpy as np
from functions_simulation_tools import empty_vectors
from functions_shared import posdata_data
from position_setups import simple_cubic_8


def input_ftsuoe(n, posdata, frameno, timestep, last_velocities, input_form='undefined', video=False, grand_resistance_matrix_fte=0):
    # Initialise all vectors in the left and right-hand sides. Then define num_spheres and num_dumbbells
    (Fa_in, Ta_in, Sa_in, Sa_c_in, Fb_in, DFb_in, Ua_in, Oa_in, Ea_in, Ea_c_in, Ub_in, DUb_in) = empty_vectors(posdata)
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)

    # Fa_in:  Forces on spheres
    # Ta_in:  Torque on spheres
    # Sa_in:  Stresslets on spheres
    # Fb_in:  Forces on dumbbells (total force on dumbbell, F1+F2)
    # DFb_in: Internal force on dumbbells (Delta F = F2-F1)
    # Ua_in:  Velocity of spheres
    # Oa_in:  Angular velocity of spheres
    # Ea_in:  Rate of strain, E^infinity
    # Ub_in:  Velocity of dumbbells
    # DUb_in: HALF the velocity difference of the dumbbells ((U2-U1)/2)

    # Give values. You must give at least half the total number of U/O/E/F/T/S values.
    # If you are giving a mix of F and U values for spheres,   you must put label the spheres   s.t. the fixed velocity spheres   are numbered first.
    # If you are giving a mix of F and U values for dumbbells, you must put label the dumbbells s.t. the fixed velocity dumbbells are numbered first.

    # Defaults
    mu = 1
    desc = ""
    box_bottom_left = np.array([0, 0, 0])
    box_top_right = np.array([0, 0, 0])
    # Background velocity is given by u^infinity = U^infinity + Omega^infinity cross x + E^infinity dot x
    # U^infinity and O^infinity are reset here and are changed for each case if required.
    # E^infinity is input for each case as Ea_in, and is also reset here if you're using FTE form.
    U_infinity = np.array([0, 0, 0])
    O_infinity = np.array([0, 0, 0])
    if input_form == "fte":
        Fa_in[:] = [[0, 0, 0] for i in range(num_spheres)]
        Ta_in[:] = [[0, 0, 0] for i in range(num_spheres)]
        Ea_in[:] = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]] for i in range(num_spheres)]
        Fb_in[:] = [[0, 0, 0] for i in range(num_dumbbells)]
        DFb_in[:] = [[0, 0, 0] for i in range(num_dumbbells)]
    elif input_form == "fts":
        Fa_in[:] = [[0, 0, 0] for i in range(num_spheres)]
        Ta_in[:] = [[0, 0, 0] for i in range(num_spheres)]
        Sa_in[:] = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]] for i in range(num_spheres)]
        Fb_in[:] = [[0, 0, 0] for i in range(num_dumbbells)]
        DFb_in[:] = [[0, 0, 0] for i in range(num_dumbbells)]
    if input_form == "ufte":
        Ta_in[:] = [[0, 0, 0] for i in range(num_spheres)]
        Ea_in[:] = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]] for i in range(num_spheres)]
        Fb_in[:] = [[0, 0, 0] for i in range(num_dumbbells)]
        DFb_in[:] = [[0, 0, 0] for i in range(num_dumbbells)]
    centre_of_background_flow = np.array([0, 0, 0])
    amplitude = 0
    frequency = 0
    num_sphere_in_each_lid = 0

    if n == 1:
        # Gravity
        Fa_in[:] = [[0, 0, -1] for i in range(num_spheres)]
        desc = "gravity"

    elif n == 2:
        # Gravity in periodic domain
        Fa_in[:] = [[0, 0, -1] for i in range(num_spheres)]
        sphere_positions, box_bottom_left, box_top_right = simple_cubic_8(8)
        # sphere_positions is ignored here, but to activate periodicity, you have to set box_bottom_left and box_top_right.
        desc = "gravity-periodic"

    elif n == 3:
        # Oscillatory background flow, about the point (2.25,0,2.25)
        natural_deltax = 2.
        spring_constant = -1
        DFb_in[:] = [list(spring_constant*(dumbbell_deltax[i]-natural_deltax*dumbbell_deltax[i]/np.linalg.norm(dumbbell_deltax[i]))) for i in range(num_dumbbells)]
        # Simple shear with speed gammadot
        startfromframe = 0
        # amplitude is amplitude at z = 1
        (Ea_in, U_infinity, O_infinity, centre_of_background_flow, amplitude, frequency) = oscillatory_shear(amplitude=1./3., period=1, start_from_frame=0,  centre_of_background_flow=np.array([2.25, 0, 2.25]), frameno=frameno, timestep=timestep, num_spheres=num_spheres)
        desc = "oscillatory-background-flow"

    elif n == 4:
        # Repulsive force
        (Fa_in, Fb_in, DFb_in) = repulsion_forces(100, 20, num_spheres, num_dumbbells, sphere_positions, dumbbell_positions, dumbbell_deltax, sphere_sizes, dumbbell_sizes, num_sphere_in_each_lid, Fa_in, Fb_in, DFb_in)
        desc = "repulsion"

    elif n == 5:
        # Force half the spheres to move to the left with a given velocity, and force the rest to move to the right.
        Ua_in[:] = [[-1, 0, 0] for i in range(num_spheres/2)] + [[1, 0, 0] for i in range(num_spheres/2, num_spheres)]

    elif n == 6:
        # Continuous shear
        gammadot = 1
        O_infinity = np.array([0, 0.5*gammadot, 0])
        Ea_in = [[[0, 0, 0.5*gammadot], [0, 0, 0], [0.5*gammadot, 0, 0]] for i in range(max(1, num_spheres))]
        desc = "continuous-shear"

    elif n == 7:
        # Gravity
        Fa_in[1] = [0, 0, -1]
        desc = "gravity"

    else:
        Fa_in = np.array([[99999, -31415, 21718]])  # Just something to flag up on the other side that there's a problem

    return Fa_in, Ta_in, Sa_in, Sa_c_in, Fb_in, DFb_in, Ua_in, Oa_in, Ea_in, Ea_c_in, Ub_in, DUb_in, desc, U_infinity, O_infinity, centre_of_background_flow, amplitude, frequency, box_bottom_left, box_top_right, mu


def repulsion_forces(strength, tau, num_spheres, num_dumbbells, sphere_positions, dumbbell_positions, dumbbell_deltax, sphere_sizes, dumbbell_sizes, num_sphere_in_each_lid, Fa_in, Fb_in, DFb_in, last_velocities=[0, 0, 0]):
    bead_force = [[0, 0, 0] for i in range(num_spheres + 2*num_dumbbells)]
    bead_positions = np.concatenate([sphere_positions, dumbbell_positions - 0.5*dumbbell_deltax, dumbbell_positions + 0.5*dumbbell_deltax])
    bead_sizes = np.concatenate([sphere_sizes, dumbbell_sizes, dumbbell_sizes])
    distance_matrix = np.linalg.norm(bead_positions-bead_positions[:, None], axis=2)
    average_size_matrix = 0.5*(bead_sizes+bead_sizes[:, None])
    scaled_distance_matrix = distance_matrix/average_size_matrix

    index_matrix = range(num_spheres+2*num_dumbbells)-np.array(range(num_spheres+2*num_dumbbells))[:, None]

    cutoff = 2.2
    overlapping_or_close = np.where(np.logical_and(scaled_distance_matrix > 0, np.logical_and(scaled_distance_matrix < cutoff, index_matrix >= 0)))
    overlapping_or_close_pairs = zip(overlapping_or_close[0], overlapping_or_close[1])

    numpairs = str(len(overlapping_or_close_pairs))
    constant = strength

    for pair in overlapping_or_close_pairs:
        scaled_overlap = scaled_distance_matrix[pair]
        h = scaled_overlap - 2.
        unit_vector = (bead_positions[pair[1]] - bead_positions[pair[0]])/np.linalg.norm((bead_positions[pair[1]] - bead_positions[pair[0]]))
        a1 = bead_sizes[pair[0]]
        a2 = bead_sizes[pair[1]]
        repulsion_force_length = 0

        '''
        # Dratler et al. repulsion potential
        constant = 0.008
        tau =1000
        repulsion_force_length = 0.5*constant*tau*np.exp(-tau*h)/(1+np.exp(-tau*h))
        '''

        # Mari et al. electrostatic repulsion
        if h >= 0:
            repulsion_force_length += strength*(2.*a1*a2/(a1+a2))*np.exp(-tau*h)
        else:
            repulsion_force_length += strength*(2.*a1*a2/(a1+a2))

        bead_force[pair[0]] = np.array(bead_force[pair[0]]) + repulsion_force_length*unit_vector
        bead_force[pair[1]] = np.array(bead_force[pair[1]]) - repulsion_force_length*unit_vector
    Fa_in[2*num_sphere_in_each_lid:num_spheres] = Fa_in[2*num_sphere_in_each_lid:num_spheres] - np.array(bead_force[2*num_sphere_in_each_lid:num_spheres])
    Fb_in = Fb_in - (np.array(bead_force[num_spheres:num_spheres+num_dumbbells]) + np.array(bead_force[num_spheres+num_dumbbells:]))
    DFb_in = DFb_in - (np.array(bead_force[num_spheres+num_dumbbells:]) - np.array(bead_force[num_spheres:num_spheres+num_dumbbells]))
    return Fa_in, Fb_in, DFb_in


def oscillatory_shear(amplitude, period, start_from_frame, frameno, timestep, centre_of_background_flow=np.array([0, 0, 0]), num_spheres=1, unused_axis=1, transpose_shear=False, opposite_direction=False):
    # unused_axis=1 means shear /_/ looking side-on.
    # unused_axis=2 means shear /_/ looking top-down.
    #                        /|                     _
    # transpose_shear means |/  ("zx") rather than /_/ ("xz"), for example.
    angular_frequency = 2*np.pi/(period)  # Angular frequency is omega = 2pi/T,  frequency is f = 1/T
    t = ((frameno+start_from_frame)*timestep)       # why the devil was there a 0.5 there?!
    gammadot = (amplitude*angular_frequency)*np.cos(t*angular_frequency)
    U_infinity = np.array([0, 0, 0])
    if transpose_shear:
        transpose_minus = -1
    else:
        transpose_minus = 1
    if opposite_direction:
        opposite_minus = -1
    else:
        opposite_minus = 1
    if unused_axis == 1:
        O_infinity = np.array([0, transpose_minus*0.5*gammadot, 0])
        Ea_in = [[[0, 0, 0.5*gammadot], [0, 0, 0], [0.5*gammadot, 0, 0]] for i in range(max(1, num_spheres))]
    elif unused_axis == 2:
        O_infinity = np.array([0, 0, opposite_minus*transpose_minus*-0.5*gammadot])
        Ea_in = [[[0, opposite_minus*0.5*gammadot, 0], [opposite_minus*0.5*gammadot, 0, 0], [0, 0, 0]] for i in range(max(1, num_spheres))]
    return Ea_in, U_infinity, O_infinity, centre_of_background_flow, amplitude, angular_frequency
