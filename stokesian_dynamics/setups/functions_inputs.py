#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 08/02/2024

"""Useful functions describing force and velocity inputs for inputs.py.
"""

import numpy as np


def repulsion_forces(strength, tau, num_spheres, num_dumbbells,
                     sphere_positions, dumbbell_positions, dumbbell_deltax,
                     sphere_sizes, dumbbell_sizes, num_sphere_in_each_lid,
                     Fa_in, Fb_in, DFb_in, last_velocities=[0, 0, 0]):
    """Add repulsion forces between close particles.

    Args:
        strength, tau: Parameters in the force function.
        num_spheres, ..., num_sphere_in_each_lid: Positions and counts of
            particles.
        Fa_in, Fb_in, DFb_in: Current forces acting on spheres and dumbbells.
        last_velocities: Velocities of particles at the previous timestep.

    Returns:
        Fa_in, Fb_in, DFb_in: Forces acting on the spheres and dumbbells after
            repulsion added.
    """

    bead_force = [[0, 0, 0] for _ in range(num_spheres + 2*num_dumbbells)]
    bead_positions = np.concatenate([sphere_positions,
                                     dumbbell_positions - 0.5*dumbbell_deltax,
                                     dumbbell_positions + 0.5*dumbbell_deltax])
    bead_sizes = np.concatenate([sphere_sizes, dumbbell_sizes, dumbbell_sizes])
    distance_matrix = np.linalg.norm(bead_positions-bead_positions[:, None],
                                     axis=2)
    average_size_matrix = 0.5*(bead_sizes+bead_sizes[:, None])
    scaled_distance_matrix = distance_matrix/average_size_matrix

    index_matrix = (range(num_spheres+2*num_dumbbells)
                    - np.array(range(num_spheres+2*num_dumbbells))[:, None])

    cutoff = 2.2
    overlapping_or_close = np.where(
        np.logical_and(scaled_distance_matrix > 0,
                       np.logical_and(scaled_distance_matrix < cutoff,
                                      index_matrix >= 0)))
    overlapping_or_close_pairs = zip(overlapping_or_close[0],
                                     overlapping_or_close[1])

    for pair in overlapping_or_close_pairs:
        scaled_overlap = scaled_distance_matrix[pair]
        h = scaled_overlap - 2.
        r = bead_positions[pair[1]] - bead_positions[pair[0]]
        unit_vector = r/np.linalg.norm(r)
        a1 = bead_sizes[pair[0]]
        a2 = bead_sizes[pair[1]]
        repulsion_force_length = 0

        # # Dratler et al. repulsion potential
        # constant = 0.008
        # tau = 1000
        # repulsion_force_length = 0.5*constant*tau*np.exp(-tau*h)/(1+np.exp(-tau*h))

        # Mari et al. electrostatic repulsion
        if h >= 0:
            repulsion_force_length += strength*(2.*a1*a2/(a1+a2))*np.exp(-tau*h)
        else:
            repulsion_force_length += strength*(2.*a1*a2/(a1+a2))

        bead_force[pair[0]] = (np.array(bead_force[pair[0]])
                               + repulsion_force_length*unit_vector)
        bead_force[pair[1]] = (np.array(bead_force[pair[1]])
                               - repulsion_force_length*unit_vector)
    Fa_in[2*num_sphere_in_each_lid:num_spheres] = (Fa_in[
        2*num_sphere_in_each_lid:num_spheres]
        - np.array(bead_force[2*num_sphere_in_each_lid:num_spheres]))
    Fb_in = Fb_in - (np.array(bead_force[num_spheres:num_spheres+num_dumbbells])
                     + np.array(bead_force[num_spheres+num_dumbbells:]))
    DFb_in = DFb_in - (np.array(bead_force[num_spheres+num_dumbbells:])
                       - np.array(bead_force[num_spheres:num_spheres+num_dumbbells]))
    return Fa_in, Fb_in, DFb_in


def oscillatory_shear(amplitude, period, frameno, timestep, phase=0,
                      centre_of_background_flow=np.array([0, 0, 0]),
                      num_spheres=1, unused_axis=1, transpose_shear=False,
                      opposite_direction=False):
    """Define background flow for an oscillatory shear at a given time,
    "a omega cos(omega t + phi)".

    Args:
        amplitude: Amplitude of oscillation. Directly returned.
        period: Period of oscillation.
        frameno: Frame number.
        timestep: Timestep size.
        phase: Phase offset in the oscillation, phi in cos(omega t + phi).
        centre_of_background_flow: Coordinates of background flow centre point.
        num_spheres: Number of spheres in the simulation (used for size of Ea_in).
        unused_axis: 1 means shear /_/ looking side-on.
                     2 means shear /_/ looking top-down.
        transpose_shear (bool): True means e.g. /| ("zx") rather than  _  ("xz").
                                               |/                     /_/

    Returns:
        Ea_in: E^infinity for each spherical particle (at least 1).
        U_infinity, O_infinity: Background flow.
        centre_of_background_flow: As input.
        Ot_infinity, Et_infinity: Integral of O_infinity and E_infinity dt.
    """

    # Angular frequency is omega = 2pi/T,  frequency is f = 1/T
    angular_frequency = 2*np.pi/(period)
    t = frameno*timestep
    gamma = amplitude*np.sin(t*angular_frequency + phase)
    gammadot = (amplitude*angular_frequency)*np.cos(t*angular_frequency + phase)
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
        Ot_infinity = np.array([0, transpose_minus*0.5*gamma, 0])
        E_infinity = [[0, 0, 0.5*gammadot],
                      [0, 0, 0],
                      [0.5*gammadot, 0, 0]]
        Et_infinity = [[0, 0, 0.5*gamma],
                       [0, 0, 0],
                       [0.5*gamma, 0, 0]]
        Ea_in = [E_infinity for _ in range(max(1, num_spheres))]
    elif unused_axis == 2:
        O_infinity = np.array([0, 0, opposite_minus*transpose_minus*-0.5*gammadot])
        Ot_infinity = np.array([0, 0, opposite_minus*transpose_minus*-0.5*gamma])
        E_infinity = [[0, opposite_minus*0.5*gammadot, 0],
                      [opposite_minus*0.5*gammadot, 0, 0],
                      [0, 0, 0]]
        Et_infinity = [[0, opposite_minus*0.5*gamma, 0],
                       [opposite_minus*0.5*gamma, 0, 0],
                       [0, 0, 0]]
        Ea_in = [E_infinity for _ in range(max(1, num_spheres))]
    return (Ea_in, U_infinity, O_infinity, centre_of_background_flow,
            Ot_infinity, Et_infinity)


def constant_shear(gammadot, frameno, timestep,
                   centre_of_background_flow=np.array([0, 0, 0]),
                   num_spheres=1):
    """Define background flow for an constant shear at a given time.

    Args:
        gammadot: Constant shear rate.
        frameno: Frame number.
        timestep: Timestep size.
        centre_of_background_flow: Coordinates of background flow centre point.
        num_spheres: Number of spheres in the simulation (used for size of Ea_in).

    Returns:
        Ea_in: E^infinity for each spherical particle (at least 1).
        U_infinity, O_infinity: Background flow.
        centre_of_background_flow: As input.
        Ot_infinity, Et_infinity: Integral of O_infinity and E_infinity dt.
    """
    t = frameno*timestep
    U_infinity = np.array([0, 0, 0])
    O_infinity = np.array([0, 0.5*gammadot, 0])
    Ot_infinity = np.array([0, 0.5*gammadot*t, 0])
    E_infinity = [[0, 0, 0.5*gammadot],
                  [0, 0, 0],
                  [0.5*gammadot, 0, 0]]
    Et_infinity = [[0, 0, 0.5*gammadot*t],
                   [0, 0, 0],
                   [0.5*gammadot*t, 0, 0]]
    Ea_in = [E_infinity for _ in range(max(1, num_spheres))]
    return (Ea_in, U_infinity, O_infinity, centre_of_background_flow,
            Ot_infinity, Et_infinity)
