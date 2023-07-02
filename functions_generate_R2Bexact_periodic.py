#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 19/10/2016

# Reference: A. K. Townsend, 2017. The mechanics of suspensions. PhD thesis, UCL.

import numpy as np
import math
from functions_shared import posdata_data, close_particles
from scipy import sparse
from inputs import lam_range_with_reciprocals, fully_2d_problem, bead_bead_interactions
from functions_generate_R2Bexact import Af, Bf, Cf, con_Gf, con_Hf, con_Mf


def generate_R2Bexact_periodic(posdata,  box_bottom_left, box_top_right, printout=0, cutoff_factor=2, frameno=0, checkpoint_start_from_frame=0, feed_every_n_timesteps=0, O_infinity=np.array([0, 0, 0]), E_infinity=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), timestep=0.1, centre_of_background_flow=np.array([0, 0, 0]), mu=1, frequency=1, amplitude=1):
    global fully_2d_problem, size_ratio_matrix, average_size_matrix, upper_triangle
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)
    R2Bexact_sidelength = 11*num_spheres + 6*num_dumbbells
    R2Bexact = sparse.lil_matrix((R2Bexact_sidelength, R2Bexact_sidelength), dtype=float)
    bead_positions = np.concatenate([sphere_positions, dumbbell_positions - 0.5*dumbbell_deltax, dumbbell_positions + 0.5*dumbbell_deltax])
    bead_sizes = np.concatenate([sphere_sizes, dumbbell_sizes, dumbbell_sizes])

    closer_than_cutoff_pairs_scaled, displacements_pairs_scaled, distances_pairs_scaled, size_ratios = close_particles(bead_positions, bead_sizes, cutoff_factor, box_bottom_left, box_top_right, O_infinity, E_infinity, frameno, timestep, frequency=frequency, amplitude=amplitude)

    uv_power = [[1, 2, 2, 1, 1], [2, 3, 3, 2, 2], [2, 3, 3, 2, 2], [1, 2, 2, 1, 1], [1, 2, 2, 1, 1]]
    ii = 0
    for (a1_index, a2_index) in closer_than_cutoff_pairs_scaled:
        r = displacements_pairs_scaled[ii]  # vector r
        s_dash = distances_pairs_scaled[ii]  # np.linalg.norm(x)
        if a1_index != a2_index:
            d = r/s_dash
        lam = size_ratios[ii]
        lam_index = np.where(lam_range_with_reciprocals == lam)[0][0]
        lam_index_recip = np.where(lam_range_with_reciprocals == 1./lam)[0][0]
        largest_size = max(bead_sizes[a1_index], bead_sizes[a2_index])
        if a1_index < num_spheres and a2_index < num_spheres:
            # Sphere to sphere
            A_coords = np.s_[a1_index*3:(a1_index+1)*3, a2_index*3:(a2_index+1)*3]
            Bt_coords = np.s_[a1_index*3:(a1_index+1)*3, 3*num_spheres+a2_index*3: 3*num_spheres+(a2_index+1)*3]
            Bt_coords_21 = np.s_[a2_index*3:(a2_index+1)*3, 3*num_spheres+a1_index*3: 3*num_spheres+(a1_index+1)*3]
            Gt_coords = np.s_[a1_index*3:(a1_index+1)*3, 6*num_spheres+a2_index*5: 6*num_spheres+(a2_index+1)*5]
            Gt_coords_21 = np.s_[a2_index*3:(a2_index+1)*3, 6*num_spheres+a1_index*5: 6*num_spheres+(a1_index+1)*5]
            C_coords = np.s_[3*num_spheres+a1_index*3: 3*num_spheres+(a1_index+1)*3, 3*num_spheres+a2_index*3: 3*num_spheres+(a2_index+1)*3]
            Ht_coords = np.s_[3*num_spheres+a1_index*3: 3*num_spheres+(a1_index+1)*3, 6*num_spheres+a2_index*5: 6*num_spheres+(a2_index+1)*5]
            Ht_coords_21 = np.s_[3*num_spheres+a2_index*3: 3*num_spheres+(a2_index+1)*3, 6*num_spheres+a1_index*5: 6*num_spheres+(a1_index+1)*5]
            M_coords = np.s_[6*num_spheres+a1_index*5: 6*num_spheres+(a1_index+1)*5, 6*num_spheres+a2_index*5: 6*num_spheres+(a2_index+1)*5]
            if a1_index == a2_index:

                nearby_beads = []
                nearby_beads_displacements = []
                nearby_beads_distances = []
                for kk in range(len(closer_than_cutoff_pairs_scaled)):
                    (i, j) = closer_than_cutoff_pairs_scaled[kk]
                    if (i == a1_index and i != j):
                        nearby_bead = a2_index
                        nearby_beads_displacements.append(displacements_pairs_scaled[kk])
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])
                    if (j == a1_index and i != j):
                        nearby_bead = a1_index
                        nearby_beads_displacements.append(-displacements_pairs_scaled[kk])  # Note minus sign
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])
                A_sum = 0
                Bt_sum = 0
                C_sum = 0
                Gt_sum = 0
                Ht_sum = 0
                M_sum = 0
                pp = 0
                for p_index in nearby_beads:
                    lam_p = bead_sizes[a1_index]/bead_sizes[p_index]
                    largest_size_p = max(bead_sizes[a1_index], bead_sizes[p_index])
                    if lam_p not in lam_range_with_reciprocals:
                        print("ERROR (Code point D): lambda not in the table of calculated values")
                    lam_index_p = np.where(lam_range_with_reciprocals == lam_p)[0][0]
                    r_p = nearby_beads_displacements[pp]
                    s_dash_p = nearby_beads_distances[pp]
                    d_p = r_p/s_dash_p
                    A_sum += np.asarray([[Af(0, d_p, lam_index_p, s_dash_p, i, j, fully_2d_problem)*largest_size_p**uv_power[0][0] for j in range(3)] for i in range(3)])
                    Bt_sum += np.asarray([[Bf(0, d_p, lam_index_p, s_dash_p, j, i, fully_2d_problem)*largest_size_p**uv_power[0][1] for j in range(3)] for i in range(3)])
                    C_sum += np.asarray([[Cf(0, d_p, lam_index_p, s_dash_p, i, j, fully_2d_problem)*largest_size_p**uv_power[1][1] for j in range(3)] for i in range(3)])
                    Gt_sum += np.asarray([[con_Gf(0, d_p, lam_index_p, s_dash_p, j, i, fully_2d_problem)*largest_size_p**uv_power[0][2] for j in range(5)] for i in range(3)])
                    Ht_sum += np.asarray([[con_Hf(0, d_p, lam_index_p, s_dash_p, j, i, fully_2d_problem)*largest_size_p**uv_power[1][2] for j in range(5)] for i in range(3)])
                    M_sum += np.asarray([[con_Mf(0, d_p, lam_index_p, s_dash_p, i, j, fully_2d_problem)*largest_size_p**uv_power[2][2] for j in range(5)] for i in range(5)])
                    pp = pp + 1
                R2Bexact[A_coords] = A_sum
                R2Bexact[Bt_coords] = Bt_sum
                R2Bexact[C_coords] = C_sum
                R2Bexact[Gt_coords] = Gt_sum
                R2Bexact[Ht_coords] = Ht_sum
                R2Bexact[M_coords] = M_sum

            else:
                R2Bexact[A_coords] = [[Af(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]
                R2Bexact[Bt_coords] = [[Bf(1, -d, lam_index_recip, s_dash, j, i, fully_2d_problem)*largest_size**uv_power[0][1] for j in range(3)] for i in range(3)]
                R2Bexact[C_coords] = [[Cf(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[1][1] for j in range(3)] for i in range(3)]
                R2Bexact[Gt_coords] = [[con_Gf(1, -d, lam_index_recip, s_dash, j, i, fully_2d_problem)*largest_size**uv_power[0][2] for j in range(5)] for i in range(3)]
                R2Bexact[Ht_coords] = [[con_Hf(1, -d, lam_index_recip, s_dash, j, i, fully_2d_problem)*largest_size**uv_power[1][2] for j in range(5)] for i in range(3)]
                R2Bexact[M_coords] = [[con_Mf(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[2][2] for j in range(5)] for i in range(5)]
                if lam == 1:
                    R2Bexact[Bt_coords_21] = -R2Bexact[Bt_coords]
                    R2Bexact[Gt_coords_21] = -R2Bexact[Gt_coords]
                    R2Bexact[Ht_coords_21] = R2Bexact[Ht_coords]
                else:
                    R2Bexact[Bt_coords_21] = [[Bf(1, d, lam_index, s_dash, j, i, fully_2d_problem)*largest_size**uv_power[0][1] for j in range(3)] for i in range(3)]
                    R2Bexact[Gt_coords_21] = [[con_Gf(1, d, lam_index, s_dash, j, i, fully_2d_problem)*largest_size**uv_power[0][2] for j in range(5)] for i in range(3)]
                    R2Bexact[Ht_coords_21] = [[con_Hf(1, d, lam_index, s_dash, j, i, fully_2d_problem)*largest_size**uv_power[1][2] for j in range(5)] for i in range(3)]

        elif a1_index < num_spheres and a2_index >= num_spheres and a2_index < num_spheres + num_dumbbells:
            # Sphere to dumbbell bead 1
            a2_index_d = a2_index-num_spheres
            R14_coords = np.s_[a1_index*3:(a1_index+1)*3,                             11*num_spheres+a2_index_d*3: 11*num_spheres + (a2_index_d+1)*3]
            R24_coords = np.s_[3*num_spheres+a1_index*3:3*num_spheres+(a1_index+1)*3, 11*num_spheres+a2_index_d*3: 11*num_spheres + (a2_index_d+1)*3]
            R34_coords = np.s_[6*num_spheres+a1_index*5:6*num_spheres+(a1_index+1)*5, 11*num_spheres+a2_index_d*3: 11*num_spheres + (a2_index_d+1)*3]

            R2Bexact[R14_coords] = [[Af(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]
            R2Bexact[R24_coords] = [[Bf(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[0][1] for j in range(3)] for i in range(3)]
            R2Bexact[R34_coords] = [[con_Gf(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[0][2] for j in range(3)] for i in range(5)]

        elif a1_index < num_spheres and a2_index >= num_spheres + num_dumbbells:
            # Sphere to dumbbell bead 2
            a2_index_d = a2_index-num_spheres-num_dumbbells
            R15_coords = np.s_[a1_index*3:(a1_index+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3: 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
            R25_coords = np.s_[3*num_spheres+a1_index*3:3*num_spheres+(a1_index+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3: 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
            R35_coords = np.s_[6*num_spheres+a1_index*5:6*num_spheres+(a1_index+1)*5, 11*num_spheres+3*num_dumbbells+a2_index_d*3: 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]

            R2Bexact[R15_coords] = [[Af(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]
            R2Bexact[R25_coords] = [[Bf(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[0][1] for j in range(3)] for i in range(3)]
            R2Bexact[R35_coords] = [[con_Gf(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[0][2] for j in range(3)] for i in range(5)]

        elif a1_index >= num_spheres and a1_index < num_spheres + num_dumbbells and a2_index >= num_spheres and a2_index < num_spheres + num_dumbbells:
            # Dumbbell bead 1 to dumbbell bead 1
            a1_index_d = a1_index-num_spheres
            a2_index_d = a2_index-num_spheres
            R44_coords = np.s_[11*num_spheres+a1_index_d*3:11*num_spheres+(a1_index_d+1)*3, 11*num_spheres+a2_index_d*3: 11*num_spheres+(a2_index_d+1)*3]
            if a1_index == a2_index:
                nearby_beads = []
                nearby_beads_displacements = []
                nearby_beads_distances = []
                for kk in range(len(closer_than_cutoff_pairs_scaled)):
                    (i, j) = closer_than_cutoff_pairs_scaled[kk]
                    if (i == a1_index and i != j):
                        nearby_bead = j  # a2_index
                        nearby_beads_displacements.append(displacements_pairs_scaled[kk])
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])
                    if (j == a1_index and i != j):
                        nearby_bead = i  # a1_index
                        nearby_beads_displacements.append(-displacements_pairs_scaled[kk])  # Note minus sign
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])
                A_sum = 0
                pp = 0
                for p_index in nearby_beads:
                    lam_p = bead_sizes[a1_index]/bead_sizes[p_index]
                    largest_size_p = max(bead_sizes[a1_index], bead_sizes[p_index])
                    if lam_p not in lam_range_with_reciprocals:
                        print("ERROR (Code point D): lambda not in the table of calculated values")
                    lam_index_p = np.where(lam_range_with_reciprocals == lam_p)[0][0]
                    r_p = nearby_beads_displacements[pp]
                    s_dash_p = nearby_beads_distances[pp]
                    d_p = r_p/s_dash_p
                    A_sum = A_sum + np.asarray([[Af(0, d_p, lam_index_p,      s_dash_p, i, j, fully_2d_problem)*largest_size_p**uv_power[0][0] for j in range(3)] for i in range(3)])
                    pp = pp + 1
                R2Bexact[R44_coords] = A_sum
            else:
                if bead_bead_interactions:
                    R2Bexact[R44_coords] = [[Af(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]

        elif a1_index >= num_spheres and a1_index < num_spheres + num_dumbbells and a2_index >= num_spheres + num_dumbbells:
            # Dumbbell bead 1 to dumbbell bead 2
            if bead_bead_interactions:
                a1_index_d = a1_index-num_spheres
                a2_index_d = a2_index-num_spheres-num_dumbbells
                R45_coords = np.s_[11*num_spheres+a1_index_d*3:11*num_spheres+(a1_index_d+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3: 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
                R2Bexact[R45_coords] = [[Af(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]

        else:
            # Dumbbell bead 2 to dumbbell bead 2
            a1_index_d = a1_index-num_spheres-num_dumbbells
            a2_index_d = a2_index-num_spheres-num_dumbbells
            R55_coords = np.s_[11*num_spheres+3*num_dumbbells+a1_index_d*3:11*num_spheres+3*num_dumbbells+(a1_index_d+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3: 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
            if a1_index == a2_index:
                nearby_beads = []
                nearby_beads_displacements = []
                nearby_beads_distances = []
                for kk in range(len(closer_than_cutoff_pairs_scaled)):
                    (i, j) = closer_than_cutoff_pairs_scaled[kk]
                    if (i == a1_index and i != j):
                        nearby_bead = j  # a2_index
                        nearby_beads_displacements.append(displacements_pairs_scaled[kk])
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])
                    if (j == a1_index and i != j):
                        nearby_bead = i  # a1_index
                        nearby_beads_displacements.append(-displacements_pairs_scaled[kk])  # Note minus sign
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])

                A_sum = 0
                pp = 0
                for p_index in nearby_beads:
                    lam_p = bead_sizes[a1_index]/bead_sizes[p_index]
                    largest_size_p = max(bead_sizes[a1_index], bead_sizes[p_index])
                    if lam_p not in lam_range_with_reciprocals:
                        print("ERROR (Code point D): lambda not in the table of calculated values")
                    lam_index_p = np.where(lam_range_with_reciprocals == lam_p)[0][0]
                    r_p = nearby_beads_displacements[pp]
                    s_dash_p = nearby_beads_distances[pp]
                    d_p = r_p/s_dash_p

                    A_sum = A_sum + np.asarray([[Af(0, d_p, lam_index_p,      s_dash_p, i, j, fully_2d_problem)*largest_size_p**uv_power[0][0] for j in range(3)] for i in range(3)])
                    pp = pp + 1
                R2Bexact[R55_coords] = A_sum
            else:
                if bead_bead_interactions:
                    R2Bexact[R55_coords] = [[Af(1, d, lam_index, s_dash, i, j, fully_2d_problem)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]
        ii = ii + 1

    # Scale by 6pi
    R2Bexact = R2Bexact * 6 * math.pi

    # symmetrise
    R2Bexact = sparse.triu(R2Bexact) + sparse.triu(R2Bexact, k=1).transpose()

    # Row and column ops I want are equivalent to doing
    # [ 1  0  0 ]   [ a b c ]   [ 1  0  0 ]
    # [ 0  1  1 ] . [ d e f ] . [ 0  1 -1 ]
    # [ 0 -1  1 ]   [ g h i ]   [ 0  1  1 ]
    #   "L"                         "R"

    # I know that we could generate L and R elsewhere rather than doing it every timestep but it takes 0.01s for a few thousand dumbbells so for now I don't mind
    Lrow = np.array([i for i in range(11*num_spheres + 6*num_dumbbells)] + [i + 11*num_spheres for i in range(3*num_dumbbells)] + [i + 11*num_spheres + 3*num_dumbbells for i in range(3*num_dumbbells)])
    Lcol = np.array([i for i in range(11*num_spheres + 6*num_dumbbells)] + [i + 11*num_spheres + 3*num_dumbbells for i in range(3*num_dumbbells)] + [i + 11*num_spheres for i in range(3*num_dumbbells)])
    Ldata = np.array([1 for i in range(11*num_spheres + 9*num_dumbbells)] + [-1 for i in range(3*num_dumbbells)])
    L = sparse.coo_matrix((Ldata, (Lrow, Lcol)), shape=(11*num_spheres+6*num_dumbbells, 11*num_spheres+6*num_dumbbells))
    R = L.transpose()

    return (mu*(L*R2Bexact*R), "R2Bexact")
