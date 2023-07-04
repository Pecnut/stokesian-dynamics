#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

# Reference: A. K. Townsend, 2017. The mechanics of suspensions. PhD thesis, UCL.
# See section 2.4.2 for definitions, which are adapted from
# section 7.2 of Kim & Karrila, 2005. Microhydrodynamics.

import numpy as np
from functions_shared import posdata_data, levi, close_particles, s2, s3
from scipy import sparse
from inputs import s_dash_range, lam_range_with_reciprocals, XYZ_raw, bead_bead_interactions
from numba import njit


kronmatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

@njit
def L1(d, i, j):
    """Unit displacement tensor L1 used in A & C. See PhD thesis (2.70)."""
    return d[i]*d[j]


@njit
def L2(d, i, j):
    """Unit displacement tensor L2 used in A & C. See PhD thesis (2.70)."""
    return kronmatrix[i][j] - d[i]*d[j]


@njit
def L3(d, i, j):
    """Unit displacement tensor L3 used in B. See PhD thesis (2.70)."""
    if i == j:
        return 0
    else:
        return sum([levi(i, j, k)*d[k] for k in range(3) if k not in [i, j]])


@njit
def L4(d, i, j, k):
    """Unit displacement tensor L4 used in G. See PhD thesis (2.70)."""
    return (d[i]*d[j] - kronmatrix[i][j]/3) * d[k]


@njit
def L5(d, i, j, k):
    """Unit displacement tensor L5 used in G. See PhD thesis (2.70)."""
    return (d[i]*kronmatrix[j][k] + d[j]*kronmatrix[i][k] - 2*d[i]*d[j]*d[k])


@njit
def L6(d, i, j, k):
    """Unit displacement tensor L6 used in H. See PhD thesis (2.70)."""
    return sum([
        levi(i, k, l)*d[l]*d[j] for l in range(3) if l not in [i, k]
    ]) + sum([
        levi(j, k, l)*d[l]*d[i] for l in range(3) if l not in [k, j]])


@njit
def L7(d, i, j, k, l):
    """Unit displacement tensor L7 used in M. See PhD thesis (2.70)."""
    return 1.5 * (d[i]*d[j] - kronmatrix[i][j]/3) * (d[k]*d[l] - kronmatrix[k][l]/3)


@njit
def L8(d, i, j, k, l):
    """Unit displacement tensor L8 used in M. See PhD thesis (2.70)."""
    return 0.5 * (d[i]*kronmatrix[j][l]*d[k]
                  + d[j]*kronmatrix[i][l]*d[k]
                  + d[i]*kronmatrix[j][k]*d[l]
                  + d[j]*kronmatrix[i][k]*d[l]
                  - 4*d[i]*d[j]*d[k]*d[l])


@njit
def L9(d, i, j, k, l):
    """Unit displacement tensor L9 used in M. See PhD thesis (2.70)."""
    return 0.5 * (kronmatrix[i][k]*kronmatrix[j][l]
                  + kronmatrix[j][k]*kronmatrix[i][l]
                  - kronmatrix[i][j]*kronmatrix[k][l]
                  + d[i]*d[j]*kronmatrix[k][l]
                  + kronmatrix[i][j]*d[k]*d[l]
                  - d[i]*kronmatrix[j][l]*d[k]
                  - d[j]*kronmatrix[i][l]*d[k]
                  - d[i]*kronmatrix[j][k]*d[l]
                  - d[j]*kronmatrix[i][k]*d[l]
                  + d[i]*d[j]*d[k]*d[l])


@njit
def XA(gam, s, lam_index):
    return XYZ(0, gam, s, lam_index)  # s:=s'


@njit
def YA(gam, s, lam_index):
    return XYZ(1, gam, s, lam_index)


@njit
def YB(gam, s, lam_index):
    return XYZ(2, gam, s, lam_index)


@njit
def XC(gam, s, lam_index):
    return XYZ(3, gam, s, lam_index)


@njit
def YC(gam, s, lam_index):
    return XYZ(4, gam, s, lam_index)


@njit
def XG(gam, s, lam_index):
    return XYZ(5, gam, s, lam_index)


@njit
def YG(gam, s, lam_index):
    return XYZ(6, gam, s, lam_index)


@njit
def YH(gam, s, lam_index):
    return XYZ(7, gam, s, lam_index)


@njit
def XM(gam, s, lam_index):
    return XYZ(8, gam, s, lam_index)


@njit
def YM(gam, s, lam_index):
    return XYZ(9, gam, s, lam_index)


@njit
def ZM(gam, s, lam_index):
    return XYZ(10, gam, s, lam_index)


@njit
def Af(gamma, d, lam_index, ss, i, j):
    """Element ij of R2Bexact submatrix A. See PhD thesis (2.69)."""
    XAg = XA(gamma, ss, lam_index)
    YAg = YA(gamma, ss, lam_index)
    return XAg * L1(d, i, j) + YAg * L2(d, i, j)


@njit
def Bf(gamma, d, lam_index, ss, i, j):
    """Element ij of R2Bexact submatrix B. See PhD thesis (2.69)."""
    YBg = YB(gamma, ss, lam_index)
    return YBg * L3(d, i, j)


@njit
def Cf(gamma, d, lam_index, ss, i, j):
    """Element ij of R2Bexact submatrix C. See PhD thesis (2.69)."""
    XCg = XC(gamma, ss, lam_index)
    YCg = YC(gamma, ss, lam_index)
    return XCg * L1(d, i, j) + YCg * L2(d, i, j)


@njit
def Gf(gamma, d, lam_index, ss, i, j, k):
    """Element ijk of uncondensed R2Bexact submatrix G. See PhD thesis (2.69)."""
    XGg = XG(gamma, ss, lam_index)
    YGg = YG(gamma, ss, lam_index)
    return XGg * L4(d, i, j, k) + YGg * L5(d, i, j, k)


@njit
def Hf(gamma, d, lam_index, ss, i, j, k):
    """Element ijk of uncondensed R2Bexact submatrix H. See PhD thesis (2.69)."""
    YHg = YH(gamma, ss, lam_index)
    return YHg * L6(d, i, j, k)


@njit
def Mf(gamma, d, lam_index, ss, i, j, k, l):
    """Element ijkl of uncondensed R2Bexact submatrix M. See PhD thesis (2.69)."""
    XMg = XM(gamma, ss, lam_index)
    YMg = YM(gamma, ss, lam_index)
    ZMg = ZM(gamma, ss, lam_index)
    return XMg * L7(d, i, j, k, l) + YMg * L8(d, i, j, k, l) + ZMg * L9(d, i, j, k, l)


@njit
def con_Gf(gamma, d, lam_index, s_dash, m, i):
    """Element mi of (condensed) R2Bexact submatrix G. See PhD thesis (2.69), 
    and see section 2.4.4 for condensation details."""
    if m == 0:
        return (0.5 * (s3+1) * Gf(gamma, d, lam_index, s_dash, 0, 0, i)
                + 0.5 * (s3-1) * Gf(gamma, d, lam_index, s_dash, 1, 1, i))
    elif m == 1:
        return s2 * Gf(gamma, d, lam_index, s_dash, 0, 1, i)
    elif m == 2:
        return (0.5 * (s3-1) * Gf(gamma, d, lam_index, s_dash, 0, 0, i)
                + 0.5 * (s3+1) * Gf(gamma, d, lam_index, s_dash, 1, 1, i))
    elif m == 3:
        return s2 * Gf(gamma, d, lam_index, s_dash, 0, 2, i)
    else:
        return s2 * Gf(gamma, d, lam_index, s_dash, 1, 2, i)


@njit
def con_Hf(gamma, d, lam_index, s_dash, m, i):
    """Element mi of (condensed) R2Bexact submatrix H. See PhD thesis (2.69), 
    and see section 2.4.4 for condensation details."""
    if m == 0:
        return (0.5 * (s3+1) * Hf(gamma, d, lam_index, s_dash, 0, 0, i)
                + 0.5 * (s3-1) * Hf(gamma, d, lam_index, s_dash, 1, 1, i))
    elif m == 1:
        return s2 * Hf(gamma, d, lam_index, s_dash, 0, 1, i)
    elif m == 2:
        return (0.5 * (s3-1) * Hf(gamma, d, lam_index, s_dash, 0, 0, i)
                + 0.5 * (s3+1) * Hf(gamma, d, lam_index, s_dash, 1, 1, i))
    elif m == 3:
        return s2 * Hf(gamma, d, lam_index, s_dash, 0, 2, i)
    else:
        return s2 * Hf(gamma, d, lam_index, s_dash, 1, 2, i)


@njit
def con1_Mf(gamma, d, lam_index, s_dash, n, k, l):
    """Element nkl of partially condensed R2Bexact submatrix M. See PhD thesis 
    (2.69), and see section 2.4.4 for condensation details."""
    if n == 0:
        return (0.5 * (s3+1) * Mf(gamma, d, lam_index, s_dash, 0, 0, k, l)
                + 0.5 * (s3-1) * Mf(gamma, d, lam_index, s_dash, 1, 1, k, l))
    elif n == 1:
        return s2 * Mf(gamma, d, lam_index, s_dash, 0, 1, k, l)
    elif n == 2:
        return (0.5 * (s3-1) * Mf(gamma, d, lam_index, s_dash, 0, 0, k, l)
                + 0.5 * (s3+1) * Mf(gamma, d, lam_index, s_dash, 1, 1, k, l))
    elif n == 3:
        return s2 * Mf(gamma, d, lam_index, s_dash, 0, 2, k, l)
    else:
        return s2 * Mf(gamma, d, lam_index, s_dash, 1, 2, k, l)


@njit
def con_Mf(gamma, d, lam_index, s_dash, n, m):
    """Element nm of (condensed) R2Bexact submatrix M. See PhD thesis (2.69),
    and see section 2.4.4 for condensation details."""
    if m == 0:
        return (0.5 * (s3+1) * con1_Mf(gamma, d, lam_index, s_dash, n, 0, 0)
                + 0.5 * (s3-1) * con1_Mf(gamma, d, lam_index, s_dash, n, 1, 1))
    elif m == 1:
        return s2 * con1_Mf(gamma, d, lam_index, s_dash, n, 0, 1)
    elif m == 2:
        return (0.5 * (s3-1) * con1_Mf(gamma, d, lam_index, s_dash, n, 0, 0)
                + 0.5 * (s3+1) * con1_Mf(gamma, d, lam_index, s_dash, n, 1, 1))
    elif m == 3:
        return s2 * con1_Mf(gamma, d, lam_index, s_dash, n, 0, 2)
    else:
        return s2 * con1_Mf(gamma, d, lam_index, s_dash, n, 1, 2)


@njit
def XYZ(scalar_index, gamma, s_dash, lam_index):
    """Look up value of X11A, X12A,..., Z12M scalar from pre-computed table.

    Args:
        scalar_index (int): 0 to 10, corresponds to X__A to Z__M.
        gamma (int): 0 correpsonds to _11_, 1 corresponds to _12_.
        s_dash: scaled centre-to-centre distance, 2s/(a+b).
        lam_index: index of the size ratio b/a, depending on precomputed ratios.
    """
    interp_y = XYZ_raw[scalar_index, gamma, :, lam_index]
    if s_dash > s_dash_range[-1]:
        print("S DASH OUT OF RANGE, functions_generate_R2Bexact.py")
    return np.interp(s_dash, s_dash_range, interp_y)
    # return np.interp(s_dash, s_dash_range, interp_y, right=0) # Numba only has a version of interp for the first three arguments. (Originally from non-periodic.)
    # return np.interp(s_dash,s_dash_range,interp_y,left=XYZ_raw[scalar_index,gamma,0,lam_index],right=0) # Numba only has a version of interp for the first three arguments. (Originally from periodic.)


def generate_R2Bexact(posdata,
                      printout=0, cutoff_factor=2, frameno=0, mu=1,
                      box_bottom_left=np.array([0, 0, 0]),
                      box_top_right=np.array([0, 0, 0]),
                      O_infinity=np.array([0, 0, 0]),
                      E_infinity=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                      timestep=0.1,
                      centre_of_background_flow=np.array([0, 0, 0]),
                      frequency=1, amplitude=1):
    """Generate R2Bexact matrix, for either nonperiodic or periodic domain.

    Args:
        posdata: Particle position, size and count data
        printout: (Unused) flag which allows you to put in debug statements
        cutoff_factor: Separation distance, multiple of (a1+a2) below which 
            R2Bexact applies
        frameno: Frame number
        mu: viscosity
        [All arguments below should be ignored for non-periodic domains]
        box_bottom_left, box_top_right: Periodic box coordinates. If equal, 
            domain is assumed to be non-periodic and all remaining args are 
            ignored. 
        O_infinity, ..., amplitude: Periodic/constant shear parameters

    Returns:
        mu*(L*R2Bexact*R): R2Bexact matrix
        "R2Bexact": Human readable name of the matrix
    """
    global average_size_matrix, upper_triangle
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)
    R2Bexact_sidelength = 11 * num_spheres + 6 * num_dumbbells
    R2Bexact = sparse.lil_matrix((R2Bexact_sidelength, R2Bexact_sidelength), dtype=float)
    bead_positions = np.concatenate([sphere_positions, dumbbell_positions - 0.5 * dumbbell_deltax, dumbbell_positions + 0.5 * dumbbell_deltax])
    bead_sizes = np.concatenate([sphere_sizes, dumbbell_sizes, dumbbell_sizes])

    closer_than_cutoff_pairs_scaled, displacements_pairs_scaled, distances_pairs_scaled, size_ratios = close_particles(bead_positions, bead_sizes, cutoff_factor, box_bottom_left, box_top_right, O_infinity, E_infinity, frameno, timestep, frequency=frequency, amplitude=amplitude)

    uv_power = [[1, 2, 2, 1, 1], [2, 3, 3, 2, 2], [2, 3, 3, 2, 2], [1, 2, 2, 1, 1], [1, 2, 2, 1, 1]]

    ii = 0
    for a1_index, a2_index in closer_than_cutoff_pairs_scaled:
        r = displacements_pairs_scaled[ii]  # vector r. Convention is a2-a1
        s_dash = distances_pairs_scaled[ii]  # np.linalg.norm(r)
        if a1_index != a2_index:
            d = r / s_dash
        lam = size_ratios[ii]  # Convention is a2/a1
        lam_index = np.where(lam_range_with_reciprocals == lam)[0][0]
        lam_index_recip = np.where(lam_range_with_reciprocals == 1. / lam)[0][0]
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
                        nearby_bead = j
                        nearby_beads_displacements.append(displacements_pairs_scaled[kk])
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])
                    if (j == a1_index and i != j):
                        nearby_bead = i
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
                    lam_p = bead_sizes[p_index] / bead_sizes[a1_index]
                    largest_size_p = max(bead_sizes[a1_index], bead_sizes[p_index])
                    if lam_p not in lam_range_with_reciprocals:
                        print("ERROR (Code point D): lambda not in the table of calculated values")
                    lam_index_p = np.where(lam_range_with_reciprocals == lam_p)[0][0]
                    r_p = nearby_beads_displacements[pp]
                    s_dash_p = nearby_beads_distances[pp]
                    d_p = r_p / s_dash_p
                    A_sum += np.asarray([[Af(0, d_p, lam_index_p, s_dash_p, i, j)*largest_size_p**uv_power[0][0] for j in range(3)] for i in range(3)])
                    Bt_sum += np.asarray([[Bf(0, d_p, lam_index_p, s_dash_p, j, i)*largest_size_p**uv_power[0][1] for j in range(3)] for i in range(3)])
                    C_sum += np.asarray([[Cf(0, d_p, lam_index_p, s_dash_p, i, j)*largest_size_p**uv_power[1][1] for j in range(3)] for i in range(3)])
                    Gt_sum += np.asarray([[con_Gf(0, d_p, lam_index_p, s_dash_p, j, i)*largest_size_p**uv_power[0][2] for j in range(5)] for i in range(3)])
                    Ht_sum += np.asarray([[con_Hf(0, d_p, lam_index_p, s_dash_p, j, i)*largest_size_p**uv_power[1][2] for j in range(5)] for i in range(3)])
                    M_sum += np.asarray([[con_Mf(0, d_p, lam_index_p, s_dash_p, i, j)*largest_size_p**uv_power[2][2] for j in range(5)] for i in range(5)])
                    pp += 1
                R2Bexact[A_coords] = A_sum
                R2Bexact[Bt_coords] = Bt_sum
                R2Bexact[C_coords] = C_sum
                R2Bexact[Gt_coords] = Gt_sum
                R2Bexact[Ht_coords] = Ht_sum
                R2Bexact[M_coords] = M_sum

            else:
                R2Bexact[A_coords] = [[Af(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]
                R2Bexact[Bt_coords] = [[Bf(1, -d, lam_index_recip, s_dash, j, i)*largest_size**uv_power[0][1] for j in range(3)] for i in range(3)]
                R2Bexact[C_coords] = [[Cf(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[1][1] for j in range(3)] for i in range(3)]
                R2Bexact[Gt_coords] = [[con_Gf(1, -d, lam_index_recip, s_dash, j, i)*largest_size**uv_power[0][2] for j in range(5)] for i in range(3)]
                R2Bexact[Ht_coords] = [[con_Hf(1, -d, lam_index_recip, s_dash, j, i)*largest_size**uv_power[1][2] for j in range(5)] for i in range(3)]
                R2Bexact[M_coords] = [[con_Mf(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[2][2] for j in range(5)] for i in range(5)]
                if lam == 1:
                    R2Bexact[Bt_coords_21] = -R2Bexact[Bt_coords]
                    R2Bexact[Gt_coords_21] = -R2Bexact[Gt_coords]
                    R2Bexact[Ht_coords_21] = R2Bexact[Ht_coords]
                else:
                    R2Bexact[Bt_coords_21] = [[Bf(1, d, lam_index, s_dash, j, i)*largest_size**uv_power[0][1] for j in range(3)] for i in range(3)]
                    R2Bexact[Gt_coords_21] = [[con_Gf(1, d, lam_index, s_dash, j, i)*largest_size**uv_power[0][2] for j in range(5)] for i in range(3)]
                    R2Bexact[Ht_coords_21] = [[con_Hf(1, d, lam_index, s_dash, j, i)*largest_size**uv_power[1][2] for j in range(5)] for i in range(3)]

        elif a1_index < num_spheres and a2_index >= num_spheres and a2_index < num_spheres + num_dumbbells:
            # Sphere to dumbbell bead 1
            a2_index_d = a2_index - num_spheres
            R14_coords = np.s_[a1_index*3:(a1_index+1)*3,                             11*num_spheres+a2_index_d*3: 11*num_spheres + (a2_index_d+1)*3]
            R24_coords = np.s_[3*num_spheres+a1_index*3:3*num_spheres+(a1_index+1)*3, 11*num_spheres+a2_index_d*3: 11*num_spheres + (a2_index_d+1)*3]
            R34_coords = np.s_[6*num_spheres+a1_index*5:6*num_spheres+(a1_index+1)*5, 11*num_spheres+a2_index_d*3: 11*num_spheres + (a2_index_d+1)*3]

            R2Bexact[R14_coords] = [[Af(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]
            R2Bexact[R24_coords] = [[Bf(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[0][1] for j in range(3)] for i in range(3)]
            R2Bexact[R34_coords] = [[con_Gf(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[0][2] for j in range(3)] for i in range(5)]

        elif a1_index < num_spheres and a2_index >= num_spheres + num_dumbbells:
            # Sphere to dumbbell bead 2
            a2_index_d = a2_index-num_spheres-num_dumbbells
            R15_coords = np.s_[a1_index*3:(a1_index+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3: 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
            R25_coords = np.s_[3*num_spheres+a1_index*3:3*num_spheres+(a1_index+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3: 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
            R35_coords = np.s_[6*num_spheres+a1_index*5:6*num_spheres+(a1_index+1)*5, 11*num_spheres+3*num_dumbbells+a2_index_d*3: 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]

            R2Bexact[R15_coords] = [[Af(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]
            R2Bexact[R25_coords] = [[Bf(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[0][1] for j in range(3)] for i in range(3)]
            R2Bexact[R35_coords] = [[con_Gf(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[0][2] for j in range(3)] for i in range(5)]

        elif a1_index >= num_spheres and a1_index < num_spheres + num_dumbbells and a2_index >= num_spheres and a2_index < num_spheres + num_dumbbells:
            # Dumbbell bead 1 to dumbbell bead 1
            a1_index_d = a1_index - num_spheres
            a2_index_d = a2_index - num_spheres
            R44_coords = np.s_[11*num_spheres+a1_index_d*3:11*num_spheres+(a1_index_d+1)*3, 11*num_spheres+a2_index_d*3: 11*num_spheres+(a2_index_d+1)*3]
            if a1_index == a2_index:
                nearby_beads = []
                nearby_beads_displacements = []
                nearby_beads_distances = []
                for kk in range(len(closer_than_cutoff_pairs_scaled)):
                    (i, j) = closer_than_cutoff_pairs_scaled[kk]
                    if (i == a1_index and i != j):
                        nearby_bead = j
                        nearby_beads_displacements.append(displacements_pairs_scaled[kk])
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])
                    if (j == a1_index and i != j):
                        nearby_bead = i
                        nearby_beads_displacements.append(-displacements_pairs_scaled[kk])  # Note minus sign
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])
                A_sum = 0
                pp = 0
                for p_index in nearby_beads:
                    lam_p = bead_sizes[p_index] / bead_sizes[a1_index]
                    largest_size_p = max(bead_sizes[a1_index], bead_sizes[p_index])
                    if lam_p not in lam_range_with_reciprocals:
                        print("ERROR (Code point D): lambda not in the table of calculated values")
                    lam_index_p = np.where(lam_range_with_reciprocals == lam_p)[0][0]
                    r_p = nearby_beads_displacements[pp]
                    s_dash_p = nearby_beads_distances[pp]
                    d_p = r_p / s_dash_p
                    A_sum = A_sum + np.asarray([[Af(0, d_p, lam_index_p, s_dash_p, i, j) * largest_size_p**uv_power[0][0] for j in range(3)] for i in range(3)])
                    pp = pp + 1
                R2Bexact[R44_coords] = A_sum
            else:
                if bead_bead_interactions:
                    R2Bexact[R44_coords] = [[Af(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]

        elif a1_index >= num_spheres and a1_index < num_spheres + num_dumbbells and a2_index >= num_spheres + num_dumbbells:
            # Dumbbell bead 1 to dumbbell bead 2
            if bead_bead_interactions:
                a1_index_d = a1_index - num_spheres
                a2_index_d = a2_index - num_spheres - num_dumbbells
                R45_coords = np.s_[11*num_spheres+a1_index_d*3:11*num_spheres+(a1_index_d+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3: 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
                R2Bexact[R45_coords] = [[Af(1, d, lam_index, s_dash, i, j)*largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]

        else:
            # Dumbbell bead 2 to dumbbell bead 2
            a1_index_d = a1_index - num_spheres - num_dumbbells
            a2_index_d = a2_index - num_spheres - num_dumbbells
            R55_coords = np.s_[11*num_spheres+3*num_dumbbells+a1_index_d*3:11*num_spheres+3*num_dumbbells+(a1_index_d+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3: 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
            if a1_index == a2_index:
                nearby_beads = []
                nearby_beads_displacements = []
                nearby_beads_distances = []
                for kk in range(len(closer_than_cutoff_pairs_scaled)):
                    (i, j) = closer_than_cutoff_pairs_scaled[kk]
                    if (i == a1_index and i != j):
                        nearby_bead = j
                        nearby_beads_displacements.append(displacements_pairs_scaled[kk])
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])
                    if (j == a1_index and i != j):
                        nearby_bead = i
                        nearby_beads_displacements.append(-displacements_pairs_scaled[kk])  # Note minus sign
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(distances_pairs_scaled[kk])
                A_sum = 0
                pp = 0
                for p_index in nearby_beads:
                    lam_p = bead_sizes[p_index] / bead_sizes[a1_index]
                    largest_size_p = max(bead_sizes[a1_index], bead_sizes[p_index])
                    if lam_p not in lam_range_with_reciprocals:
                        print("ERROR (Code point D): lambda not in the table of calculated values")
                    lam_index_p = np.where(lam_range_with_reciprocals == lam_p)[0][0]
                    r_p = nearby_beads_displacements[pp]
                    s_dash_p = nearby_beads_distances[pp]
                    d_p = r_p / s_dash_p
                    A_sum = A_sum + np.asarray([[Af(0, d_p, lam_index_p, s_dash_p, i, j) * largest_size_p**uv_power[0][0] for j in range(3)] for i in range(3)])
                    pp = pp + 1
                R2Bexact[R55_coords] = A_sum
            else:
                if bead_bead_interactions:
                    R2Bexact[R55_coords] = [[Af(1, d, lam_index, s_dash, i, j) * largest_size**uv_power[0][0] for j in range(3)] for i in range(3)]
        ii = ii + 1

    # Scale by 6pi
    R2Bexact = R2Bexact * 6 * np.pi

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
