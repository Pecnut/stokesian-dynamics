#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

# Reference: A. K. Townsend, 2017. The mechanics of suspensions,
#                                  PhD thesis, UCL.
# See section 2.4.2 for definitions, which are adapted from
# section 7.2 of Kim & Karrila, 2005. Microhydrodynamics.

import numpy as np
from functions.shared import (posdata_data, levi, close_particles, s2, hs3p1,
                              hs3m1, cond_E, cond_idx, submatrix_coords_tuple,
                              is_sphere, is_dumbbell_bead_1,
                              is_dumbbell_bead_2)
from scipy import sparse
from settings import bead_bead_interactions
from resistance_scalars.data import (s_dash_range, lam_range_with_reciprocals,
                                     XYZ_raw)
from numba import njit


kronmatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


# @njit(float64(float64[:],int64,int64),cache=True)
@njit(cache=True)
def L1(d, i, j):
    """Unit displacement tensor L1 used in A & C. See PhD thesis (2.70)."""
    return d[i]*d[j]


@njit(cache=True)
def L2(d, i, j):
    """Unit displacement tensor L2 used in A & C. See PhD thesis (2.70)."""
    return kronmatrix[i][j] - d[i]*d[j]


@njit(cache=True)
def L3(d, i, j):
    """Unit displacement tensor L3 used in B. See PhD thesis (2.70)."""
    if i == j:
        return 0
    else:
        return sum([levi(i, j, k)*d[k] for k in range(3) if k not in [i, j]])


@njit(cache=True)
def L4(d, i, j, k):
    """Unit displacement tensor L4 used in G. See PhD thesis (2.70)."""
    return (d[i]*d[j] - kronmatrix[i][j]/3) * d[k]


@njit(cache=True)
def L5(d, i, j, k):
    """Unit displacement tensor L5 used in G. See PhD thesis (2.70)."""
    return (d[i]*kronmatrix[j][k] + d[j]*kronmatrix[i][k] - 2*d[i]*d[j]*d[k])


@njit(cache=True)
def L6(d, i, j, k):
    """Unit displacement tensor L6 used in H. See PhD thesis (2.70)."""
    return sum([
        levi(i, k, l)*d[l]*d[j] for l in range(3) if l not in [i, k]
    ]) + sum([
        levi(j, k, l)*d[l]*d[i] for l in range(3) if l not in [k, j]])


@njit(cache=True)
def L7(d, i, j, k, l):
    """Unit displacement tensor L7 used in M. See PhD thesis (2.70)."""
    return (1.5
            * (d[i]*d[j] - kronmatrix[i][j]/3)
            * (d[k]*d[l] - kronmatrix[k][l]/3))


@njit(cache=True)
def L8(d, i, j, k, l):
    """Unit displacement tensor L8 used in M. See PhD thesis (2.70)."""
    return 0.5 * (d[i]*kronmatrix[j][l]*d[k]
                  + d[j]*kronmatrix[i][l]*d[k]
                  + d[i]*kronmatrix[j][k]*d[l]
                  + d[j]*kronmatrix[i][k]*d[l]
                  - 4*d[i]*d[j]*d[k]*d[l])


@njit(cache=True)
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


@njit(cache=True)
def XA(gam, s, lam_index):
    return XYZ(0, gam, s, lam_index)  # s:=s'


@njit(cache=True)
def YA(gam, s, lam_index):
    return XYZ(1, gam, s, lam_index)


@njit(cache=True)
def YB(gam, s, lam_index):
    return XYZ(2, gam, s, lam_index)


@njit(cache=True)
def XC(gam, s, lam_index):
    return XYZ(3, gam, s, lam_index)


@njit(cache=True)
def YC(gam, s, lam_index):
    return XYZ(4, gam, s, lam_index)


@njit(cache=True)
def XG(gam, s, lam_index):
    return XYZ(5, gam, s, lam_index)


@njit(cache=True)
def YG(gam, s, lam_index):
    return XYZ(6, gam, s, lam_index)


@njit(cache=True)
def YH(gam, s, lam_index):
    return XYZ(7, gam, s, lam_index)


@njit(cache=True)
def XM(gam, s, lam_index):
    return XYZ(8, gam, s, lam_index)


@njit(cache=True)
def YM(gam, s, lam_index):
    return XYZ(9, gam, s, lam_index)


@njit(cache=True)
def ZM(gam, s, lam_index):
    return XYZ(10, gam, s, lam_index)


@njit(cache=True)
def Af(gamma, d, lam_index, ss, i, j):
    """Element ij of R2Bexact submatrix A. See PhD thesis (2.69)."""
    XAg = XA(gamma, ss, lam_index)
    YAg = YA(gamma, ss, lam_index)
    return XAg * L1(d, i, j) + YAg * L2(d, i, j)


@njit(cache=True)
def Bf(gamma, d, lam_index, ss, i, j):
    """Element ij of R2Bexact submatrix B. See PhD thesis (2.69)."""
    YBg = YB(gamma, ss, lam_index)
    return YBg * L3(d, i, j)


@njit(cache=True)
def Cf(gamma, d, lam_index, ss, i, j):
    """Element ij of R2Bexact submatrix C. See PhD thesis (2.69)."""
    XCg = XC(gamma, ss, lam_index)
    YCg = YC(gamma, ss, lam_index)
    return XCg * L1(d, i, j) + YCg * L2(d, i, j)


@njit(cache=True)
def Gf(gamma, d, lam_index, ss, i, j, k):
    """Element ijk of uncondensed R2Bexact submatrix G. See PhD thesis
    (2.69)."""
    XGg = XG(gamma, ss, lam_index)
    YGg = YG(gamma, ss, lam_index)
    return XGg * L4(d, i, j, k) + YGg * L5(d, i, j, k)


@njit(cache=True)
def Hf(gamma, d, lam_index, ss, i, j, k):
    """Element ijk of uncondensed R2Bexact submatrix H. See PhD thesis
    (2.69)."""
    YHg = YH(gamma, ss, lam_index)
    return YHg * L6(d, i, j, k)


@njit(cache=True)
def Mf(gamma, d, lam_index, ss, i, j, k, l):
    """Element ijkl of uncondensed R2Bexact submatrix M. See PhD thesis
    (2.69)."""
    XMg = XM(gamma, ss, lam_index)
    YMg = YM(gamma, ss, lam_index)
    ZMg = ZM(gamma, ss, lam_index)
    return (XMg * L7(d, i, j, k, l)
            + YMg * L8(d, i, j, k, l)
            + ZMg * L9(d, i, j, k, l))


@njit(cache=True)
def con_Gf_row(gamma, d, lam_index, s_dash, i):
    """Elements [0:5]i of (condensed) R2Bexact submatrix G. See PhD thesis
    (2.69), and see section 2.4.4 for condensation details."""
    A = Gf(gamma, d, lam_index, s_dash, 0, 0, i)
    B = Gf(gamma, d, lam_index, s_dash, 1, 1, i)
    return np.array([
        (hs3p1*A + hs3m1*B),
        s2*Gf(gamma, d, lam_index, s_dash, 0, 1, i),
        (hs3m1*A + hs3p1*B),
        s2*Gf(gamma, d, lam_index, s_dash, 0, 2, i),
        s2*Gf(gamma, d, lam_index, s_dash, 1, 2, i)
    ])


@njit(cache=True)
def con_Hf_row(gamma, d, lam_index, s_dash, i):
    """Elements [0:5]i of (condensed) R2Bexact submatrix H. See PhD thesis
    (2.69), and see section 2.4.4 for condensation details."""
    A = Hf(gamma, d, lam_index, s_dash, 0, 0, i)
    B = Hf(gamma, d, lam_index, s_dash, 1, 1, i)
    return np.array([
        (hs3p1*A + hs3m1*B),
        s2*Hf(gamma, d, lam_index, s_dash, 0, 1, i),
        (hs3m1*A + hs3p1*B),
        s2*Hf(gamma, d, lam_index, s_dash, 0, 2, i),
        s2*Hf(gamma, d, lam_index, s_dash, 1, 2, i)
    ])


@njit(cache=True)
def XYZ(scalar_index, gamma, s_dash, lam_index):
    """Look up value of X11A, X12A,..., Z12M scalar from pre-computed table.

    Args:
        scalar_index (int): 0 to 10, corresponds to X__A to Z__M.
        gamma (int): 0 correpsonds to _11_, 1 corresponds to _12_.
        s_dash: scaled centre-to-centre distance, 2s/(a+b).
        lam_index: index of the size ratio b/a, depending on precomputed ratios
    """
    interp_y = XYZ_raw[scalar_index, gamma, :, lam_index]
    if s_dash > s_dash_range[-1]:
        print("S DASH OUT OF RANGE, functions/generate_R2Bexact.py")
    return np.interp(s_dash, s_dash_range, interp_y)
    # Numba only has a version of interp for the first three arguments.
    # Originally from non-periodic:
    #   return np.interp(s_dash, s_dash_range, interp_y, right=0)
    # Originally from periodic:
    #   return np.interp(s_dash,s_dash_range,interp_y,
    #                    left=XYZ_raw[scalar_index,gamma,0,lam_index],right=0)


def generate_R2Bexact(posdata,
                      printout=0, cutoff_factor=2, frameno=0, mu=1,
                      box_bottom_left=np.array([0, 0, 0]),
                      box_top_right=np.array([0, 0, 0]),
                      Ot_infinity=np.array([0, 0, 0]),
                      Et_infinity=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])):
    """Generate R2Bexact matrix, for either nonperiodic or periodic domain.

    Args:
        posdata: Particle position, size and count data.
        printout: (Unused) flag which allows you to put in debug statements.
        cutoff_factor: Separation distance, multiple of (a1+a2) below which
            R2Bexact applies.
        frameno: (Unused) frame number you can put in debug statements.
        mu: Viscosity.
        [All arguments below should be ignored for non-periodic domains]
        box_bottom_left, box_top_right: Periodic box coordinates. If equal,
            domain is assumed to be non-periodic and all remaining args are
            ignored.
        Ot_infinity, Et_infinity: Integral of O_infinity and E_infinity dt.

    Returns:
        mu*(L*R2Bexact*R): R2Bexact matrix.
        "R2Bexact": Human readable name of the matrix.
    """
    global average_size_matrix, upper_triangle
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
        dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
        element_sizes, element_positions, element_deltax, num_elements,
        num_elements_array, element_type, uv_start, uv_size,
        element_start_count) = posdata_data(posdata)

    R2Bexact_sidelength = 11 * num_spheres + 6 * num_dumbbells
    R2Bexact = np.zeros((R2Bexact_sidelength, R2Bexact_sidelength),
                        dtype=float)
    bead_positions = np.concatenate([sphere_positions,
                                     dumbbell_positions - 0.5*dumbbell_deltax,
                                     dumbbell_positions + 0.5*dumbbell_deltax])
    bead_sizes = np.concatenate([sphere_sizes, dumbbell_sizes, dumbbell_sizes])

    (closer_than_cutoff_pairs_scaled, displacements_pairs_scaled,
        distances_pairs_scaled, size_ratios) = close_particles(
            bead_positions, bead_sizes, cutoff_factor, box_bottom_left,
            box_top_right, Ot_infinity=Ot_infinity, Et_infinity=Et_infinity)

    R2Bexact = generate_R2Bexact_loop(
        R2Bexact, bead_sizes, num_spheres, num_dumbbells,
        closer_than_cutoff_pairs_scaled, displacements_pairs_scaled,
        distances_pairs_scaled, size_ratios)

    if num_dumbbells > 0:
        # Row and column ops I want are equivalent to doing
        # [ 1  0  0 ]   [ a b c ]   [ 1  0  0 ]
        # [ 0  1  1 ] . [ d e f ] . [ 0  1 -1 ]
        # [ 0 -1  1 ]   [ g h i ]   [ 0  1  1 ]
        #   "L"                         "R"

        # I know that we could generate L and R elsewhere rather than doing it
        # every timestep but it takes 0.01s for a few thousand dumbbells so for
        # now I don't mind
        Lrow = np.array([i for i in range(11*num_spheres + 6*num_dumbbells)]
                        + [i + 11*num_spheres for i in range(3*num_dumbbells)]
                        + [i + 11*num_spheres + 3*num_dumbbells
                           for i in range(3*num_dumbbells)])
        Lcol = np.array([i for i in range(11*num_spheres + 6*num_dumbbells)]
                        + [i + 11*num_spheres + 3*num_dumbbells
                           for i in range(3*num_dumbbells)]
                        + [i + 11*num_spheres for i in range(3*num_dumbbells)])
        Ldata = np.array([1 for i in range(11*num_spheres + 9*num_dumbbells)]
                         + [-1 for i in range(3*num_dumbbells)])
        L = sparse.coo_matrix((Ldata, (Lrow, Lcol)),
                              shape=(11*num_spheres+6*num_dumbbells,
                                     11*num_spheres+6*num_dumbbells))
        R = L.transpose()

        return (mu*(L*R2Bexact*R), "R2Bexact")
    else:
        return (mu*R2Bexact, "R2Bexact")


@njit(cache=True)
def generate_R2Bexact_loop(R2Bexact, bead_sizes, num_spheres, num_dumbbells,
                           closer_than_cutoff_pairs_scaled,
                           displacements_pairs_scaled, distances_pairs_scaled,
                           size_ratios):
    """Helper function for generating R2Bexact matrix.
    This function can be Numba-d.

    Args:
        R2Bexact: Empty R2Bexact matrix.
        mu: Viscosity.
        bead_sizes: Sizes of all beads.
        num_spheres: Number of spheres.
        num_dumbbells: Number of dumbbells.
        closer_than_cutoff_pairs_scaled: Pairs of beads closer than cutoff.
        displacements_pairs_scaled: Displacements of pairs of beads.
        distances_pairs_scaled: Distances between pairs of beads.
        size_ratios: Ratios of bead sizes.

    Returns:
        R2Bexact: R2Bexact matrix with elements filled in, but not yet
                  symmetrised.
    """
    R33_temp = np.empty((5,5), dtype=float)
    if num_dumbbells > 0:
        R34_temp = np.empty((5,3), dtype=float)
        R35_temp = np.empty((5,3), dtype=float)

    ii = 0
    for a1_index, a2_index in closer_than_cutoff_pairs_scaled:
        # Displacement convention for the R2Bexact formulae is a2-a1
        # (inconveniently the opposite of in Minfinity)
        r = displacements_pairs_scaled[ii]  # vector r. Convention is a2-a1
        s_dash = distances_pairs_scaled[ii]  # np.linalg.norm(r)
        if a1_index != a2_index:
            d = r / s_dash
        lam = size_ratios[ii]  # Convention is a2/a1
        lam_index = np.where(np.isclose(lam_range_with_reciprocals,lam))[0][0]
        lam_index_recip = np.where(
            np.isclose(lam_range_with_reciprocals,1/lam))[0][0]

        a1 = bead_sizes[a1_index]
        scale_A = 6*np.pi*a1
        scale_B = 4*np.pi*a1**2
        scale_C = 8*np.pi*a1**3
        scale_G = 4*np.pi*a1**2
        scale_H = 8*np.pi*a1**3
        scale_M = 20/3*np.pi*a1**3
        a2 = bead_sizes[a2_index]
        scale_B2 = 4*np.pi*a2**2
        scale_G2 = 4*np.pi*a2**2
        scale_H2 = 8*np.pi*a2**3

        (A_c, Bt_c, Bt21_c, Gt_c, Gt21_c, C_c, Ht_c, Ht21_c, M_c,
         R14_c, R24_c, R34_c, R44_c, R15_c, R25_c, R35_c, R45_c,
         R55_c) = submatrix_coords_tuple(a1_index, a2_index, num_spheres,
                                         num_dumbbells)

        if (is_sphere(a1_index, num_spheres)
                and is_sphere(a2_index, num_spheres)):
            # Sphere to sphere
            if a1_index == a2_index:
                nearby_beads = []
                nearby_beads_displacements = []
                nearby_beads_distances = []
                for kk in range(len(closer_than_cutoff_pairs_scaled)):
                    (i, j) = closer_than_cutoff_pairs_scaled[kk]
                    if (i == a1_index and i != j):
                        nearby_bead = j
                        nearby_beads_displacements.append(
                            displacements_pairs_scaled[kk])
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(
                            distances_pairs_scaled[kk])
                    if (j == a1_index and i != j):
                        nearby_bead = i
                        nearby_beads_displacements.append(
                            -displacements_pairs_scaled[kk])  # Note minus sign
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(
                            distances_pairs_scaled[kk])
                A_sum = np.zeros((3, 3))
                Bt_sum = np.zeros((3, 3))
                C_sum = np.zeros((3, 3))
                Gt_sum = np.zeros((3, 5))
                Ht_sum = np.zeros((3, 5))
                M_sum = np.zeros((5, 5))
                pp = 0
                for p_index in nearby_beads:
                    lam_p = bead_sizes[p_index] / bead_sizes[a1_index]
                    lam_index_p = np.where(
                        np.isclose(lam_range_with_reciprocals,lam_p))[0][0]
                    r_p = nearby_beads_displacements[pp]
                    s_dash_p = nearby_beads_distances[pp]
                    d_p = r_p / s_dash_p
                    A_sum += np.array([[
                        Af(0, d_p, lam_index_p, s_dash_p, i, j)
                        for j in range(3)] for i in range(3)])*scale_A
                    Bt_sum += np.array([[
                        Bf(0, d_p, lam_index_p, s_dash_p, j, i)
                        for j in range(3)] for i in range(3)])*scale_B
                    C_sum += np.array([[
                        Cf(0, d_p, lam_index_p, s_dash_p, i, j)
                        for j in range(3)] for i in range(3)])*scale_C
                    for i in range(3):
                        Gt_sum[i] += con_Gf_row(
                            0, d_p, lam_index_p, s_dash_p, i)*scale_G
                        Ht_sum[i] += con_Hf_row(
                            0, d_p, lam_index_p, s_dash_p, i)*scale_H
                    for i in range(5):
                        for j in range(5):
                            R33_temp[i,j] = Mf(0, d_p, lam_index_p, s_dash_p,
                                               *cond_idx[i], *cond_idx[j])
                    M_sum += cond_E @ R33_temp @ cond_E * scale_M
                    pp += 1
                R2Bexact[A_c[0]:A_c[0]+3,
                         A_c[1]:A_c[1]+3] = A_sum
                R2Bexact[Bt_c[0]:Bt_c[0]+3,
                         Bt_c[1]:Bt_c[1]+3] = Bt_sum
                R2Bexact[C_c[0]:C_c[0]+3,
                         C_c[1]:C_c[1]+3] = C_sum
                R2Bexact[Gt_c[0]:Gt_c[0]+3,
                         Gt_c[1]:Gt_c[1]+5] = Gt_sum
                R2Bexact[Ht_c[0]:Ht_c[0]+3,
                         Ht_c[1]:Ht_c[1]+5] = Ht_sum
                R2Bexact[M_c[0]:M_c[0]+5,
                         M_c[1]:M_c[1]+5] = M_sum

            else:
                R2Bexact[A_c[0]:A_c[0]+3,
                         A_c[1]:A_c[1]+3] = np.array([[
                             Af(1, d, lam_index, s_dash, i, j)
                             for j in range(3)] for i in range(3)]) * scale_A
                R2Bexact[Bt_c[0]:Bt_c[0]+3,
                         Bt_c[1]:Bt_c[1]+3] = np.array([[
                             Bf(1, -d, lam_index_recip, s_dash, j, i)
                             for j in range(3)] for i in range(3)]) * scale_B2
                R2Bexact[C_c[0]:C_c[0]+3,
                         C_c[1]:C_c[1]+3] = np.array([[
                             Cf(1, d, lam_index, s_dash, i, j)
                             for j in range(3)] for i in range(3)]) * scale_C
                for i in range(3):
                    R2Bexact[Gt_c[0]+i,
                             Gt_c[1]:Gt_c[1]+5] = (
                                 con_Gf_row(1, -d, lam_index_recip, s_dash, i)
                                 * scale_G2)
                    R2Bexact[Ht_c[0]+i,
                             Ht_c[1]:Ht_c[1]+5] = (
                                 con_Hf_row(1, -d, lam_index_recip, s_dash, i)
                                 * scale_H2)
                for i in range(5):
                    for j in range(5):
                        R33_temp[i,j] = Mf(1, d, lam_index, s_dash,
                                           *cond_idx[i], *cond_idx[j])
                R2Bexact[M_c[0]:M_c[0]+5,
                         M_c[1]:M_c[1]+5] = (cond_E @ R33_temp @ cond_E
                                             * scale_M)

                if lam == 1:
                    R2Bexact[Bt21_c[0]:Bt21_c[0]+3,
                             Bt21_c[1]:Bt21_c[1]+3] = -R2Bexact[
                                 Bt_c[0]:Bt_c[0]+3,
                                 Bt_c[1]:Bt_c[1]+3]
                    R2Bexact[Gt21_c[0]:Gt21_c[0]+3,
                             Gt21_c[1]:Gt21_c[1]+5] = -R2Bexact[
                                 Gt_c[0]:Gt_c[0]+3,
                                 Gt_c[1]:Gt_c[1]+5]
                    R2Bexact[Ht21_c[0]:Ht21_c[0]+3,
                             Ht21_c[1]:Ht21_c[1]+5] = R2Bexact[
                                 Ht_c[0]:Ht_c[0]+3,
                                 Ht_c[1]:Ht_c[1]+5]
                else:
                    R2Bexact[Bt21_c[0]:Bt21_c[0]+3,
                             Bt21_c[1]:Bt21_c[1]+3] = np.array([[
                                 Bf(1, d, lam_index, s_dash, j, i)
                                 for j in range(3)] for i in range(3)])*scale_B
                    for i in range(3):
                        R2Bexact[Gt21_c[0]+i,
                                 Gt21_c[1]:Gt21_c[1]+5] = (
                            con_Gf_row(1, d, lam_index, s_dash, i)
                            * scale_G)
                        R2Bexact[Ht21_c[0]+i,
                                 Ht21_c[1]:Ht21_c[1]+5] = (
                            con_Hf_row(1, d, lam_index, s_dash, i)
                            * scale_H)

        elif (is_sphere(a1_index, num_spheres)
              and is_dumbbell_bead_1(a2_index, num_spheres, num_dumbbells)):
            # Sphere to dumbbell bead 1
            R2Bexact[R14_c[0]:R14_c[0]+3,
                     R14_c[1]:R14_c[1]+3] = np.array([[
                         Af(1, d, lam_index, s_dash, i, j)
                         for j in range(3)] for i in range(3)])*scale_A
            R2Bexact[R24_c[0]:R24_c[0]+3,
                     R24_c[1]:R24_c[1]+3] = np.array([[
                         Bf(1, d, lam_index, s_dash, i, j)
                         for j in range(3)] for i in range(3)])*scale_B
            for i in range(5):
                for j in range(3):
                    R34_temp[i,j] = Gf(
                        1, d, lam_index, s_dash, *cond_idx[i], j)
            R2Bexact[R34_c[0]:R34_c[0]+5,
                     R34_c[1]:R34_c[1]+3] = cond_E @ R34_temp * scale_G

        elif (is_sphere(a1_index, num_spheres)):
            # Sphere to dumbbell bead 2
            R2Bexact[R15_c[0]:R15_c[0]+3,
                     R15_c[1]:R15_c[1]+3] = np.array([[
                         Af(1, d, lam_index, s_dash, i, j)
                         for j in range(3)] for i in range(3)])*scale_A
            R2Bexact[R25_c[0]:R25_c[0]+3,
                     R25_c[1]:R25_c[1]+3] = np.array([[
                         Bf(1, d, lam_index, s_dash, i, j)
                         for j in range(3)] for i in range(3)])*scale_B
            for i in range(5):
                for j in range(3):
                    R35_temp[i,j] = Gf(
                        1, d, lam_index, s_dash, *cond_idx[i], j)
            R2Bexact[R35_c[0]:R35_c[0]+5,
                     R35_c[1]:R35_c[1]+3] = cond_E @ R35_temp * scale_G

        elif (is_dumbbell_bead_1(a1_index, num_spheres, num_dumbbells)
              and is_dumbbell_bead_1(a2_index, num_spheres, num_dumbbells)):
            # Dumbbell bead 1 to dumbbell bead 1
            if a1_index == a2_index:
                nearby_beads = []
                nearby_beads_displacements = []
                nearby_beads_distances = []
                for kk in range(len(closer_than_cutoff_pairs_scaled)):
                    (i, j) = closer_than_cutoff_pairs_scaled[kk]
                    if (i == a1_index and i != j):
                        nearby_bead = j
                        nearby_beads_displacements.append(
                            displacements_pairs_scaled[kk])
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(
                            distances_pairs_scaled[kk])
                    if (j == a1_index and i != j):
                        nearby_bead = i
                        nearby_beads_displacements.append(
                            -displacements_pairs_scaled[kk])  # Note minus sign
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(
                            distances_pairs_scaled[kk])
                A_sum = np.zeros((3, 3))
                pp = 0
                for p_index in nearby_beads:
                    lam_p = bead_sizes[p_index] / bead_sizes[a1_index]
                    lam_index_p = np.where(
                        np.isclose(lam_range_with_reciprocals,lam_p))[0][0]
                    r_p = nearby_beads_displacements[pp]
                    s_dash_p = nearby_beads_distances[pp]
                    d_p = r_p / s_dash_p
                    A_sum += np.asarray([[
                        Af(0, d_p, lam_index_p, s_dash_p, i, j)
                        for j in range(3)] for i in range(3)])*scale_A
                    pp = pp + 1
                R2Bexact[R44_c[0]:R44_c[0]+3,
                         R44_c[1]:R44_c[1]+3] = A_sum
            else:
                if bead_bead_interactions:
                    R2Bexact[R44_c[0]:R44_c[0]+3,
                             R44_c[1]:R44_c[1]+3] = np.array([[
                                 Af(1, d, lam_index, s_dash, i, j)
                                 for j in range(3)] for i in range(3)])*scale_A

        elif (is_dumbbell_bead_1(a1_index, num_spheres, num_dumbbells)
              and is_dumbbell_bead_2(a2_index, num_spheres, num_dumbbells)):
            # Dumbbell bead 1 to dumbbell bead 2
            if bead_bead_interactions:
                R2Bexact[R45_c[0]:R45_c[0]+3,
                         R45_c[1]:R45_c[1]+3] = np.array([[
                             Af(1, d, lam_index, s_dash, i, j)
                             for j in range(3)] for i in range(3)])*scale_A

        else:
            # Dumbbell bead 2 to dumbbell bead 2
            if a1_index == a2_index:
                nearby_beads = []
                nearby_beads_displacements = []
                nearby_beads_distances = []
                for kk in range(len(closer_than_cutoff_pairs_scaled)):
                    (i, j) = closer_than_cutoff_pairs_scaled[kk]
                    if (i == a1_index and i != j):
                        nearby_bead = j
                        nearby_beads_displacements.append(
                            displacements_pairs_scaled[kk])
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(
                            distances_pairs_scaled[kk])
                    if (j == a1_index and i != j):
                        nearby_bead = i
                        nearby_beads_displacements.append(
                            -displacements_pairs_scaled[kk])  # Note minus sign
                        nearby_beads.append(nearby_bead)
                        nearby_beads_distances.append(
                            distances_pairs_scaled[kk])
                A_sum = np.zeros((3, 3))
                pp = 0
                for p_index in nearby_beads:
                    lam_p = bead_sizes[p_index] / bead_sizes[a1_index]
                    lam_index_p = np.where(
                        np.isclose(lam_range_with_reciprocals,lam_p))[0][0]
                    r_p = nearby_beads_displacements[pp]
                    s_dash_p = nearby_beads_distances[pp]
                    d_p = r_p / s_dash_p
                    A_sum += np.asarray([[
                        Af(0, d_p, lam_index_p, s_dash_p, i, j)
                        for j in range(3)] for i in range(3)])*scale_A
                    pp = pp + 1
                R2Bexact[R55_c[0]:R55_c[0]+3,
                         R55_c[1]:R55_c[1]+3] = A_sum
            else:
                if bead_bead_interactions:
                    R2Bexact[R55_c[0]:R55_c[0]+3,
                             R55_c[1]:R55_c[1]+3] = np.array([[
                                 Af(1, d, lam_index, s_dash, i, j)
                                 for j in range(3)] for i in range(3)])*scale_A
        ii = ii + 1

    # symmetrise
    R2Bexact = np.triu(R2Bexact) + np.triu(R2Bexact, k=1).transpose()

    return R2Bexact
