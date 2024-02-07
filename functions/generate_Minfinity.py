#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

# Reference: A. K. Townsend, 2017. The mechanics of suspensions. PhD thesis, UCL.
# See section 2.5 in particular, and also section 2 generally, and appendix A.

import numpy as np
from numpy import pi
from functions.shared import (posdata_data, levi, norm, s2, s3,
                              submatrix_coords, is_sphere, is_dumbbell_bead_1,
                              is_dumbbell_bead_2)
from inputs import bead_bead_interactions
from scipy.sparse import coo_matrix
from numba import njit

kronmatrix = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
kronkronmatrix = [[[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
                  [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
                  [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
                  [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
                  [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]]]
kronmatrix = np.array(kronmatrix)
kronkronmatrix = np.array(kronkronmatrix)


# O(J)
@njit
def J(ri, rj, i, j, ss):
    """Oseen tensor J_ij. See PhD thesis section A.2.3.

    Args:
        ri: ith element of vector r
        rj: jth element of vector r
        ss: r ( = |vector r|)
        i, j: coordinate indices"""
    return kronmatrix[i][j]/ss + ri*rj/ss**3

# O(D J)


@njit
def R(r, ss, i, j):
    """Rotlet R_ij. See PhD thesis section A.2.3."""
    return -0.5*sum(
        [levi(j, k, l)*D_J(r, ss, k, i, l) for k in range(3) for l in range(3)
         if k != l and j != k and j != l]
    )


@njit
def K(r, ss, i, j, k):
    """Tensor K_ijk. See PhD thesis section A.2.3."""
    return 0.5*(D_J(r, ss, k, i, j) + D_J(r, ss, j, i, k))


@njit
def D_J(r, ss, l, i, j):
    """Derivative of Oseen tensor, D_l J_ij. See PhD thesis section A.2.3."""
    return (-kronmatrix[i][j]*r[l] + kronmatrix[i][l]*r[j]
            + kronmatrix[j][l]*r[i])/ss**3 - 3*r[i]*r[j]*r[l]/ss**5

# O(D^2 J)


@njit
def DD_J(r, ss, m, l, i, j):
    """2nd derivative of Oseen tensor, D_m D_l J_ij. See PhD thesis section
    A.2.3."""
    # return ((-(i==j)*(l==m) + (i==l)*(j==m) + (j==l)*(i==m))/ss**3
    #         - 3*(-(i==j)*r[l]*r[m] + (i==l)*r[j]*r[m] + (j==l)*r[i]*r[m]
    #              + (i==m)*r[j]*r[l] + r[i]*(j==m)*r[l]
    #              + r[i]*r[j]*(l==m))/ss**5
    #         + 15*r[i]*r[j]*r[l]*r[m]/ss**7)
    return ((-kronkronmatrix[i][j][l][m]
             + kronkronmatrix[i][l][j][m]
             + kronkronmatrix[j][l][i][m])/ss**3
            - 3*(-kronmatrix[i][j]*r[l]*r[m]
                 + kronmatrix[i][l]*r[j]*r[m]
                 + kronmatrix[j][l]*r[i]*r[m]
                 + kronmatrix[i][m]*r[j]*r[l]
                 + r[i]*kronmatrix[j][m]*r[l]
                 + r[i]*r[j]*kronmatrix[l][m])/ss**5
            + 15*r[i]*r[j]*r[l]*r[m]/ss**7)


@njit
def D_R(r, ss, l, i, j):
    """Derivative of rotlet, D_l R_ij. See PhD thesis section A.2.3."""
    return -0.5 * sum([
        levi(j, m, n) * DD_J(r, ss, l, m, i, n)
        for m in range(3) for n in range(3) if m != n and m != j and n != j])


@njit
def D_K(r, ss, l, i, j, k):
    """Derivative of K tensor, D_l K_ijk. See PhD thesis section A.2.3."""
    return 0.5*(DD_J(r, ss, l, k, i, j) + DD_J(r, ss, l, j, i, k))


@njit
def Lap_J(ri, rj, i, j, ss):
    """Laplacian of Oseen tensor, D^2 J_ij. See PhD thesis section A.2.3."""
    return 2*kronmatrix[i][j]/ss**3 - 6*ri*rj/ss**5

# O(D^3 J)


@njit
def DLap_J(r, ss, k, i, j):
    """Derivative of Laplacian of Oseen tensor, D_k D^2 J_ij. See PhD thesis
    section A.2.3."""
    return (-6./ss**5)*(kronmatrix[i][j]*r[k] + kronmatrix[i][k]*r[j]
                        + kronmatrix[j][k]*r[i]) + (30./ss**7)*r[i]*r[j]*r[k]


@njit
def Lap_R(r, ss, i, j):
    '''Laplacian of rotlet, D^2 R_ij. See PhD thesis section A.2.3.'''
    return -0.5*sum([
        levi(j, k, l) * DLap_J(r, ss, k, i, l)
        for k in range(3) for l in range(3) if k != l and j != k and j != k])


@njit
def Lap_K(r, ss, i, j, k):
    '''Laplacian of K tensor, D^2 K_ijk. See PhD thesis section A.2.3.'''
    return DLap_J(r, ss, i, j, k)

# O(D^4 J)


@njit
def DLap_K(r, ss, l, i, j, k):
    """Derivative of Laplacian of K tensor D_l D^2 K_ijk. See PhD thesis
    section A.2.3."""
    return ((-6./ss**5)*(kronkronmatrix[i][j][k][l]
                         + kronkronmatrix[i][k][j][l]
                         + kronkronmatrix[j][k][i][l])
            - (210./ss**9)*r[i]*r[j]*r[k]*r[l]
            + (30./ss**7)*(kronmatrix[i][j]*r[k]*r[l]
                           + kronmatrix[i][k]*r[j]*r[l]
                           + kronmatrix[j][k]*r[i]*r[l]
                           + kronmatrix[i][l]*r[j]*r[k]
                           + kronmatrix[j][l]*r[i]*r[k]
                           + kronmatrix[k][l]*r[i]*r[j]))


@njit
def M11(ri, rj, s, a1, a2, i, j, c, mu):
    """Element ij of Minfinity submatrix a. See PhD thesis table 2.1."""
    if s > 1e-10:
        return c*(J(ri, rj, i, j, s)
                  + (a1**2 + a2**2)/6. * Lap_J(ri, rj, i, j, s))
    else:
        return kronmatrix[i][j]/(6*pi*mu*a1)


@njit
def M12(r, s, a1, a2, i, j, c, mu):
    """Element ij of Minfinity submatrix b tilde. See PhD thesis table 2.1."""
    if s > 1e-10:
        return c*(R(r, s, i, j) + a1**2/6. * Lap_R(r, s, i, j))
    else:
        return 0


@njit
def M13(r, s, a1, a2, i, j, k, c, mu):
    """Element ijk of uncondensed Minfinity submatrix g tilde. See PhD thesis
    table 2.1."""
    if s > 1e-10:
        return -c*(K(r, s, i, j, k)
                   + (a1**2/6. + a2**2/10.) * Lap_K(r, s, i, j, k))
    else:
        return 0


@njit
def M22(r, s, a1, a2, i, j, c, mu):
    """Element ij of Minfinity submatrix c. See PhD thesis table 2.1."""
    if abs(r[0]) + abs(r[1]) + abs(r[2]) > 1e-10:
        return c*0.5*sum([
            levi(i, k, l)*D_R(r, s, k, l, j)
            for k in range(3) for l in range(3)
            if k != l and i != k and i != l])
    else:
        return kronmatrix[i][j]/(8*pi*mu*a1**3)


@njit
def M23(r, s, a1, a2, i, j, k, c, mu):
    """Element ijk of uncondensed Minfinity submatrix h tilde. See PhD thesis
    table 2.1."""
    if abs(r[0]) + abs(r[1]) + abs(r[2]) > 1e-10:
        return c*-0.5*sum([
            levi(i, l, m) * (D_K(r, s, l, m, j, k)
                             + (a2**2/6.)*DLap_K(r, s, l, m, j, k))
            for l in range(3) for m in range(3)
            if l != m and i != l and i != m])
    else:
        return 0


@njit
def M33(r, s, a1, a2, i, j, k, l, c, mu):
    """Element ijkl of uncondensed Minfinity submatrix m. See PhD thesis table
    2.1."""
    return -0.5*c*((D_K(r, s, j, i, k, l) + D_K(r, s, i, j, k, l))
                   + (a1**2 + a2**2)/10. * (DLap_K(r, s, j, i, k, l)
                                            + DLap_K(r, s, i, j, k, l)))


@njit
def con_M13(r, s, a1, a2, i, m, c, mu):
    """Element im of (condensed) Minfinity submatrix g tilde. See PhD thesis
    table 2.1, and see section 2.4.4 for condensation details."""
    if m == 0:
        return (0.5*(s3+1)*M13(r, s, a1, a2, i, 0, 0, c, mu)
                + 0.5*(s3-1)*M13(r, s, a1, a2, i, 1, 1, c, mu))
    elif m == 1:
        return s2*M13(r, s, a1, a2, i, 0, 1, c, mu)
    elif m == 2:
        return (0.5*(s3-1)*M13(r, s, a1, a2, i, 0, 0, c, mu)
                + 0.5*(s3+1)*M13(r, s, a1, a2, i, 1, 1, c, mu))
    elif m == 3:
        return s2*M13(r, s, a1, a2, i, 0, 2, c, mu)
    else:
        return s2*M13(r, s, a1, a2, i, 1, 2, c, mu)


@njit
def con_M23(r, s, a1, a2, i, m, c, mu):
    """Element im of (condensed) Minfinity submatrix h tilde. See PhD thesis
    table 2.1, and see section 2.4.4 for condensation details."""
    if m == 0:
        return (0.5*(s3+1)*M23(r, s, a1, a2, i, 0, 0, c, mu)
                + 0.5*(s3-1)*M23(r, s, a1, a2, i, 1, 1, c, mu))
    elif m == 1:
        return s2*M23(r, s, a1, a2, i, 0, 1, c, mu)
    elif m == 2:
        return (0.5*(s3-1)*M23(r, s, a1, a2, i, 0, 0, c, mu)
                + 0.5*(s3+1)*M23(r, s, a1, a2, i, 1, 1, c, mu))
    elif m == 3:
        return s2*M23(r, s, a1, a2, i, 0, 2, c, mu)
    else:
        return s2*M23(r, s, a1, a2, i, 1, 2, c, mu)


@njit
def con1_M33(r, s, a1, a2, n, k, l, c, mu):
    """Element nkl of partially condensed Minfinity submatrix m. See PhD thesis
    table 2.1, and see section 2.4.4 for condensation details."""
    if n == 0:
        return (0.5*(s3+1)*M33(r, s, a1, a2, 0, 0, k, l, c, mu)
                + 0.5*(s3-1)*M33(r, s, a1, a2, 1, 1, k, l, c, mu))
    elif n == 1:
        return s2*M33(r, s, a1, a2, 0, 1, k, l, c, mu)
    elif n == 2:
        return (0.5*(s3-1)*M33(r, s, a1, a2, 0, 0, k, l, c, mu)
                + 0.5*(s3+1)*M33(r, s, a1, a2, 1, 1, k, l, c, mu))
    elif n == 3:
        return s2*M33(r, s, a1, a2, 0, 2, k, l, c, mu)
    else:
        return s2*M33(r, s, a1, a2, 1, 2, k, l, c, mu)


@njit
def con_M33(r, s, a1, a2, n, m, c, mu):
    """Element nm of (condensed) Minfinity submatrix m. See PhD thesis
    table 2.1, and see section 2.4.4 for condensation details."""
    if s <= 1e-10:
        return kronmatrix[n][m]/((20./3)*pi*mu*a1**3)
    if m == 0:
        return (0.5*(s3+1)*con1_M33(r, s, a1, a2, n, 0, 0, c, mu)
                + 0.5*(s3-1)*con1_M33(r, s, a1, a2, n, 1, 1, c, mu))
    elif m == 1:
        return s2*con1_M33(r, s, a1, a2, n, 0, 1, c, mu)
    elif m == 2:
        return (0.5*(s3-1)*con1_M33(r, s, a1, a2, n, 0, 0, c, mu)
                + 0.5*(s3+1)*con1_M33(r, s, a1, a2, n, 1, 1, c, mu))
    elif m == 3:
        return s2*con1_M33(r, s, a1, a2, n, 0, 2, c, mu)
    else:
        return s2*con1_M33(r, s, a1, a2, n, 1, 2, c, mu)


def generate_Minfinity(posdata, printout=0, frameno=0, mu=1):
    """Generate Minfinity matrix for nonperiodic periodic domain.

    Args:
        posdata: Particle position, size and count data
        printout: (Unused) flag which allows you to put in debug statements
        frameno: (Unused) frame number which you can use in debug statements
        mu: viscosity

    Returns:
        L*Minfinity*R: Minfinity matrix
        "Minfinity": Human readable name of the matrix
    """
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
        dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
        element_sizes, element_positions, element_deltax, num_elements,
        num_elements_array, element_type, uv_start, uv_size,
        element_start_count) = posdata_data(posdata)
    Minfinity_sidelength = 11*num_spheres + 6*num_dumbbells
    Minfinity = np.zeros((Minfinity_sidelength, Minfinity_sidelength),
                         dtype=float)
    bead_positions = np.concatenate([sphere_positions,
                                     dumbbell_positions - 0.5*dumbbell_deltax,
                                     dumbbell_positions + 0.5*dumbbell_deltax])
    bead_sizes = np.concatenate([sphere_sizes, dumbbell_sizes, dumbbell_sizes])

    c = 1./(8*pi*mu)

    for a1_index, a2_index in [(u, v)
                               for u in range(len(bead_sizes))
                               for v in range(u, len(bead_sizes))]:
        r = -(bead_positions[a2_index] - bead_positions[a1_index])
        a1 = bead_sizes[a1_index]
        a2 = bead_sizes[a2_index]
        s = norm(r)
        if s > 1e-8 and 2*s/(a1+a2) < 2.001:
            ss_out = 2.001*(a1+a2)/2
            r = np.array([r[0]*ss_out/s, r[1]*ss_out/s, r[2]*ss_out/s])
            s = ss_out

        (A_coords, Bt_coords, Bt_coords_21, Gt_coords, Gt_coords_21,
         C_coords, Ht_coords, Ht_coords_21, M_coords,
         M14_coords, M24_coords, M34_coords, M44_coords,
         M15_coords, M25_coords, M35_coords, M45_coords,
         M55_coords) = submatrix_coords(a1_index, a2_index, num_spheres,
                                        num_dumbbells)

        if is_sphere(a1_index, num_spheres) and is_sphere(a2_index, num_spheres):
            # Sphere to sphere
            Minfinity[A_coords] = [[M11(r[i], r[j], s, a1, a2, i, j, c, mu)
                                    for j in range(3)] for i in range(3)]
            Minfinity[Bt_coords] = [[M12(r, s, a1, a2, i, j, c, mu)
                                     for j in range(3)] for i in range(3)]
            Minfinity[C_coords] = [[M22(r, s, a1, a2, i, j, c, mu)
                                    for j in range(3)] for i in range(3)]
            Minfinity[Gt_coords] = [[con_M13(r, s, a1, a2, i, j, c, mu)
                                     for j in range(5)] for i in range(3)]
            Minfinity[Ht_coords] = [[con_M23(r, s, a1, a2, i, j, c, mu)
                                     for j in range(5)] for i in range(3)]
            Minfinity[M_coords] = [[con_M33(r, s, a1, a2, i, j, c, mu)
                                    for j in range(5)] for i in range(5)]
            if a1 == a2:
                Minfinity[Bt_coords_21] = -Minfinity[Bt_coords]
                Minfinity[Gt_coords_21] = -Minfinity[Gt_coords]
                Minfinity[Ht_coords_21] = Minfinity[Ht_coords]
            else:
                Minfinity[Bt_coords_21] = [[M12(-r, s, a2, a1, i, j, c, mu)
                                            for j in range(3)] for i in range(3)]
                Minfinity[Gt_coords_21] = [[con_M13(-r, s, a2, a1, i, j, c, mu)
                                            for j in range(5)] for i in range(3)]
                Minfinity[Ht_coords_21] = [[con_M23(-r, s, a2, a1, i, j, c, mu)
                                            for j in range(5)] for i in range(3)]

        elif (is_sphere(a1_index, num_spheres)
              and is_dumbbell_bead_1(a2_index, num_spheres, num_dumbbells)):
            # Sphere to dumbbell bead 1
            mr = [-r[0], -r[1], -r[2]]
            Minfinity[M14_coords] = [[M11(r[i], r[j], s, a1, a2, i, j, c, mu)
                                      for j in range(3)] for i in range(3)]
            Minfinity[M24_coords] = [[M12(mr, s, a2, a1, j, i, c, mu)
                                      for j in range(3)] for i in range(3)]
            Minfinity[M34_coords] = [[con_M13(mr, s, a1, a2, j, i, c, mu)
                                      for j in range(3)] for i in range(5)]

        elif is_sphere(a1_index, num_spheres):
            # Sphere to dumbbell bead 2
            mr = [-r[0], -r[1], -r[2]]
            Minfinity[M15_coords] = [[M11(r[i], r[j], s, a1, a2, i, j, c, mu)
                                      for j in range(3)] for i in range(3)]
            Minfinity[M25_coords] = [[M12(mr, s, a2, a1, j, i, c, mu)
                                      for j in range(3)] for i in range(3)]
            Minfinity[M35_coords] = [[con_M13(mr, s, a1, a2, j, i, c, mu)
                                      for j in range(3)] for i in range(5)]

        elif (is_dumbbell_bead_1(a1_index, num_spheres, num_dumbbells)
              and is_dumbbell_bead_1(a2_index, num_spheres, num_dumbbells)):
            # Dumbbell bead 1 to dumbbell bead 1
            if bead_bead_interactions or a1_index == a2_index:
                Minfinity[M44_coords] = [[M11(r[i], r[j], s, a1, a2, i, j, c, mu)
                                          for j in range(3)] for i in range(3)]

        elif (is_dumbbell_bead_1(a1_index, num_spheres, num_dumbbells)
              and is_dumbbell_bead_2(a2_index, num_spheres, num_dumbbells)):
            if bead_bead_interactions:
                # Dumbbell bead 1 to dumbbell bead 2
                Minfinity[M45_coords] = [[M11(r[i], r[j], s, a1, a2, i, j, c, mu)
                                          for j in range(3)] for i in range(3)]

        else:
            # Dumbbell bead 2 to dumbbell bead 2
            if bead_bead_interactions or a1_index == a2_index:
                Minfinity[M55_coords] = [[M11(r[i], r[j], s, a1, a2, i, j, c, mu)
                                          for j in range(3)] for i in range(3)]

    # symmetrise
    Minfinity = np.triu(Minfinity) + np.triu(Minfinity, k=1).transpose()

    # Row and column ops I want are equivalent to doing
    #   [ 1    0    0 ]   [ a b c ]   [ 1    0    0 ]
    #   [ 0  1/2  1/2 ] . [ d e f ] . [ 0  1/2 -1/2 ]
    #   [ 0 -1/2  1/2 ]   [ g h i ]   [ 0  1/2  1/2 ]
    #        "L"                       "R"

    # I know that we could generate L and R elsewhere rather than doing it
    # every timestep but it takes 0.01s for a few thousand dumbbells so for
    # now I don't mind
    Lrow = np.array([i for i in range(11*num_spheres + 6*num_dumbbells)]
                    + [i + 11*num_spheres for i in range(3*num_dumbbells)]
                    + [i + 11*num_spheres + 3*num_dumbbells for i in range(3*num_dumbbells)])
    Lcol = np.array([i for i in range(11*num_spheres + 6*num_dumbbells)]
                    + [i + 11*num_spheres + 3*num_dumbbells for i in range(3*num_dumbbells)]
                    + [i + 11*num_spheres for i in range(3*num_dumbbells)])
    Ldata = np.array([1 for i in range(11*num_spheres)]
                     + [0.5 for i in range(9*num_dumbbells)]
                     + [-0.5 for i in range(3*num_dumbbells)])
    L = coo_matrix((Ldata, (Lrow, Lcol)),
                   shape=(11*num_spheres+6*num_dumbbells,
                          11*num_spheres+6*num_dumbbells))
    R = L.transpose()

    return ((L*Minfinity*R), "Minfinity")
