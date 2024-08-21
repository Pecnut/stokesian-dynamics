#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 10/01/2017

import numpy as np
import math
from functions.shared import (posdata_data, norm, s2, hs3p1, hs3m1, spi,
                              submatrix_coords_tuple,
                              is_sphere, is_dumbbell_bead_1, is_dumbbell_bead_2,
                              shear_basis_vectors, cond_idx, cond_E)
from settings import how_far_to_reproduce_gridpoints, bead_bead_interactions
from scipy.sparse import coo_matrix
from math import erfc, pi, exp
from numba import njit


# === DERIVATIVES OF r erfc (lambda r) ===
# 1st derivatives

# def D_rerfclr(rj, ss, erfc0, erfc1):
#     return rj/ss*erfc0 + rj*erfc1

# 2nd derivatives
@njit(cache=True)
def DD_rerfclr(ri, rj, ss, erfc0, erfc1, erfc2, i, j):
    """Return 2nd derivative of r erfc(lambda r), D_i D_j r erfc(lambda r).
    See PhD thesis section A.2.5.

    Args:
        ri: ith element of vector r
        rj: jth element of vector r
        ss: r ( = |vector r|)
        erfc0: 0th derivative of erfc(lambda r)
        erfc1: 1st derivative of erfc(lambda r)
        erfc2: 2nd derivative of erfc(lambda r)
        i, j: coordinate indices
    """
    return (((i == j)/ss - ri*rj/ss**3)*erfc0
            + ((i == j) + ri*rj/ss**2)*erfc1
            + ri*rj/ss*erfc2)


@njit(cache=True)
def Lap_rerfclr(ss, erfc0, erfc1, erfc2):
    """Laplacian of r erfc(lambda r). See PhD thesis section A.2.5."""
    return 2/ss*erfc0 + 4*erfc1 + ss*erfc2

# 3rd derivatives


@njit(cache=True)
def DDD_rerfclr(ri, rj, rl, ss, erfc0, erfc1, erfc2, erfc3, i, j, l):
    """3rd derivative of r erfc(lambda r), D_i D_j D_l r erfc(lambda r).
    See PhD thesis section A.2.5."""
    a = -(i == j)*rl - (i == l)*rj - (j == l)*ri + 3*ri*rj*rl/ss**2
    return (1/ss**3*(a)*erfc0
            + 1/ss**2*(-a)*erfc1
            + 1/ss*((i == j)*rl + (i == l)*rj + (j == l)*ri)*erfc2
            + ri*rj*rl/ss**2*erfc3)


@njit(cache=True)
def DLap_rerfclr(rl, ss, erfc0, erfc1, erfc2, erfc3):
    """Derivative of Laplacian of r erfc(lambda r), D_l D^2 r erfc(lambda r).
    See PhD thesis section A.2.5."""
    return -2*rl/ss**3*erfc0 + 2*rl/ss**2*erfc1 + 5*rl/ss*erfc2 + rl*erfc3

# 4th derivatives


@njit(cache=True)
def DDDD_rerfclr(ri, rj, rl, rm, ss, erfc0, erfc1, erfc2, erfc3, erfc4,
                 i, j, l, m):
    """4th derivative of r erfc(lambda r), D_i D_j D_l D_m r erfc(lambda r).
    See PhD thesis section A.2.5."""
    kron_rr = ((i == j)*rl*rm + (i == l)*rj*rm + (l == j)*ri*rm + (i == m)*rj*rl
               + (j == m)*ri*rl + (l == m)*ri*rj)
    kron3 = (i == j)*(l == m) + (j == l)*(i == m) + (i == l)*(j == m)
    rirjrlrm = ri*rj*rl*rm

    a = - kron3/ss**3 + 3*kron_rr/ss**5 - 15*rirjrlrm/ss**7

    return ((a)*erfc0
            + (-a*ss)*erfc1
            + (kron3/ss - 3*rirjrlrm/ss**5)*erfc2
            + (kron_rr/ss**2 - 2*rirjrlrm/ss**4)*erfc3
            + rirjrlrm/ss**3 * erfc4)


@njit(cache=True)
def DDLap_rerfclr(rl, rm, ss, erfc0, erfc1, erfc2, erfc3, erfc4, l, m):
    """2nd derivative of Laplacian of r erfc(lambda r),
    D_l D_m D^2 r erfc(lambda r). See PhD thesis section A.2.5."""
    kron_lm = (l == m)
    rlrm = rl*rm
    a = -2*kron_lm/ss**3 + 6*rlrm/ss**5
    return ((a)*erfc0
            + (-a*ss)*erfc1
            + (-3*rlrm/ss**3 + 5*kron_lm/ss)*erfc2
            + (5*rlrm/ss**2 + kron_lm)*erfc3
            + rlrm/ss * erfc4)


@njit(cache=True)
def LapLap_rerfclr(ss, erfc2, erfc3, erfc4):
    """Double Laplacian of r erfc(lambda r), D^4 r erfc(lambda r).
    See PhD thesis section A.2.5."""
    return 12/ss*erfc2 + 8*erfc3 + ss*erfc4

# 5th derivatives


@njit(cache=True)
def DDDLap_rerfclr(ri, rj, rk, ss, erfc0, erfc1, erfc2, erfc3, erfc4,
                   erfc5, i, j, k):
    """3rd derivative of Laplacian of r erfc(lambda r),
    D_i D_j D_k D^2 r erfc(lambda r). See PhD thesis section A.2.5."""
    kron_r = (i == j)*rk + (i == k)*rj + (j == k)*ri
    a = 6*kron_r/ss**5 - 30*ri*rj*rk/ss**7
    rirjrk = ri*rj*rk
    return (a*erfc0
            + (-a*ss)*erfc1
            + (-3*kron_r/ss**3 + 3*rirjrk/ss**5)*erfc2
            + (5*kron_r/ss**2 - 13*rirjrk/ss**4)*erfc3
            + (kron_r/ss + 4*rirjrk/ss**3)*erfc4
            + rirjrk/ss**2 * erfc5)


@njit(cache=True)
def DLapLap_rerfclr(rk, ss, erfc2, erfc3, erfc4, erfc5):
    """Derivative of double Laplacian of r erfc(lambda r),
    D_k D^4 r erfc(lambda r). See PhD thesis section A.2.5."""
    return (-12*rk/ss**3*erfc2
            + 12*rk/ss**2*erfc3
            + 9*rk/ss*erfc4
            + rk*erfc5)

# 6th derivatives


@njit(cache=True)
def DDDDLap_rerfclr(ri, rj, rk, rl, ss, erfc0, erfc1, erfc2, erfc3, erfc4,
                    erfc5, erfc6, i, j, k, l):
    """4th derivative of Laplacian of r erfc(lambda r),
    D_i D_j D_k D_l D^2 r erfc(lambda r). See PhD thesis section A.2.5."""
    kron_rr = ((i == j)*rk*rl + (i == k)*rj*rl + (k == j)*ri*rl + (i == l)*rj*rk
               + (j == l)*ri*rk + (k == l)*ri*rj)
    kron3 = (i == j)*(k == l) + (j == k)*(i == l) + (i == k)*(j == l)
    rirjrkrl = ri*rj*rk*rl
    a = 6*kron3/ss**5 - 30*kron_rr/ss**7 + 210*rirjrkrl/ss**9
    return (a*erfc0
            + (-a*ss)*erfc1
            + (-3*kron3/ss**3 + 3*kron_rr/ss**5 + 15*rirjrkrl/ss**7)*erfc2
            + (5*kron3/ss**2 - 13*kron_rr/ss**4 + 55*rirjrkrl/ss**6)*erfc3
            + (kron3/ss + 4*kron_rr/ss**3 - 25*rirjrkrl/ss**5)*erfc4
            + (kron_rr/ss**2 + 2*rirjrkrl/ss**4)*erfc5
            + (rirjrkrl/ss**3)*erfc6)


@njit(cache=True)
def DDLapLap_rerfclr(rk, rl, ss, erfc2, erfc3, erfc4, erfc5, erfc6, k, l):
    """2nd derivative of double Laplacian of r erfc(lambda r),
    D_k D_l D^4 r erfc(lambda r). See PhD thesis section A.2.5."""
    a = -12*(k == l)/ss**3 + 36*rk*rl/ss**5
    return (a * erfc2
            + (-a)*ss*erfc3
            + (3*rk*rl/ss**3 + 9*(k == l)/ss)*erfc4
            + (9*rk*rl/ss**2 + (k == l))*erfc5
            + rk*rl/ss * erfc6)


# Derivatives of erfc (lambda r)
@njit(cache=True)
def generate_erfcs(s, lamb):
    """0th to 6th derivatives of erfc(lambda r). See PhD thesis section A.2.6."""
    E = 2/pi**0.5*exp(-s**2*lamb**2)*lamb
    erfc0 = erfc(lamb*s)
    erfc1 = -E
    erfc2 = 2*lamb**2*s*E
    erfc3 = -2*lamb**2*E*(2*lamb**2*s**2-1)
    erfc4 = 4*lamb**4*s*E*(2*lamb**2*s**2-3)
    erfc5 = -4*lamb**4*E*(4*lamb**4*s**4-12*lamb**2*s**2+3)
    erfc6 = 8*lamb**6*s*E*(4*lamb**4*s**4-20*lamb**2*s**2+15)
    return np.array([erfc0, erfc1, erfc2, erfc3, erfc4, erfc5, erfc6])

# === CONSTANTS ===

# kronmatrix = d_ij d_kl
kronmatrix = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

# kron3traceless_ijkl = d_ik d_jl + d_il d_jk - 2/3 d_ij d_kl
ft = 1.3333333333333333
mtt = -0.6666666666666667
kron3tracelessmatrix = np.array([
    [[[ft, 0, 0, 0, 0], [0, mtt, 0, 0, 0], [0, 0, mtt, 0, 0], [0, 0, 0, mtt, 0], [0, 0, 0, 0, mtt]],
     [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]],
    [[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[mtt, 0, 0, 0, 0], [0, ft, 0, 0, 0], [0, 0, mtt, 0, 0], [0, 0, 0, mtt, 0], [0, 0, 0, 0, mtt]],
     [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]],
    [[[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[mtt, 0, 0, 0, 0], [0, mtt, 0, 0, 0], [0, 0, ft, 0, 0], [0, 0, 0, mtt, 0], [0, 0, 0, 0, mtt]],
     [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]],
    [[[0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
     [[mtt, 0, 0, 0, 0], [0, mtt, 0, 0, 0], [0, 0, mtt, 0, 0], [0, 0, 0, ft, 0], [0, 0, 0, 0, mtt]],
     [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]]],
    [[[0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
     [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]],
     [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]],
     [[mtt, 0, 0, 0, 0], [0, mtt, 0, 0, 0], [0, 0, mtt, 0, 0], [0, 0, 0, mtt, 0], [0, 0, 0, 0, ft]]]])

# === DERIVATIVES OF J^r(r) ===
# O(J)


@njit(cache=True)
def J(r, ss, i, j, erfcs):
    '''Realspace periodic Oseen tensor J^r_ij(r). See PhD thesis section A.2.4.'''
    return (kronmatrix[i][j]*Lap_rerfclr(ss, erfcs[0], erfcs[1], erfcs[2])
            - DD_rerfclr(r[i], r[j], ss, erfcs[0], erfcs[1], erfcs[2], i, j))

# O(D J)


@njit(cache=True)
def R(r, ss, i, j, erfcs):
    '''Realspace periodic rotlet R^r_ij(r). See PhD thesis section A.2.4.'''
    k = (j+1) % 3
    l = (j+2) % 3
    return -0.5*(D_J(r, ss, k, i, l, erfcs) - D_J(r, ss, l, i, k, erfcs))


@njit(cache=True)
def K(r, ss, i, j, k, erfcs):
    '''Realspace periodic tensor K^r_ijk(r). See PhD thesis section A.2.4.'''
    return 0.5*(D_J(r, ss, k, i, j, erfcs) + D_J(r, ss, j, i, k, erfcs))


@njit(cache=True)
def D_J(r, ss, l, i, j, erfcs):
    '''Derivative of realspace periodic Oseen tensor, D_l J^r_ij(r).
    See PhD thesis section A.2.4.'''

    return (kronmatrix[i][j]*DLap_rerfclr(r[l], ss,
                                          erfcs[0], erfcs[1],
                                          erfcs[2], erfcs[3])
            - DDD_rerfclr(r[i], r[j], r[l], ss,
                          erfcs[0], erfcs[1],
                          erfcs[2], erfcs[3], i, j, l))

# O(D^2 J)


@njit(cache=True)
def DD_J(r, ss, m, l, i, j, erfcs):
    '''2nd derivative of realspace periodic Oseen tensor, D_m D_l J^r_ij(r).
    See PhD thesis section A.2.4.'''
    rl = r[l]
    rm = r[m]
    ri = r[i]
    rj = r[j]
    return ((i == j)*DDLap_rerfclr(rl, rm, ss,
                                   erfcs[0], erfcs[1], erfcs[2],
                                   erfcs[3], erfcs[4], l, m)
            - DDDD_rerfclr(ri, rj, rl, rm, ss,
                           erfcs[0], erfcs[1], erfcs[2],
                           erfcs[3], erfcs[4], i, j, l, m))


@njit(cache=True)
def D_R(r, ss, l, i, j, erfcs):
    '''Derivative of realspace periodic rotlet, D_l R^r_ij(r).
    See PhD thesis section A.2.4.'''
    m = (j+1) % 3
    n = (j+2) % 3
    return -0.5 * (DD_J(r, ss, l, m, i, n, erfcs)
                   - DD_J(r, ss, l, n, i, m, erfcs))


@njit(cache=True)
def D_K(r, ss, l, i, j, k, erfcs):
    '''Derivative of realspace periodic K tensor, D_l K^r_ijk(r).
    See PhD thesis section A.2.4.'''
    return 0.5*(DD_J(r, ss, l, k, i, j, erfcs) + DD_J(r, ss, l, j, i, k, erfcs))


@njit(cache=True)
def Lap_J(r, ss, i, j, erfcs):
    '''Laplacian of realspace periodic Oseen tensor, D^2 J^r_ij(r).
    See PhD thesis section A.2.4.'''
    return (kronmatrix[i][j]*LapLap_rerfclr(ss, erfcs[2], erfcs[3], erfcs[4])
            - DDLap_rerfclr(r[i], r[j], ss,
                            erfcs[0], erfcs[1], erfcs[2],
                            erfcs[3], erfcs[4], i, j))

# O(D^3 J)


@njit(cache=True)
def DLap_J(r, ss, k, i, j, erfcs):
    '''Derivative of Laplacian of periodic Oseen tensor, D_k D^2 J^r_ij(r).
    See PhD thesis section A.2.4.'''
    rk = r[k]
    return (kronmatrix[i][j]*DLapLap_rerfclr(rk, ss, erfcs[2], erfcs[3],
                                             erfcs[4], erfcs[5])
            - DDDLap_rerfclr(r[i], r[j], rk, ss,
                             erfcs[0], erfcs[1], erfcs[2],
                             erfcs[3], erfcs[4], erfcs[5], i, j, k))


@njit(cache=True)
def Lap_R(r, ss, i, j, erfcs):
    '''Laplacian of realspace periodic rotlet, D^2 J^r_ij(r).
    See PhD thesis section A.2.4.'''
    k = (j+1) % 3
    l = (j+2) % 3
    return -0.5*(DLap_J(r, ss, k, i, l, erfcs) - DLap_J(r, ss, l, i, k, erfcs))


@njit(cache=True)
def Lap_K(r, ss, i, j, k, erfcs):
    '''Laplacian of realspace periodic K tensor, D^2 K^r_ijk(r).
    See PhD thesis section A.2.4.'''
    return 0.5*(DLap_J(r, ss, k, i, j, erfcs) + DLap_J(r, ss, j, i, k, erfcs))

# O(D^4 J)


@njit(cache=True)
def DDLap_J(r, ss, l, k, i, j, erfcs):
    '''2nd derivative of Laplacian of realspace periodic Oseen tensor,
    D_l D_k D^2 J^r_ij(r). See PhD thesis section A.2.4.'''
    rk = r[k]
    rl = r[l]
    return (kronmatrix[i][j]*DDLapLap_rerfclr(rk, rl, ss,
                                              erfcs[2], erfcs[3], erfcs[4],
                                              erfcs[5], erfcs[6], k, l)
            - DDDDLap_rerfclr(r[i], r[j], rk, rl, ss,
                              erfcs[0], erfcs[1], erfcs[2], erfcs[3],
                              erfcs[4], erfcs[5], erfcs[6], i, j, k, l))


@njit(cache=True)
def DLap_K(r, ss, l, i, j, k, erfcs):
    '''Derivative of Laplacian of realspace periodic K tensor,
    D_l D^2 K^r_ijk(r). See PhD thesis section A.2.4.'''
    return 0.5*(DDLap_J(r, ss, l, k, i, j, erfcs)
                + DDLap_J(r, ss, l, j, i, k, erfcs))

# === TENSORS a^r(r) etc. ===
# a^r


@njit(cache=True)
def ar(r, s, a1, a2, i, j, erfcs, c, mu):
    """Element ij of the realspace periodic version of Minfinity submatrix a.

    Equivalent to non-periodic definition (see PhD thesis table 2.1) with J
    replaced by realspace form J^r_ij, a.k.a. J in this file. See (2.157)."""
    if s > 1e-10:
        return c*(J(r, s, i, j, erfcs)
                  + (a1**2 + a2**2)/6. * Lap_J(r, s, i, j, erfcs))
    else:
        return kronmatrix[i][j]/(6*pi*mu*a1)


@njit(cache=True)
def btr(r, s, a1, a2, i, j, erfcs, c, mu):
    """Element ij of the wavespace periodic version of Minfinity submatrix
    b tilde. See docstring for ar."""
    if s > 1e-10:
        return c*(R(r, s, i, j, erfcs) + a1**2/6. * Lap_R(r, s, i, j, erfcs))
    else:
        return 0


@njit(cache=True)
def cr(r, s, a1, a2, i, j, erfcs, c, mu):
    """Element ij of the wavespace periodic version of Minfinity submatrix c.
    See docstring for ar."""
    if abs(r[0]) + abs(r[1]) + abs(r[2]) <= 1e-10:
        return kronmatrix[i][j]/(8*pi*mu*a1**3)
    k = (i+1) % 3
    l = (i+2) % 3
    return c*0.5*(D_R(r, s, k, l, j, erfcs) - D_R(r, s, l, k, j, erfcs))


@njit(cache=True)
def gtr(r, s, a1, a2, i, j, k, erfcs, c, mu):
    """Element ijk of the uncontracted wavespace periodic version of Minfinity
    submatrix g tilde. See docstring for ar."""
    if s > 1e-10:
        return -c*(K(r, s, i, j, k, erfcs)
                   + (a1**2/6. + a2**2/10.) * Lap_K(r, s, i, j, k, erfcs))
    else:
        return 0


@njit(cache=True)
def htr(r, s, a1, a2, i, j, k, erfcs, c, mu):
    """Element ijk of the uncontracted wavespace periodic version of Minfinity
    submatrix h tilde. See docstring for ar."""
    if abs(r[0]) + abs(r[1]) + abs(r[2]) > 1e-10:
        l = (i+1) % 3
        m = (i+2) % 3
        return c*-0.5*((D_K(r, s, l, m, j, k, erfcs)
                        + (a2**2/6.)*DLap_K(r, s, l, m, j, k, erfcs))
                       - (D_K(r, s, m, l, j, k, erfcs)
                          + (a2**2/6.)*DLap_K(r, s, m, l, j, k, erfcs)))
    else:
        return 0


@njit(cache=True)
def mr(r, s, a1, a2, i, j, k, l, erfcs, c, mu):
    """Element ijkl of the uncontracted wavespace periodic version of Minfinity
    submatrix m. See docstring for ar."""
    if s > 1e-10:
        return -0.5*c*((D_K(r, s, j, i, k, l, erfcs)
                        + D_K(r, s, i, j, k, l, erfcs))
                       + (a1**2 + a2**2)/10. * (DLap_K(r, s, j, i, k, l, erfcs)
                                                + DLap_K(r, s, i, j, k, l, erfcs)))
    else:
        return 0.5*(kron3tracelessmatrix[i][j][k][l])/(20./3.*pi*mu*a1**3)


# === FOURIER TRANSFORMED J^k(k) ===
# The only ones required are Jtilde, DD_Jtilde, D_Rtilde, D_Ktilde, LapJ_tilde,
# DLapK_tilde.

@njit(cache=True)
def Jtilde(ki, kj, ss, i, j, RR):
    """Fourier transform of wavespace Oseen tensor J^k_ij(k).
    See PhD thesis section A.2.7."""
    return -((i == j)*ss**2 - ki*kj)*RR


@njit(cache=True)
def DD_Jtilde(kkm, kkl, kki, kkj, ss, i, j, RR):
    """Fourier transform of 2nd derivative of wavespace Oseen tensor,
    D_m D_l J^k_ij(k). See PhD thesis section A.2.7."""
    return -kkm*kkl * Jtilde(kki, kkj, ss, i, j, RR)


@njit(cache=True)
def Lap_Jtilde(kki, kkj, ss, i, j, RR):
    """Fourier transform of Laplacian of wavespace Oseen tensor, D^2 J^k_ij(k).
    See PhD thesis section A.2.7."""
    return -ss**2 * Jtilde(kki, kkj, ss, i, j, RR)


@njit(cache=True)
def D_Ktilde(kkl, kki, kkj, kkk, ss, i, j, k, RR):
    """Fourier transform of derivative of wavespace K tensor,
    D_l K^k_ijk(k). See PhD thesis section A.2.7."""
    return 0.5*(DD_Jtilde(kkl, kkk, kki, kkj, ss, i, j, RR)
                + DD_Jtilde(kkl, kkj, kki, kkk, ss, i, k, RR))


@njit(cache=True)
def DLap_Ktilde(kkl, kki, kkj, kkk, ss, i, j, k, RR):
    """Fourier transform of derivative of Laplacian of wavespace K tensor,
    D_l D^2 K^k_ijk(k). See PhD thesis section A.2.7."""
    return 0.5 * (kkk*kkl*ss**2 * Jtilde(kki, kkj, ss, i, j, RR)
                  + kkl*kkj*ss**2 * Jtilde(kki, kkk, ss, i, k, RR))


@njit(cache=True)
def D_Rtilde(kk, ss, l, i, j, RR):
    """Fourier transform of derivative of rotlet, D_l R^k_ij(k).
    See PhD thesis section A.2.7."""
    kkl = kk[l]
    kki = kk[i]
    m = (j+1) % 3
    n = (j+2) % 3
    kkm = kk[m]
    kkn = kk[n]
    return -0.5 * (DD_Jtilde(kkl, kkm, kki, kkn, ss, i, n, RR)
                   - DD_Jtilde(kkl, kkn, kki, kkm, ss, i, m, RR))

# === FOURIER TRANSFORMED TENSORS ===


@njit(cache=True)
def aktilde(kk, ss, a1, a2, i, j, RR, c):
    """Element ij of the wavespace periodic version of Minfinity submatrix a.

    Equivalent to non-periodic definition (see PhD thesis table 2.1) with J
    replaced by Fourier transform of J^k_ij, a.k.a. Jtilde. See (2.158). Name
    'tilde' here means Fourier transform, not as in e.g. B/Btilde symmetry."""
    kki = kk[i]
    kkj = kk[j]
    return c*(Jtilde(kki, kkj, ss, i, j, RR)
              + (a1**2 + a2**2)/6. * Lap_Jtilde(kki, kkj, ss, i, j, RR))


@njit(cache=True)
def cktilde(kk, ss, a1, a2, i, j, RR, c):
    """Element ij of the wavespace periodic version of Minfinity submatrix c.
    See docstring for aktilde."""

    k = (i+1) % 3
    l = (i+2) % 3
    return c*0.5*(D_Rtilde(kk, ss, k, l, j, RR) - D_Rtilde(kk, ss, l, k, j, RR))


@njit(cache=True)
def htktilde(kk, ss, a1, a2, i, j, k, RR, c):
    """Element ijk of the uncontracted wavespace periodic version of Minfinity
    submatrix h tilde. See docstring for aktilde."""
    kkj = kk[j]
    kkk = kk[k]
    l = (i+1) % 3
    m = (i+2) % 3
    kkl = kk[l]
    kkm = kk[m]
    return c*-0.5*((D_Ktilde(kkl, kkm, kkj, kkk, ss, m, j, k, RR)
                    + (a2**2/6.)*DLap_Ktilde(kkl, kkm, kkj, kkk, ss, m, j, k, RR))
                   - (D_Ktilde(kkm, kkl, kkj, kkk, ss, l, j, k, RR)
                      + (a2**2/6.)*DLap_Ktilde(kkm, kkl, kkj, kkk, ss, l, j, k, RR)))


@njit(cache=True)
def mktilde(kk, ss, a1, a2, i, j, k, l, RR, c):
    """Element ijkl of the uncontracted wavespace periodic version of Minfinity
    submatrix m. See docstring for aktilde."""
    kki = kk[i]
    kkj = kk[j]
    kkk = kk[k]
    kkl = kk[l]
    return -0.5*c*(D_Ktilde(kkj, kki, kkk, kkl, ss, i, k, l, RR)
                   + D_Ktilde(kki, kkj, kkk, kkl, ss, j, k, l, RR)
                   + (a1**2 + a2**2)/10. * (
                       DLap_Ktilde(kkj, kki, kkk, kkl, ss, i, k, l, RR)
                       + DLap_Ktilde(kki, kkj, kkk, kkl, ss, j, k, l, RR)
    ))

# === MATRIX DEFINITIONS ===


@njit(cache=True)
def M11(i, j, r, s, a1, a2, erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn,
        erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points,
        num_K_points, c, mu, s_lmn, erfcs_lmn):
    """Element ij of Minfinity submatrix a. See PhD thesis (2.155)-(2.156)."""
    if s > 1e-10:
        # a_ab,rep = SUM_lmn a^r (x + x_lmn) + 1/L^3 SUM'_lmn exp(i k_lmn.x) atilde^k(k_lmn)
        #            --------(2)-------------   ---------------(4)---------------------------
        #            Copies of itself real          Copies of itself wavespace

        # (2)
        sum_ar = 0
        for q in range(num_X_points):
            sum_ar += ar(r + X_lmn[q], s_lmn[q], a1, a2, i, j, erfcs_lmn[q],
                         c, mu)
        # (4)
        # Imaginary part of e(i*k.r) always cancels out over the sum I think
        # (should probably try to show this but I'm pretty certain)
        sum_ak = 0
        for q in range(num_K_points):
            sum_ak += (math.cos(np.dot(K_lmn[q], r))
                       * aktilde(K_lmn[q], Ks_lmn[q], a1, a2, i, j, RR_K[q],
                                 c))
        sum_ak = 1./L**3 * sum_ak
        return sum_ar + sum_ak
    else:
        # a_aa,rep = a_aa + SUM'_lmn a^r (x_lmn) + 1/L^3 SUM'_lmn atilde^k(k_lmn) - a^k(0)
        #            -(1)-  --------(2)---------   ---------------(4)------------   -(5)--
        #           Normal  Copies of itself real  Copies of itself wavespace  (1)+(5)=a^r(0)

        # (1)
        # Technically this calls ar rather than a but when s = 0 it's the same
        a_aa = ar(r, s, a1, a2, i, j, erfcs, c, mu)
        # (2)
        sum_ar = 0.
        for q in range(num_Xdash_points):
            sum_ar += ar(Xdash_lmn[q], Sdash_lmn[q], a1, a1, i, j,
                         erfcs_Sdash_lmn[q], c, mu)
        # (4)
        sum_ak = 0.
        for q in range(num_K_points):
            sum_ak += aktilde(K_lmn[q], Ks_lmn[q], a1, a1, i, j, RR_K[q], c)
        sum_ak *= 1./L**3
        # (5)
        a_k0 = c*kronmatrix[i][j]*(8*lamb/spi
                                   + (a1**2+a2**2)/6.*(-160*lamb**3/(3*spi)))

        return a_aa + sum_ar + sum_ak - a_k0


@njit(cache=True)
def M12(i, j, r, s, a1, a2, X_lmn, num_X_points, c, mu, s_lmn, erfcs_lmn):
    """Element ij of Minfinity submatrix b tilde. Same form as PhD thesis
    (2.155)-(2.156), and see note at bottom of page."""
    if s > 1e-10:
        # (2)
        # Return value below is just sum_btr
        bt = 0.
        for q in range(num_X_points):
            bt += btr(r + X_lmn[q], s_lmn[q], a1, a2, i, j, erfcs_lmn[q], c,
                      mu)
        return bt
        # (4)
        # I think sum_btk always ends up being 0 (given that it's imaginary,
        # over the sum it looks like it cancels out)
        # Imaginary part of e(i*k.r) always cancels out over the sum I think
        # (should probably try to show this but I'm pretty certain)
        # return sum_btr
    else:
        return 0


@njit(cache=True)
def M13(i, j, k, r, s, a1, a2, X_lmn, num_X_points, c, mu, s_lmn, erfcs_lmn):
    """Element ijk of uncontracted Minfinity submatrix g tilde. Same form as
    PhD thesis (2.155)-(2.156), and see note at bottom of page."""
    if s > 1e-10:
        # (2)
        # Return value below is just sum_gtr
        gt = 0.
        for q in range(num_X_points):
            gt += gtr(r + X_lmn[q], s_lmn[q], a1, a2, i, j, k, erfcs_lmn[q],
                      c, mu)
        return gt
        # (4)
        # I think gtk always ends up being 0 (given that it's imaginary, over
        # the sum it looks like it cancels out)
        # return sum_gtr
    else:
        return 0


@njit(cache=True)
def M22(i, j, r, s, a1, a2, erfcs, L, X_lmn, Xdash_lmn, Sdash_lmn,
        erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points,
        num_K_points, c, mu, s_lmn, erfcs_lmn):
    """Element ij of Minfinity submatrix c. Same form as PhD thesis
    (2.155)-(2.156)."""
    if s > 1e-10:
        # (2)
        sum_cr = 0.
        for q in range(num_X_points):
            sum_cr += cr(r + X_lmn[q], s_lmn[q], a1, a2, i, j, erfcs_lmn[q],
                         c, mu)
        # (4)
        sum_ck = 0.
        for q in range(num_K_points):
            sum_ck += (math.cos(np.dot(K_lmn[q], r))
                       * cktilde(K_lmn[q], Ks_lmn[q], a1, a2, i, j, RR_K[q],
                                 c))
        sum_ck *= 1./L**3
        return sum_cr + sum_ck
    else:
        # (1)
        # Technically this calls cr rather than c but when s = 0 it's the same
        c_aa = cr(r, s, a1, a2, i, j, erfcs, c, mu)
        # (2)
        sum_cr = 0.
        for q in range(num_Xdash_points):
            sum_cr += cr(Xdash_lmn[q], Sdash_lmn[q], a1, a1, i, j,
                         erfcs_Sdash_lmn[q], c, mu)
        # (4)
        sum_ck = 0.
        for q in range(num_K_points):
            sum_ck += cktilde(K_lmn[q], Ks_lmn[q], a1, a1, i, j, RR_K[q], c)
        sum_ck *= 1./L**3
        # (5) = 0
        return c_aa + sum_cr + sum_ck


@njit(cache=True)
def M23(i, j, k, r, s, a1, a2, L, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn,
        K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,
        c, mu, s_lmn, erfcs_lmn):
    """Element ijk of uncontracted Minfinity submatrix h tilde. Same form as
    PhD thesis (2.155)-(2.156)."""
    if s > 1e-10:
        # (2)
        sum_htr = 0.
        for q in range(num_X_points):
            sum_htr += htr(r + X_lmn[q], s_lmn[q], a1, a2, i, j, k,
                           erfcs_lmn[q], c, mu)
        # (4)
        sum_htk = 0.
        for q in range(num_K_points):
            sum_htk += (math.cos(np.dot(K_lmn[q], r))
                        * htktilde(K_lmn[q], Ks_lmn[q], a1, a2, i, j, k,
                                   RR_K[q], c))
        sum_htk *= 1./L**3
        return sum_htr + sum_htk
    else:
        # (2)
        sum_hr = 0.
        for q in range(num_Xdash_points):
            sum_hr += htr(Xdash_lmn[q], Sdash_lmn[q], a1, a1, i, j, k,
                          erfcs_Sdash_lmn[q], c, mu)
        # (4)
        sum_hk = 0.
        for q in range(num_K_points):
            sum_hk += htktilde(K_lmn[q], Ks_lmn[q], a1, a1, i, j, k, RR_K[q], c)
        sum_hk *= 1./L**3
        return sum_hr + sum_hk


@njit(cache=True)
def M33(i, j, k, l, r, s, a1, a2, erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn,
        erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points,
        num_K_points, c, mu, s_lmn, erfcs_lmn):
    """Element ijkl of uncontracted Minfinity submatrix m. Same form as
    PhD thesis (2.155)-(2.156)."""
    if s > 1e-10:
        # (2)
        sum_mr = 0.
        for q in range(num_X_points):
            sum_mr += mr(r + X_lmn[q], s_lmn[q], a1, a2, i, j, k, l,
                         erfcs_lmn[q], c, mu)
        # (4)
        sum_mk = 0.
        for q in range(num_K_points):
            sum_mk += (math.cos(np.dot(K_lmn[q], r))
                       * mktilde(K_lmn[q], Ks_lmn[q], a1, a2, i, j, k, l,
                                 RR_K[q], c))
        sum_mk *= 1./L**3
        return sum_mr + sum_mk
    else:
        # (1)
        # Technically this calls mr rather than m but when s = 0 it's the same
        m_aa = mr(r, s, a1, a2, i, j, k, l, erfcs, c, mu)
        # (2)
        sum_mr = 0.
        for q in range(num_Xdash_points):
            sum_mr += mr(Xdash_lmn[q], Sdash_lmn[q], a1, a1, i, j, k, l,
                         erfcs_Sdash_lmn[q], c, mu)
        # (4)
        sum_mk = 0.
        for q in range(num_K_points):
            sum_mk += mktilde(K_lmn[q], Ks_lmn[q], a1, a1, i, j, k, l,
                              RR_K[q], c)
        sum_mk *= 1./L**3
        # (5)
        m_k0 = c*(
            (
                -8*lamb**3/(3*spi)
                + ((a1**2 + a2**2)/10. * (168*lamb**5)/(5*spi))
            )
            * -3*kron3tracelessmatrix[i][j][k][l]
        )
        return m_aa + sum_mr + sum_mk - m_k0


@njit(cache=True)
def con_M13_row(i, args):
    """Elements i[0:5] of (condensed) Minfinity submatrix g tilde. See
    docstring for M13, and see section 2.4.4 for condensation details."""
    A = M13(i, 0, 0, *args)
    B = M13(i, 1, 1, *args)
    return np.array([
        (hs3p1*A + hs3m1*B),
        s2*M13(i, 0, 1, *args),
        (hs3m1*A + hs3p1*B),
        s2*M13(i, 0, 2, *args),
        s2*M13(i, 1, 2, *args)
    ])


@njit(cache=True)
def con_M23_row(i, args):
    """Elements i[0:5] of (condensed) Minfinity submatrix h tilde. See
    docstring for M23, and see section 2.4.4 for condensation details."""
    A = M23(i, 0, 0, *args)
    B = M23(i, 1, 1, *args)
    return np.array([
        (hs3p1*A + hs3m1*B),
        s2*M23(i, 0, 1, *args),
        (hs3m1*A + hs3p1*B),
        s2*M23(i, 0, 2, *args),
        s2*M23(i, 1, 2, *args)
    ])


def generate_Minfinity_periodic(posdata, box_bottom_left, box_top_right,
                                printout=0, frameno=0, mu=1,
                                Ot_infinity=np.array([0, 0, 0]),
                                Et_infinity=np.array([[0, 0, 0],
                                                      [0, 0, 0],
                                                      [0, 0, 0]])):
    """Generate Minfinity matrix for periodic domain.

    Args:
        posdata: Particle position, size and count data
        box_bottom_left, box_top_right: Periodic box coordinates.
        printout: (Unused) flag which allows you to put in debug statements
        frameno: Frame number
        mu: viscosity
        O_infinity, ..., Et_infinity: Background flow

    Returns:
        L*Minfinity*R: Minfinity matrix
        "Minfinity": Human readable name of the matrix
    """
    # NOTE: Centre of background flow currently not implemented - 27/1/2017
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
    # Set lamb, the 'switch' between real and wavespace.
    # Beenakker says set this as lambda = sqrt(pi)/L
    Lx = box_top_right[0] - box_bottom_left[0]
    Ly = box_top_right[1] - box_bottom_left[1]
    Lz = box_top_right[2] - box_bottom_left[2]
    L = (Lx*Ly*Lz)**(1./3.)
    lamb = spi/L  # sqrt(pi)/L
    gridpoints_x = [i for i in range(-how_far_to_reproduce_gridpoints,
                                     how_far_to_reproduce_gridpoints+1)]
    gridpoints_y = [i for i in range(-how_far_to_reproduce_gridpoints,
                                     how_far_to_reproduce_gridpoints+1)]
    gridpoints_z = [i for i in range(-how_far_to_reproduce_gridpoints,
                                     how_far_to_reproduce_gridpoints+1)]
    X_lmn_canonical = np.array([[ll, mm, nn]
                                for ll in gridpoints_x
                                for mm in gridpoints_y
                                for nn in gridpoints_z])

    box_dimensions = box_top_right - box_bottom_left
    basis_canonical = np.diag(box_dimensions)
    sheared_basis_vectors = shear_basis_vectors(
        basis_canonical, box_dimensions, Ot_infinity, Et_infinity)
    X_lmn_sheared = np.dot(X_lmn_canonical, sheared_basis_vectors)
    # NOTE: If you change the next line you have to change it in K_lmn as well!
    X_lmn_sheared_inside_radius = X_lmn_sheared[
        np.linalg.norm(X_lmn_sheared, axis=1) <= 1.4142*how_far_to_reproduce_gridpoints*L]

    X_lmn = X_lmn_sheared_inside_radius
    Xdash_lmn = X_lmn_sheared_inside_radius[
        np.linalg.norm(X_lmn_sheared_inside_radius, axis=1) > 0]
    Sdash_lmn = np.linalg.norm(Xdash_lmn, axis=1)
    erfcs_Sdash_lmn = np.array([generate_erfcs(s, lamb) for s in Sdash_lmn])

    num_X_points = X_lmn.shape[0]
    num_Xdash_points = Xdash_lmn.shape[0]

    k_basis_vectors = 2*np.pi*L**(-3)*np.array(
        [np.cross(sheared_basis_vectors[0], sheared_basis_vectors[2]),
         np.cross(sheared_basis_vectors[2], sheared_basis_vectors[1]),
         np.cross(sheared_basis_vectors[0], sheared_basis_vectors[1])])
    K_lmn = np.dot(X_lmn_canonical, k_basis_vectors)[
        np.logical_and(np.linalg.norm(X_lmn_sheared, axis=1) <= 1.4142*how_far_to_reproduce_gridpoints*L,
                       np.linalg.norm(X_lmn_sheared, axis=1) > 0)]
    Ks_lmn = np.linalg.norm(K_lmn, axis=1)
    num_K_points = K_lmn.shape[0]
    RR_K = np.array([-8*math.pi/ks**4
                     * (1 + ks**2/(4*lamb**2) + ks**4/(8*lamb**4))
                     * math.exp(-ks**2/(4*lamb**2)) for ks in Ks_lmn])

    Minfinity = generate_Minfinity_loop(Minfinity, bead_positions, bead_sizes,
                                        num_spheres, num_dumbbells, c, mu,
                                        lamb, X_lmn, Xdash_lmn, Sdash_lmn,
                                        erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K,
                                        num_X_points, num_Xdash_points,
                                        num_K_points, L)

    if num_dumbbells > 0:
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
        Ldata = np.array([1 for _ in range(11*num_spheres)]
                         + [0.5 for _ in range(9*num_dumbbells)]
                         + [-0.5 for _ in range(3*num_dumbbells)])
        L = coo_matrix((Ldata, (Lrow, Lcol)),
                       shape=(11*num_spheres+6*num_dumbbells, 11*num_spheres+6*num_dumbbells))
        R = L.transpose()
        return ((L*Minfinity*R), "Minfinity")
    else:
        return (Minfinity, "Minfinity")


@njit(cache=True)
def generate_Minfinity_loop(Minfinity, bead_positions, bead_sizes, num_spheres,
                            num_dumbbells, c, mu,
                            lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn,
                            K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points,
                            num_K_points, L):
    """Helper function for generating Minfinity matrix for nonperiodic domain.
    This function can be Numba-d.

    Args:
        Minfinity: Zeroed Minfinity matrix
        bead_positions: Bead positions
        bead_sizes: Bead sizes
        num_spheres: Number of spheres
        num_dumbbells: Number of dumbbells
        c: constant, 1/(8 pi mu)
        mu: viscosity
        lamb: The 'switch' between real and wavespace. Beenakker says set
              this as lambda = sqrt(pi)/L
        X_lmn: array of vectors pointing to the other periodic cells (the grid)
        Xdash_lmn: X_lmn with 0 removed
        Sdash_lmn: Norm of Xdash_lmn
        erfcs_Sdash_lmn: erfcs of Sdash_lmn
        K_lmn: Wavespace version of X_lmn
        Ks_lmn: Norm of K_lmn
        RR_K: Pre-computed term that appears in F[J^k](k): see thesis (A.126)
        num_X_points: length of X_lmn
        num_Xdash_points: length of Xdash_lmn
        num_K_points: length of K_lmn
        L: Periodic box side length

    Returns:
        Minfinity: Minfinity matrix with elements filled in
    """

    M33_temp = np.empty((5,5), dtype=float)
    if num_dumbbells > 0:
        M34_temp = np.empty((5,3), dtype=float)
        M35_temp = np.empty((5,3), dtype=float)

    for a1_index, a2_index in [(u, v)
                               for u in range(len(bead_sizes))
                               for v in range(u, len(bead_sizes))]:
        # Displacement convention for the Minfinity formulae is a1-a2
        # (inconveniently the opposite of in R2Bexact)
        r = bead_positions[a1_index] - bead_positions[a2_index]
        a1 = bead_sizes[a1_index]
        a2 = bead_sizes[a2_index]
        s = norm(r)

        if s > 1e-8 and 2*s/(a1+a2) < 2.001:
            ss_out = 2.001*(a1+a2)/2
            r = np.array([r[0]*ss_out/s, r[1]*ss_out/s, r[2]*ss_out/s])
            s = ss_out

        (A_c, Bt_c, Bt21_c, Gt_c, Gt21_c, C_c, Ht_c, Ht21_c, M_c,
         M14_c, M24_c, M34_c, M44_c, M15_c, M25_c, M35_c, M45_c,
         M55_c) = submatrix_coords_tuple(a1_index, a2_index, num_spheres,
                                         num_dumbbells)

        if (is_sphere(a1_index, num_spheres) and
                is_sphere(a2_index, num_spheres)):
            # Sphere to sphere

            erfcs = generate_erfcs(s, lamb)
            # Strictly only needed if s > 1e-10 but 'if's in Numba are slow
            # s_lmn = np.linalg.norm(r + X_lmn, axis=1) # Unsupported by Numba
            s_lmn = np.sqrt(np.sum((r + X_lmn)**2,axis=1))
            erfcs_lmn = [generate_erfcs(S, lamb) for S in s_lmn]

            Minfinity[A_c[0]:A_c[0]+3,
                      A_c[1]:A_c[1]+3] = np.array([[
                          M11(i, j, r, s, a1, a2, erfcs, L, lamb,
                              X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn,
                              K_lmn, Ks_lmn, RR_K,
                              num_X_points, num_Xdash_points, num_K_points,
                              c, mu, s_lmn, erfcs_lmn)
                          for j in range(3)] for i in range(3)])

            Minfinity[Bt_c[0]:Bt_c[0]+3,
                      Bt_c[1]:Bt_c[1]+3] = np.array([[
                          M12(i, j, r, s, a1, a2,
                              X_lmn, num_X_points,
                              c, mu, s_lmn, erfcs_lmn)
                          for j in range(3)] for i in range(3)])

            Minfinity[C_c[0]:C_c[0]+3,
                      C_c[1]:C_c[1]+3] = np.array([[
                          M22(i, j, r, s, a1, a2, erfcs, L,
                              X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn,
                              K_lmn, Ks_lmn, RR_K,
                              num_X_points, num_Xdash_points, num_K_points,
                              c, mu, s_lmn, erfcs_lmn)
                          for j in range(3)] for i in range(3)])

            args = (r, s, a1, a2, X_lmn, num_X_points, c, mu, s_lmn, erfcs_lmn)
            for i in range(3):
                Minfinity[Gt_c[0]+i,
                          Gt_c[1]:Gt_c[1]+5] = con_M13_row(i, args)

            args = (r, s, a1, a2, L, X_lmn, Xdash_lmn,
                    Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K,
                    num_X_points, num_Xdash_points, num_K_points,
                    c, mu, s_lmn, erfcs_lmn)
            for i in range(3):
                Minfinity[Ht_c[0]+i,
                          Ht_c[1]:Ht_c[1]+5] = con_M23_row(i, args)

            args = (r, s, a1, a2, erfcs, L, lamb, X_lmn, Xdash_lmn,
                    Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K,
                    num_X_points, num_Xdash_points, num_K_points,
                    c, mu, s_lmn, erfcs_lmn)
            for i in range(5):
                for j in range(5):
                    M33_temp[i,j] = M33(*cond_idx[i], *cond_idx[j], *args)
            Minfinity[M_c[0]:M_c[0]+5,
                      M_c[1]:M_c[1]+5] = cond_E @ M33_temp @ cond_E

            if a1 == a2:
                Minfinity[Bt21_c[0]:Bt21_c[0]+3,
                          Bt21_c[1]:Bt21_c[1]+3] = -Minfinity[
                    Bt_c[0]:Bt_c[0]+3,
                    Bt_c[1]:Bt_c[1]+3]
                Minfinity[Gt21_c[0]:Gt21_c[0]+3,
                          Gt21_c[1]:Gt21_c[1]+5] = -Minfinity[
                    Gt_c[0]:Gt_c[0]+3,
                    Gt_c[1]:Gt_c[1]+5]
                Minfinity[Ht21_c[0]:Ht21_c[0]+3,
                          Ht21_c[1]:Ht21_c[1]+5] = Minfinity[
                    Ht_c[0]:Ht_c[0]+3,
                    Ht_c[1]:Ht_c[1]+5]
            else:
                mr = np.array([-r[0], -r[1], -r[2]])
                m_s_lmn = np.sqrt(np.sum((mr + X_lmn)**2,axis=1))
                m_erfcs_lmn = [generate_erfcs(S, lamb) for S in m_s_lmn]
                Minfinity[Bt21_c[0]:Bt21_c[0]+3,
                          Bt21_c[1]:Bt21_c[1]+3] = np.array([[
                              M12(i, j, mr, s, a2, a1,
                                  X_lmn, num_X_points,
                                  c, mu, m_s_lmn, m_erfcs_lmn)
                              for j in range(3)] for i in range(3)])

                args = (mr, s, a2, a1, X_lmn, num_X_points, c, mu,
                        m_s_lmn, m_erfcs_lmn)
                for i in range(3):
                    Minfinity[Gt21_c[0]+i,
                              Gt21_c[1]:Gt21_c[1]+5
                              ] = con_M13_row(i, args)

                args = (mr, s, a2, a1, L, X_lmn, Xdash_lmn,
                        Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K,
                        num_X_points, num_Xdash_points, num_K_points,
                        c, mu, m_s_lmn, m_erfcs_lmn)
                for i in range(3):
                    Minfinity[Ht21_c[0]+i,
                              Ht21_c[1]:Ht21_c[1]+5
                              ] = con_M23_row(i, args)

        elif (is_sphere(a1_index, num_spheres)
              and is_dumbbell_bead_1(a2_index, num_spheres, num_dumbbells)):
            # Sphere to dumbbell bead 1
            mr = np.array([-r[0], -r[1], -r[2]])

            # Strictly only needed if s > 1e-10 but 'if's in Numba are slow
            # s_lmn = np.linalg.norm(r + X_lmn, axis=1)  # Unsupported by Numba
            # m_s_lmn = np.linalg.norm(mr + X_lmn, axis=1)
            s_lmn = np.sqrt(np.sum((r + X_lmn)**2,axis=1))
            m_s_lmn = np.sqrt(np.sum((mr + X_lmn)**2,axis=1))
            erfcs_lmn = [generate_erfcs(S, lamb) for S in s_lmn]
            m_erfcs_lmn = [generate_erfcs(S, lamb) for S in m_s_lmn]

            Minfinity[M14_c[0]:M14_c[0]+3,
                      M14_c[1]:M14_c[1]+3] = np.array([[
                          M11(i, j, r, s, a1, a2, erfcs, L, lamb,
                              X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn,
                              K_lmn, Ks_lmn, RR_K,
                              num_X_points, num_Xdash_points, num_K_points,
                              c, mu, s_lmn, erfcs_lmn)
                          for j in range(3)] for i in range(3)])

            Minfinity[M24_c[0]:M24_c[0]+3,
                      M24_c[1]:M24_c[1]+3] = np.array([[
                          M12(j, i, mr, s, a1, a2,
                              X_lmn, num_X_points,
                              c, mu, m_s_lmn, m_erfcs_lmn)
                          for j in range(3)] for i in range(3)])

            args = (mr, s, a1, a2, X_lmn, num_X_points, c, mu, m_s_lmn,
                    m_erfcs_lmn)
            for i in range(5):
                for j in range(3):
                    M34_temp[i,j] = M13(j, *cond_idx[i], *args)
            Minfinity[M34_c[0]:M34_c[0]+5,
                      M34_c[1]:M34_c[1]+3] = cond_E @ M34_temp

        elif is_sphere(a1_index, num_spheres):
            # Sphere to dumbbell bead 2
            mr = np.array([-r[0], -r[1], -r[2]])

            # Strictly only needed if s > 1e-10 but 'if's in Numba are slow
            # s_lmn = np.linalg.norm(r + X_lmn, axis=1)  # Unsupported by Numba
            # m_s_lmn = np.linalg.norm(mr + X_lmn, axis=1)
            s_lmn = np.sqrt(np.sum((r + X_lmn)**2,axis=1))
            m_s_lmn = np.sqrt(np.sum((mr + X_lmn)**2,axis=1))
            erfcs_lmn = [generate_erfcs(S, lamb) for S in s_lmn]
            m_erfcs_lmn = [generate_erfcs(S, lamb) for S in m_s_lmn]

            Minfinity[M15_c[0]:M15_c[0]+3,
                      M15_c[1]:M15_c[1]+3] = np.array([[
                          M11(i, j, r, s, a1, a2, erfcs, L, lamb,
                              X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn,
                              K_lmn, Ks_lmn, RR_K,
                              num_X_points, num_Xdash_points, num_K_points,
                              c, mu, s_lmn, erfcs_lmn)
                          for j in range(3)] for i in range(3)])
            Minfinity[M25_c[0]:M25_c[0]+3,
                      M25_c[1]:M25_c[1]+3] = np.array([[
                          M12(j, i, mr, s, a1, a2, X_lmn, num_X_points,
                              c, mu, m_s_lmn, m_erfcs_lmn)
                          for j in range(3)] for i in range(3)])

            args = (mr, s, a1, a2, X_lmn, num_X_points, c, mu, m_s_lmn,
                    m_erfcs_lmn)
            for i in range(5):
                for j in range(3):
                    M35_temp[i,j] = M13(j, *cond_idx[i], *args)
            Minfinity[M35_c[0]:M35_c[0]+5,
                      M35_c[1]:M35_c[1]+3] = cond_E @ M35_temp

        elif (is_dumbbell_bead_1(a1_index, num_spheres, num_dumbbells)
              and is_dumbbell_bead_1(a2_index, num_spheres, num_dumbbells)):
            # Dumbbell bead 1 to dumbbell bead 1
            a1_index_d = a1_index-num_spheres
            a2_index_d = a2_index-num_spheres
            if bead_bead_interactions or a1_index_d == a2_index_d:
                erfcs = generate_erfcs(s, lamb)
                # s_lmn = np.linalg.norm(r + X_lmn, axis=1) # Unsupported Numba
                s_lmn = np.sqrt(np.sum((r + X_lmn)**2,axis=1))
                erfcs_lmn = [generate_erfcs(S, lamb) for S in s_lmn]

                Minfinity[M44_c[0]:M44_c[0]+3,
                          M44_c[1]:M44_c[1]+3] = np.array([[
                              M11(i, j, r, s, a1, a2, erfcs, L, lamb,
                                  X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn,
                                  K_lmn, Ks_lmn, RR_K,
                                  num_X_points, num_Xdash_points, num_K_points,
                                  c, mu, s_lmn, erfcs_lmn)
                              for j in range(3)] for i in range(3)])

        elif (is_dumbbell_bead_1(a1_index, num_spheres, num_dumbbells)
              and is_dumbbell_bead_2(a2_index, num_spheres, num_dumbbells)):
            # Dumbbell bead 1 to dumbbell bead 2
            a1_index_d = a1_index-num_spheres
            a2_index_d = a2_index-num_spheres-num_dumbbells
            if bead_bead_interactions or a1_index_d == a2_index_d:
                erfcs = generate_erfcs(s, lamb)
                # s_lmn = np.linalg.norm(r + X_lmn, axis=1) # Unsupported Numba
                s_lmn = np.sqrt(np.sum((r + X_lmn)**2,axis=1))
                erfcs_lmn = [generate_erfcs(S, lamb) for S in s_lmn]

                Minfinity[M45_c[0]:M45_c[0]+3,
                          M45_c[1]:M45_c[1]+3] = np.array([[
                              M11(i, j, r, s, a1, a2, erfcs, L, lamb,
                                  X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn,
                                  K_lmn, Ks_lmn, RR_K,
                                  num_X_points, num_Xdash_points, num_K_points,
                                  c, mu, s_lmn, erfcs_lmn)
                              for j in range(3)] for i in range(3)])

        else:
            # Dumbbell bead 2 to dumbbell bead 2
            a1_index_d = a1_index-num_spheres-num_dumbbells
            a2_index_d = a2_index-num_spheres-num_dumbbells
            if bead_bead_interactions or a1_index_d == a2_index_d:
                erfcs = generate_erfcs(s, lamb)
                # s_lmn = np.linalg.norm(r + X_lmn, axis=1) # Unsupported Numba
                s_lmn = np.sqrt(np.sum((r + X_lmn)**2,axis=1))
                erfcs_lmn = [generate_erfcs(S, lamb) for S in s_lmn]

                Minfinity[M55_c[0]:M55_c[0]+3,
                          M55_c[1]:M55_c[1]+3] = np.array([[
                              M11(i, j, r, s, a1, a2, erfcs, L, lamb,
                                  X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn,
                                  K_lmn, Ks_lmn, RR_K,
                                  num_X_points, num_Xdash_points, num_K_points,
                                  c, mu, s_lmn, erfcs_lmn)
                              for j in range(3)] for i in range(3)])

    # symmetrise
    Minfinity = np.triu(Minfinity) + np.triu(Minfinity, k=1).transpose()

    return Minfinity
