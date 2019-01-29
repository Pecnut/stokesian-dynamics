#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 10/01/2017

import numpy as np
from numpy import sqrt, pi
from functions_shared import posdata_data, add_sphere_rotations_to_positions
from functions_shared import norm
from inputs import posdata, how_far_to_reproduce_gridpoints, bead_bead_interactions
from scipy.sparse import coo_matrix
from scipy.special import erfc, erf
from scipy.integrate import tplquad
import math

# === DERIVATIVES OF r erfc (lambda r) ===
# 1st derivatives
from ewald_functions.D_rerfclr import D_rerfclr
# 2nd derivatives
from ewald_functions.DD_rerfclr import DD_rerfclr
from ewald_functions.Lap_rerfclr import Lap_rerfclr
# 3rd derivatives
from ewald_functions.DDD_rerfclr import DDD_rerfclr
from ewald_functions.DLap_rerfclr import DLap_rerfclr
# 4th derivatives
from ewald_functions.DDDD_rerfclr import DDDD_rerfclr
from ewald_functions.DDLap_rerfclr import DDLap_rerfclr
from ewald_functions.LapLap_rerfclr import LapLap_rerfclr
# 5th derivatives
from ewald_functions.DDDLap_rerfclr import DDDLap_rerfclr
from ewald_functions.DLapLap_rerfclr import DLapLap_rerfclr
# 6th derivatives
from ewald_functions.DDDDLap_rerfclr import DDDDLap_rerfclr
from ewald_functions.DDLapLap_rerfclr import DDLapLap_rerfclr
# Derivatives of erfc (lambda r)
from ewald_functions.generate_erfcs import generate_erfcs

# === CONSTANTS ===

s3 = sqrt(3)
s2 = sqrt(2)
kronmatrix = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
# NOTE: kron3traceless_ijkl = d_ik d_jl + d_il d_jk - 2/3 d_ij d_kl
kron3tracelessmatrix = [[[[1.3333333333333335, 0.0, 0.0, 0.0, 0.0],   [0.0, -0.6666666666666666, 0.0, 0.0, 0.0],   [0.0, 0.0, -0.6666666666666666, 0.0, 0.0],   [0.0, 0.0, 0.0, -0.6666666666666666, 0.0],   [0.0, 0.0, 0.0, 0.0, -0.6666666666666666]],
  [[0.0, 1.0, 0.0, 0.0, 0.0],   [1.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 1.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [1.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 1.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [1.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 1.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [1.0, 0.0, 0.0, 0.0, 0.0]]],
 [[[0.0, 1.0, 0.0, 0.0, 0.0],   [1.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[-0.6666666666666666, 0.0, 0.0, 0.0, 0.0],   [0.0, 1.3333333333333335, 0.0, 0.0, 0.0],   [0.0, 0.0, -0.6666666666666666, 0.0, 0.0],   [0.0, 0.0, 0.0, -0.6666666666666666, 0.0],   [0.0, 0.0, 0.0, 0.0, -0.6666666666666666]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 1.0, 0.0, 0.0],   [0.0, 1.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 1.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 1.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 1.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 1.0, 0.0, 0.0, 0.0]]],
 [[[0.0, 0.0, 1.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [1.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 1.0, 0.0, 0.0],   [0.0, 1.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[-0.6666666666666666, 0.0, 0.0, 0.0, 0.0],   [0.0, -0.6666666666666666, 0.0, 0.0, 0.0],   [0.0, 0.0, 1.3333333333333335, 0.0, 0.0],   [0.0, 0.0, 0.0, -0.6666666666666666, 0.0],   [0.0, 0.0, 0.0, 0.0, -0.6666666666666666]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 1.0, 0.0],   [0.0, 0.0, 1.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 1.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 1.0, 0.0, 0.0]]],
 [[[0.0, 0.0, 0.0, 1.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [1.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 1.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 1.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 1.0, 0.0],   [0.0, 0.0, 1.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0]],
  [[-0.6666666666666666, 0.0, 0.0, 0.0, 0.0],   [0.0, -0.6666666666666666, 0.0, 0.0, 0.0],   [0.0, 0.0, -0.6666666666666666, 0.0, 0.0],   [0.0, 0.0, 0.0, 1.3333333333333335, 0.0],   [0.0, 0.0, 0.0, 0.0, -0.6666666666666666]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 1.0],   [0.0, 0.0, 0.0, 1.0, 0.0]]],
 [[[0.0, 0.0, 0.0, 0.0, 1.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [1.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 1.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 1.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 1.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 1.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 0.0],   [0.0, 0.0, 0.0, 0.0, 1.0],   [0.0, 0.0, 0.0, 1.0, 0.0]],
  [[-0.6666666666666666, 0.0, 0.0, 0.0, 0.0],   [0.0, -0.6666666666666666, 0.0, 0.0, 0.0],   [0.0, 0.0, -0.6666666666666666, 0.0, 0.0],   [0.0, 0.0, 0.0, -0.6666666666666666, 0.0],   [0.0, 0.0, 0.0, 0.0, 1.3333333333333335]]]]

# === DERIVATIVES OF J^r(r) ===
# O(J)
def J(r,ss,i,j,erfcs):
    return kronmatrix[i][j]*Lap_rerfclr(ss,erfcs[0],erfcs[1],erfcs[2]) - DD_rerfclr(r[i],r[j],ss,erfcs[0],erfcs[1],erfcs[2],i,j)

# O(D J)
def R(r,ss,i,j,erfcs):
    k = (j+1)%3
    l = (j+2)%3
    return -0.5*( D_J(r,ss,k,i,l,erfcs) - D_J(r,ss,l,i,k,erfcs))

def K(r,ss,i,j,k,erfcs):
    return 0.5*(D_J(r,ss,k,i,j,erfcs) + D_J(r,ss,j,i,k,erfcs))

def D_J(r,ss,l,i,j,erfcs):
    return kronmatrix[i][j]*DLap_rerfclr(r[l],ss,erfcs[0],erfcs[1],erfcs[2],erfcs[3]) - DDD_rerfclr(r[i],r[j],r[l],ss,erfcs[0],erfcs[1],erfcs[2],erfcs[3],i,j,l)

# O(D^2 J)
from ewald_functions.DD_J import DD_J

def D_R(r,ss,l,i,j,erfcs):
    m = (j+1)%3
    n = (j+2)%3
    return -0.5 * ( DD_J(r,ss,l,m,i,n,erfcs) - DD_J(r,ss,l,n,i,m,erfcs) )

def D_K(r,ss,l,i,j,k,erfcs):
    return 0.5*(DD_J(r,ss,l,k,i,j,erfcs) + DD_J(r,ss,l,j,i,k,erfcs))

def Lap_J(r,ss,i,j,erfcs):
    return kronmatrix[i][j]*LapLap_rerfclr(ss,erfcs[2],erfcs[3],erfcs[4]) - DDLap_rerfclr(r[i],r[j],ss,erfcs[0],erfcs[1],erfcs[2],erfcs[3],erfcs[4],i,j)

# O(D^3 J)
def DLap_J(r,ss,k,i,j,erfcs):
    rk = r[k]
    return kronmatrix[i][j]*DLapLap_rerfclr(rk,ss,erfcs[2],erfcs[3],erfcs[4],erfcs[5]) - DDDLap_rerfclr(r[i],r[j],rk,ss,erfcs[0],erfcs[1],erfcs[2],erfcs[3],erfcs[4],erfcs[5],i,j,k)

def Lap_R(r,ss,i,j,erfcs):
    k = (j+1)%3
    l = (j+2)%3
    return -0.5*( DLap_J(r,ss,k,i,l,erfcs) - DLap_J(r,ss,l,i,k,erfcs))

def Lap_K(r,ss,i,j,k,erfcs):
    return 0.5*(DLap_J(r,ss,k,i,j,erfcs) + DLap_J(r,ss,j,i,k,erfcs))

# O(D^4 J)
def DDLap_J(r,ss,l,k,i,j,erfcs):
    rk = r[k]
    rl = r[l]
    return kronmatrix[i][j]*DDLapLap_rerfclr(rk,rl,ss,erfcs[2],erfcs[3],erfcs[4],erfcs[5],erfcs[6],k,l) - DDDDLap_rerfclr(r[i],r[j],rk,rl,ss,erfcs[0],erfcs[1],erfcs[2],erfcs[3],erfcs[4],erfcs[5],erfcs[6],i,j,k,l)

def DLap_K(r,ss,l,i,j,k,erfcs):
    return 0.5*(DDLap_J(r,ss,l,k,i,j,erfcs) + DDLap_J(r,ss,l,j,i,k,erfcs))

# === TENSORS a^r(r) etc. ===
#a^r
def ar(r,s,a1,a2, i, j,erfcs,c,mu):
    if s > 1e-10:
        return c*(J(r,s,i,j,erfcs) + (a1**2 + a2**2)/6. * Lap_J(r,s,i,j,erfcs))
    else:
        return kronmatrix[i][j]/(6*pi*mu*a1)

def btr(r,s,a1,a2,i,j,erfcs,c,mu):
    if s > 1e-10:
        return c*(R(r,s,i,j,erfcs) + a1**2/6. * Lap_R(r,s,i,j,erfcs))
    else:
        return 0

def cr(r,s,a1,a2, i, j,erfcs,c,mu):
    if abs(r[0]) + abs(r[1]) + abs(r[2]) > 1e-10:
        k = (i+1)%3
        l = (i+2)%3
        return c*0.5*( D_R(r,s,k,l,j,erfcs) - D_R(r,s,l,k,j,erfcs) )
    else:
        return kronmatrix[i][j]/(8*pi*mu*a1**3)

def gtr(r,s,a1,a2, i, j, k,erfcs,c,mu):
    if s > 1e-10:
        return -c*(K(r,s,i,j,k,erfcs) + (a1**2/6. + a2**2/10.) * Lap_K(r,s,i,j,k,erfcs))
    else:
        return 0

def htr(r,s,a1,a2,i,j,k,erfcs,c,mu):
    if abs(r[0]) + abs(r[1]) + abs(r[2]) > 1e-10:
        l = (i+1)%3
        m = (i+2)%3
        return c*-0.5*( (D_K(r,s,l,m,j,k,erfcs) + (a2**2/6.)*DLap_K(r,s,l,m,j,k,erfcs))
                       -(D_K(r,s,m,l,j,k,erfcs) + (a2**2/6.)*DLap_K(r,s,m,l,j,k,erfcs))     )
    else:
        return 0

def mr(r,s,a1,a2, i, j, k, l,erfcs,c,mu):
    if s > 1e-10:
        return -0.5*c*((D_K(r,s,j,i,k,l,erfcs) + D_K(r,s,i,j,k,l,erfcs)) + (a1**2 + a2**2)/10. * (DLap_K(r,s,j,i,k,l,erfcs) + DLap_K(r,s,i,j,k,l,erfcs)))
    else:
        return 0.5*(kron3tracelessmatrix[i][j][k][l])/(20./3.*pi*mu*a1**3)


# === FOURIER TRANSFORMED J^k(k) ===
# The only ones required are Jtilde, DD_Jtilde, D_Rtilde, D_Ktilde, LapJ_tilde, DLapK_tilde.
# All apart from D_Rtilde (for no good reason tbh) are in Cython files in the ewald_functions folder.
from ewald_functions.Jtilde import Jtilde
from ewald_functions.DD_Jtilde import DD_Jtilde
from ewald_functions.Lap_Jtilde import Lap_Jtilde
from ewald_functions.D_Ktilde import D_Ktilde
from ewald_functions.DLap_Ktilde import DLap_Ktilde

def D_Rtilde(kk,ss,l,i,j,RR):
    kkl = kk[l]
    kki = kk[i]
    m = (j+1)%3
    n = (j+2)%3
    kkm = kk[m]
    kkn = kk[n]
    return -0.5 * (DD_Jtilde(kkl,kkm,kki,kkn,ss,l,m,i,n,RR) - DD_Jtilde(kkl,kkn,kki,kkm,ss,l,n,i,m,RR))

# === FOURIER TRANSFORMED TENSORS ===
def aktilde(kk,ss,a1,a2,i,j,RR,c,mu):
    kki = kk[i]
    kkj = kk[j]
    return c*(Jtilde(kki,kkj,ss,i,j,RR) + (a1**2 + a2**2)/6. * Lap_Jtilde(kki,kkj,ss,i,j,RR))

def cktilde(kk,ss,a1,a2,i,j,RR,c,mu):
    k = (i+1)%3
    l = (i+2)%3
    return c*0.5*(D_Rtilde(kk,ss,k,l,j,RR) - D_Rtilde(kk,ss,l,k,j,RR))

def htktilde(kk,ss,a1,a2,i,j,k,RR,c,mu):
    kki = kk[i]
    kkj = kk[j]
    kkk = kk[k]
    l = (i+1)%3
    m = (i+2)%3
    kkl = kk[l]
    kkm = kk[m]
    return c*-0.5*(   (D_Ktilde(kkl,kkm,kkj,kkk,ss,l,m,j,k,RR) + (a2**2/6.)*DLap_Ktilde(kkl,kkm,kkj,kkk,ss,l,m,j,k,RR))
                    - (D_Ktilde(kkm,kkl,kkj,kkk,ss,m,l,j,k,RR) + (a2**2/6.)*DLap_Ktilde(kkm,kkl,kkj,kkk,ss,m,l,j,k,RR))    )

def mktilde(kk,ss,a1,a2,i,j,k,l,RR,c,mu):
    kki = kk[i]
    kkj = kk[j]
    kkk = kk[k]
    kkl = kk[l]
    return -0.5*c*(  D_Ktilde(kkj,kki,kkk,kkl,ss,j,i,k,l,RR) + D_Ktilde(kki,kkj,kkk,kkl,ss,i,j,k,l,RR) + (a1**2 + a2**2)/10. * (DLap_Ktilde(kkj,kki,kkk,kkl,ss,j,i,k,l,RR) + DLap_Ktilde(kki,kkj,kkk,kkl,ss,i,j,k,l,RR))  )

# === MATRIX DEFINITIONS ===
def M11(r,s,a1,a2, i, j, erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu):
    if s > 1e-10:
        # a_ab,rep = SUM_lmn a^r (x + x_lmn) + 1/L^3 SUM'_lmn exp(i k_lmn.x) atilde^k(k_lmn)
        #            --------(2)-------------   ---------------(4)---------------------------
        #            Copies of itself real          Copies of itself wavespace

        # (2)
        s_lmn = np.linalg.norm(r + X_lmn,axis=1)
        erfcs_lmn = [generate_erfcs(S,lamb) for S in s_lmn]
        sum_ar = sum([  ar(r + X_lmn[q],s_lmn[q],a1,a2,i,j,erfcs_lmn[q],c,mu)   for q in range(num_X_points) ])
        # (4)
        # Imaginary part of e(i*k.r) always cancels out over the sum I think (should probably try to show this but I'm pretty certain)
        sum_ak = 1./L**3 * (sum([  math.cos(np.dot(K_lmn[q],r)) * aktilde(K_lmn[q],Ks_lmn[q],a1,a2,i,j,RR_K[q],c,mu)   for q in range(num_K_points) ]))
        M11_ab = sum_ar + sum_ak
        return M11_ab

    else:
        # a_aa,rep = a_aa + SUM'_lmn a^r (x_lmn) + 1/L^3 SUM'_lmn atilde^k(k_lmn) - a^k(0)
        #            -(1)-  --------(2)---------   ---------------(4)------------   -(5)--
        #           Normal  Copies of itself real  Copies of itself wavespace       Forgotten lol

        # (1)
        a_aa = ar(r,s,a1,a2,i,j,erfcs,c,mu) # Technically this calls ar rather than a but when s = 0 it's the same response
        # (2)
        sum_ar = sum([   ar(Xdash_lmn[q],Sdash_lmn[q],a1,a1,i,j,erfcs_Sdash_lmn[q],c,mu)   for q in range(num_Xdash_points) ])
        # (4)
        sum_ak = 1./L**3 * (sum([  aktilde(K_lmn[q],Ks_lmn[q],a1,a1,i,j,RR_K[q],c,mu)   for q in range(num_K_points) ]))
        # (5)
        a_k0 = c*kronmatrix[i][j]*(8*lamb/math.sqrt(math.pi) + (a1**2+a2**2)/6.*(-160*lamb**3/( 3*math.sqrt(math.pi))))
        M11_aa = a_aa + sum_ar + sum_ak - a_k0
        return M11_aa

def M12(r,s,a1,a2, i, j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu):
    if s > 1e-10:
        # (2)
        s_lmn = np.linalg.norm(r + X_lmn,axis=1)
        erfcs_lmn = [generate_erfcs(S,lamb) for S in s_lmn]
        sum_btr = sum([  btr(r + X_lmn[q],s_lmn[q],a1,a2,i,j,erfcs_lmn[q],c,mu)   for q in range(num_X_points) ])
        # (4)
        # I think sum_btk always ends up being 0 (given that it's imaginary, over the sum it looks like it cancels out)
        # Imaginary part of e(i*k.r) always cancels out over the sum I think (should probably try to show this but I'm pretty certain)
        M12_ab = sum_btr
        return M12_ab
    else:
        return 0

def M13(r,s,a1,a2, i, j, k,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu):
    if s > 1e-10:
        # (2)
        s_lmn = np.linalg.norm(r + X_lmn,axis=1)
        erfcs_lmn = [generate_erfcs(S,lamb) for S in s_lmn]
        sum_gtr = sum([  gtr(r + X_lmn[q],s_lmn[q],a1,a2,i,j,k,erfcs_lmn[q],c,mu)   for q in range(num_X_points) ])
        # (4)
        #I think gtk always ends up being 0 (given that it's imaginary, over the sum it looks like it cancels out)
        M13_ab = sum_gtr
        return M13_ab
    else:
        return 0

def M22(r,s,a1,a2, i, j, erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu):
    if s > 1e-10:
        # (2)
        s_lmn = np.linalg.norm(r + X_lmn,axis=1)
        erfcs_lmn = [generate_erfcs(S,lamb) for S in s_lmn]
        sum_cr = sum([  cr(r + X_lmn[q],s_lmn[q],a1,a2,i,j,erfcs_lmn[q],c,mu)   for q in range(num_X_points) ])
        # (4)
        sum_ck = 1./L**3 * (sum([  math.cos(np.dot(K_lmn[q],r)) * cktilde(K_lmn[q],Ks_lmn[q],a1,a2,i,j,RR_K[q],c,mu)   for q in range(num_K_points) ]))
        M22_ab = sum_cr + sum_ck
        return M22_ab
    else:
        # (1)
        c_aa = cr(r,s,a1,a2,i,j,erfcs,c,mu) # Technically this calls ar rather than a but when s = 0 it's the same response
        # (2)
        sum_cr = sum([  cr(Xdash_lmn[q],Sdash_lmn[q],a1,a1,i,j,erfcs_Sdash_lmn[q],c,mu)   for q in range(num_Xdash_points) ])
        # (4)
        sum_ck = 1./L**3 * (sum([  cktilde(K_lmn[q],Ks_lmn[q],a1,a1,i,j,RR_K[q],c,mu)   for q in range(num_K_points) ]))
        # (5) = 0
        M22_aa = c_aa + sum_cr + sum_ck
        return M22_aa


def M23(r,s,a1,a2, i, j, k,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu):
    if s > 1e-10:
        # (2)
        s_lmn = np.linalg.norm(r + X_lmn,axis=1)
        erfcs_lmn = [generate_erfcs(S,lamb) for S in s_lmn]
        sum_htr = sum([  htr(r + X_lmn[q],s_lmn[q],a1,a2,i,j,k,erfcs_lmn[q],c,mu)   for q in range(num_X_points) ])
        # (4)
        sum_htk = 1./L**3 * (sum([  math.cos(np.dot(K_lmn[q],r)) * htktilde(K_lmn[q],Ks_lmn[q],a1,a2,i,j,k,RR_K[q],c,mu)   for q in range(num_K_points) ]))
        M23_ab = sum_htr + sum_htk
        return M23_ab
    else:
        # (2)
        sum_hr = sum([   htr(Xdash_lmn[q],Sdash_lmn[q],a1,a1,i,j,k,erfcs_Sdash_lmn[q],c,mu)   for q in range(num_Xdash_points) ])
        # (4)
        sum_hk = 1./L**3 * (sum([  htktilde(K_lmn[q],Ks_lmn[q],a1,a1,i,j,k,RR_K[q],c,mu)   for q in range(num_K_points) ]))
        M23_aa = sum_hr + sum_hk
        return M23_aa

def M33(r,s,a1,a2, i, j, k, l,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu):
    if s > 1e-10:
        # (2)
        s_lmn = np.linalg.norm(r + X_lmn,axis=1)
        erfcs_lmn = [generate_erfcs(S,lamb) for S in s_lmn]
        sum_mr = sum([  mr(r + X_lmn[q],s_lmn[q],a1,a2,i,j,k,l,erfcs_lmn[q],c,mu)   for q in range(num_X_points) ])
        # (4)
        sum_mk = 1./L**3 * (sum([  math.cos(np.dot(K_lmn[q],r)) * mktilde(K_lmn[q],Ks_lmn[q],a1,a2,i,j,k,l,RR_K[q],c,mu)   for q in range(num_K_points) ]))
        M33_ab = sum_mr + sum_mk
        return M33_ab
    else:
        # (1)
        m_aa = mr(r,s,a1,a2,i,j,k,l,erfcs,c,mu) # Technically this calls ar rather than a but when s = 0 it's the same response
        # (2)
        sum_mr = sum([  mr(Xdash_lmn[q],Sdash_lmn[q],a1,a1,i,j,k,l,erfcs_Sdash_lmn[q],c,mu)   for q in range(num_Xdash_points) ])
        # (4)
        sum_mk = 1./L**3 * (sum([  mktilde(K_lmn[q],Ks_lmn[q],a1,a2,i,j,k,l,RR_K[q],c,mu)   for q in range(num_K_points) ]))
        # (5)
        m_k0 = c*(-8*lamb**3/(3*math.sqrt(math.pi)) + (a1**2 + a2**2)/10. * (168*lamb**5)/(5*math.sqrt(math.pi)) ) * -3*kron3tracelessmatrix[i][j][k][l]
        M33_aa = m_aa + sum_mr + sum_mk - m_k0
        return M33_aa

def con_M13(r,s,a1,a2, i, m,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu):
    if m == 0:
        return 0.5*(s3+1)*M13(r,s,a1,a2,i,0,0,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) + 0.5*(s3-1)*M13(r,s,a1,a2,i,1,1,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif m == 1:
        return s2*M13(r,s,a1,a2,i,0,1,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif m == 2:
        return 0.5*(s3-1)*M13(r,s,a1,a2,i,0,0,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) + 0.5*(s3+1)*M13(r,s,a1,a2,i,1,1,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif m == 3:
        return s2*M13(r,s,a1,a2,i,0,2,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    else:
        return s2*M13(r,s,a1,a2,i,1,2,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)

def con_M23(r,s,a1,a2, i, m,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu):
    if m == 0:
        return 0.5*(s3+1)*M23(r,s,a1,a2,i,0,0,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) + 0.5*(s3-1)*M23(r,s,a1,a2,i,1,1,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif m == 1:
        return s2*M23(r,s,a1,a2,i,0,1,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif m == 2:
        return 0.5*(s3-1)*M23(r,s,a1,a2,i,0,0,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) + 0.5*(s3+1)*M23(r,s,a1,a2,i,1,1,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif m == 3:
        return s2*M23(r,s,a1,a2,i,0,2,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    else:
        return s2*M23(r,s,a1,a2,i,1,2,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)

def con1_M33(r,s,a1,a2, n, k, l,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu):
    if n == 0:
        return 0.5*(s3+1)*M33(r,s,a1,a2,0,0, k, l,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) + 0.5*(s3-1)*M33(r,s,a1,a2,1,1,k, l,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif n == 1:
        return s2*M33(r,s,a1,a2,0,1,k,l,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif n == 2:
        return 0.5*(s3-1)*M33(r,s,a1,a2,0,0,k,l,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) + 0.5*(s3+1)*M33(r,s,a1,a2,1,1,k,l,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif n == 3:
        return s2*M33(r,s,a1,a2,0,2,k,l,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    else:
        return s2*M33(r,s,a1,a2,1,2,k,l,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)

def con_M33(r,s,a1,a2, n, m,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu):
    if m == 0:
        return 0.5*(s3+1)*con1_M33(r,s,a1,a2,n,0,0,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) + 0.5*(s3-1)*con1_M33(r,s,a1,a2,n,1,1,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)

    elif m == 1:
        return s2*con1_M33(r,s,a1,a2,n,0,1,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif m == 2:
        return 0.5*(s3-1)*con1_M33(r,s,a1,a2,n,0,0,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) + 0.5*(s3+1)*con1_M33(r,s,a1,a2,n,1,1,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    elif m == 3:
        return s2*con1_M33(r,s,a1,a2,n,0,2,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)
    else:
        return s2*con1_M33(r,s,a1,a2,n,1,2,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)




def generate_Minfinity_periodic(posdata, box_bottom_left, box_top_right, printout=0, cutoff_factor=2,frameno=0,O_infinity=np.array([0,0,0]),E_infinity=np.array([[0,0,0],[0,0,0],[0,0,0]]),timestep=0.1,centre_of_background_flow=np.array([0,0,0]),mu=1,frequency=1,amplitude=1):
    # NOTE: Centre of background flow currently not implemented - 27/1/2017
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)
    R2Bexact_sidelength = 11*num_spheres + 6*num_dumbbells
    R2Bexact = np.zeros((R2Bexact_sidelength, R2Bexact_sidelength), dtype=np.float)
    bead_positions = np.concatenate([sphere_positions,dumbbell_positions - 0.5*dumbbell_deltax, dumbbell_positions + 0.5*dumbbell_deltax])
    bead_sizes = np.concatenate([sphere_sizes, dumbbell_sizes, dumbbell_sizes])
    c = 1./(8*pi*mu)
    # Set lamb, the 'switch' between real and wavespace. Beenakker says set this as lambda = sqrt(pi)/L
    Lx = box_top_right[0] - box_bottom_left[0]
    Ly = box_top_right[1] - box_bottom_left[1]
    Lz = box_top_right[2] - box_bottom_left[2]
    L = (Lx*Ly*Lz)**(1./3.)
    lamb = math.sqrt(math.pi)/L
    gridpoints_x = [i for i in range(-how_far_to_reproduce_gridpoints,how_far_to_reproduce_gridpoints+1)]
    gridpoints_y = [i for i in range(-how_far_to_reproduce_gridpoints,how_far_to_reproduce_gridpoints+1)]
    gridpoints_z = [i for i in range(-how_far_to_reproduce_gridpoints,how_far_to_reproduce_gridpoints+1)]
    X_lmn_canonical = np.array([[ll,mm,nn] for ll in gridpoints_x for mm in gridpoints_y for nn in gridpoints_z])

    basis_canonical = np.array([[Lx,0,0],[0,Ly,0],[0,0,Lz]])
    # NOTE: For CONTINUOUS shear, set the following
    #time_t = frameno*timestep
    #sheared_basis_vectors_add_on = (np.cross(np.array(O_infinity)*time_t,basis_canonical).transpose() + np.dot(np.array(E_infinity)*time_t,(basis_canonical).transpose())).transpose()# + basis_canonical
    # NOTE: For OSCILLATORY shear, set the following (basically there isn't a way to find out shear given E)
    time_t = frameno*timestep
    gamma = amplitude*np.sin(time_t*frequency)
    Ot_infinity = np.array([0,0.5*gamma,0])
    Et_infinity = [[0,0,0.5*gamma],[0,0,0],[0.5*gamma,0,0]]
    sheared_basis_vectors_add_on = (np.cross(Ot_infinity,basis_canonical).transpose() + np.dot(Et_infinity,(basis_canonical).transpose())).transpose()

    sheared_basis_vectors_add_on_mod  = np.mod(sheared_basis_vectors_add_on,[Lx,Ly,Lz])
    sheared_basis_vectors = basis_canonical + sheared_basis_vectors_add_on_mod
    X_lmn_sheared = np.dot(X_lmn_canonical,sheared_basis_vectors)
    X_lmn_sheared_inside_radius = X_lmn_sheared[np.linalg.norm(X_lmn_sheared,axis=1)<=1.4142*how_far_to_reproduce_gridpoints*L] # NOTE: If you change this you have to change it in K_lmn as well!

    X_lmn = X_lmn_sheared_inside_radius
    Xdash_lmn = X_lmn_sheared_inside_radius[np.linalg.norm(X_lmn_sheared_inside_radius,axis=1)>0]
    Sdash_lmn = np.linalg.norm(Xdash_lmn,axis=1)
    erfcs_Sdash_lmn = [generate_erfcs(s,lamb) for s in Sdash_lmn]

    num_X_points = X_lmn.shape[0]
    num_Xdash_points = Xdash_lmn.shape[0]

    k_basis_vectors = 2*np.pi*L**(-3)*np.array([np.cross(sheared_basis_vectors[0],sheared_basis_vectors[2]),np.cross(sheared_basis_vectors[2],sheared_basis_vectors[1]),np.cross(sheared_basis_vectors[0],sheared_basis_vectors[1])])
    K_lmn = np.dot(X_lmn_canonical,k_basis_vectors)[np.logical_and(np.linalg.norm(X_lmn_sheared,axis=1)<=1.4142*how_far_to_reproduce_gridpoints*L,np.linalg.norm(X_lmn_sheared,axis=1)>0)]
    Ks_lmn = np.linalg.norm(K_lmn,axis=1)
    num_K_points = K_lmn.shape[0]
    RR_K = [-8*math.pi/ks**4 * (1 + ks**2/(4*lamb**2) + ks**4/(8*lamb**4)) * math.exp(-ks**2/(4*lamb**2)) for ks in Ks_lmn]

    for a1_index,a2_index in [(u,v) for u in range(len(bead_sizes)) for v in range(u,len(bead_sizes))]:
        r = (bead_positions[a2_index] - bead_positions[a1_index])
        a1 = bead_sizes[a1_index]
        a2 = bead_sizes[a2_index]
        s = norm(r)

        if s > 1e-8 and 2*s/(a1+a2) < 2.001:
            ss_out = 2.001*(a1+a2)/2
            r = [r[0]*ss_out/s,r[1]*ss_out/s,r[2]*ss_out/s]
            s = ss_out

        if a1_index < num_spheres and a2_index < num_spheres:
            # Sphere to sphere
            A_coords =  np.s_[              a1_index*3 :               (a1_index+1)*3,               a2_index*3 :               (a2_index+1)*3]
            Bt_coords = np.s_[              a1_index*3 :               (a1_index+1)*3, 3*num_spheres+a2_index*3 : 3*num_spheres+(a2_index+1)*3]
            Bt_coords_21 = np.s_[           a2_index*3 :               (a2_index+1)*3, 3*num_spheres+a1_index*3 : 3*num_spheres+(a1_index+1)*3]
            Gt_coords = np.s_[              a1_index*3 :               (a1_index+1)*3, 6*num_spheres+a2_index*5 : 6*num_spheres+(a2_index+1)*5]
            Gt_coords_21 = np.s_[           a2_index*3 :               (a2_index+1)*3, 6*num_spheres+a1_index*5 : 6*num_spheres+(a1_index+1)*5]
            C_coords =  np.s_[3*num_spheres+a1_index*3 : 3*num_spheres+(a1_index+1)*3, 3*num_spheres+a2_index*3 : 3*num_spheres+(a2_index+1)*3]
            Ht_coords = np.s_[3*num_spheres+a1_index*3 : 3*num_spheres+(a1_index+1)*3, 6*num_spheres+a2_index*5 : 6*num_spheres+(a2_index+1)*5]
            Ht_coords_21 = np.s_[3*num_spheres+a2_index*3 : 3*num_spheres+(a2_index+1)*3, 6*num_spheres+a1_index*5 : 6*num_spheres+(a1_index+1)*5]
            M_coords =  np.s_[6*num_spheres+a1_index*5 : 6*num_spheres+(a1_index+1)*5, 6*num_spheres+a2_index*5 : 6*num_spheres+(a2_index+1)*5]

            erfcs = generate_erfcs(s,lamb)
            R2Bexact[A_coords] =  [[M11(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)  for j in range(3)] for i in range(3)]
            R2Bexact[Bt_coords] = [[M12(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) for j in range(3)] for i in range(3)]
            R2Bexact[Bt_coords_21] = -R2Bexact[Bt_coords]
            R2Bexact[C_coords] =  [[M22(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) for j in range(3)] for i in range(3)]
            R2Bexact[Gt_coords] = [[con_M13(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) for j in range(5)] for i in range(3)]
            R2Bexact[Gt_coords_21] = -R2Bexact[Gt_coords]
            R2Bexact[Ht_coords] = [[con_M23(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) for j in range(5)] for i in range(3)]
            R2Bexact[Ht_coords_21] = R2Bexact[Ht_coords]
            R2Bexact[M_coords] = [[con_M33(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu) for j in range(5)] for i in range(5)]


        elif a1_index < num_spheres and a2_index >= num_spheres and a2_index < num_spheres + num_dumbbells:
            # Sphere to dumbbell bead 1
            mr = [-r[0],-r[1],-r[2]]
            a2_index_d = a2_index-num_spheres
            R14_coords = np.s_[a1_index*3:(a1_index+1)*3,                             11*num_spheres+a2_index_d*3 : 11*num_spheres +(a2_index_d+1)*3]
            R24_coords = np.s_[3*num_spheres+a1_index*3:3*num_spheres+(a1_index+1)*3, 11*num_spheres+a2_index_d*3 : 11*num_spheres +(a2_index_d+1)*3]
            R34_coords = np.s_[6*num_spheres+a1_index*5:6*num_spheres+(a1_index+1)*5, 11*num_spheres+a2_index_d*3 : 11*num_spheres +(a2_index_d+1)*3]

            R2Bexact[R14_coords] = [[M11(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)  for j in range(3)] for i in range(3)]
            R2Bexact[R24_coords] = [[M12(mr,s,a2,a1,j,i,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)  for j in range(3)] for i in range(3)]
            R2Bexact[R34_coords] = [[con_M13(mr,s,a1,a2,j,i,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)  for j in range(3)] for i in range(5)]


        elif a1_index < num_spheres and a2_index >= num_spheres + num_dumbbells:
            # Sphere to dumbbell bead 2
            mr = [-r[0],-r[1],-r[2]]
            a2_index_d = a2_index-num_spheres-num_dumbbells
            R15_coords = np.s_[a1_index*3:(a1_index+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3 : 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
            R25_coords = np.s_[3*num_spheres+a1_index*3:3*num_spheres+(a1_index+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3 : 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
            R35_coords = np.s_[6*num_spheres+a1_index*5:6*num_spheres+(a1_index+1)*5, 11*num_spheres+3*num_dumbbells+a2_index_d*3 : 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]

            R2Bexact[R15_coords] = [[M11(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)  for j in range(3)] for i in range(3)]
            R2Bexact[R25_coords] = [[M12(mr,s,a2,a1,j,i,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)  for j in range(3)] for i in range(3)]
            R2Bexact[R35_coords] = [[con_M13(mr,s,a1,a2,j,i,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)  for j in range(3)] for i in range(5)]

        elif a1_index >= num_spheres and a1_index < num_spheres + num_dumbbells and a2_index >= num_spheres and a2_index < num_spheres + num_dumbbells:
            # Dumbbell bead 1 to dumbbell bead 1
            a1_index_d = a1_index-num_spheres
            a2_index_d = a2_index-num_spheres
            if bead_bead_interactions or a1_index_d == a2_index_d:
                R44_coords = np.s_[11*num_spheres+a1_index_d*3:11*num_spheres+(a1_index_d+1)*3, 11*num_spheres+a2_index_d*3 : 11*num_spheres+(a2_index_d+1)*3]
                erfcs = generate_erfcs(s,lamb)
                R2Bexact[R44_coords] = [[M11(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)  for j in range(3)] for i in range(3)]

                s_lmn = np.linalg.norm(r + X_lmn,axis=1)

        elif a1_index >= num_spheres and a1_index < num_spheres + num_dumbbells and a2_index >= num_spheres + num_dumbbells:
            # Dumbbell bead 1 to dumbbell bead 2
            a1_index_d = a1_index-num_spheres
            a2_index_d = a2_index-num_spheres-num_dumbbells
            if bead_bead_interactions or a1_index_d == a2_index_d:
                R45_coords = np.s_[11*num_spheres+a1_index_d*3:11*num_spheres+(a1_index_d+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3 : 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
                erfcs = generate_erfcs(s,lamb)
                R2Bexact[R45_coords] = [[M11(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)  for j in range(3)] for i in range(3)]

        else:
            # Dumbbell bead 2 to dumbbell bead 2
            a1_index_d = a1_index-num_spheres-num_dumbbells
            a2_index_d = a2_index-num_spheres-num_dumbbells
            if bead_bead_interactions or a1_index_d == a2_index_d:
                R55_coords = np.s_[11*num_spheres+3*num_dumbbells+a1_index_d*3:11*num_spheres+3*num_dumbbells+(a1_index_d+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3 : 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
                erfcs = generate_erfcs(s,lamb)
                R2Bexact[R55_coords] = [[M11(r,s,a1,a2,i,j,erfcs, L, lamb, X_lmn, Xdash_lmn, Sdash_lmn, erfcs_Sdash_lmn, K_lmn, Ks_lmn, RR_K, num_X_points, num_Xdash_points, num_K_points,c,mu)  for j in range(3)] for i in range(3)]

    #symmetrise
    R2Bexact = np.triu(R2Bexact) + np.triu(R2Bexact,k=1).transpose()

    # Row and column ops I want are equivalent to doing
    #   [ 1    0    0 ]   [ a b c ]   [ 1    0    0 ]
    #   [ 0  1/2  1/2 ] . [ d e f ] . [ 0  1/2 -1/2 ]
    #   [ 0 -1/2  1/2 ]   [ g h i ]   [ 0  1/2  1/2 ]
    #        "L"                       "R"

    # I know that we could generate L and R elsewhere rather than doing it every timestep but it takes 0.01s for a few thousand dumbbells so for now I don't mind
    Lrow = np.array([i for i in range(11*num_spheres + 6*num_dumbbells)] + [i + 11*num_spheres for i in range(3*num_dumbbells)] + [i + 11*num_spheres + 3*num_dumbbells for i in range(3*num_dumbbells)])
    Lcol = np.array([i for i in range(11*num_spheres + 6*num_dumbbells)] + [i + 11*num_spheres + 3*num_dumbbells for i in range(3*num_dumbbells)] + [i + 11*num_spheres for i in range(3*num_dumbbells)])
    Ldata = np.array([1 for i in range(11*num_spheres)] + [0.5 for i in range(9*num_dumbbells)] + [-0.5 for i in range(3*num_dumbbells)])
    L = coo_matrix((Ldata, (Lrow, Lcol)), shape=(11*num_spheres+6*num_dumbbells, 11*num_spheres+6*num_dumbbells))
    R = L.transpose()
    return ((L*R2Bexact*R), "Minfinity")
