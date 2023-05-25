#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

# Reference: A. K. Townsend, 2017. The mechanics of suspensions. PhD thesis, UCL. Appendix A.

import numpy as np
from numpy import sqrt, pi
from functions_shared import posdata_data, add_sphere_rotations_to_positions, levi
#from functions_shared import norm
from Minfinity_functions.norm import norm
#from inputs import mu, cutoff_factor, num_frames, text_only, viewbox_bottomleft_topright, printout, setup_number, posdata, s_dash_range, lam_range, lam_range_with_reciprocals, XYZ_raw, view_labels, fps, viewing_angle, timestep, trace_paths, two_d_plot, save_positions_every_n_timesteps, save_forces_every_n_timesteps, XYZf, use_XYZd_values, input_form, invert_m_every
from inputs import posdata, bead_bead_interactions
from scipy.sparse import coo_matrix

s3 = sqrt(3)
s2 = sqrt(2)
kronmatrix = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
kronkronmatrix = [[[[1, 0, 0, 0, 0],   [0, 1, 0, 0, 0],   [0, 0, 1, 0, 0],   [0, 0, 0, 1, 0],   [0, 0, 0, 0, 1]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]]],
 [[[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[1, 0, 0, 0, 0],   [0, 1, 0, 0, 0],   [0, 0, 1, 0, 0],   [0, 0, 0, 1, 0],   [0, 0, 0, 0, 1]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]]],
 [[[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[1, 0, 0, 0, 0],   [0, 1, 0, 0, 0],   [0, 0, 1, 0, 0],   [0, 0, 0, 1, 0],   [0, 0, 0, 0, 1]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]]],
 [[[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[1, 0, 0, 0, 0],   [0, 1, 0, 0, 0],   [0, 0, 1, 0, 0],   [0, 0, 0, 1, 0],   [0, 0, 0, 0, 1]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]]],
 [[[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 0]],
  [[1, 0, 0, 0, 0],   [0, 1, 0, 0, 0],   [0, 0, 1, 0, 0],   [0, 0, 0, 1, 0],   [0, 0, 0, 0, 1]]]]

# O(J)
from Minfinity_functions.J import J
#def J(r,ss,i,j):
#    return kronmatrix[i][j]/ss + r[i]*r[j]/ss**3

# O(D J)
def R(r,ss,i,j):
    return -0.5*sum([ levi(j,k,l)*D_J(r,ss,k,i,l) for k in range (3) for l in range(3) if k!=l and j!=k and j!=l])

def K(r,ss,i,j,k):
    return 0.5*(D_J(r,ss,k,i,j) + D_J(r,ss,j,i,k))

def D_J(r,ss,l,i,j):
    return (-kronmatrix[i][j]*r[l] + kronmatrix[i][l]*r[j] + kronmatrix[j][l]*r[i])/ss**3 - 3*r[i]*r[j]*r[l]/ss**5

# O(D^2 J)
def DD_J(r,ss,m,l,i,j):
    #return (-kronmatrix[i][j]*kronmatrix[l][m] + kronmatrix[i][l]*kronmatrix[j][m] + kronmatrix[j][l]*kronmatrix[i][m])/ss**3 - 3*(-kronmatrix[i][j]*r[l]*r[m] + kronmatrix[i][l]*r[j]*r[m] + kronmatrix[j][l]*r[i]*r[m] + kronmatrix[i][m]*r[j]*r[l] + r[i]*kronmatrix[j][m]*r[l] + r[i]*r[j]*kronmatrix[l][m])/ss**5 + 15*r[i]*r[j]*r[l]*r[m]/ss**7
    return (-kronkronmatrix[i][j][l][m] + kronkronmatrix[i][l][j][m] + kronkronmatrix[j][l][i][m])/ss**3 - 3*(-kronmatrix[i][j]*r[l]*r[m] + kronmatrix[i][l]*r[j]*r[m] + kronmatrix[j][l]*r[i]*r[m] + kronmatrix[i][m]*r[j]*r[l] + r[i]*kronmatrix[j][m]*r[l] + r[i]*r[j]*kronmatrix[l][m])/ss**5 + 15*r[i]*r[j]*r[l]*r[m]/ss**7

def D_R(r,ss,l,i,j):
    return -0.5 * sum([levi(j,m,n) * DD_J(r,ss,l,m,i,n) for m in range(3) for n in range(3) if m!=n and m!=j and n!=j])

def D_K(r,ss,l,i,j,k):
    return 0.5*(DD_J(r,ss,l,k,i,j) + DD_J(r,ss,l,j,i,k))

from Minfinity_functions.Lap_J import Lap_J
#def Lap_J(r,ss,i,j):
#    return 2*kronmatrix[i][j]/ss**3 - 6*r[i]*r[j]/ss**5

# O(D^3 J)
def DLap_J(r,ss,k,i,j):
    return (-6./ss**5)*(kronmatrix[i][j]*r[k] + kronmatrix[i][k]*r[j] + kronmatrix[j][k]*r[i]) + (30./ss**7)*r[i]*r[j]*r[k]

def Lap_R(r,ss,i,j):
    return -0.5*sum([levi(j,k,l) * DLap_J(r,ss,k,i,l) for k in range(3) for l in range(3) if k!=l and j!=k and j!=k])

def Lap_K(r,ss,i,j,k):
    return DLap_J(r,ss,i,j,k)

# O(D^4 J)
def DLap_K(r,ss,l,i,j,k):
    return (-6./ss**5)*(kronkronmatrix[i][j][k][l] + kronkronmatrix[i][k][j][l] + kronkronmatrix[j][k][i][l]) - (210./ss**9)*r[i]*r[j]*r[k]*r[l] + (30./ss**7)*(kronmatrix[i][j]*r[k]*r[l] + kronmatrix[i][k]*r[j]*r[l] + kronmatrix[j][k]*r[i]*r[l] + kronmatrix[i][l]*r[j]*r[k] + kronmatrix[j][l]*r[i]*r[k] + kronmatrix[k][l]*r[i]*r[j])

from Minfinity_functions.M11 import M11

'''
def M11(r,s,a1,a2, i, j,c,mu):
    if s > 1e-10:
        return c*(J(r[i],r[j],i,j,s) + (a1**2 + a2**2)/6. * Lap_J(r[i],r[j],i,j,s))
    else:
        return kronmatrix[i][j]/(6*pi*mu*a1)
'''

def M12(r,s,a1,a2, i, j,c,mu):
    if s > 1e-10:
        return c*(R(r,s,i,j) + a1**2/6. * Lap_R(r,s,i,j))
    else:
        return 0

def M13(r,s,a1,a2, i, j, k,c,mu):
    if s > 1e-10:
        return -c*(K(r,s,i,j,k) + (a1**2/6. + a2**2/10.) * Lap_K(r,s,i,j,k))
    else:
        return 0

def M22(r,s,a1,a2, i, j,c,mu):
    if abs(r[0]) + abs(r[1]) + abs(r[2]) > 1e-10:
        return c*0.5*sum([ levi(i,k,l)*D_R(r,s,k,l,j) for k in range(3) for l in range(3) if k!=l and i!=k and i!=l])
    else:
        return kronmatrix[i][j]/(8*pi*mu*a1**3)

def M23(r,s,a1,a2, i, j, k,c,mu):
    if abs(r[0]) + abs(r[1]) + abs(r[2]) > 1e-10:
        return c*-0.5*sum([levi(i,l,m) * (D_K(r,s,l,m,j,k) + (a2**2/6.)*DLap_K(r,s,l,m,j,k)) for l in range(3) for m in range(3) if l!=m and i!=l and i!=m])
    else:
        return 0

def M33(r,s,a1,a2, i, j, k, l,c,mu):
    return -0.5*c*((D_K(r,s,j,i,k,l) + D_K(r,s,i,j,k,l)) + (a1**2 + a2**2)/10. * (DLap_K(r,s,j,i,k,l) + DLap_K(r,s,i,j,k,l)))


def con_M13(r,s,a1,a2, i, m,c,mu):
    if m == 0:
        return 0.5*(s3+1)*M13(r,s,a1,a2,i,0,0,c,mu) + 0.5*(s3-1)*M13(r,s,a1,a2,i,1,1,c,mu)
    elif m == 1:
        return s2*M13(r,s,a1,a2,i,0,1,c,mu)
    elif m == 2:
        return 0.5*(s3-1)*M13(r,s,a1,a2,i,0,0,c,mu) + 0.5*(s3+1)*M13(r,s,a1,a2,i,1,1,c,mu)
    elif m == 3:
        return s2*M13(r,s,a1,a2,i,0,2,c,mu)
    else:
        return s2*M13(r,s,a1,a2,i,1,2,c,mu)

def con_M23(r,s,a1,a2, i, m,c,mu):
    if m == 0:
        return 0.5*(s3+1)*M23(r,s,a1,a2,i,0,0,c,mu) + 0.5*(s3-1)*M23(r,s,a1,a2,i,1,1,c,mu)
    elif m == 1:
        return s2*M23(r,s,a1,a2,i,0,1,c,mu)
    elif m == 2:
        return 0.5*(s3-1)*M23(r,s,a1,a2,i,0,0,c,mu) + 0.5*(s3+1)*M23(r,s,a1,a2,i,1,1,c,mu)
    elif m == 3:
        return s2*M23(r,s,a1,a2,i,0,2,c,mu)
    else:
        return s2*M23(r,s,a1,a2,i,1,2,c,mu)

def con1_M33(r,s,a1,a2, n, k, l,c,mu):
    if n == 0:
        return 0.5*(s3+1)*M33(r,s,a1,a2,0,0, k, l,c,mu) + 0.5*(s3-1)*M33(r,s,a1,a2,1,1,k, l,c,mu)
    elif n == 1:
        return s2*M33(r,s,a1,a2,0,1,k,l,c,mu)
    elif n == 2:
        return 0.5*(s3-1)*M33(r,s,a1,a2,0,0,k,l,c,mu) + 0.5*(s3+1)*M33(r,s,a1,a2,1,1,k,l,c,mu)
    elif n == 3:
        return s2*M33(r,s,a1,a2,0,2,k,l,c,mu)
    else:
        return s2*M33(r,s,a1,a2,1,2,k,l,c,mu)

def con_M33(r,s,a1,a2, n, m,c,mu):
    if s > 1e-10:
        if m == 0:
            return 0.5*(s3+1)*con1_M33(r,s,a1,a2,n,0,0,c,mu) + 0.5*(s3-1)*con1_M33(r,s,a1,a2,n,1,1,c,mu)
        elif m == 1:
            return s2*con1_M33(r,s,a1,a2,n,0,1,c,mu)
        elif m == 2:
            return 0.5*(s3-1)*con1_M33(r,s,a1,a2,n,0,0,c,mu) + 0.5*(s3+1)*con1_M33(r,s,a1,a2,n,1,1,c,mu)
        elif m == 3:
            return s2*con1_M33(r,s,a1,a2,n,0,2,c,mu)
        else:
            return s2*con1_M33(r,s,a1,a2,n,1,2,c,mu)
    else:
        return kronmatrix[n][m]/((20./3)*pi*mu*a1**3)


def generate_Minfinity(posdata, printout=0,cutoff_factor=2,frameno=0, mu=1):
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, element_sizes, element_positions, element_deltax,  num_elements, num_elements_array, element_type, uv_start, uv_size, element_start_count) = posdata_data(posdata)
    R2Bexact_sidelength = 11*num_spheres + 6*num_dumbbells
    R2Bexact = np.zeros((R2Bexact_sidelength, R2Bexact_sidelength), dtype=np.float)
    bead_positions = np.concatenate([sphere_positions,dumbbell_positions - 0.5*dumbbell_deltax, dumbbell_positions + 0.5*dumbbell_deltax])
    bead_sizes = np.concatenate([sphere_sizes, dumbbell_sizes, dumbbell_sizes])

    c = 1./(8*pi*mu)

    for a1_index,a2_index in [(u,v) for u in range(len(bead_sizes)) for v in range(u,len(bead_sizes))]:
        r = -(bead_positions[a2_index] - bead_positions[a1_index])
        a1 = bead_sizes[a1_index]
        a2 = bead_sizes[a2_index]
        s = norm(r)
        if s > 1e-8 and 2*s/(a1+a2) < 2.001:
            ss_out = 2.001*(a1+a2)/2
            r = np.array([r[0]*ss_out/s,r[1]*ss_out/s,r[2]*ss_out/s])
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

            R2Bexact[A_coords] =  [[M11(r[i],r[j],s,a1,a2,i,j,c,mu)  for j in range(3)] for i in range(3)]
            R2Bexact[Bt_coords] = [[M12(r,s,a1,a2,i,j,c,mu) for j in range(3)] for i in range(3)]
            R2Bexact[C_coords] =  [[M22(r,s,a1,a2,i,j,c,mu) for j in range(3)] for i in range(3)]
            R2Bexact[Gt_coords] = [[con_M13(r,s,a1,a2,i,j,c,mu) for j in range(5)] for i in range(3)]
            R2Bexact[Ht_coords] = [[con_M23(r,s,a1,a2,i,j,c,mu) for j in range(5)] for i in range(3)]
            R2Bexact[M_coords] = [[con_M33(r,s,a1,a2,i,j,c,mu) for j in range(5)] for i in range(5)]
            # NOTE Next line - and indeed all 12/21s stuff - is patently false if a1 != a2
            if a1 == a2:
                R2Bexact[Bt_coords_21] = -R2Bexact[Bt_coords]
                R2Bexact[Gt_coords_21] = -R2Bexact[Gt_coords]
                R2Bexact[Ht_coords_21] = R2Bexact[Ht_coords]
            else:
                R2Bexact[Bt_coords_21] = [[M12(-r,s,a2,a1,i,j,c,mu) for j in range(3)] for i in range(3)]
                R2Bexact[Gt_coords_21] = [[con_M13(-r,s,a2,a1,i,j,c,mu) for j in range(5)] for i in range(3)]
                R2Bexact[Ht_coords_21] = [[con_M23(-r,s,a2,a1,i,j,c,mu) for j in range(5)] for i in range(3)]

        elif a1_index < num_spheres and a2_index >= num_spheres and a2_index < num_spheres + num_dumbbells:
            # Sphere to dumbbell bead 1
            mr = [-r[0],-r[1],-r[2]]
            a2_index_d = a2_index-num_spheres
            R14_coords = np.s_[a1_index*3:(a1_index+1)*3,                             11*num_spheres+a2_index_d*3 : 11*num_spheres +(a2_index_d+1)*3]
            R24_coords = np.s_[3*num_spheres+a1_index*3:3*num_spheres+(a1_index+1)*3, 11*num_spheres+a2_index_d*3 : 11*num_spheres +(a2_index_d+1)*3]
            R34_coords = np.s_[6*num_spheres+a1_index*5:6*num_spheres+(a1_index+1)*5, 11*num_spheres+a2_index_d*3 : 11*num_spheres +(a2_index_d+1)*3]

            R2Bexact[R14_coords] = [[M11(r[i],r[j],s,a1,a2,i,j,c,mu)  for j in range(3)] for i in range(3)]
            R2Bexact[R24_coords] = [[M12(mr,s,a2,a1,j,i,c,mu)  for j in range(3)] for i in range(3)]
            R2Bexact[R34_coords] = [[con_M13(mr,s,a1,a2,j,i,c,mu)  for j in range(3)] for i in range(5)]

        elif a1_index < num_spheres and a2_index >= num_spheres + num_dumbbells:
            # Sphere to dumbbell bead 2
            mr = [-r[0],-r[1],-r[2]]
            a2_index_d = a2_index-num_spheres-num_dumbbells
            R15_coords = np.s_[a1_index*3:(a1_index+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3 : 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
            R25_coords = np.s_[3*num_spheres+a1_index*3:3*num_spheres+(a1_index+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3 : 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
            R35_coords = np.s_[6*num_spheres+a1_index*5:6*num_spheres+(a1_index+1)*5, 11*num_spheres+3*num_dumbbells+a2_index_d*3 : 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]

            R2Bexact[R15_coords] = [[M11(r[i],r[j],s,a1,a2,i,j,c,mu)  for j in range(3)] for i in range(3)]
            R2Bexact[R25_coords] = [[M12(mr,s,a2,a1,j,i,c,mu)  for j in range(3)] for i in range(3)]
            R2Bexact[R35_coords] = [[con_M13(mr,s,a1,a2,j,i,c,mu)  for j in range(3)] for i in range(5)]

        elif a1_index >= num_spheres and a1_index < num_spheres + num_dumbbells and a2_index >= num_spheres and a2_index < num_spheres + num_dumbbells:
            # Dumbbell bead 1 to dumbbell bead 1
            a1_index_d = a1_index-num_spheres
            a2_index_d = a2_index-num_spheres
            if bead_bead_interactions or a1_index_d == a2_index_d:
                R44_coords = np.s_[11*num_spheres+a1_index_d*3:11*num_spheres+(a1_index_d+1)*3, 11*num_spheres+a2_index_d*3 : 11*num_spheres+(a2_index_d+1)*3]
                R2Bexact[R44_coords] = [[M11(r[i],r[j],s,a1,a2,i,j,c,mu)  for j in range(3)] for i in range(3)]

        elif a1_index >= num_spheres and a1_index < num_spheres + num_dumbbells and a2_index >= num_spheres + num_dumbbells:
            if bead_bead_interactions:
                # Dumbbell bead 1 to dumbbell bead 2
                a1_index_d = a1_index-num_spheres
                a2_index_d = a2_index-num_spheres-num_dumbbells
                R45_coords = np.s_[11*num_spheres+a1_index_d*3:11*num_spheres+(a1_index_d+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3 : 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
                R2Bexact[R45_coords] = [[M11(r[i],r[j],s,a1,a2,i,j,c,mu)  for j in range(3)] for i in range(3)]

        else:
            # Dumbbell bead 2 to dumbbell bead 2
            a1_index_d = a1_index-num_spheres-num_dumbbells
            a2_index_d = a2_index-num_spheres-num_dumbbells
            if bead_bead_interactions or a1_index_d == a2_index_d:
                R55_coords = np.s_[11*num_spheres+3*num_dumbbells+a1_index_d*3:11*num_spheres+3*num_dumbbells+(a1_index_d+1)*3, 11*num_spheres+3*num_dumbbells+a2_index_d*3 : 11*num_spheres+3*num_dumbbells+(a2_index_d+1)*3]
                R2Bexact[R55_coords] = [[M11(r[i],r[j],s,a1,a2,i,j,c,mu)  for j in range(3)] for i in range(3)]

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
