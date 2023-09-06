#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 17/10/2014

import numpy as np
from functions_shared import posdata_data, contraction


def empty_vectors(posdata):
    """Initialise all vectors in the left and right-hand sides.

    The string Pippa (my dog) serves as a placeholder which will be overwritten
    by input functions. If you spot her in an error message, she is serving as 
    a warning that you have not defined enough inputs.

    Args:
        posdata: Position, size and count data for the particles.

    Returns:
        An array of placeholder arrays to fit the sizes of
        [Fa, Ta, Sa, Sa condensed, Fb, DFb,
         Ua, Oa, Ea, Ea condensed, Ub, HalfDUb].
    """
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes, 
     dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, 
     element_sizes, element_positions, element_deltax,  num_elements, 
     num_elements_array, element_type, uv_start, uv_size, 
     element_start_count) = posdata_data(posdata)
    return (
        [['pippa' for _ in range(3)] for _ in range(num_spheres)],
        [['pippa' for _ in range(3)] for _ in range(num_spheres)],
        [
            [['pippa' for _ in range(3)] for _ in range(3)]
            for _ in range(num_spheres)
        ],
        [['pippa' for _ in range(5)] for _ in range(num_spheres)],
        [['pippa' for _ in range(3)] for _ in range(num_dumbbells)],
        [['pippa' for _ in range(3)] for _ in range(num_dumbbells)],
        [['pippa' for _ in range(3)] for _ in range(num_spheres)],
        [['pippa' for _ in range(3)] for _ in range(num_spheres)],
        [
            [['pippa' for _ in range(3)] for _ in range(3)]
            for _ in range(num_spheres)
        ],
        [['pippa' for _ in range(5)] for _ in range(num_spheres)],
        [['pippa' for _ in range(3)] for _ in range(num_dumbbells)],
        [['pippa' for _ in range(3)] for _ in range(num_dumbbells)],
    )


def condense(s_spheres, num_spheres):
    '''Condense 3x3 matrix to 5-element vector for each sphere.'''
    s_spheres_condensed = [
        ['pippa' for _ in range(5)] for _ in range(num_spheres)
    ]
    for i in range(num_spheres):
        for j in range(5):
            s_spheres_condensed[i][j] = sum(
                sum(
                    contraction(j, k, l) * s_spheres[i][k][l] for k in range(3)
                )
                for l in range(3)
            )
    return s_spheres_condensed


def uncondense(e_spheres_condensed, num_spheres):
    '''Uncondense 5-element vector back to 3x3 matrix for each sphere.

    Undoes `condense` function.'''
    e_spheres = np.zeros([num_spheres, 3, 3])
    for i in range(num_spheres):
        e_spheres[i, 0, 0] = ((np.sqrt(3)+3)/6. * e_spheres_condensed[i, 0] 
                              + (np.sqrt(3)-3)/6. * e_spheres_condensed[i, 2])
        e_spheres[i, 0, 1] = e_spheres_condensed[i, 1]/np.sqrt(2)
        e_spheres[i, 0, 2] = e_spheres_condensed[i, 3]/np.sqrt(2)
        e_spheres[i, 1, 0] = e_spheres[i, 0, 1]
        e_spheres[i, 1, 1] = ((np.sqrt(3)-3)/6. * e_spheres_condensed[i, 0] 
                              + (np.sqrt(3)+3)/6. * e_spheres_condensed[i, 2])
        e_spheres[i, 1, 2] = e_spheres_condensed[i, 4]/np.sqrt(2)
        e_spheres[i, 2, 0] = e_spheres[i, 0, 2]
        e_spheres[i, 2, 1] = e_spheres[i, 1, 2]
        e_spheres[i, 2, 2] = -e_spheres[i, 1, 1] - e_spheres[i, 0, 0]
    return e_spheres


def construct_force_vector_from_fts(posdata, f_spheres, t_spheres, s_spheres,
                                    f_dumbbells, deltaf_dumbbells):
    """Construct a single 'grand' force vector from 
    'Fa,Ta,Sa condensed,Fb,DFb'.

    Condenses `s_spheres` and then returns the concatenated 
    [f_spheres, t_spheres, s_spheres_dumbbells, f_dumbbells, deltaf_dumbbells]. 
    """
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes, 
     dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, 
     element_sizes, element_positions, element_deltax,  num_elements, 
     num_elements_array, element_type, uv_start, uv_size, 
     element_start_count) = posdata_data(posdata)
    s_spheres_condensed = condense(s_spheres, num_spheres)
    if num_spheres == 0 and num_dumbbells == 0:
        force_vector = np.array([])
    if num_spheres > 0 and num_dumbbells == 0:
        # Converts from numpy array (possibly) to list
        fs = [item for sublist in f_spheres for item in sublist]
        ts = [item for sublist in t_spheres for item in sublist]
        ss = [item for sublist in s_spheres_condensed for item in sublist]
        force_vector = fs + ts + ss
    if num_spheres == 0 and num_dumbbells > 0:
        force_vector = list(np.array([f_dumbbells, 
                                      deltaf_dumbbells]).flatten())
    if num_spheres > 0 and num_dumbbells > 0:
        fs = [item for sublist in f_spheres for item in sublist]
        ts = [item for sublist in t_spheres for item in sublist]
        ss = [item for sublist in s_spheres_condensed for item in sublist]
        fd = [item for sublist in f_dumbbells for item in sublist]
        dfd = [item for sublist in deltaf_dumbbells for item in sublist]
        force_vector = fs + ts + ss + fd + dfd
    return force_vector


def deconstruct_velocity_vector_for_fts(posdata, velocity_vector):
    """Take 'grand' velocity vector and deconstruct into 
    'Ua,Oa,Ea,Ub,HalfDUb'."""
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
     element_sizes, element_positions, element_deltax,  num_elements,
     num_elements_array, element_type, uv_start, uv_size, 
     element_start_count) = posdata_data(posdata)
    N1 = 3*num_spheres
    N2 = 6*num_spheres
    N3 = 11*num_spheres
    N4 = 11*num_spheres + 3*num_dumbbells
    N5 = 11*num_spheres + 6*num_dumbbells
    u_spheres = velocity_vector[:N1].reshape(num_spheres, 3)
    o_spheres = velocity_vector[N1:N2].reshape(num_spheres, 3)
    e_spheres_condensed = velocity_vector[N2:N3].reshape(num_spheres, 5)
    u_dumbbells = velocity_vector[N3:N4].reshape(num_dumbbells, 3)
    half_deltau_dumbbells = velocity_vector[N4:N5].reshape(num_dumbbells, 3)
    e_spheres = uncondense(e_spheres_condensed, num_spheres)
    return (u_spheres, o_spheres, e_spheres, u_dumbbells, 
            half_deltau_dumbbells)


def vecmat_mat_vecmat(A, B, C, matrix_size, items_in_vector, starts):
    '''{Vector of matrices} times {matrix} times {vector of matrices}.

    For vector of matrices A[i] and vector of matrices C[j], returns
        D[i,j] = (A[i] . B) . C[j].

    In context of Townsend (2017) PhD thesis, eq (A.22), this corresponds to
        [[ g~ t g,  g~ t h,  g~ t m,  g~ t m34,  g~ t m35], ...].
        Note that since t = m^-1,
            g~ t m = g~
        which means that M - {this matrix} is now zero for this element of D.
        Hence we add vec_mat_cross afterwards.
    '''
    D = np.zeros([matrix_size, matrix_size])
    for Ai in range(items_in_vector):
        for Ci in range(items_in_vector):
            D[starts[Ai]:starts[Ai+1], starts[Ci]:starts[Ci+1]] = np.dot(
                np.dot(A[Ai], B), C[Ci])
    return D


def vec_mat_cross(A, B, C, matrix_size, items_in_vector, starts, crosspoint):
    '''{Vector of matrices} times matrix; and matrix times {vector of matrices}
    in a cross.

    For vectors of matrices A[i] and C[i], returns
        D[i,crosspoint] = B.C[i]    (horizontal of the cross)
        D[crosspoint,i] = -A[i].B   (vertical of the cross)
        D[crosspoint,crosspoint] = -B

    In context of Townsend (2017) PhD thesis, eq (A.22), this corresponds to
        D = [[ 0,  0, -gtilde t,     0,     0], 
             [ 0,  0, -htilde t,     0,     0]
             [tg, th,        -t, t m34, t m35], 
             [ 0,  0,    -m43 t,     0,     0], 
             [ 0,  0,    -m53 t,     0,     0]].

    '''
    D = np.zeros([matrix_size, matrix_size])
    for Ai in range(items_in_vector):
        for Ci in range(items_in_vector):
            if Ai == crosspoint and Ci != crosspoint:  
                # it's on the horizontal cross but not the meeting point
                D[starts[Ai]:starts[Ai+1], 
                  starts[Ci]:starts[Ci+1]] = np.dot(B, C[Ci])
            elif Ai != crosspoint and Ci == crosspoint:  
                # it's on the vertical cross but not the meeting point
                D[starts[Ai]:starts[Ai+1], 
                  starts[Ci]:starts[Ci+1]] = -np.dot(A[Ai], B)
            elif Ai == crosspoint:  # and Ci == crosspoint:
                D[starts[Ai]:starts[Ai+1], 
                  starts[Ci]:starts[Ci+1]] = -B
    return D


def fts_to_fte_matrix(posdata, grand_mobility_matrix):
    '''Convert mobility matrix expecting to multiply a grand force vector of 
    FTS form to one expecting FTE form.

    See Townsend (2017) PhD thesis, section A.1.3 for details.
    '''
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes, 
     dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells, 
     element_sizes, element_positions, element_deltax,  num_elements, 
     num_elements_array, element_type, uv_start, uv_size, 
     element_start_count) = posdata_data(posdata)
    M = grand_mobility_matrix
    C0 = 0
    C1 = 3*num_spheres
    C2 = 6*num_spheres
    C3 = 11*num_spheres
    C4 = 11*num_spheres + 3*num_dumbbells
    C5 = 11*num_spheres + 6*num_dumbbells

    gt = M[C0:C1, C2:C3]
    g = gt.transpose()
    ht = M[C1:C2, C2:C3]
    h = ht.transpose()
    m = M[C2:C3, C2:C3]
    m34 = M[C2:C3, C3:C4]
    m43 = m34.transpose()
    m35 = M[C2:C3, C4:C5]
    m53 = m35.transpose()

    # Not possible to have dumbbells only because this is an FTS to FTE 
    # conversion, and E is only there for spheres.
    items_in_vector = 5 if num_spheres > 0 and num_dumbbells > 0 else 3
    starts = [C0, C1, C2, C3, C4, C5]

    m_inv = np.linalg.inv(m)
    crosspoint = 2  # corresponds to the ghm row/col
    if num_spheres > 0:
        if num_dumbbells > 0:
            vec1 = [gt, ht, m, m43, m53]
            vec2 = [g, h, m, m34, m35]
        if num_dumbbells == 0:
            vec1 = [gt, ht, m]
            vec2 = [g, h, m]

    return (
        M
        - vecmat_mat_vecmat(vec1, m_inv, vec2, C5, items_in_vector, starts)
        - vec_mat_cross(
            vec1, m_inv, vec2, C5, items_in_vector, starts, crosspoint
        )
    )


def fte_to_ufte_matrix(num_fixed_velocity_spheres, posdata, 
                       grand_mobility_matrix_fte):
    '''Convert mobility matrix expecting to multiply a grand force vector of
    FTE form to one expecting UFTE form.

    Used for when some spheres are given prescribed velocities, rather than
    forces.
    '''

    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
     element_sizes, element_positions, element_deltax,  num_elements,
     num_elements_array, element_type, uv_start, uv_size,
     element_start_count) = posdata_data(posdata)
    M = grand_mobility_matrix_fte
    C0 = 0
    C05 = 3*num_fixed_velocity_spheres
    C1 = 3*num_spheres
    C2 = 6*num_spheres
    C3 = 11*num_spheres
    C4 = 11*num_spheres + 3*num_dumbbells
    C5 = 11*num_spheres + 6*num_dumbbells

    # a has been split up into x (fixed velocity particles) and r (free
    # velocity particles).
    # M = [axx axr btx gtx m14x m15x]
    #     [arx arr btr gtr m14r m15r]
    #     [bx  br  c   ht  m24  m25 ]
    #     [gx  gr  h   m   m34  m35

    axx = M[C0:C05, C0:C05]
    axr = M[C0:C05, C05:C1]
    arx = M[C05:C1, C0:C05]
    btx = M[C0:C05, C1:C2]
    bx = btx.transpose()
    gtx = M[C0:C05, C2:C3]
    gx = -gtx.transpose()  # note MFTE has slightly different sign symmetries
    m14x = M[C0:C05, C3:C4]
    m15x = M[C0:C05, C4:C5]
    m51x = m15x.transpose()  # these don't need a minus sign.
    m41x = m14x.transpose()

    # Not possible to have dumbbells only because this is an FTS to FTE
    # conversion, and E is only there for spheres.
    items_in_vector = 5+1 if num_spheres > 0 and num_dumbbells > 0 else 3+1
    starts = [C0, C05, C1, C2, C3, C4, C5]

    axx_inv = np.linalg.inv(axx)
    gx = np.asarray(gx)

    crosspoint = 0  # corresponds to the ghm row/col
    if num_spheres > 0:
        if num_dumbbells > 0:
            vec1 = [axx, arx, bx, gx, m41x, m51x]
            vec2 = [axx, axr, btx, gtx, m14x, m15x]
        if num_dumbbells == 0:
            vec1 = [axx, arx, bx, gx]
            vec2 = [axx, axr, btx, gtx]

    return (
        M
        - vecmat_mat_vecmat(vec1, axx_inv, vec2, C5, items_in_vector, starts)
        - vec_mat_cross(
            vec1, axx_inv, vec2, C5, items_in_vector, starts, crosspoint
        )
    )


def ufte_to_ufteu_matrix(num_fixed_velocity_dumbbells,
                         num_fixed_velocity_spheres, posdata, 
                         grand_mobility_matrix_ufte):
    '''Convert mobility matrix expecting to multiply a grand force vector of 
    UFTE form to one expecting UFTEU form.

    Used for when some spheres and dumbbells are given prescribed velocities, 
    rather than forces.
    '''
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
     element_sizes, element_positions, element_deltax,  num_elements,
     num_elements_array, element_type, uv_start, uv_size, 
     element_start_count) = posdata_data(posdata)
    M = grand_mobility_matrix_ufte
    C0 = 0
    C05 = 3*num_fixed_velocity_spheres
    C1 = 3*num_spheres
    C2 = 6*num_spheres
    C3 = 11*num_spheres
    C35 = 11*num_spheres + 3*num_fixed_velocity_dumbbells
    C4 = 11*num_spheres + 3*num_dumbbells
    C45 = 11*num_spheres + 3*num_dumbbells + 3*num_fixed_velocity_dumbbells
    C5 = 11*num_spheres + 6*num_dumbbells

    starts = [C0, C05, C1, C2, C3, C35, C4, C45, C5]

    # now m44 and m55 have been split up into x (fixed velocity particles) and
    # r (free velocity particles).
    #
    # M = [axx   axr   btx  gtx  m14xx m14xr m15xx m15xr]
    #     [arx   arr   btr  gtr  m14rx m14rr m15rx m15rr]
    #     [bx    br    c    ht   m24x  m24r  m25x  m25r ]
    #     [gx    gr    h    m    m34x  m34r  m35x  m35r ]
    #     [m41xx m41rx m42x m43x m44xx m44xr m45xx m45xr]
    #     etc

    # Step 1. Swap the m44xx row and column

    m41xx = M[C3:C35, C0:C05]
    m41rx = M[C3:C35, C05:C1]
    m42x = M[C3:C35, C1:C2]
    m43x = M[C3:C35, C2:C3]
    m44xx = M[C3:C35, C3:C35]
    m44xr = M[C3:C35, C35:C4]
    m45xx = M[C3:C35, C4:C45]
    m45xr = M[C3:C35, C45:C5]
    m14xx = -m41xx.transpose()
    m14rx = m41rx.transpose()
    m24x = m42x.transpose()
    m34x = -m43x.transpose()
    m44rx = m44xr.transpose()
    m54xx = m45xx.transpose()
    m54rx = m45xr.transpose()

    # Currently I haven't put in dumbbells only because there's no point I
    # think. Since this comes from an FTE conversion, and E is only for
    # spheres, that would need dealing with first.
    # If there are no dumbbells then doing this is pointless, so I am assuming
    # there are dumbbells.
    items_in_vector = 7+1

    m44xx_inv = np.linalg.inv(m44xx)

    crosspoint = 4  # corresponds to the m44xx row/col
    vec1 = [m14xx, m14rx, m24x, m34x, m44xx, m44rx, m54xx, m54rx]  # down
    vec2 = [m41xx, m41rx, m42x, m43x, m44xx, m44xr, m45xx, m45xr]  # across

    MUFTEU1 = (M 
               - vecmat_mat_vecmat(vec1, m44xx_inv, vec2, C5, items_in_vector,
                                   starts) 
               - vec_mat_cross(vec1, m44xx_inv, vec2, C5, items_in_vector,
                               starts, crosspoint))

    # Step 2. Swap the m55xx row and column

    M = MUFTEU1

    m51xx = M[C4:C45, C0:C05]
    m51rx = M[C4:C45, C05:C1]
    m52x = M[C4:C45, C1:C2]
    m53x = M[C4:C45, C2:C3]
    m54xx = M[C4:C45, C3:C35]
    m54xr = M[C4:C45, C35:C4]
    m55xx = M[C4:C45, C4:C45]
    m55xr = M[C4:C45, C45:C5]
    m15xx = -m51xx.transpose()
    m15rx = m51rx.transpose()
    m25x = m52x.transpose()
    m35x = -m53x.transpose()
    m45xx = -m54xx.transpose()
    m45rx = m54xr.transpose()
    m55rx = m55xr.transpose()

    m55xx_inv = np.linalg.inv(m55xx)

    crosspoint = 6  # corresponds to the m55xx row/col
    vec1 = [m15xx, m15rx, m25x, m35x, m45xx, m45rx, m55xx, m55rx]  # down
    vec2 = [m51xx, m51rx, m52x, m53x, m54xx, m54xr, m55xx, m55xr]  # across

    return (
        M
        - vecmat_mat_vecmat(vec1, m55xx_inv, vec2, C5, items_in_vector, starts)
        - vec_mat_cross(
            vec1, m55xx_inv, vec2, C5, items_in_vector, starts, crosspoint
        )
    )


def fts_to_duf_matrix(num_fixed_velocity_dumbbells, posdata,
                      grand_mobility_matrix_fts):
    '''Convert mobility matrix expecting to multiply a grand force vector of 
    FTS form to one expecting DUF form.

    Used for dumbbell-only sims where some dumbbells are given a prescribed 
    velocity, rather than a force.
    '''
    (sphere_sizes, sphere_positions, sphere_rotations,  dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax, num_spheres, num_dumbbells,
     element_sizes, element_positions, element_deltax,  num_elements,
     num_elements_array, element_type, uv_start, uv_size, 
     element_start_count) = posdata_data(posdata)
    M = grand_mobility_matrix_fts
    C3 = 0
    C35 = 3*num_fixed_velocity_dumbbells
    C4 = 3*num_dumbbells
    C45 = 3*num_dumbbells + 3*num_fixed_velocity_dumbbells
    C5 = 6*num_dumbbells

    starts = [C3, C35, C4, C45, C5]

    # now m44 and m55 have been split up into x (fixed velocity particles) and
    # r (free velocity particles).
    #
    # M = [m44xx m44xr m45xx m45xr]
    #     [m44rx m44rr m45rx m45rr]
    #     etc

    # Step 1. Swap the m44xx row and column

    m44xx = M[C3:C35, C3:C35]
    m44xr = M[C3:C35, C35:C4]
    m45xx = M[C3:C35, C4:C45]
    m45xr = M[C3:C35, C45:C5]
    m44rx = m44xr.transpose()
    m54xx = m45xx.transpose()
    m54rx = m45xr.transpose()

    # Currently I haven't put in dumbbells only because there's no point I
    # think. Since this comes from an FTE conversion, and E is only for
    # spheres, that would need dealing with first.
    # If there are no dumbbells then doing this is pointless, so I am assuming
    # there are dumbbells.
    items_in_vector = 4

    m44xx_inv = np.linalg.inv(m44xx)

    crosspoint = 0  # corresponds to the m44xx row/col
    vec1 = [m44xx, m44rx, m54xx, m54rx]  # down
    vec2 = [m44xx, m44xr, m45xx, m45xr]  # across

    MDUF1 = (M 
             - vecmat_mat_vecmat(vec1, m44xx_inv, vec2, C5, items_in_vector,
                                 starts) 
             - vec_mat_cross(vec1, m44xx_inv, vec2, C5, items_in_vector,
                             starts, crosspoint))

    # Step 2. Swap the m55xx row and column

    M = MDUF1

    m54xx = M[C4:C45, C3:C35]
    m54xr = M[C4:C45, C35:C4]
    m55xx = M[C4:C45, C4:C45]
    m55xr = M[C4:C45, C45:C5]
    m45xx = -m54xx.transpose()
    m45rx = m54xr.transpose()
    m55rx = m55xr.transpose()

    m55xx_inv = np.linalg.inv(m55xx)

    crosspoint = 2  # corresponds to the m55xx row/col
    vec1 = [m45xx, m45rx, m55xx, m55rx]  # down
    vec2 = [m54xx, m54xr, m55xx, m55xr]  # across

    return (
        M
        - vecmat_mat_vecmat(vec1, m55xx_inv, vec2, C5, items_in_vector, starts)
        - vec_mat_cross(
            vec1, m55xx_inv, vec2, C5, items_in_vector, starts, crosspoint
        )
    )
