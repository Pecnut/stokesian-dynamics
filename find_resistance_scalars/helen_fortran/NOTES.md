# Notes on two-sphere code

*Last updated: 26 August 2023*

The two-sphere Fortran code solves the Kim mobility problem,

    (U Ω S) = M (F T E),

where

* U contains the particle velocities,
* Ω containts the particle angular velocities,
* S contains the particle stresslet tensors,
* F contains the particle forces,
* T contains the particle torques,
* E contains the background rate of strain tensor,
* M is the Kim mobility tensor (see Kim & Karrila, sec. 7.2).

The two-sphere code uses the same method described for a three-sphere version of the problem in Wilson (2013). 

Note that that paper only considers the (U Ω) = M (F T) problem, so the extension here to S and E is not explicitly documented or tested in the paper.

I think, in the original version of the code:

1. For a given F, that S_xx, S_yy and S_zz may have the wrong sign in the output. I believe their magnitude is correct. I think S_xy, S_xz and S_yz are fine, having the correct sign and magnitude. I think, for a given E, *all* elements of S are correct, so if there is indeed a problem, it will be in the term associated with M_SF.

2. For a given E_xx, E_yy, E_zz, that U has the wrong sign in the output. I believe their magnitude is correct. I think given E_xy, E_xz and E_yz, that the sign and magnitude of U is correct. Again, I think, for a given F, *all* elements of U are correct, so if there is indeed a problem, it will be in the term associated with M_UE.

## A post-hoc fix has been implemented

**NOTE:** This behaviour has been fixed post-hoc, I believe, by introducing the variable `xg_sign = -1` in `parameters.f`. This variable is used in five lines in `2sphere.f` to flip the sign of relevant inputs/outputs. However, as I am not the author of the original paper and am not familiar with the method, I cannot say whether this is a permanent fix or whether it only fixes the cases we are interested in here. This bugfix can be removed by setting `xg_sign = 1` in `parameters.f` instead. 

## Evidence for the wrong sign of S_xx, S_yy, S_zz given F

### 1. Mobility tensors

Consider the mobility tensors x11g and y11g.

Kim & Karrila gives far-field expressions for these in table 11.25 (p. 302) table 11.27 (p. 304). In particular, consider a = b = 1 and R = 4.5. Then

    x11g = 2 * (f_5/9^5 + f_7/9^7 + f_9/9^9 + ...)
         = 2 * 0.00311915,
    
    y11g = 2 * (f_7/9^7 + f_9/9^9 + ...)
         = 2 * 0.00001833,

i.e. x11g and y11g have the same sign.

Now where do x11g and y11g appear in the mobility matrix? They appear in the g submatrix: recall that

    S = gF + hT + mE.

If T = E = 0, then in index notation,

    S_ij = g_ijk F_k.

And

    g_ijk = x11g (d_i d_j - 1/3 δ_ij) d_k
            + y11g (d_i δ_jk + d_j δ_ik - 2 d_i d_j d_k)

Let the two spheres be separated along the x-axis. Consider the case we'll call "F11":

    sphere 1: F = (1, 0, 0)  (i.e. force towards sphere 2)
    sphere 2: F = (0, 0, 0).

Then on sphere 1,

    S_ij = g_ij1
               ⎡ 2x11g    0     0  ⎤
         = 1/3 ⎢   0   -x11g    0  ⎥ .
               ⎣   0      0  -x11g ⎦

Meanwhile, also consider the case we'll call "F12":

    sphere 1: F = (0, 1, 0)
    sphere 2: F = (0, 0, 0), 

then on sphere 1,

    S_ij = g_ij2
            ⎡   0  y11g  0 ⎤
         =  ⎢ y11g   0   0 ⎥ .
            ⎣   0    0   0 ⎦

If you ask this code these two problems (which you can run by supplying "F11" and "F12" as arguments when running the code), you get the following signs in the output:

    For F11: S on sphere 1 = diag(-, +, +),
    For F12: S on sphere 1 = [[0 + 0],[+ 0 0],[0 0 0]],

i.e. x11g < 0 and y11g > 0. In particular, you get x11g with the same magnitude but the opposite sign to that given by the K&K far-field expression above. You get a y11g which matches in sign and magnitude with K&K.

### 2. S given F

So given F11, what should S be? Consider the problem as just two points and a single point force.

This is actually harder to figure out, even though surely it is a basic question which really just asks about the definition of a stresslet. I can't find an explicit answer anywhere. I am not sure I have this right - I would really like someone to confirm this - but I think it might go like:

    S_ij = 1/2 (x_j σ_ik n_k + x_i σ_jk n_k) - 1/3 (x_k σ_kl n_l) δ_ij
         = 1/2 (x_j f_i + x_i f_j) - 1/3 x_k f_k δ_ij,

where x is the separation distance between two point forces.

If x = (1,0,0) and f = (1,0,0) then

               ⎡ 2  0  0 ⎤
      S  = 1/3 ⎢ 0 -1  0 ⎥ .
               ⎣ 0  0 -1 ⎦

And if x = (1,0,0) and f = (0,1,0) then

               ⎡ 0  1  0 ⎤
      S  = 1/2 ⎢ 1  0  0 ⎥ .
               ⎣ 0  0  0 ⎦

I think there is a constant missing at the beginning of the expression for S_ij, and I'm not sure of the sign of that constant. 

BUT this does at least appear to back up the fact that x11g and y11g should have the same sign.

## Evidence for the wrong sign of U given E_xx, E_yy, E_zz

A priori, it is not immediately clear if E in the code represents E or -E, but either way, it should be consistent between the diagonal and off-diagonal terms of E.

### 1. E1 on both particles

Given "E1" on both particles, i.e. F = T = 0 and 

    E = diag(1, -1/2, -1/2),

the code says the sign of the velocities and angular velocites should be

    U of particle 1 = (+, 0, 0)
    U of particle 2 = (-, 0, 0)
    Ω of particle 1 = (0, 0, 0)
    Ω of particle 2 = (0, 0, 0).

E.x looks like

    E.x = (x, -y/2, -z/2)

      ↓
    ← · →
      ↑

which *disagrees* with the U given as the answer. This could be OK if E actually meant –E, but in that case we would expect sign disagreement on the next case as well...

### 2. E4 on both particles

Given "E4" on both particles, i.e. F = T = 0 and 

            ⎡ 0  1  0 ⎤
      E  =  ⎢ 1  0  0 ⎥ ,
            ⎣ 0  0  0 ⎦

the code says the sign of the velocities and angular velocities should be

    U of particle 1 = (0, -, 0)
    U of particle 2 = (0, +, 0)
    Ω of particle 1 = (0, 0, -)
    Ω of particle 2 = (0, 0, -)

E.x looks like

    E.x = (y, x, 0)

      →
    ↓ · ↑
      ←    

which *agrees* with both the U and Ω given as the answer. So there is inconsistency between inputs involving E_xx, E_yy and E_zz, and inputs involving E_xy, E_yz and E_xz.

Coupled with the analysis about the wrong sign of S given F, we can be pretty confident that there is a sign error on _xx, _yy and _zz terms.

## Where an error might lie

A problem for another day, but here might be a starting point.

### Checking the U output given E

I think this might be the easier route to go down. Given the symmetry of any error it has to be something 'shared' between E and S?

### Checking the S output given F

Stresslets are calculated using SSt() in the code. Element-wise, for particle i:
    
* SSt(1,i, 0) → xx and yy
* SSt(1,i,±2) → yy
* SSt(1,i,±1) → xy
* SSt(2,i,±1) → xz
* SSt(2,i,±2) → yz.

Given the wrong sign on xx, yy and zz, maybe looking at specific terms where the first argument of SSt is 1, and the last argument is even might be a start.

The evidence points towards a mistake in code relating to the M_SF term, but without diving into the method I'm not sure exactly where that is.

There might be something going on with the H term which is in the definition of SSt(). It is supposed to be the h in (6) in the paper, but where there is a square root over the factorials in the paper, there isn't in the code. Adding the factorial in the code produces values of S which I believe have the wrong magnitude, so I suspect this is a typo in the paper? Not sure.

## Red herrings

There can be confusion in these problems, and the literature has evolved over time, as to whether F, T, S represent actions on the particle by the fluid, or on the fluid by the particle. Even E can sometimes mean -E. But I think this is a red herring: any sign error would be consistent across all elements of S.

The sign of the H term makes no difference to the output. But there is something

## References
* Kim & Karrila, 2005. *Microhydrodynamics: Principles and selected applications.* Dover Publications, Mineola, NY, USA.
* [Wilson, 2013](http://www.ucl.ac.uk/~ucahhwi/publist/papers/2013-W.pdf). Stokes flow past three spheres, *Journal of Computational Physics* **245**, 302–316.
