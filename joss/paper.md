---
title: 'Stokesian Dynamics in Python'
tags:
  - Python
  - Stokesian Dynamics
  - low Reynolds number flow
  - suspension mechanics
  - fluid dynamics
  - non-Newtonian fluids
authors:
  - name: Adam K. Townsend
    orcid: 0000-0003-1700-2873
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Mathematical Sciences, Durham University, Upper Mountjoy, Stockton Road, Durham DH1 3LE, United Kingdom
   index: 1
date: 6 September 2023
bibliography: paper.bib
---

# Summary

Stokesian Dynamics [@Brady:1988] is a microhydrodynamic, [low Reynolds number approach](https://en.wikipedia.org/wiki/Stokes_flow#Stokes_equations) to modelling the movement of suspensions of particles in fluids which considers the interaction of particles with each other against a Newtonian background solvent. It is typically chosen for its suitability for three-dimensional simulation with low calculation and time penalty.

In the most basic case, Stokes’ law states that a single sphere of radius $a$, travelling with a velocity $U$ in an unbounded Newtonian fluid of viscosity $\mu$, in a low Reynolds number regime, experiences a drag force, $F$, of $F=-6 \pi \mu a U$.

Stokesian Dynamics, at its heart, is an extension of this linear relationship between the force acting on a particle and the velocity at which it travels. As a method, it is adaptable and continues to be used in the field, providing some interesting insight into the behaviour of particle suspensions. Validations with experiments have shown it to provide results within acceptable error.

The Stokesian Dynamics recipe can be summarised as follows:
* Compute long-range interactions between particles. This is done by using a truncated expansion of the boundary integral equation for unbounded Stokes flow (@Ladyzhenskaya:1964, @Durlofsky:1987). For periodic domains, this is done by using the Ewald summation method (@Ewald:1921, @Brady:1988).
* Compute short-range lubrication between particles. This is done by interpolating pre-computed results on a pairwise basis (@Jeffrey:1984, @Jeffrey:1992, @Kim:2005, @Wilson:2013, @Townsend:2023).
* Combining the two.

The method is explained and described in detail in chapter 2 of @Townsend:2017.



# Statement of need

This is a Python 3 implementation of the Stokesian Dynamics method for polydisperse spheres suspended in a 3D Newtonian background fluid. The fluid-filled domain may be unbounded or a periodic cube. Physical setups and custom forces on the particles are easily implemented using Python; meanwhile, the computational speed is handled by Numba.

This software is aimed at researchers in low Reynolds number fluid dynamics who are looking for an easy-to-use yet flexible implementation of this popular method. The startup cost of writing a Stokesian Dynamics code is high, given the need for pre-computed resistance scalars. There are also a number of minus sign errors in the literature which need resolving first. These have been resolved and validated before being implemented in this code. The hope is that many months of future researchers' time (often PhD students' time) will be saved by no longer reinventing the wheel.



# Description of the software

This software allows you to place spherical particles in a fluid, apply some forces to them, and see how they move. You can also move the fluid in some way and see how the particles react to that. [A lovely Javascript 2D implementation by the Swan group at MIT](http://web.mit.edu/swangroup/sd-game.shtml) shows how it's done.

In particular, this software has the following features:

* Fully 3D simulation
* Free choice of number of particles
* Choice of simulating particles in an unbounded fluid ('non-periodic') or in a periodic domain
* Free choice of spherical particles sizes (up to $n$ different sizes for low $n$)
* Choice to include bead-and-spring dumbbells, formed of pairs of (usually smaller) spherical particles, as a way to introduce viscoelasticity into the background fluid
* Free choice of particle interaction forces
* Choice of whether to include long-range hydrodynamic forces or not ($\mathcal{M}^\infty$)
* Choice of whether to include lubrication forces or not ($\mathcal{R}^{\text{2B,exact}}$)
* Choice of Euler, two-step Adams–Bashforth or RK4 timestepping
* Choice of how often to find $(\mathcal{M}^\infty)^{-1}$, a matrix which varies slowly and takes a long time to compute, hence is typically computed every 10 timesteps.
* For each spherical, non-bead-and-spring particle in the system, the software takes the forces and torques you want to apply to it, as well as the background fluid shear rate ($\mathbfit{F}$, $\mathbfit{T}$ and $\mathbfsfit{E}$), and gives you the particle velocities, angular velocities and stresslets ($\mathbfit{U}$, $\mathbfit{\Omega}$ and $\mathbfsfit{S}$). It can also do $\mathbfit{FT}\mathbfsfit{S}$ to $\mathbfit{U\Omega}\mathbfsfit{E}$, for when you want to specify stresslets instead of the background shear rate; and $\mathbfit{U}_1\mathbfit{F}_2\mathbfit{T}\mathbfsfit{E}$ to $\mathbfit{F}_1\mathbfit{U}_2\mathbfit{\Omega}\mathbfsfit{S}$, for when there are some particles whose velocities you want to fix while letting some other particles move under specified forces (e.g. when you want to fix some particles as walls). For the beads in the bead-and-spring dumbbells, which are normally much smaller than the other particles in the system, the only option is to apply the force and extract the velocity ($\mathbfit{F}$ to $\mathbfit{U}$).
* Time-to-completion estimates
* Emails you on completion
* Output stored in convenient `.npz` format
* Video generation scripts to watch the particle behaviour



# Related software

* [Hydrolib](https://doi.org/10.1016/0010-4655(95)00029-F) is a Fortran 77 implementation of Stokesian Dynamics from 1995. It solves the resistance and mobility problem in $\mathbfit{F}$ or $\mathbfit{FT}$ form for monodisperse particles in an infinite or cubically-periodic domain. It can additionally handle fixed particles and rigid body configurations. It continues to receive some citations in recent literature but lacks the flexibility or onboarding of more modern implementations.
* [libstokes](https://github.com/kichiki/libstokes) is a C implementation of Stokesian Dynamics from 1999, last updated 2013. It contains functions which solve the resistance and mobility problem in $\mathbfit{F}$, $\mathbfit{FT}$ or $\mathbfit{FT}\mathbfsfit{S}$ form in a periodic domain. These functions can additionally handle fixed particles. The project was important in cataloguing some errors in the papers related to lubrication, which in turn helped in the development of @Townsend:2023. The code does not appear to be actively maintained and lacks documentation.
* [StokesDT](https://github.com/xing-liu/stokesdt) is a C++, parallel toolkit for Stokes flow problems from 2014. The documentation is very sparse, but it appears to solve the resistance and mobility problem in $\mathbfit{F}$ form for monodisperse particles in an infinite or periodic domain. It does not appear to include lubrication, preferring instead to use the Rotne--Prager--Yamakawa mobility approximation. The code does not appear to be actively maintained and documentation remains incomplete.
* [PyStokes](https://github.com/rajeshrinet/pystokes) is a Python (with Cython) library for Stokesian Dynamics from 2014, last updated 2023. It solves the resistance and mobility problem in $\mathbfit{F}$, $\mathbfit{FT}$ or $\mathbfit{FT}\mathbfsfit{S}$ form in an unbounded or periodic domain, as well as a number of more interesting domains such as flow near a wall. It additionally handles autophoresis through solution of Laplace's equation. It does not appear to include lubrication.



# Published work which has used this software

* @Townsend:2017b
* @Townsend:2017c
* @Townsend:2018a
* @Townsend:2018b
* @Townsend:2017
* @Townsend:2023



# References