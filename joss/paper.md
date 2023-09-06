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
* For each spherical, non-bead-and-spring particle in the system, the software takes the forces and torques you want to apply to it, as well as the background fluid shear rate ($\mathbfit{F}$, $\mathbfit{T}$ and $\mathbfsfit{E}$), and gives you the particle velocities, angular velocities and stresslets ($\mathbfit{U}$, $\mathbfit{\Omega}$ and $\mathbfsfit{S}$). It can also do $\boldsymbol{FT}\mathbfsfit{S}$ to $\mathbfit{U\Omega}\mathbfsfit{E}$, for when you want to specify stresslets instead of the background shear rate; and $\mathbfit{U}_1\mathbfit{F}_2\mathbfit{T}\mathbfsfit{E}$ to $\mathbfit{F}_1\mathbfit{U}_2\mathbfit{\Omega}\mathbfsfit{S}$, for when there are some particles whose velocities you want to fix while letting some other particles move under specified forces (e.g. when you want to fix some particles as walls). For the beads in the bead-and-spring dumbbells, which are normally much smaller than the other particles in the system, the only option is to apply the force and extract the velocity ($\mathbfit{F}$ to $\mathbfit{U}$).
* Time-to-completion estimates
* Emails you on completion
* Output stored in convenient `.npz` format
* Video generation scripts to watch the particle behaviour


# Published work which has used this software

* @Townsend:2017b
* @Townsend:2017c
* @Townsend:2018a
* @Townsend:2018b
* @Townsend:2017


# References