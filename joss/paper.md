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
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Mathematics, University College London, Gower Street, London WC1E 6BT, United Kingdom
   index: 1
 - name: Department of Mathematical Sciences, Durham University, Upper Mountjoy, Stockton Road, Durham DH1 3LE, United Kingdom
   index: 2
date: 16 June 2023
bibliography: paper.bib
---

# Summary

Stokesian Dynamics [@Brady:1988] is a microhydrodynamic, [low Reynolds number approach](https://en.wikipedia.org/wiki/Stokes_flow#Stokes_equations) to modelling the movement of suspensions of particles in fluids which considers the interaction of particles with each other against a Newtonian background solvent. It is typically chosen for its suitability for three-dimensional simulation with low calculation and time penalty.

In the most basic case, Stokes’ law states that a single sphere of radius $a$, travelling with a velocity $U$ in an unbounded Newtonian fluid of viscosity $\mu$, in a low Reynolds number regime, experiences a drag force, $F$, of $F=-6 \pi \mu a U$.

Stokesian Dynamics, at its heart, is an extension of this linear relationship between the force acting on a particle and the velocity at which it travels. As a method, it is adaptable and continues to be used in the field, providing some interesting insight into the behaviour of particle suspensions. Validations with experiments have shown it to provide results within acceptable error.

The method is explained and described in detail in chapter 2 of @Townsend:2017.



# Statement of need



what problems the software is designed to solve, who the target audience is, and its relation to other work?

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

[STATE OF THE FIELD]



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
* For each spherical, non-bead-and-spring particle in the system, the software takes the forces and torques you want to apply to it, as well as the background fluid shear rate ($\boldsymbol{F}$, $\boldsymbol{T}$ and $\mathbfsfit{E}$), and gives you the particle velocities, angular velocities and stresslets ($\boldsymbol{U}$, $\boldsymbol{\Omega}$ and $\mathbfsfit{S}$). It can also do $\boldsymbol{FT}\mathbfsfit{S}$ to $\boldsymbol{U\Omega}\mathbfsfit{E}$, for when you want to specify stresslets instead of the background shear rate; and $\boldsymbol{U}_1\boldsymbol{F}_2\boldsymbol{T}\mathbfsfit{E}$ to $\boldsymbol{F}_1\boldsymbol{U}_2\boldsymbol{\Omega}\mathbfsfit{S}$, for when there are some particles whose velocities you want to fix while letting some other particles move under specified forces (e.g. when you want to fix some particles as walls). For the beads in the bead-and-spring dumbbells, which are normally much smaller than the other particles in the system, the only option is to apply the force and extract the velocity ($\boldsymbol{F}$ to $\boldsymbol{U}$).
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








# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References