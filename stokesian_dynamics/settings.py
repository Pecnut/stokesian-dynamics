#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

"""Settings for a Stokesian Dynamics simulation.

See docs/settings.md for a full explanation of each setting. 

This file is imported by all other scripts in the Stokesian Dynamics package,
so it is the place to set up your simulation. You can override the inputs 
    setup_number, input_number, timestep, num_frames
by passing them in as arguments to run_simulation.py from the command line. 
See the README for more information.
"""

import sys
import numpy as np
from numba import config
from functions.shared import throw_error


# ----------------------------------------------------------------------------|
# BASIC SETTINGS

setup_number = 1
input_number = 1
fully_2d_problem = False

# Timestep (default = 0.1)
timestep = 0.1

# Number of frames to run simulation for
num_frames = 100

# How to solve the equation U=MF. {'fts','fte','ufte','ufteu','duf'}
input_form = 'fte'

# ----------------------------------------------------------------------------|
# BASIC SETTINGS: OVERRIDES FROM COMMAND LINE

args = sys.argv[1:]
number_of_args = len(args)
running_on_legion = 0

try:
    if number_of_args >= 6:
        extract_force_on_wall_due_to_dumbbells = (
            args[5] in ["True", "true", "t", "T", "1"])
    if number_of_args >= 5:
        input_form = args[4]
    if number_of_args >= 4:
        num_frames = int(args[3])
    if number_of_args >= 3:
        timestep = float(args[2])
    if number_of_args >= 2:
        input_number = int(args[1])
    if number_of_args >= 1:
        setup_number = int(args[0])
except ValueError:
    throw_error("Invalid input. Command-line input should come in the "
                + "order: setup number, input number, timestep, no. frames, "
                + "input form, extract dumbbell force Boolean.")

# ----------------------------------------------------------------------------|
# ADVANCED SETTINGS

# How many timesteps to wait in between finding (Minfinity)^-1 (10 is normally
# OK). In between these timesteps we use the previously computed value.
invert_m_every = 1

# When to start using R2bexact. cutoff_factor = r*/(a1+a2). Default 2.
cutoff_factor = 2

# Timestepping scheme: Choose explicit timestep from: ['euler', 'ab2', 'rk4']
# (ab2=Adams Bashforth)
timestepping_scheme = 'euler'

# If using RK4, do you want to generate Minfinity for each of the 4 stages
# (True), or just for the initial stage (False)? In the absence of R2Bexact, 
# choosing False makes the timestep equivalent to Euler; but if
# R2Bexact is present, and invert_m_every>1, this might be an OK approximation.
rk4_generate_minfinity_for_each_stage = True

# Are we going to be feeding in new particles underneath the box? (Really just
# for certain types of simulations)
feed_every_n_timesteps = 0  # 0 to turn off
feed_from_file = 'feed-from-file-name'

# For periodic boxes, how many repeats of the box do we want to consider before
# the contribution decays away?
# If Lx is box width, '1' corresponds to [-Lx, 0, Lx]. '2' corresponds to
# [-2Lx, -Lx, 0, Lx, 2Lx] etc.
# '2' is normally sufficient. '3' gives you a bit more accuracy.
how_far_to_reproduce_gridpoints = 2

# Set level of output to screen. # 0 = minimal output, 1 = separation distance
# and velocities, 2 = matrices, 3 = individual calculations
printout = 0

# Send email on completion? Set this up in functions/email.py.
email_on_completion = False

# View graphics? (Note: doesn't save a video. To do that, use
# plot_particle_positions_video.py after the fact.)
view_graphics = True

# Bead-bead interactions? (Should really always be true)
bead_bead_interactions = True

# Explosion protection? (Dumbbells with dx > 5 stops the execution)
explosion_protection = False

# Save positions/forces every n timesteps?
start_saving_after_first_n_timesteps = 0
save_positions_every_n_timesteps = 1
save_forces_every_n_timesteps = 1
save_forces_and_positions_to_temp_file_as_well = True
save_to_temp_file_every_n_timesteps = 120

if start_saving_after_first_n_timesteps > num_frames:
    print("WARNING: start_saving_after_first_n_timesteps > num_frames. "
          + "Saving will fail.")

# Work out the force on any particles whose velocities are specified in the
# input file. (Typical use case: these 'fixed' particles are acting as walls.)
# This is saved to `force_on_wall_due_to_dumbbells` in the output NPZ file.
extract_force_on_wall_due_to_dumbbells = False

# Just use diagonal terms of Minfinity. Good(?) approximation for dense
# suspensions only.
use_drag_Minfinity = False
# Only use Minfinity, i.e. turn off R2Bexact. Mostly put in for a little
# discussion.
use_Minfinity_only = False

# Only relevant if  use_drag_Minfinity = True.
# The precomputed XYZ scalars in the resistance matrix already have R2binfinity
# subtracted off them (it's quicker). These data files are labelled with '-d'.
# If use_drag_Minfinity = True, there is a choice of using '-d' scalars where
# the drag-only R2Binfinity is subtracted, or using '-d' scalars where the full
# R2Binfinity is subtracted. You might pick the drag version if you wanted
# to make sure that the two-sphere case works, but you might pick the full
# version if you have a dense suspension.
use_drag_R2Binfinity = False

# Turn Numba on or off
config.DISABLE_JIT = True

# ----------------------------------------------------------------------------|
# GRAPHICAL SETTINGS

# If generating video, what should we be able to see?
viewbox_bottomleft_topright = np.array([[-15, 0, -15], [15, 1, 15]])

# Viewing angle on video, (elev,azim).
# e.g. (0,-90) = x-z plane; (30,-60) = Matplotlib default.
viewing_angle = (0, -90)

# View arrows on spheres in the video?
view_labels = False

# Trace the paths on the video? 0 off, n>=1 on. It will generate a line between
# every n timesteps.
trace_paths = 0

# 2D Plot? This removes the third axis.
two_d_plot = True
