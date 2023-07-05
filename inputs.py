#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

import sys  # Overrides from command line at the bottom of this script
import glob
import numpy as np
from numba import config
from position_setups import pos_setup
from functions_shared import add_sphere_rotations_to_positions, throw_error


# --------------------
# PHYSICAL INPUTS
setup_number = 1
input_number = 1
fully_2d_problem = False

# Timestep (default = 0.1)
timestep = 0.04

# Number of frames to run simulation for
num_frames = 100

# How to solve the equation U=MF. {'fts','fte','ufte','ufteu','duf'}
input_form = 'fte'
extract_force_on_wall_due_to_dumbbells = False

# --------------------------------------------------|
# OVERRIDES FROM COMMAND LINE
args = sys.argv[1:]
number_of_args = len(args)
running_on_legion = 0

try:
    if number_of_args >= 6:
        extract_force_on_wall_due_to_dumbbells = (args[5] in ["True", "true", "t", "T", "1"])
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
    throw_error("Invalid input. Command-line input should come in the order: setup number, input number, timestep, no. frames, input form, extract dumbbell force Boolean.")
# --------------------------------------------------|


posdata, setup_description = pos_setup(setup_number)

# Checkpointing ---------|

# Checkpoint override. If a file exists with the same input number, position number, number of frames and timestep, AND it ends in _TEMP, try to continue it.

# Initialise variables
checkpoint_filename = ''
checkpoint_start_from_frame = 0

# NOTE: comment this section to disable Checkpointing
# Otherwise read in list of temporary files
checkpoint_start_from_frame = 0
for filename in sorted(glob.glob("output/*_TEMP.npz"), reverse=True):
    checkpoint_setup_number = int(filename.split("-")[1][1:])
    checkpoint_input_number = int(filename.split("-")[2][1:])
    checkpoint_num_frames = int(filename.split("-")[3][:-2])
    if filename.split("-")[4][-1] == "e":
        raw_timestep = filename.split("-")[4][1:] + "-" + filename.split("-")[5]
    else:
        raw_timestep = filename.split("-")[4][1:]
    checkpoint_timestep = float(raw_timestep.replace("p", "."))
    if checkpoint_setup_number == setup_number and checkpoint_input_number == input_number and checkpoint_num_frames == num_frames and checkpoint_timestep == timestep:
        ignored_posdata, setup_description = pos_setup(checkpoint_setup_number)
        try:
            with np.load(filename) as data1:
                positions_centres = data1['centres']
                positions_deltax = data1['deltax']
            num_particles = positions_centres.shape[1]
            num_dumbbells = positions_deltax.shape[1]
            num_spheres = num_particles - num_dumbbells
            sphere_positions = positions_centres[-1, 0:num_spheres, :]
            dumbbell_positions = positions_centres[-1, num_spheres:num_particles, :]
            dumbbell_deltax = positions_deltax[-1, :, :]
            sphere_sizes = ignored_posdata[0]
            dumbbell_sizes = ignored_posdata[3]
            sphere_rotations = add_sphere_rotations_to_positions(sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
            posdata = (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax)

            checkpoint_filename = filename
            checkpoint_start_from_frame = positions_centres.shape[0]

            word_checkpoint = "\033[42m\033[01m CHECKPOINT FOUND \033[0m\033[49m "
            print(
                f"{word_checkpoint}Continuing file '{checkpoint_filename}' from frame {str(checkpoint_start_from_frame + 1)}/{num_frames}"
            )
        except Exception:
            print("Checkpoint possibly found but the file was corrupt (normally because power was lost midway through saving). Starting from beginning.")

# End checkpointing -----|


# --------------------
# GLOBAL INPUTS

# How many timesteps to wait in between finding (Minfinity)^-1 (10 normally OK). In between these timesteps we use the previously computed value.
invert_m_every = 1

# When to start using R2bexact. cutoff_factor = r*/(a1+a2). Default 2.
cutoff_factor = 2

# Timestepping scheme: Choose explicit timestep from: ['euler', 'ab2', 'rk4']  (ab2=Adams Bashforth)
timestepping_scheme = 'rk4'

# If using RK4, can choose to use the same Minfinity for each of the 4 stages. 
# In absence of R2Bexact, this makes the timestep equivalent to Euler; but if R2Bexact is present, and invert_m_every>1, this might be an OK approximation.
rk4_generate_minfinity_for_each_stage = True

# Are we going to be feeding in new particles underneath the box? (Really just for certain types of simulations)
feed_every_n_timesteps = 0  # 0 to turn off
feed_from_file = 'feed-from-file-name'

# For periodic boxes, how many repeats of the box do we want to consider before the contribution decays away?
# If Lx is box width, '1' corresponds to [-Lx, 0, Lx]. '2' corresponds to [-2Lx, -Lx, 0, Lx, 2Lx] etc.
# '2' is normally sufficient. '3' gives you a bit more accuracy.
how_far_to_reproduce_gridpoints = 2

# Set level of output to screen. # 0 = minimal output, 1 = separation distance and velocities, 2 = matrices, 3 = individual calculations
printout = 0

# Send email on completion? Set this up in functions_email.py.
send_email = False

# View graphics? (Note: doesn't save a video. To do that, use plot_particle_positions_video.py after the fact.)
view_graphics = True
text_only = 1 - int(view_graphics)

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
    print("WARNING: start_saving_after_first_n_timesteps > num_frames. Saving will fail.")

# Just use diagonal terms of Minfinity. Good(?) approximation for dense suspensions only.
use_drag_Minfinity = False
# Only use Minfinity, i.e. turn off R2Bexact. Mostly put in for a little discussion.
use_Minfinity_only = False

# Only relevant if  use_drag_Minfinity = True.
# The precomputed XYZ scalars in the resistance matrix already have R2binfinity
# subtracted off them (it's quicker). For historical reasons, they are called 
# 'd' scalars and the data files are labelled with '-d'.
# If use_drag_Minfinity = True, there is a choice of using 'd' scalars where 
# the drag Minfinity is subtracted, or using 'd' scalars where the full 
# Minfinity is subtracted. You might pick the drag version if you wanted
# to make sure that the two-sphere case works, but you might pick the full 
# version if you have a dense suspension.
use_full_Minfinity_scalars_with_drag_Minfinity = False

# --------------------
# VIDEO-RELATED INPUTS

# If generating video, what should we be able to see?
viewbox_bottomleft_topright = np.array([[-15, 0, -15], [15, 1, 15]])

# Viewing angle on video, (elev,azim). e.g. (0,-90) = x-z plane; (30,-60) = default
viewing_angle = (0, -90)

# View labels and arrows on spheres in the video?
view_labels = 0

# Frames per second
fps = 8

# Trace the paths on the video? 0 off, >=1 on. It will generate a line between every n timesteps.
trace_paths = 0

# 2D Plot? This removes the third axis.
two_d_plot = 1

# -----------
# DATA INPUTS

resistance_scalars_folder = "find_resistance_scalars/"

# IMPORTANT NOTE: if you change s_range or lam_range, you must rerun find_resistance_scalars_looper().
# Note: the s_range is the same currently for all values of lambda. This may be a bad idea in the long run. Not sure.
s_dash_range = np.loadtxt(f'{resistance_scalars_folder}/values_of_s_dash.txt')
range_s_dash_range = range(s_dash_range.shape[0])
range_len_of_s_dash_range = range(s_dash_range.shape[0])
lam_range = np.loadtxt(f'{resistance_scalars_folder}/values_of_lambda.txt')
lam_range_with_reciprocals = lam_range
for lam in lam_range_with_reciprocals:
    if (1./lam) not in lam_range_with_reciprocals:
        lam_range_with_reciprocals = np.append(lam_range_with_reciprocals, (1./lam))
lam_range_with_reciprocals.sort()

# Decide here whether we are going to use the d values or not
if not use_drag_Minfinity or (use_drag_Minfinity and use_full_Minfinity_scalars_with_drag_Minfinity):
    filename = f'{resistance_scalars_folder}/scalars_general_resistance_blob_d.txt'
else:
    filename = f'{resistance_scalars_folder}/scalars_general_resistance_blob_dnominf.txt'

try:
    with open(filename, 'rb') as inputfile:
        XYZ_raw = np.load(inputfile)
except IOError:
    XYZ_raw = []  # Only likely to be in this position when generating the resistance scalars.

filename = f'{resistance_scalars_folder}/scalars_general_resistance_blob.txt'
with open(filename, 'rb') as inputfile:
    XYZ_raw_no_d = np.load(inputfile)
s_dash_range = np.loadtxt(f'{resistance_scalars_folder}/values_of_s_dash.txt')

XYZf = 0

# ------------
# TURN NUMBA ON OR OFF
config.DISABLE_JIT = True
