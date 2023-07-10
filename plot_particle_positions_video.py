#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

"""Create MP4 video of either the most recently created NPZ file in the
output folder or an NPZ file specified in the script. 

Video is created in the output_videos folder.
"""

import glob
import os
import sys
import time
from input_setups import input_ftsuoe
from position_setups import pos_setup
from functions_shared import shear_basis_vectors, format_elapsed_time
from functions_timestepping import format_time_left
from functions_graphics import *
from matplotlib import animation, rcParams
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
# Use newest file
newest = max(glob.iglob('output/*.[Nn][Pp][Zz]'), key=os.path.getctime)
filename = newest[len("output/"):-len(".npz")]
# Overwrite filename here if you want a specific file

timestep = float(filename.split('-')[4][1:].replace('p', '.'))
input_number = int(filename.split('-')[2][1:])
setup_number = int(filename.split('-')[1][1:])

# Find out info from input_number
posdata, setup_description = pos_setup(setup_number)

print("Generating video [" + filename + ".mp4]")
print("[Timestep " + str(timestep) + " | ]")

num_frames_override_start = 0
num_frames_override_end = 0  # Set to 0 to not override number of frames
display_every_n_frames = 1

viewing_angle = (0, -90)
viewbox_bottomleft_topright = np.array([[-10, -10, -10], [10, 10, 10]])
two_d_plot = 0
view_labels = 0

# Fixed inputs
trace_paths = 0

data1 = np.load("output/" + filename + ".npz")
positions_centres = data1['centres']
positions_deltax = data1['deltax']
Fa_out = data1['Fa']
Fb_out = data1['Fb']
DFb_out = data1['DFb']
positions_sphere_rotations = data1['sphere_rotations']

if num_frames_override_end == 0:
    num_frames = positions_centres.shape[0]//display_every_n_frames
else:
    num_frames = (num_frames_override_end-num_frames_override_start)//display_every_n_frames
num_particles = positions_centres.shape[1]
num_dumbbells = positions_deltax.shape[1]
num_spheres = num_particles - num_dumbbells
sphere_sizes = np.array([1 for i in range(num_spheres)])
dumbbell_sizes = np.array([0.1 for i in range(num_dumbbells)])
max_DFb_out = 1

# Pictures initialise
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 120, 'figure.figsize': [11, 11],
                 'savefig.dpi': 140})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(viewing_angle[0], viewing_angle[1])
spheres = list()
dumbbell_spheres = list()
dumbbell_lines = list()
force_lines = list()
force_text = list()
torque_lines = list()
velocity_lines = list()
velocity_text = list()
sphere_labels = list()
angular_velocity_lines = list()
sphere_lines = list()
sphere_trace_lines = list()
dumbbell_trace_lines = list()
v = viewbox_bottomleft_topright.transpose()
ax.auto_scale_xyz(v[0], v[1], v[2])
ax.set_xlim3d(v[0, 0], v[0, 1])
ax.set_ylim3d(v[1, 0], v[1, 1])
ax.set_zlim3d(v[2, 0], v[2, 1])
ax.set_box_aspect((1, 1, 1), zoom=1.4)
if two_d_plot == 1:
    ax.set_proj_type('ortho')
    ax.set_yticks([])
else:
    ax.set_ylabel("$y$")
ax.set_xlabel("$x$")
ax.set_zlabel("$z$")
fig.tight_layout()
times = [0 for _ in range(num_frames)]

# Pictures
if viewbox_bottomleft_topright.size == 0:
    sphere_positions = positions_centres[0, 0:num_spheres, :]
    dumbbell_positions = positions_centres[0, num_spheres:num_particles, :]
    if num_spheres > 0 and num_dumbbells > 0:
        m = np.array([abs(sphere_positions).max(),
                      abs(dumbbell_positions).max()]).max()
    elif num_spheres > 0 and num_dumbbells == 0:
        m = abs(sphere_positions).max()
    elif num_dumbbells > 0 and num_spheres == 0:
        m = abs(dumbbell_positions).max()
    else:
        print("No viewbox defined, and no spheres or dumbbells detected")
        m = 3
    viewbox_bottomleft_topright = np.array([[-m, -m, -m], [m, m, m]])


def generate_frame(frameno, viewbox_bottomleft_topright=np.array([]),
                   view_labels=1, timestep=0.1, trace_paths=0,
                   num_frames_override_start=0):
    global posdata, previous_step_posdata, sphere_sizes, dumbbell_sizes
    global spheres, dumbbell_lines, sphere_lines, sphere_trace_lines
    global dumbbell_trace_lines, dumbbell_spheres
    global force_lines, force_text, torque_lines
    global velocity_lines, velocity_text, angular_velocity_lines, sphere_labels
    global error, previous_timestamp, saved_element_positions, saved_deltax
    global saved_Fa_out, saved_Fb_out, saved_DFb_out
    global fig, ax
    global linec1, linec2, linec3, linec4

    frame_start_time = time.time()
    real_frameno = frameno*display_every_n_frames + num_frames_override_start

    print("Generating frame "
          + ("{:" + str(len(str(num_frames))) + ".0f}").format(real_frameno)
          + " ("
          + ("{:" + str(len(str(num_frames))) + ".0f}").format(frameno)
          + "/" + str(num_frames) + ")...", end=" ")

    sphere_positions = positions_centres[real_frameno, 0:num_spheres, :]
    sphere_rotations = positions_sphere_rotations[real_frameno, 0:num_spheres, :]
    dumbbell_positions = positions_centres[real_frameno, num_spheres:num_particles, :]
    dumbbell_deltax = positions_deltax[real_frameno, :, :]

    no_line = False
    if frameno == num_frames - 1:
        FaX = Fa_out[frameno-1]
        FbX = Fb_out[frameno-1]
        DFbX = DFb_out[frameno-1]
    else:
        FaX = Fa_out[frameno]
        FbX = Fb_out[frameno]
        DFbX = DFb_out[frameno]

    # FOR PERIODIC, let's get it to repeat itself left and right
    (Fa_in, Ta_in, Sa_in, Sa_c_in, Fb_in, DFb_in, Ua_in, Oa_in, Ea_in,
     Ea_c_in, Ub_in, HalfDUb_in, desc, U_infinity, O_infinity,
     centre_of_background_flow, Ot_infinity, Et_infinity,
     box_bottom_left, box_top_right, mu) = input_ftsuoe(
        input_number, posdata, real_frameno, timestep, [[], [], [], []],
        skip_computation=True)
    if np.linalg.norm(box_top_right - box_bottom_left) > 0:  # Periodic
        Lx = box_top_right[0] - box_bottom_left[0]
        Ly = box_top_right[1] - box_bottom_left[1]
        Lz = box_top_right[2] - box_bottom_left[2]
        L = (Lx*Ly*Lz)**(1./3.)
        # How to interpret `how_far_to_reproduce_gridpoints`:
        #   1 corresponds to [-Lx, 0, Lx].
        #   2 corresponds to [-2Lx, -Lx, 0, Lx, 2Lx] etc.
        how_far_to_reproduce_gridpoints = 1
        gridpoints_x = [i for i in range(-how_far_to_reproduce_gridpoints,
                                         how_far_to_reproduce_gridpoints+1)]
        gridpoints_y = [0]
        gridpoints_z = [i for i in range(-how_far_to_reproduce_gridpoints,
                                         how_far_to_reproduce_gridpoints+1)]
        X_lmn_canonical = np.array([[ll, mm, nn] for ll in gridpoints_x
                                    for mm in gridpoints_y
                                    for nn in gridpoints_z])

        # Then shear the basis vectors
        box_dimensions = box_top_right - box_bottom_left
        basis_canonical = np.diag(box_dimensions)
        sheared_basis_vectors = shear_basis_vectors(
            basis_canonical, box_dimensions, Ot_infinity, Et_infinity)
        X_lmn_sheared = np.dot(X_lmn_canonical, sheared_basis_vectors)

        # Draw the periodic box in red.
        corners = np.array([[box_bottom_left[0], 0, box_bottom_left[2]],
                            [box_bottom_left[0], 0, box_top_right[2]],
                            [box_top_right[0], 0, box_top_right[2]],
                            [box_top_right[0], 0, box_bottom_left[2]]])
        # box corners, sheared:
        #  (i) Transformed into the canonical basis (box length = 1 unit)
        # (ii) Transformed into the sheared basis
        corners_sheared = np.dot(sheared_basis_vectors, 
                                 np.linalg.solve(basis_canonical,
                                                 corners.T)).T
        cor1 = corners_sheared[0]
        cor2 = corners_sheared[1]
        cor3 = corners_sheared[2]
        cor4 = corners_sheared[3]
        if frameno > 0:
            linec1.remove()
            linec2.remove()
            linec3.remove()
            linec4.remove()
        linec1 = plt.plot(*np.transpose([cor1,cor2]), color='r', linewidth=1)[0]
        linec2 = plt.plot(*np.transpose([cor2,cor3]), color='r', linewidth=1)[0]
        linec3 = plt.plot(*np.transpose([cor3,cor4]), color='r', linewidth=1)[0]
        linec4 = plt.plot(*np.transpose([cor4,cor1]), color='r', linewidth=1)[0]

        # NOTE: If you change the next line you have to change it in K_lmn too!
        X_lmn_sheared_inside_radius = X_lmn_sheared[
            np.linalg.norm(X_lmn_sheared, axis=1) <= 1.4142*how_far_to_reproduce_gridpoints*L]
        X_lmn = X_lmn_sheared_inside_radius
        if num_spheres > 0:
            sphere_positions = np.concatenate(
                np.array([sphere_positions + X for X in X_lmn]), axis=0)
            sphere_rotations = np.concatenate(
                np.array([sphere_rotations + X for X in X_lmn]), axis=0)
            sphere_sizes_periodic = np.concatenate(
                np.array([sphere_sizes for _ in X_lmn]), axis=0)
        else:
            sphere_sizes_periodic = sphere_sizes
        if num_dumbbells > 0:
            dumbbell_sizes_periodic = np.concatenate(
                np.array([dumbbell_sizes for _ in X_lmn]), axis=0)
            dumbbell_positions = np.concatenate(
                np.array([dumbbell_positions + X for X in X_lmn]), axis=0)
            dumbbell_deltax = np.concatenate(
                np.array([dumbbell_deltax for _ in X_lmn]), axis=0)
            FbX = np.concatenate([FbX for _ in X_lmn], axis=0)
            DFbX = np.concatenate([DFbX for _ in X_lmn], axis=0)
        else:
            dumbbell_sizes_periodic = dumbbell_sizes
        FaX = np.concatenate([FaX for _ in X_lmn], axis=0)
        posdata = [sphere_sizes_periodic, sphere_positions, sphere_rotations,
                   dumbbell_sizes_periodic, dumbbell_positions, dumbbell_deltax]
        previous_step_posdata = posdata
    else:
        posdata = [sphere_sizes, sphere_positions, sphere_rotations,
                   dumbbell_sizes, dumbbell_positions, dumbbell_deltax]
        previous_step_posdata = posdata

    Ta_out = [[0, 0, 0] for _ in range(num_spheres)]
    Oa_out = [[0, 0, 0] for _ in range(num_spheres)]
    Ua_out = [[0, 0, 0] for _ in range(num_spheres)]

    for q in (spheres + force_lines + torque_lines + velocity_lines
              + angular_velocity_lines + sphere_lines + dumbbell_lines
              + dumbbell_spheres):
        q.remove()

    if num_spheres > 0:
        (spheres, sphere_lines, sphere_trace_lines) = plot_all_spheres(
            ax, real_frameno, viewbox_bottomleft_topright, posdata,
            previous_step_posdata, trace_paths, spheres, sphere_lines,
            sphere_trace_lines, FaX)
    if num_dumbbells > 0:
        (dumbbell_spheres, dumbbell_lines, dumbbell_trace_lines) = plot_all_dumbbells(
            ax, real_frameno, viewbox_bottomleft_topright, posdata,
            previous_step_posdata, trace_paths, dumbbell_spheres, dumbbell_lines,
            dumbbell_trace_lines, FbX, DFbX, max_DFb_out=max_DFb_out,
            no_line=no_line)

    if view_labels == 1:
        (force_lines, force_text) = plot_all_force_lines(
            ax, viewbox_bottomleft_topright, posdata, Fa_out, force_lines)
        torque_lines = plot_all_torque_lines(
            ax, viewbox_bottomleft_topright, posdata, Ta_out, torque_lines)
        (velocity_lines, velocity_text, sphere_labels) = plot_all_velocity_lines(
            ax, viewbox_bottomleft_topright, posdata, Ua_out,
            velocity_lines)   # Velocity in green
        angular_velocity_lines = plot_all_angular_velocity_lines(
            ax, viewbox_bottomleft_topright, posdata, Oa_out,
            angular_velocity_lines)  # Ang vel in white with green edging

    ax.set_title("  frame "
                 + ("{:" + str(len(str(num_frames))) + ".0f}").format(frameno + 1)
                 + "/" + str(num_frames),
                 loc='left', y=0.97)
    ax.set_title(filename, loc='center', y=1.05)

    pic_elapsed_time = time.time() - frame_start_time
    print("[[" + "\033[1m" + format_elapsed_time(pic_elapsed_time)
          + "\033[0m" + "]]", end=" ")

    times[frameno] = pic_elapsed_time
    timeaverage = sum(times) / (frameno+1)
    numberoftimesleft = num_frames - frameno - 1
    timeleft = (numberoftimesleft * timeaverage) * 1.05
    print("[" + format_time_left(timeleft, "") + "]")


total_time_start = time.time()
generate_frame_args = [viewbox_bottomleft_topright, view_labels, timestep,
                       trace_paths, num_frames_override_start]
ani = animation.FuncAnimation(fig, generate_frame, frames=num_frames,
                              fargs=generate_frame_args, repeat=False,
                              interval=200, save_count=num_frames)
mywriter = animation.FFMpegWriter(fps=8)
try:
    ani.save('output_videos/' + filename + '.mp4', writer=mywriter)
except FileNotFoundError as e:
    print(e)
    print("ERROR: You need to install FFMPEG.")
    print("If your Python distribution is Anaconda, see https://anaconda.org/conda-forge/ffmpeg")
    sys.exit()
total_elapsed_time = time.time() - total_time_start
print("[Total time to run " + format_elapsed_time(total_elapsed_time) + "]")
print("Completed [" + filename + ".mp4]")
