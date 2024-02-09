#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

"""Plot particles at a given frame number for an NPZ file specified in the
script.

Does not plot any periodic copies. If you want to do this, see the code in
plot_particle_positions_video.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pylab import rcParams
sys.path.append("..")  # Allows importing from SD directory
from functions.graphics import (plot_all_spheres, plot_all_dumbbells,
                                plot_all_torque_lines, plot_all_velocity_lines,
                                plot_all_angular_velocity_lines)
from functions.shared import add_sphere_rotations_to_positions


filename = 'filename_here'
frameno = 0
viewing_angle = (0, -90)
viewbox_bottomleft_topright = np.array([[-15, -15, -15], [15, 15, 15]])
two_d_plot = False
view_labels = False
trace_paths = 0

# Naming the folders like this means you can run this script from any directory
this_folder = os.path.dirname(os.path.abspath(__file__))
output_folder = this_folder + "/../output/"

data1 = np.load(output_folder + filename + ".npz")
positions_centres = data1['centres']
positions_deltax = data1['deltax']
Fa_out = data1['Fa']
Fb_out = data1['Fb']
DFb_out = data1['DFb']

num_frames = positions_centres.shape[0]
num_particles = positions_centres.shape[1]
num_dumbbells = positions_deltax.shape[1]
num_spheres = num_particles - num_dumbbells
sphere_positions = positions_centres[frameno, 0:num_spheres, :]
dumbbell_positions = positions_centres[frameno, num_spheres:num_particles, :]
dumbbell_deltax = positions_deltax[frameno, :, :]

sphere_sizes = np.array([1 for _ in range(num_spheres)])
dumbbell_sizes = np.array([0.1 for _ in range(num_dumbbells)])
sphere_rotations = add_sphere_rotations_to_positions(
    sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
Ta_out = [[0, 0, 0] for _ in range(num_spheres)]
Oa_out = [[0, 0, 0] for _ in range(num_spheres)]
Ua_out = [[0, 0, 0] for _ in range(num_spheres)]

posdata = [sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
           dumbbell_positions, dumbbell_deltax]
previous_step_posdata = posdata

# Pictures initialise
rcParams.update({'font.size': 11})
rcParams.update({'figure.dpi': 120, 'figure.figsize': [6, 6],
                 'savefig.dpi': 140})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(viewing_angle[0], viewing_angle[1])
spheres = list()
dumbbell_lines = list()
dumbbell_spheres = list()
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
if two_d_plot:
    ax.set_proj_type('ortho')
    ax.set_yticks([])
else:
    ax.set_ylabel("$y$")
ax.set_xlabel("$x$")
ax.set_zlabel("$z$")
fig.tight_layout()

# Pictures
if viewbox_bottomleft_topright.size == 0:
    if num_spheres > 0 and num_dumbbells > 0:
        m = np.array([abs(sphere_positions).max(),
                      abs(dumbbell_positions).max()]).max()
    elif num_spheres > 0 and num_dumbbells == 0:
        m = abs(sphere_positions).max()
    elif num_dumbbells > 0 and num_spheres == 0:
        m = abs(dumbbell_positions).max()
    else:
        print("PROBLEM")
        m = 3
    viewbox_bottomleft_topright = np.array([[-m, -m, -m], [m, m, m]])

if num_spheres > 0:
    (spheres, sphere_lines, sphere_trace_lines) = plot_all_spheres(
        ax, frameno, viewbox_bottomleft_topright, posdata, previous_step_posdata,
        trace_paths, spheres, sphere_lines, sphere_trace_lines, Fa_out[frameno])
if num_dumbbells > 0:
    (dumbbell_spheres, dumbbell_lines, dumbbell_trace_lines) = plot_all_dumbbells(
        ax, frameno, viewbox_bottomleft_topright, posdata, previous_step_posdata,
        trace_paths, dumbbell_spheres, dumbbell_lines, dumbbell_trace_lines,
        Fb_out[frameno], DFb_out[frameno])
if view_labels:
    torque_lines = plot_all_torque_lines(ax, viewbox_bottomleft_topright,
                                         posdata, Ta_out, torque_lines)
    (velocity_lines, velocity_text, sphere_labels) = plot_all_velocity_lines(
        ax, viewbox_bottomleft_topright, posdata, Ua_out,
        velocity_lines)   # Velocity in green
    angular_velocity_lines = plot_all_angular_velocity_lines(
        ax, viewbox_bottomleft_topright, posdata, Oa_out,
        angular_velocity_lines)  # Ang vel in white with green edging

for q in (dumbbell_lines):
    q.remove()

ax.set_title("  frame "
             + ("{:" + str(len(str(num_frames))) + ".0f}").format(frameno)
             + "/" + str(num_frames-1), loc='left', y=0.97, fontsize=11)
ax.set_title(filename, loc='center', y=1.055, fontsize=11)

plt.show()
