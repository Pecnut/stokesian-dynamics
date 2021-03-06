#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, rcParams
from mpl_toolkits.mplot3d import proj3d
from functions_graphics import *
from functions_shared import add_sphere_rotations_to_positions, format_elapsed_time
import time, math, sys, os, glob
# Use newest file
newest = max(glob.iglob('output/*.[Nn][Pp][Zz]'), key=os.path.getctime)
filename = newest[len("output/"):-len(".npz")]
# Overwrite filename here if you want a specific file

timestep = float(filename.split('-')[4][1:].replace('p','.'))
input_number = np.int(filename.split('-')[2][1:])
setup_number = np.int(filename.split('-')[1][1:])

# Find out info from input_number
from position_setups import pos_setup
from input_setups import input_ftsuoe
posdata, setup_description = pos_setup(setup_number)

print "Generating video [" + filename + ".mp4]"
print "[Timestep " + str(timestep) + " | ]"

num_frames_override_start = 0
num_frames_override_end = 5
display_every_n_frames = 1

viewing_angle = (0,-90)
viewbox_bottomleft_topright = np.array([[-10,-10,-10],[10,10,10]])
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
    num_frames = positions_centres.shape[0]/display_every_n_frames
else:
    num_frames = (num_frames_override_end - num_frames_override_start)/display_every_n_frames
num_particles = positions_centres.shape[1]
num_dumbbells = positions_deltax.shape[1]
num_spheres = num_particles - num_dumbbells
sphere_sizes = np.array([1 for i in range(num_spheres)])
dumbbell_sizes = np.array([0.1 for i in range (num_dumbbells)])
max_DFb_out = 1


def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,a,b],
                        [0,0,-0.0001,zback]])

# Pictures initialise
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 120, 'figure.figsize': [11,11], 'savefig.dpi': 140, 'savefig.jpeg_quality': 140})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(viewing_angle[0],viewing_angle[1])
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
ax.auto_scale_xyz(v[0],v[1],v[2])
ax.set_xlim3d(v[0,0],v[0,1])
ax.set_ylim3d(v[1,0],v[1,1])
ax.set_zlim3d(v[2,0],v[2,1])
if two_d_plot == 1:
    proj3d.persp_transformation = orthogonal_proj
    ax.set_yticks([])
else:
    ax.set_ylabel("$y$")
ax.set_xlabel("$x$")
ax.set_zlabel("$z$")
fig.tight_layout()
ax.dist = 6.85
times = [0 for i in range(num_frames)]

#Pictures
if viewbox_bottomleft_topright.size == 0:
    if num_spheres > 0 and num_dumbbells > 0:
        m = np.array([abs(sphere_positions).max(),abs(dumbbell_positions).max()]).max()
    elif num_spheres > 0 and num_dumbbells == 0:
        m = abs(sphere_positions).max()
    elif num_dumbbells > 0 and num_spheres == 0:
        m = abs(dumbbell_positions).max()
    else:
        print "PROBLEM"
        m = 3
    viewbox_bottomleft_topright = np.array([[-m,-m,-m],[m,m,m]])

def generate_frame(frameno, viewbox_bottomleft_topright=np.array([]), view_labels=1, timestep=0.1, trace_paths = 0, num_frames_override_start=0):
    global posdata, previous_step_posdata, sphere_sizes, dumbbell_sizes
    global spheres, dumbbell_lines, sphere_lines, sphere_trace_lines, dumbbell_trace_lines, dumbbell_spheres
    global force_lines, force_text, torque_lines
    global velocity_lines, velocity_text, angular_velocity_lines, sphere_labels
    global error, previous_timestamp, saved_element_positions, saved_deltax, saved_Fa_out, saved_Fb_out, saved_DFb_out
    global fig,ax
    global linec1, linec2, linec3, linec4

    frame_start_time = time.time()
    real_frameno = frameno*display_every_n_frames + num_frames_override_start

    print "Generating frame " + ("{:" + str(len(str(num_frames))) + ".0f}").format(real_frameno) + " (" + ("{:" + str(len(str(num_frames))) + ".0f}").format(frameno) + "/" + str(num_frames) + ")...",

    sphere_positions = positions_centres[real_frameno,0:num_spheres, :]
    sphere_rotations = positions_sphere_rotations[real_frameno,0:num_spheres, :]
    dumbbell_positions = positions_centres[real_frameno,num_spheres:num_particles,:]
    dumbbell_deltax = positions_deltax[real_frameno,:,:]

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
    Fa_in, Ta_in, Sa_in, Sa_c_in, Fb_in, DFb_in, Ua_in, Oa_in, Ea_in, Ea_c_in, Ub_in, DUb_in, desc, U_infinity, O_infinity, centre_of_background_flow, amplitude, frequency,box_bottom_left,box_top_right,mu = input_ftsuoe(input_number,posdata,real_frameno,timestep,[[],[],[],[]],video=True)
    if np.linalg.norm(box_top_right - box_bottom_left) > 0: #Periodic
        E_infinity = Ea_in[0]

        Lx = box_top_right[0] - box_bottom_left[0]
        Ly = box_top_right[1] - box_bottom_left[1]
        Lz = box_top_right[2] - box_bottom_left[2]
        L = (Lx*Ly*Lz)**(1./3.)

        lamb = math.sqrt(math.pi)/L
        how_far_to_reproduce_gridpoints = 1 # 1 corresponds to [-Lx, 0, Lx]. 2 corresponds to [-2Lx, -Lx, 0, Lx, 2Lx] etc.
        gridpoints_x = [i for i in range(-how_far_to_reproduce_gridpoints,how_far_to_reproduce_gridpoints+1)]
        gridpoints_y = [0]
        gridpoints_z = [i for i in range(-how_far_to_reproduce_gridpoints,how_far_to_reproduce_gridpoints+1)]
        X_lmn_canonical = np.array([[ll,mm,nn] for ll in gridpoints_x for mm in gridpoints_y for nn in gridpoints_z])

        # Then shear the basis vectors
        basis_canonical = np.array([[Lx,0,0],[0,Ly,0],[0,0,Lz]])

        # NOTE: For CONTINUOUS shear, set the following
        #time_t = (frameno*display_every_n_frames+num_frames_override_start)*timestep
        #sheared_basis_vectors_add_on = (np.cross(np.array(O_infinity)*time_t,basis_canonical).transpose() + np.dot(np.array(E_infinity)*time_t,(basis_canonical).transpose())).transpose()# + basis_canonical
        # NOTE: For OSCILLATORY shear, set the following (basically there isn't a way to find out shear given E)
        time_t = (frameno*display_every_n_frames+num_frames_override_start)*timestep
        gamma = amplitude*np.sin(time_t*frequency)
        Ot_infinity = np.array([0,0.5*gamma,0])
        Et_infinity = [[0,0,0.5*gamma],[0,0,0],[0.5*gamma,0,0]]
        sheared_basis_vectors_add_on = (np.cross(Ot_infinity,basis_canonical).transpose() + np.dot(Et_infinity,(basis_canonical).transpose())).transpose()

        sheared_basis_vectors_add_on_mod  = np.mod(sheared_basis_vectors_add_on,[Lx,Ly,Lz])
        sheared_basis_vectors = basis_canonical + sheared_basis_vectors_add_on_mod
        X_lmn_sheared = np.dot(X_lmn_canonical,sheared_basis_vectors)

        corners = np.array([[box_bottom_left[0],0,box_bottom_left[2]],
                            [box_bottom_left[0],0,box_top_right[2]],
                            [box_top_right[0],0,box_top_right[2]],
                            [box_top_right[0],0,box_bottom_left[2]]])/4.
        corners_sheared = np.dot(corners,sheared_basis_vectors)
        cor1 = corners_sheared[0]
        cor2 = corners_sheared[1]
        cor3 = corners_sheared[2]
        cor4 = corners_sheared[3]
        if frameno>0:
            linec1.remove()
            linec2.remove()
            linec3.remove()
            linec4.remove()
        linec1 = plt.plot((cor1[0],cor2[0]),(cor1[1],cor2[1]),(cor1[2],cor2[2]),color='r',linewidth=1)[0]
        linec2 = plt.plot((cor2[0],cor3[0]),(cor2[1],cor3[1]),(cor2[2],cor3[2]),color='r',linewidth=1)[0]
        linec3 = plt.plot((cor3[0],cor4[0]),(cor3[1],cor4[1]),(cor3[2],cor4[2]),color='r',linewidth=1)[0]
        linec4 = plt.plot((cor4[0],cor1[0]),(cor4[1],cor1[1]),(cor4[2],cor1[2]),color='r',linewidth=1)[0]

        X_lmn_sheared_inside_radius = X_lmn_sheared[np.linalg.norm(X_lmn_sheared,axis=1)<=1.4142*how_far_to_reproduce_gridpoints*L] # NOTE: If you change this you have to change it in K_lmn as well!
        X_lmn = X_lmn_sheared_inside_radius
        if num_spheres > 0:
            sphere_positions = np.concatenate(np.array([sphere_positions + X for X in X_lmn]),axis=0)
            sphere_rotations = np.concatenate(np.array([sphere_rotations + X for X in X_lmn]),axis=0)
            sphere_sizes_periodic = np.concatenate(np.array([sphere_sizes for X in X_lmn]),axis=0)
        else:
            sphere_sizes_periodic = sphere_sizes
        if num_dumbbells > 0:
            dumbbell_sizes_periodic = np.concatenate(np.array([dumbbell_sizes for X in X_lmn]),axis=0)
            dumbbell_positions = np.concatenate(np.array([dumbbell_positions + X for X in X_lmn]),axis=0)
            dumbbell_deltax = np.concatenate(np.array([dumbbell_deltax for X in X_lmn]),axis=0)
            FbX = np.concatenate([FbX for X in X_lmn],axis=0)
            DFbX = np.concatenate([DFbX for X in X_lmn],axis=0)
        else:
            dumbbell_sizes_periodic = dumbbell_sizes
        FaX = np.concatenate([FaX for X in X_lmn],axis=0)
        posdata = [sphere_sizes_periodic, sphere_positions, sphere_rotations, dumbbell_sizes_periodic, dumbbell_positions, dumbbell_deltax]
        previous_step_posdata = posdata
    else:
        posdata = [sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax]
        previous_step_posdata = posdata

    Ta_out = [[0,0,0] for i in range(num_spheres)]
    Oa_out = [[0,0,0] for i in range(num_spheres)]
    Ua_out = [[0,0,0] for i in range(num_spheres)]

    for q in (spheres + force_lines + torque_lines + velocity_lines + angular_velocity_lines + sphere_lines + dumbbell_lines + dumbbell_spheres):
        q.remove()

    (spheres, sphere_lines, sphere_trace_lines) = plot_all_spheres(ax,real_frameno,viewbox_bottomleft_topright,posdata,previous_step_posdata,trace_paths,spheres,sphere_lines,sphere_trace_lines,FaX)
    (dumbbell_spheres, dumbbell_lines, dumbbell_trace_lines) = plot_all_dumbbells(ax,real_frameno,viewbox_bottomleft_topright,posdata,previous_step_posdata,trace_paths,dumbbell_spheres,dumbbell_lines, dumbbell_trace_lines,FbX,DFbX,max_DFb_out=max_DFb_out,no_line=no_line)

    if view_labels == 1:
        (force_lines, force_text) = plot_all_force_lines(ax,viewbox_bottomleft_topright,posdata,Fa_out,force_lines)
        torque_lines = plot_all_torque_lines(ax,viewbox_bottomleft_topright,posdata,Ta_out,torque_lines)
        (velocity_lines, velocity_text, sphere_labels) = plot_all_velocity_lines(ax,viewbox_bottomleft_topright,posdata,Ua_out,velocity_lines)   # Velocity in green
        angular_velocity_lines = plot_all_angular_velocity_lines(ax,viewbox_bottomleft_topright,posdata,Oa_out,angular_velocity_lines) # Ang vel in white with green edging

    ax.set_title("  frame " + ("{:" + str(len(str(num_frames))) + ".0f}").format(frameno+1) + "/" + str(num_frames),loc='left',y=0.97)
    ax.set_title(filename, loc='center',y=1.05)

    pic_elapsed_time = time.time() - frame_start_time
    print "[[" + "\033[1m" + format_elapsed_time(pic_elapsed_time) + "\033[0m" + "]]",

    times[frameno] = pic_elapsed_time
    timeaverage = sum(times) / (frameno+1)
    numberoftimesleft = num_frames - frameno - 1

    if (numberoftimesleft * timeaverage) * 1.05 > 86400: #24 hours
        start_color = "\033[94m"
    elif (numberoftimesleft * timeaverage) * 1.05 > 18000: #5 hours
        start_color = "\033[95m"
    elif (numberoftimesleft * timeaverage) * 1.05 > 3600: #1 hour
        start_color = "\033[91m"
    elif (numberoftimesleft * timeaverage) * 1.05 > 600: #10 mins
        start_color = "\033[93m"
    else:
        start_color = "\033[92m"
    end_color = "\033[0m"

    elapsed_time_formatted = start_color + format_elapsed_time((numberoftimesleft * timeaverage) * 1.05) + end_color

    print "[" + elapsed_time_formatted + "]"


total_time_start = time.time()
generate_frame_args = [viewbox_bottomleft_topright, view_labels, timestep, trace_paths, num_frames_override_start]
ani = animation.FuncAnimation(fig, generate_frame, frames=num_frames, fargs=generate_frame_args, repeat=False, interval=200, save_count=num_frames)
mywriter = animation.FFMpegWriter(fps=8)
ani.save('output_videos/' + filename + '.mp4', writer=mywriter)
total_elapsed_time = time.time() - total_time_start
print "[Total time to run " + format_elapsed_time(total_elapsed_time) + "]"
print "Completed [" + filename + ".mp4]"
