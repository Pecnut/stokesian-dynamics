#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 17/10/2014

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def rotation_matrix(d):
    """
    Calculate a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[0, d[2], -d[1]],
                     [-d[2], 0, d[0]],
                     [d[1], -d[0], 0]], dtype=np.float64)

    return ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew


def pathpatch_2d_to_3d(pathpatch, z=0, normal='z'):
    """
    Transform a 2D patch to a 3D patch using the given normal vector.

    The patch is projected into the XY plane, rotated about the origin
    and finally translated by z.
    """
    if isinstance(normal, str):  # Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1, 0, 0), index)

    normal /= np.linalg.norm(normal)  # Make sure the vector is normalised

    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face colour

    verts = path.vertices  # Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1))  # Obtain the rotation vector
    M = rotation_matrix(d)  # Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z)
                                     for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    """Translate the 3D pathpatch by the amount delta."""
    pathpatch._segment3d += delta


def between0and1(num):
    """Forces `num` to be between 0 and 1, else returns 0 or 1."""
    if num < 0:
        return 0
    elif num > 1:
        return 1
    else:
        return num


def plot_sphere(ax, frameno, position, previous_position, trace_paths, radius,
                sphere_rotations, dumbbell=0, sphere_colour='b'):
    n = normal(ax)

    if dumbbell == 0:
        shading = 1
        colour = sphere_colour
    if dumbbell == 1:
        shading = 1
        colour = 'r'

    # Make invisible for a slice through a 3D sample
    # if position[1] >= 0.05 or position[1] < -1:
    #    shading = 0

    p = Circle((0, 0), radius, facecolor=colour, alpha=shading)

    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z=0, normal=n)
    # posn of ball
    pathpatch_translate(p, (position[0], position[1], position[2]))

    pos1 = position
    distance_from_screen_1 = np.dot(pos1, np.array(n)) * sign(ax.elev)
    max_distance_from_screen = distance_from_screen_1 + radius
    min_distance_from_screen = distance_from_screen_1 - radius

    pos2 = sphere_rotations[0]
    distance_from_screen_2 = np.dot(pos2, np.array(n)) * sign(ax.elev)
    shading_2 = (distance_from_screen_2 - min_distance_from_screen)/(
        max_distance_from_screen - min_distance_from_screen)
    sh2 = str(0.5*between0and1(shading_2))

    pos3 = sphere_rotations[1]
    distance_from_screen_3 = np.dot(pos3, np.array(n)) * sign(ax.elev)
    shading_3 = (distance_from_screen_3 - min_distance_from_screen)/(
        max_distance_from_screen - min_distance_from_screen)
    sh3 = str(0.5*between0and1(shading_3))

    # Orientation lines
    display_orientation_lines = True
    if not display_orientation_lines:
        pos2 = pos1
        pos3 = pos1
    line1 = plt.plot((pos1[0], pos2[0]), (pos1[1], pos2[1]),
                     (pos1[2], pos2[2]),sh2, linewidth=1, zorder=500)[0]
    line2 = plt.plot((pos1[0], pos3[0]), (pos1[1], pos3[1]),
                     (pos1[2], pos3[2]), sh3, linewidth=1, zorder=500)[0]

    linetrace = None
    if trace_paths > 0 and frameno % trace_paths == 0:
        linetrace = plt.plot((previous_position[0], position[0]),
                             (previous_position[1], position[1]),
                             (previous_position[2], position[2]),
                             color=sphere_colour, linewidth=1)[0]
    return (p, line1, line2, linetrace)


def plot_dumbbell(ax, position, trace_paths, radius, dx, rot_theta=0, rot_phi=0,
                  dumbbell_colour=['b', 'b'], no_line=False):
    pos1 = position - 0.5*dx
    pos2 = position + 0.5*dx

    if not no_line:
        line = plt.plot((pos1[0], pos2[0]), (pos1[1], pos2[1]),
                        (pos1[2], pos2[2]), color=dumbbell_colour[0],
                        linewidth=1)[0]

    n = normal(ax)
    alpha1 = 1
    alpha2 = 1
    # Make invisible if we want a slice through a 3D sample
    '''
    if pos1[1] >= 0.05 or pos1[1] <= -1:
        alpha1 = 0
    if pos2[1] >= 0.05 or pos2[1] <= -1:
        alpha2 = 0
    '''

    p1 = Circle((0, 0), radius, facecolor=dumbbell_colour[0], alpha=alpha1)
    ax.add_patch(p1)
    pathpatch_2d_to_3d(p1, z=0, normal=n)
    pathpatch_translate(p1, (pos1[0], pos1[1], pos1[2]))  # posn of ball
    p2 = Circle((0, 0), radius, facecolor=dumbbell_colour[1], alpha=alpha2)
    ax.add_patch(p2)
    pathpatch_2d_to_3d(p2, z=0, normal=n)
    pathpatch_translate(p2, (pos2[0], pos2[1], pos2[2]))  # posn of ball

    return (p1, p2) if no_line else (p1, p2, line)


def plot_all_spheres(ax, frameno, posdata, previous_step_posdata, trace_paths,
                     sphere_trace_lines, f_spheres):
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax) = posdata
    (previous_sphere_sizes, previous_sphere_positions,
     previous_sphere_rotations, previous_dumbbell_sizes,
     previous_dumbbell_positions,
     previous_dumbbell_deltax) = previous_step_posdata
    spheres = []
    sphere_lines = []
    for i in range(sphere_positions.shape[0]):
        C = between0and1(np.linalg.norm(np.array(f_spheres[i], float)))
        sphere_colour = [C, 0, 1-C]
        (p, l1, l2, ltrace) = plot_sphere(
            ax, frameno, sphere_positions[i, :],
            previous_sphere_positions[i, :], trace_paths, sphere_sizes[i],
            sphere_rotations[i], sphere_colour=sphere_colour)
        spheres.append(p)
        sphere_lines.extend((l1, l2))
        if trace_paths > 0 and frameno % trace_paths == 0:
            sphere_trace_lines.append(ltrace)
    return spheres, sphere_lines, sphere_trace_lines


def plot_all_dumbbells(ax, posdata, trace_paths, dumbbell_trace_lines,
                       f_dumbbells, deltaf_dumbbells, no_line=False):
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax) = posdata
    dumbbell_lines = []
    dumbbell_spheres = []

    force_on_beads1 = 0.5*(f_dumbbells - deltaf_dumbbells)
    force_on_beads2 = 0.5*(f_dumbbells + deltaf_dumbbells)
    max_force_on_bead = 0.5*np.max([np.max(force_on_beads1),
                                    np.max(force_on_beads2)])

    for i in range(dumbbell_sizes.size):
        if max_force_on_bead == 0:
            C = 0
        else:
            C = between0and1(
                np.linalg.norm(force_on_beads1[i])/max_force_on_bead)
            C = C**2
        dumbbell_colour1 = [C, 1-C, 0]  # go red, else be blue
        if max_force_on_bead == 0:
            C = 0
        else:
            C = between0and1(
                np.linalg.norm(force_on_beads2[i])/max_force_on_bead)
            C = C**2
        dumbbell_colour2 = [C, 1-C, 0]  # go red, else be blue
        dumbbell_colour = [dumbbell_colour1, dumbbell_colour2]
        if no_line:
            sphere1, sphere2 = plot_dumbbell(
                ax, dumbbell_positions[i, :], trace_paths, dumbbell_sizes[i],
                dumbbell_deltax[i, :], dumbbell_colour=dumbbell_colour,
                no_line=no_line)
        else:
            sphere1, sphere2, line = plot_dumbbell(
                ax, dumbbell_positions[i, :], trace_paths, dumbbell_sizes[i],
                dumbbell_deltax[i, :], dumbbell_colour=dumbbell_colour,
                no_line=no_line)
            dumbbell_lines.append(line)
        dumbbell_spheres.extend((sphere1, sphere2))
        # NOTE: CURRENTLY, DUMBBELL TRACE LINES ARE NOT IMPLEMENTED

    return dumbbell_spheres, dumbbell_lines, dumbbell_trace_lines


def plot_all_force_lines(ax, posdata, f_spheres):
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax) = posdata
    force_lines = force_text = []
    for i in range(sphere_sizes.size):
        # Make invisible if hidden
        if ((f_spheres[i, :] != np.array([0, 0, 0])).any()
                and (sphere_positions[i, 1] < 0.05 and sphere_positions[i, 1] > -1)):
            f_spheres = f_spheres*6  # so we can see it
            line = Arrow3D([sphere_positions[i, 0],
                            sphere_positions[i, 0]+f_spheres[i, 0]],
                           [sphere_positions[i, 1],
                            sphere_positions[i, 1]+f_spheres[i, 1]],
                           [sphere_positions[i, 2],
                            sphere_positions[i, 2]+f_spheres[i, 2]],
                           mutation_scale=20, lw=1,
                           arrowstyle="-|>", color="r")
            ax.add_artist(line)
            force_lines.append(line)
            ftext = ax.text(
                sphere_positions[i, 0] + f_spheres[i, 0],
                sphere_positions[i, 1] + f_spheres[i, 1],
                sphere_positions[i, 2] + f_spheres[i, 2],
                f"{str(i)}",
                color='r',
            )
            force_text.append(ftext)
    return force_lines, force_text


def plot_all_torque_lines(ax, posdata, t_spheres):
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax) = posdata
    torque_lines = []
    for i in range(sphere_sizes.size):
        if (t_spheres[i, :] != np.array([0, 0, 0])).any():
            line = Arrow3D([sphere_positions[i, 0],
                            sphere_positions[i, 0]+t_spheres[i, 0]],
                           [sphere_positions[i, 1],
                            sphere_positions[i, 1]+t_spheres[i, 1]],
                           [sphere_positions[i, 2],
                            sphere_positions[i, 2]+t_spheres[i, 2]],
                           mutation_scale=20, lw=1, arrowstyle="-|>",
                           edgecolor="r", facecolor="w")
            ax.add_artist(line)
            torque_lines.append(line)
    return torque_lines


def plot_all_velocity_lines(ax, posdata, u_spheres):
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax) = posdata
    velocity_lines = velocity_text = sphere_labels = []
    for i in range(sphere_sizes.size):
        if (u_spheres[i, :] != np.array([0, 0, 0])).any():
            line = Arrow3D([sphere_positions[i, 0],
                            sphere_positions[i, 0]+u_spheres[i, 0]],
                           [sphere_positions[i, 1],
                            sphere_positions[i, 1]+u_spheres[i, 1]],
                           [sphere_positions[i, 2],
                            sphere_positions[i, 2]+u_spheres[i, 2]],
                           mutation_scale=20, lw=1, arrowstyle="-|>",
                           color="g")
            ax.add_artist(line)
            velocity_lines.append(line)

            vtext = ax.text(
                sphere_positions[i, 0] + u_spheres[i, 0],
                sphere_positions[i, 1] + u_spheres[i, 1],
                sphere_positions[i, 2] + u_spheres[i, 2],
                f"{str(i)}",
                color='g',
            )
            velocity_text.append(vtext)

        slabel = ax.text(
            sphere_positions[i, 0],
            sphere_positions[i, 1],
            sphere_positions[i, 2],
            f"{str(i)}",
            color='k',
        )
        sphere_labels.append(slabel)
    return velocity_lines, velocity_text, sphere_labels


def plot_all_angular_velocity_lines(ax, posdata, o_spheres):
    (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
     dumbbell_positions, dumbbell_deltax) = posdata
    angular_velocity_lines = []
    for i in range(sphere_sizes.size):
        if (o_spheres[i, :] != np.array([0, 0, 0])).any():
            line = Arrow3D([sphere_positions[i, 0],
                            sphere_positions[i, 0]+o_spheres[i, 0]],
                           [sphere_positions[i, 1],
                            sphere_positions[i, 1]+o_spheres[i, 1]],
                           [sphere_positions[i, 2],
                            sphere_positions[i, 2]+o_spheres[i, 2]],
                           mutation_scale=20, lw=1, arrowstyle="-|>",
                           edgecolor="g", facecolor="w")
            ax.add_artist(line)
            angular_velocity_lines.append(line)
    return angular_velocity_lines


def from0topi(n):
    if n < 0:
        return n + np.pi
    else:
        return n


def sign(n):
    if n > 0:
        return 1
    elif n == 0:
        return 0
    else:
        return -1


def normal(ax):
    theta = from0topi(ax.elev*np.pi/180)
    phi = ax.azim * np.pi/180
    return np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)
