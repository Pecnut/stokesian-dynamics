#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com

"""Solve FTE non-periodic mobility problem using Lamb's solution for a few
different scenarios and place the results in a text file for us to compare our
Stokesian Dynamics results against. Lamb's solution works in the mid-field.

Requires the Fortran code in ../find_resistance_scalars/helen_fortran to be
compiled. See README in ../find_resistance_scalars/helen_fortran/.

Writes:
    lambs_solution_compare_with.txt.

This text file has already been generated for a number of scenarios used by
pytest to verify the results of the simulations. This script does not need to
be executed if you are happy with the existing test cases.
"""
import subprocess
import sys
import numpy as np
import csv


def run_fortran(s_dash, a1, a2, case):
    """Run Helen's two-sphere Fortran code directly.

    The Fortran code solves the dimensional Kim & Karilla mobility problem,
        (U O S) = M (F T E),
    for given F, T and E, and sphere separation and size ratio. Works for
    a2/a1 <=1  and  s_dash < ~2.014.

    Args:
        s_dash: Scaled separation distance s'.
        a1: Radius of first particle
        a2: Radius of second particle.
        case: A code, "F11", "F12" etc., which is understood in `2sphere.f` to 
            represent, e.g., "a force of 1 parallel to the separation vector".

    Returns:
        1D array (U O S) in dimensional form (i.e. U ~ 1/(6 pi a) F).
    """
    s = 0.5 * (a1 + a2) * s_dash  # Dimensionalise the distance
    print("  using s = " + str(s) + " and case = " + case)
    if sys.platform == "win32":
        fortran_code_directory = "helen_fortran\\"
        starter = ""
    else:
        fortran_code_directory = "../find_resistance_scalars/helen_fortran/"
        starter = "./"
    command_to_run = (starter + "lamb.exe " + str(s) + " " + str(r1) + " "
                      + str(r2) + " " + case + " short")
    output_text = subprocess.check_output(
        fortran_code_directory + command_to_run, shell=True)
    return output_text.split()


# Reorder output
rp = 0.5 * (1 + np.sqrt(3))
rm = 0.5 * (-1 + np.sqrt(3))
r2 = np.sqrt(2)
reorder = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, rp, rm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, r2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, rm, rp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, r2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, rp, rm, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, rm, rp, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r2]])

s_dash_range = [2.1, 3.9]
r_range = [(1, 1), (1, 0.1), (0.1, 1), (0.1, 0.1)]
cases = ["F11", "F12", "F21", "F22",
         "T11", "T12", "T21", "T22",
         "E1 ", "E3 ", "E4 "]

data = []
for rs in r_range:
    r1, r2 = rs
    print("r1,r2=" + str(rs))
    for s_dash in s_dash_range:
        for case in cases:
            # Run Helen's Fortran code
            UOS_output_text = np.array(run_fortran(s_dash, r1, r2, case))
            UOS_output = UOS_output_text.astype(float)
            reordered_UOS = np.dot(reorder, UOS_output.transpose()).transpose()
            data.append([r1, r2, s_dash, case] + list(reordered_UOS))

heading = "  r1   r2   s' cas " + " ".join(["%13s" % i for i in [
                                            "U1x", "U1y", "U1z",
                                            "U2x", "U2y", "U2z",
                                            "Omega1x", "Omega1y", "Omega1z",
                                            "Omega2x", "Omega2y", "Omega2z",
                                            "S11", "S12", "S13", "S14", "S15",
                                            "S21", "S22", "S23", "S24", "S25"
                                            ]])

with open('lambs_solution_compare_with.txt', 'w') as file_:
    file_.write(heading + "\n")
    for row in data:
        formatted_row = []
        for i in range(3):
            formatted_row.append("%.2f" % row[i])
        formatted_row.append(row[3])
        for i in range(4, len(row)):
            formatted_row.append("% .6e" % row[i])
        file_.write(" ".join(formatted_row) + "\n")
