import numpy as np
import matplotlib.pyplot as plt

'''
Example (c)

(c) Fig. 9 of Sierou & Brady, 2001 (periodic)
https://doi.org/10.1017/S0022112001005912
which is a correction to
Fig. 1 of Brady et al., 1988
https://doi.org/10.1017/S0022112088002411

A simple cubic array sediments vertically under a constant force. The velocity
is measured for different particle concentrations. Each simulation will produce
one data point, specifying the velocity for a given concentration. Vary the 
concentration by altering the periodic box size and the cubic lattice size.

Change the periodic box size and cubic lattice size by altering the side length
in the argument of 
    simple_cubic_8(side_length)
in BOTH `input_setups.py` and `position_setups.py`. Make sure 
    how_far_to_reproduce_gridpoints ≥ 2 
for accurate results.

The timestep here is assumed to be an Euler timestep. Make sure
    timestepping_scheme = 'euler'
in `inputs.py`.

Run
    python run_simulation.py 3 2 1 1 fte

'''

# Specify the name of the output file created from the above simulation
output_filename = 'name_of_output_file_without_extension'

data1 = np.load("../output/" + output_filename + ".npz")
timestep_size = 1

# Euler timestep reversed. Speeds aren't saved by default so this suffices.
speeds_of_particles = (data1['centres'][1]-data1['centres'][0])/timestep_size 
# Pick out the z components of all spheres
U = speeds_of_particles[:,2]
# All sphere should sediment at the same rate, so pick the first, switch the sign,
# and normalise by Stokes' law for a single sedimenting particle (6 pi)
sedimentation_speed = -U[0]*6*np.pi

side_length = 2*(data1['centres'][0,1,2]-data1['centres'][0,0,2]) # True for 2x2x2 cubic domains
phi = 8*4/3*np.pi/side_length**3

print (phi, sedimentation_speed)

plt.plot([phi],[sedimentation_speed],'X',markersize=10)

# Data from the paper
phi_sb = [0.524427481,
0.449618321,
0.341984733,
0.214503817,
0.124427481,
0.06259542,
0.026717557,
0.001526718,
0]
u_sb = [0.104900728,
0.102376862,
0.111104125,
0.160271692,
0.242719753,
0.35848019,
0.49642408,
0.825050035,
1]

plt.plot(phi_sb,u_sb,'.',markersize=10, linestyle="--",zorder=0)

plt.legend(['Single result from code','Scraped from paper'])

plt.axis([-0.02, 0.6, 0, 1.03])
plt.xlabel('Concentration, $\phi$')#,rotation='horizontal')
plt.ylabel('Sedimentation velocity')
plt.show()
