import numpy as np
import matplotlib.pyplot as plt

'''
Example (b) 

Fig. 5 of Durlofsky et al. (1987) (non-periodic)
https://doi.org/10.1017/S002211208700171X

This test case considers three horizontally-aligned particles sedimenting 
vertically, and looks at their interesting paths over a large number of 
timesteps. Use a small timestep - with a timestep of 0.25 you need 50000
frames - and set
    invert_m_every = 1
in `inputs.py` (instead of the default of 10), in order to recover the same
particle paths. Set 
    config.DISABLE_JIT = False
in `inputs.py` for speed.

This system is very sensitive to initial conditions (see
Jánosi et al. 1997, https://doi.org/10.1103/PhysRevE.56.2858) and is also
sensitive to the interpolation in R2Bexact, so you can expect small deviation
towards the bottom of the graph.

The timestep here can be Euler or RK4.

Run
    python run_simulation.py 2 1 0.25 50000 fte

'''

# Specify the name of the output file created from the above simulation
output_filename = 'name_of_output_file_without_extension'

data1 = np.load("../output/" + output_filename + ".npz")
particle_positions = data1['centres']


plt.plot(particle_positions[:,:,0],particle_positions[:,:,2])

durlofsky_fig5_data = np.genfromtxt("data/durlofsky_fig5_data.txt",delimiter=",")
plt.plot(durlofsky_fig5_data[:,0],durlofsky_fig5_data[:,1],'.',color='gray',ms=2,zorder=0)

plt.legend(['Particle 1','Particle 2','Particle 3','Scraped from paper'])
plt.xlabel('x')
plt.ylabel('y',rotation=0)
plt.xlim([-6,13])
plt.ylim([-810,10])
plt.show()
