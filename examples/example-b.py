import numpy as np
import matplotlib.pyplot as plt

'''
Example (b) 

Fig. 5 of Durlofsky et al. (1987) (non-periodic)
https://doi.org/10.1017/S002211208700171X

This test case considers three horizontally-aligned particles sedimenting 
vertically, and looks at their interesting paths over a large number of 
timesteps. Set
    invert_m_every = 1
in `inputs.py` (instead of the default of 10), in order to recover the same
particle paths. Set 
    config.DISABLE_JIT = False
in `inputs.py` for speed.

Choosing RK4 timestepping with a timestep of 128 for 100 frames is sufficient
to find the solution that this software will converge to. Make sure
    timestepping_scheme = 'rk4'
in `inputs.py`. The solution deviates slightly at the bottom of the graph from
the 1987 paper. This could be a small error in the paper, but really this 
system is (a) very sensitive to initial conditions (see Jánosi et al. 1997, 
https://doi.org/10.1103/PhysRevE.56.2858) and (b) is also sensitive to the 
interpolation in R2Bexact.

Run
    python run_simulation.py 2 1 128 100 fte

Perhaps reflecting the timestepping scheme used in the paper (which is not
stated), we find that you can use AB2 timestepping for 800 frames with a 
timestep of 16 to recover a solution which matches the paper well. Make sure
    timestepping_scheme = 'ab2'
in `inputs.py`. Note that shrinking the timestep converges to the same solution
as in the RK4 solution above.

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
