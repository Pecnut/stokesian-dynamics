import numpy as np
import matplotlib.pyplot as plt

'''
Example (a) 

Fig. 1 of Durlofsky et al. (1987) (non-periodic)
https://doi.org/10.1017/S002211208700171X

This test case looks at horizontal chains of 5, 9 and 15 spheres sedimenting 
vertically. The instantaneous drag coefficient, 
    λ = F/(6πμaU),
is measured for each sphere in the chain, in each case. Here we set up the 
chain of length 15. Running for 1, reading the velocity  U  and calculating
λ reproduces this graph.

The timestep here is assumed to be an Euler timestep. Make sure
    timestepping_scheme = 'euler'
in `inputs.py`.

Run
    python run_simulation.py 1 1 1 1 fte
(position setup number 1, forces input 1, with a timestep of 1 [arbitrary
 choice] for 1 timestep, specifying forces, torques and rate of strain).
'''

# Specify the name of the output file created from the above simulation
output_filename = 'name_of_output_file_without_extension'

data1 = np.load("../output/" + output_filename + ".npz")
forces_on_particles = data1['Fa'][0] # At timestep 0
number_of_particles = forces_on_particles.shape[0]
sphere_size = 1
timestep_size = 1

# Euler timestep reversed. Speeds aren't saved by default so this suffices.
speeds_of_particles = (data1['centres'][1]-data1['centres'][0])/timestep_size 

# Pick out the z components of all spheres
F = forces_on_particles[:,2]
U = speeds_of_particles[:,2]

drag_coefficient = F/(6*np.pi*sphere_size*U)
N = (number_of_particles+1)//2

durlofsky_data = [0.5018, 0.5029, 0.5054, 0.5102,
                  0.5183, 0.5321, 0.5559, 0.6170] # Extracted from the paper

plt.plot(range(0,N),drag_coefficient[N-1:])
plt.plot(range(0,N),durlofsky_data,'x')
plt.legend(['From code','Scraped from paper'])
plt.xlabel('Sphere number')
plt.ylabel('λ',rotation=0)
plt.show()
