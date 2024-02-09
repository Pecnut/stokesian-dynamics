# Stokesian Dynamics settings documentation

Before reading this documentation, make sure you have read the repository [README](../README.md).

## Introduction
The **settings.py** file contains various configuration settings for the Stokesian Dynamics simulation software in the **stokesian_dynamics** folder.

## Basic settings

### `setup_number`
- Type: `int`
- Default value: `1`
- Description: References which initial particle position setup to use, which includes the number of particles, the size of the particles, and the the type of the particle (sphere or dumbbell). The setup number corresponds to a specific particle configuration in **setups/positions.py**. This file can be edited to add new setups.

This can be overridden from the command line by passing the setup number as the first argument.

### `input_number`
- Type: `int`
- Default value: `1`
- Description: References which forces and/or velocities to impose on the particles (the right-hand side of the mobility problem). The input number corresponds to a specific force and velocity configuration in **setups/inputs.py**. This file can be edited to add new force/velocity setups.

This can be overridden from the command line by passing the input number as the second argument.

### `fully_2d_problem`
- Type: `bool`
- Default value: `False`
- Description: Specifies whether the simulation is a fully 2D problem. If set to `True`, the simulation will always set the value of the particles' *y*-coordinates to 0.

### `timestep`
- Type: `float`
- Default value: `0.1`
- Description: Specifies the timestep size for the simulation. Smaller values result in more accurate simulations but can increase computational cost if you need more of them.

This can be overridden from the command line by passing the timestep as the third argument.

### `num_frames`
- Type: `int`
- Default value: `100`
- Description: Specifies the number of frames to run the simulation for. This value determines the duration of the simulation.

This can be overridden from the command line by passing the number of frames as the fourth argument.

### `input_form`
- Type: `str`
- Default value: `'fte'`
- Description: Specifies the type of force and/or velocity to impose on the particles, i.e. what goes on the left and right-hand sides of the mobility problem. The available options are:
  - `'fte'`: 
    - Input: Forces, torques and background rate of strain
    - Output: Velocities, angular velocities and stresslets
  - `'fts'`: 
    - Input: Forces, torques and stresslets
    - Output: Velocities, angular velocities and background rate of strain
  - `'ufte'`: 
    - Input: Velocities of some spheres; forces on the rest of the spheres; torques and background rate of strain for all spheres; forces on the dumbbells
    - Output: Forces on the fixed-velocity spheres; velocities of the free spheres; angular velocities and stresslets for all spheres; velocities of the dumbbells
  - `'ufteu'`: 
    - Input: Velocities of some spheres; forces on the rest of the spheres; torques and background rate of strain for all spheres; velocities of some dumbbells; forces on the rest of the dumbbells
    - Output: Forces on the fixed-velocity spheres; velocities of the free spheres; angular velocities and stresslets for all spheres; forces on the fixed-velocity dumbbells; velocities for the free dumbbells
  - `'duf'`: 
    - Input: (Dumbbells only) Velocities of some dumbbells; forces on the rest of the dumbbells
    - Output: Forces on the fixed-velocity dumbbells; velocities of the free dumbbells

This can be overridden from the command line by passing the input form as the fifth argument.

## Advanced settings

### `invert_m_every`
- Type: `int`
- Default value: `1`
- Description: Specifies how many timesteps to wait in between finding the inverse of Minfinity, the far-field mobility matrix. In between these timesteps, the previously computed value is used. Computing Minfinity can be the most computationally expensive part of the simulation, so setting this value to a higher number can reduce the computational cost.

### `cutoff_factor`
- Type: `float`
- Default value: `2`
- Description: Specifies when to start using R2bexact. The cutoff factor is the ratio of the distance between the particle centres to the sum of the particle radii.

### `timestepping_scheme`
- Type: `str`
- Default value: `'euler'`
- Description: Specifies the timestepping scheme to use. The available options are:
  - `'euler'`: Euler method
  - `'ab2'`: Adams–Bashforth 2nd order method
  - `'rk4'`: Runge–Kutta 4th order method

### `rk4_generate_minfinity_for_each_stage`
- Type: `bool`
- Default value: `True`
- Description: Specifies whether to work out Minfinity for each of the 4 stages when using the Runge–Kutta 4th order method, or to approximate them by just using the Minfinity from the first stage. In the absence of R2Bexact, setting this to `False` makes the timestep equivalent to the Euler method. If R2Bexact is present and `invert_m_every` is greater than 1, choosing `False` might be an OK approximation.

### `feed_every_n_timesteps`
- Type: `int`
- Default value: `0` (off)
- Description: Specifies how often to feed in new particles to a simulation. Useful if the simulation involves large domains where an object of interest moves through a sea of obstacles, and you only need to actually include the obstacles when the object gets near. This will require some configuration in `run_simulation.py`: just search for `feed_every_n_timesteps` in the source code. If set to `0`, feeding is turned off.

### `feed_from_file`
- Type: `str`
- Default value: None
- Description: Specifies the name of the file to feed new particles from. This is only relevant if `feed_every_n_timesteps` is greater than 0.

### `how_far_to_reproduce_gridpoints`
- Type: `int`
- Default value: `2`
- Description: Specifies how many repeats of the box to consider before the contribution decays away for periodic boxes. If the box width is `Lx`, `1` corresponds to `[-Lx, 0, Lx]`, `2` corresponds to `[-2Lx, -Lx, 0, Lx, 2Lx]`, and so on. `2` is normally sufficient, but `3` gives a bit more accuracy.

### `printout`
- Type: `int`
- Default value: `0`
- Description: Specifies the level of output to the screen. The available options are:
  - `0`: Minimal output
  - `1`: Separation distance and velocities
  - `2`: Matrices
  - `3`: Individual calculations

### `email_on_completion`
- Type: `bool`
- Default value: `False`
- Description: Specifies whether to send an email on completion of the simulation. This requires setting up the email settings in **functions/email.py**.

### `view_graphics`
- Type: `bool`
- Default value: `True`
- Description: Specifies whether to view graphics during the simulation. If set to `True`, the simulation will display the particles' positions and velocities in a 3D plot.

### `bead_bead_interactions`
- Type: `bool`
- Default value: `True`
- Description: Specifies whether to include interactions between dumbbell beads in the simulation. This should really always be set to `True`.

### `explosion_protection`
- Type: `bool`
- Default value: `False`
- Description: Specifies whether to include 'explosion protection' in the simulation. Specifically, dumbbells with a separation distance greater than 5 times the default sphere radius (i.e. 5 in simulation units) will stop the execution of the simulation. Typically this would suggest something has gone wrong.

### `save_positions_every_n_timesteps`
- Type: `int`
- Default value: `1`
- Description: Specifies how often to save the particles' positions to a file. If set to `1`, the positions will be saved every timestep.

### `save_forces_every_n_timesteps`
- Type: `int`
- Default value: `1`
- Description: Specifies how often to save the forces on the particles to a file. If set to `1`, the forces will be saved every timestep.

### `save_forces_and_positions_to_temp_file_as_well`
- Type: `bool`
- Default value: `True`
- Description: Specifies whether to save the forces and positions to a temporary file as well as to the main output file. This is useful if the simulation is expected to run for a long time and you want to make a running backup in case the power fails, allowing you to restart the simulation from the last saved state.

### `save_to_temp_file_every_n_timesteps`
- Type: `int`
- Default value: `120`
- Description: Specifies how often to save the forces and positions to a temporary file. If set to `120`, the forces and positions will be saved at least every 120 timesteps, if `save_positions_every_n_timesteps` is not smaller.

### `extract_force_on_wall_due_to_dumbbells`
- Type: `bool`
- Default value: `False`
- Description: Specifies whether to extract the total force on any fixed particles (typically useful when fixed particles are representing a wall) due to the dumbbells. If set to `True`, the force on the wall due to the dumbbells will be saved to the output NPZ file as `force_on_wall_due_to_dumbbells`.

### `use_drag_Minfinity`
- Type: `bool`
- Default value: `False`
- Description: Specifies whether to use the drag Minfinity, i.e. diagonal self-terms only. This might a good approximation for dense suspensions only.

### `use_Minfinity_only`
- Type: `bool`
- Default value: `False`
- Description: Specifies whether to use Minfinity only, i.e. turn off R2Bexact. This is mostly put in for a little discussion.

### `use_drag_R2Binfinity`
- Type: `bool`
- Default value: `False`
- Description: Only relevant if `use_drag_Minfinity = True`. The precomputed XYZ scalars in the resistance matrix already have R2Binfinity subtracted off them (it's quicker). These data files are labelled with '-d'. If `use_drag_Minfinity` is set to `True`, there is a choice of using '-d' scalars where the drag-only R2Binfinity is subtracted, or using '-d' scalars where the full R2Binfinity is subtracted. You might pick the drag version if you wanted to make sure that the two-sphere case is consistent, but you might pick the full version if you have a dense suspension.

### `config.DISABLE_JIT`
- Type: `bool`
- Default value: `True` (i.e. Numba is disabled)
- Description: Specifies whether to turn Numba on (`False`) or off (`True`).

## Graphical settings

These are only relevant if `view_graphics` is set to `True`.

### `viewbox_bottomleft_topright`
- Type: `np.ndarray`
- Default value: `np.array([[-15, 0, -15], [15, 1, 15]])`
- Description: Specifies the bottom-left and top-right corners of the viewbox for the 3D plot. The viewbox is the region of space that is visible in the 3D plot.

### `viewing_angle`
- Type: `tuple`
- Default value: `(0, -90)`
- Description: Specifies the viewing angle for the 3D plot. The first value is the elevation angle and the second value is the azimuthal angle. `(0,-90)` represents the x-z plane; `(30,-60)` is the matplotlib default.

### `view_labels`
- Type: `bool`
- Default value: `False`
- Description: Specifies whether to view arrows on the spheres in the 3D plot representing forces. If set to `True`, arrows will be displayed.

### `trace_paths`
- Type: `int`
- Default value: `0` (off)
- Description: Specifies whether to trace the paths of the particles in the 3D plot. If set to `n` > 0, a line will be drawn between every `n` timesteps.

### `two_d_plot`
- Type: `bool`
- Default value: `True`
- Description: Specifies whether to remove the third axis in the plot and display as a 2D plot instead.


