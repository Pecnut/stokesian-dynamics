# Stokes flow past two spheres

Two-sphere version of the Fortran code associated with 
* [Wilson, 2013](http://www.ucl.ac.uk/~ucahhwi/publist/papers/2013-W.pdf). Stokes flow past three spheres, *Journal of Computational Physics* **245**, 302–316.

For use in finding resistance scalars, compile the code into an executable called **lamb.exe**. To do this, run `gfortran 2sphere.f base.f reflect.f -o lamb.exe` from this directory, where `gfortran` is the name of your version of Fortran. Another common alternative is `g95`. (Mac/Linux users: no need to be suspicious of the `.exe` ending; this will still work on your system.)

For full instructions, see section 3 in the README of the parent directory. 

I believe there is a small bug in the original version of this code which means that the mobility scalars xg come out with the wrong sign. This behaviour has been fixed here by introducing the parameter `xg_sign`, which flips the sign of a number of inputs and outputs in `2sphere.f`. However, as I am not the author of the original paper and am not familiar with the method, I cannot say whether this is a permanent fix or whether it only fixes the cases we are interested in here. To remove this bugfix, let `xg_sign = 1` in `parameters.f` instead of `xg_sign = -1`.