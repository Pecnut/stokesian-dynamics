# Code for generating lubrication resistance functions: Readme #

This code forms part of a [Python 3 implementation of Stokesian Dynamics](http://github.com/Pecnut/stokesian-dynamics) for spherical particles of different sizes, but can be run on its own. To run it alone, it requires **Python** and **Fortran**.

**If you are here from the Python Stokesian Dynamics implementation,** you do not need to run these scripts if you are happy with particle ratios of 1, 0.1 and 0.01. Data files for these ratios are already computed. If, however, you wish to change or augment these ratios, please proceed.

This code generates lubrication resistance functions for two unequal rigid particles which are both close (near-field) and quite close (mid-field). These functions are commonly notated *X*<sub>11</sub><sup>*A*</sup>, *X*<sub>12</sub><sup>*A*</sup>, *Y*<sub>11</sub><sup>*A*</sup>, etc.

 The near-field resistance functions were first given analytically in [Jeffrey & Onishi (1984)](https://doi.org/10.1017/S0022112084000355) and [Jeffrey (1992)](https://doi.org/10.1063/1.858494). The expressions in these articles have been completely corrected in [Townsend (2018)](https://arxiv.org/abs/1802.08226), and these corrected expressions are used here. The mid-field resistance functions are computed using a version of Lamb's method described in [Wilson (2013)](http://www.ucl.ac.uk/~ucahhwi/publist/papers/2013-W.pdf).

## 0. Contact details and how to contribute <a name="s0"></a> ##

* The code is written by Adam Townsend ([adamtownsend.com](http://adamtownsend.com/), [@Pecnut](https://twitter.com/pecnut)). The Fortran files in `helen_fortran` are by Helen Wilson ([ucl.ac.uk/~ucahhwi](https://www.ucl.ac.uk/~ucahhwi/), [@profhelenwilson](https://twitter.com/profhelenwilson)).
* Feel free to [post in the discussion forum of the main repository](https://github.com/Pecnut/stokesian-dynamics/discussions) with questions or ideas if you are happy to comment publicly.
* You can also [create a new issue in the main repository](https://github.com/Pecnut/stokesian-dynamics/issues) in the GitHub repository if you want to report a bug.

## 1. Notes ##

The resistance scalars are computed by two different methods. For two spheres of radius *a*<sub>1</sub> and *a*<sub>2</sub>, a distance *s* apart, we define the non-dimensional distance apart *s*' as *s*' = 2*s*/(*a*<sub>1</sub> + *a*<sub>2</sub>) and size ratio *λ* as *λ* = *a*<sub>2</sub>/*a*<sub>1</sub>.

We use a mid-field method and a near-field method. The mid-field method works for *s*' ≳ 2.01386.

The near-field method works for:

| *λ*         | *s*'         |
| ----------- | ------------ |
|       1     |      < 2.01  |
|       0.1   |      < 2.001 |
|       0.01  |      < 2.001 |

This is because the near-field equations are only good for *s*' – 2 ≪ *λ*.

For values of *s*' between 2.001 and 2.01, in the Stokesian Dynamics code, we generate near-field scalars up to 2.001, and then interpolate between 2.001 and 2.01. This gives much better data than trying to get the ‘arbitrary values’ expressions in Jeffrey & Onishi (1984) to give you something that is smooth between the upper end of the near-field and the lower end of the mid-field. Since we’re already interpolating between data points anyway, this is neat and easy.

## 2. Inputs and outputs

**If you are running this as part of the Stokesian Dynamics implementation,** the only input you may wish to change here is the list of values of *λ*. After changing this, you can simply follow the instructions in section 3 with no further regard to any of the inputs and outputs of the scripts.

**Inputs:**

* Values of *λ* should be placed in **values_of_lambda.txt**. Note that only *λ* ≤ 1 need to be entered; resistance scalars for the reciprocals are automatically generated.

* Values of *s*' which correspond to the near-field, and therefore will need their resistance scalars computed using the near-field method, should be listed in **values_of_s_dash_nearfield.txt**.

* Values corresponding to the midfield should be listed in **values_of_s_dash_midfield.txt**

**Outputs:**

(Some highlights of the outputs only)

* **scalars_general_resistance.txt**: Resistance scalars for different values of *s'* and *λ*. Scalars commonly labelled '11' are in the row where `gamma` = 0; scalars commonly labelled '12' are in the row where `gamma` = 1. Scalars corresponding to '21' and '22' can be found as '12' and '11' respectively for the reciprocal value of *λ*.

* **scalars_general_resistance.npy**: Numpy NPY data file of the above table.

* **scalars_general_resistance_d.txt**: Resistance scalars with the two-body long-range mobility interaction subtracted off.

* **scalars_general_resistance_d.npy**: Numpy NPY data file of the above table: this is the data file used by the Stokesian Dynamics code by default.

* **scalars_pairs_resistance_midfield.txt**: More verbose version of **scalars_general_resistance.txt** for midfield values of *s*' only.

* **scalars_pairs_mobility_midfield.txt**: Verbose list of the mobility scalars x11a, x12a etc. for midfield values of *s*' only.

## 3. Method ##

1.  Compile the Fortran code in **/helen_fortran/** into an executable called **/helen_fortran/lamb.exe**. To do this, enter the **/helen_fortran/** folder and run `gfortran 2sphere.f base.f reflect.f -o lamb.exe`, where `gfortran` is the name of your version of Fortran. Another common alternative is `g95`. *You can safely ignore warnings of the form 'Array reference at (1) out of bounds (0 < 1) in loop beginning at (2)'.* (Mac/Linux users: no need to be suspicious of the `.exe` ending; this will still work on your system.)

2.  Run **generate_midfield_scalars.py** to generate scalars in the mid-field (*s*' from **values_of_s_dash_midfield.txt**) using code from Wilson (2013). This code will work for *s*’ as low as about 2.01386 but no less. Running this takes about an hour on my laptop. This generates all the files ending in **\_midfield**.

3.  Run **generate_nearfield_scalars.py** to generate scalars in the near-field (*s*' from **values_of_s_dash_nearfield.txt**). This takes about five minutes on my laptop. This generates both files with **\_nearfield\_** in them.

4.  Run **combine_resistance_scalars.py** to combine these two results. This is quick!

5.  Run **../convert_resistance_scalars_to_d_form.py** to convert these results into a form where the mobility interaction is already subtracted off (for unclear reasons called 'd' form). This is also quick!

## 4. References ##

* [Townsend, 2018](https://arxiv.org/abs/1802.08226). Generating, from scratch, the near-field asymptotic forms of scalar resistance functions for two unequal rigid spheres in low-Reynolds-number flow, *arXiv:1802.08226 [physics.flu-dyn]*
* [Wilson, 2013](http://www.ucl.ac.uk/~ucahhwi/publist/papers/2013-W.pdf). Stokes flow past three spheres, *Journal of Computational Physics* **245**, 302–316.
