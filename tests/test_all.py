import numpy as np
import sys
sys.path.append("stokesian_dynamics")  # Allows import from SD directory
from functions.timestepping import generate_output_FTSUOE
from settings import cutoff_factor, printout, timestep
from setups.tests.positions import pos_setup_tests
from functions.simulation_tools import condense, uncondense


def fte_mobility_solve_nonperiodic(setup_number, input_number):
    frameno = 0
    regenerate_Minfinity = True
    last_velocities = []
    last_generated_Minfinity_inverse = []
    last_velocity_vector = []

    posdata, setup_description = pos_setup_tests(setup_number)

    (Fa_out, Ta_out, Sa_out, Fb_out, DFb_out,
     Ua_out, Oa_out, Ea_out, Ub_out, HalfDUb_out,
     last_generated_Minfinity_inverse, gen_times,
     U_infinity, O_infinity, centre_of_background_flow,
     Ot_infinity, Et_infinity,
     force_on_wall_due_to_dumbbells, last_velocity_vector
     ) = generate_output_FTSUOE(
        posdata, frameno, timestep, input_number,
        last_generated_Minfinity_inverse, regenerate_Minfinity, 'fte',
        cutoff_factor, printout, False, False,
        False, last_velocities, last_velocity_vector, feed_every_n_timesteps=0)

    return (Ua_out, Oa_out, Sa_out)


def uos_compare(line, Ua, Oa, Sa):
    is_wrong = 0
    if not np.all(np.isclose(line[0:6], Ua.flatten(), rtol=1e-3)):
        is_wrong += 1
        print("Velocity output from SD code is not close enough to output",
              "from Lamb code.")
        print("   From Lamb: U1,U2=", line[0:6])
        print("   SD code:   U1,U2=", list(Ua.flatten()))
    if not np.all(np.isclose(line[6:12], Oa.flatten(), rtol=1e-3)):
        is_wrong += 1
        print("Angular velocity output from SD code is not close enough to",
              "output from Lamb code.")
        print("   From Lamb: O1,O2=", line[6:12])
        print("   SD code:   O1,O2=", list(Oa.flatten()))
    if not np.all(np.isclose(line[12:22], np.array(condense(Sa, 2)).flatten(),
                             rtol=1e-3)):
        is_wrong += 1
        print("Stresslet output from SD code is not close enough to output",
              "from Lamb code.")
        print("   From Lamb: S1,S2=")
        print(uncondense(np.reshape(line[12:22], (2, 5)), 2))
        print("   SD code:   S1,S2=")
        print(Sa)
    return is_wrong


def test_against_lamb():
    print("Test output of SD code against output from Lamb code.")
    setup_numbers = range(-1, -9, -1)
    input_numbers = range(-1, -12, -1)
    num_wrong_values = 0
    with open('tests/lambs_solution_compare_with.txt') as f:
        for i, line in enumerate(f.readlines()[1:]):
            line = line.split()
            setup_number = setup_numbers[i // 11]
            input_number = input_numbers[i % 11]
            Ua, Oa, Sa = fte_mobility_solve_nonperiodic(
                setup_number, input_number
            )
            line_floats = [float(i) for i in line[4:]]
            print("Test no.", i)
            print("a1 = " + line[0] + ", " +
                  "a2 = " + line[1] + ", " +
                  "s = " + line[2] + ", " +
                  "test: " + line[3])
            num_wrong_values += uos_compare(line_floats, Ua, Oa, Sa)
            print()
    assert num_wrong_values == 0
