import numpy as np

values_of_s_dash_nearfield = np.loadtxt('values_of_s_dash_nearfield.txt', ndmin=1)
values_of_s_dash_midfield = np.loadtxt('values_of_s_dash_midfield.txt', ndmin=1)
values_of_s_dash = np.concatenate((values_of_s_dash_nearfield, values_of_s_dash_midfield))
values_of_s_dash.sort()
np.savetxt('values_of_s_dash.txt', values_of_s_dash, fmt="% .8e")
lam_range = np.loadtxt('values_of_lambda.txt')
s_dash_range = values_of_s_dash
if lam_range.shape == tuple():
    lam_length = 1
    lam_range = np.array([lam_range + 0])
else:
    lam_length = lam_range.shape[0]
if s_dash_range.shape == tuple():
    s_dash_length = 1
    s_dash_range = np.array([s_dash_range + 0])
else:
    s_dash_length = s_dash_range.shape[0]

with open('scalars_general_resistance_blob_midfield.txt', 'rb') as inputfile:
    XYZ_mid_raw = np.load(inputfile)

with open('scalars_general_resistance_text_nearfield_mathematica_for_python.txt', 'rb') as inputfile:
    mathematica = inputfile.read()
    # square_brackets = mathematica.replace("{","[").replace("}","]").replace("*^","E").replace("\r\n",",")  # Windows version
    square_brackets = mathematica.replace("{", "[").replace("}", "]").replace("*^", "E").replace("\n", ",")      # Unix/Mac version
    XYZ_near_raw = np.array(eval(square_brackets)).transpose()

print XYZ_mid_raw.shape
print XYZ_near_raw.shape


XYZ_general_table = np.concatenate((XYZ_near_raw, XYZ_mid_raw), axis=2)

general_resistance_scalars_names = np.array(["XA", "YA", "YB", "XC", "YC", "XG", "YG", "YH", "XM", "YM", "ZM"])
general_scalars_length = len(general_resistance_scalars_names)

with open('scalars_general_resistance_blob.txt', 'wb') as outputfile:
    np.save(outputfile, XYZ_general_table)


# Write XYZ_table and xyz_table to file (human readable)
lam_range_with_reciprocals = lam_range
for lam in lam_range:
    if (1. / lam) not in lam_range_with_reciprocals:
        lam_range_with_reciprocals = np.append(lam_range_with_reciprocals, (1. / lam))
lam_range_with_reciprocals.sort()
lam_wr_length = lam_range_with_reciprocals.shape[0]
XYZ_general_human = np.zeros((s_dash_length * lam_wr_length * 2, general_scalars_length + 3))
for s_dash in s_dash_range:
    s_dash_index = np.argwhere(s_dash_range == s_dash)[0, 0]
    s_dash_length = s_dash_range.shape[0]
    lam_length = lam_range.shape[0]
    lam_wr_length = lam_range_with_reciprocals.shape[0]
    for lam in lam_range_with_reciprocals:
        lam_wr_index = np.argwhere(lam_range_with_reciprocals == lam)[0, 0]
        for gam in range(2):
            XYZ_outputline = np.append([s_dash, lam, gam], XYZ_general_table[:, gam, s_dash_index, lam_wr_index])
            XYZ_general_human[(lam_wr_index * s_dash_length + s_dash_index) * 2 + gam, :] = XYZ_outputline


with open('scalars_general_resistance_text.txt', 'a') as outputfile:
    heading = "Resistance scalars, combined " + time.strftime("%d/%m/%Y %H:%M:%S") + "."
    np.savetxt(outputfile, np.array([heading]), fmt="%s")
    np.savetxt(outputfile, np.append(["s'", "lambda", "gamma"], general_resistance_scalars_names), newline=" ", fmt="%15s")
    outputfile.write("\n")
    np.savetxt(outputfile, XYZ_general_human, newline="\n", fmt="% .8e")
