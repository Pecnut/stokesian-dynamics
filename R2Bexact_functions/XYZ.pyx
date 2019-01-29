import numpy as np
cimport numpy as np
from inputs import s_dash_range, range_s_dash_range, XYZ_raw

cpdef XYZ(int scalar_index, int gamma, double s_dash, int lam_index):
  cdef double[:] interp_y = XYZ_raw[scalar_index, gamma, range_s_dash_range, lam_index]
  return np.interp(s_dash,s_dash_range,interp_y,left=XYZ_raw[scalar_index,gamma,0,lam_index],right=0)
