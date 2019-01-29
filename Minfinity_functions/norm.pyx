cimport numpy as np
cpdef norm(np.ndarray[double, ndim=1] x):
  return (x[0]**2 + x[1]**2 + x[2]**2)**0.5
