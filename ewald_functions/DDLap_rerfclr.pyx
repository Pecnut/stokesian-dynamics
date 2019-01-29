def DDLap_rerfclr(double rl, double rm, double ss, double erfc0, double erfc1, double erfc2, double erfc3, double erfc4, int l, int m):
  cdef double rlrm, a
  cdef int kron_lm
  kron_lm = (l==m)
  rlrm = rl*rm
  a = -2*kron_lm/ss**3 + 6*rlrm/ss**5
  return (                           a)*erfc0 \
       + (                       -a*ss)*erfc1 \
       + (-3*rlrm/ss**3 + 5*kron_lm/ss)*erfc2 \
       + ( 5*rlrm/ss**2 +   kron_lm   )*erfc3 \
       +     rlrm/ss                   *erfc4
