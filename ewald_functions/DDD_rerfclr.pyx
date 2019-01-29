def DDD_rerfclr(double ri, double rj, double rl, double ss, double erfc0, double erfc1, double erfc2, double erfc3, int i, int j, int l):
  cdef double a
  a = -(i==j)*rl - (i==l)*rj - (j==l)*ri + 3*ri*rj*rl/ss**2
  return 1/ss**3*(a)*erfc0 \
      + 1/ss**2*(-a)*erfc1 \
      + 1/ss*((i==j)*rl + (i==l)*rj + (j==l)*ri)*erfc2 \
      + ri*rj*rl/ss**2*erfc3
