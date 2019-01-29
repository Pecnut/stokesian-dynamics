def DDLapLap_rerfclr(double rk, double rl, double ss, double erfc2, double erfc3, double erfc4, double erfc5, double erfc6, int k, int l):
   cdef double a
   a = -12*(k==l)/ss**3 + 36*rk*rl/ss**5
   return                         a    *erfc2 \
        + (                      -a)*ss*erfc3 \
        + (3*rk*rl/ss**3 + 9*(k==l)/ss)*erfc4 \
        + (9*rk*rl/ss**2 + (k==l)     )*erfc5 \
        +    rk*rl/ss                  *erfc6
