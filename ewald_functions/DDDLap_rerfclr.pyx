def DDDLap_rerfclr(double ri, double rj, double rk, double ss, double erfc0, double erfc1, double erfc2, double erfc3, double erfc4, double erfc5, int i, int j, int k):
   cdef double kron_r, rirjrk, a
   kron_r = (i==j)*rk + (i==k)*rj + (j==k)*ri
   a = 6*kron_r/ss**5 - 30*ri*rj*rk/ss**7
   rirjrk = ri*rj*rk
   return a*erfc0 \
        + (-a*ss)*erfc1 \
        + (-3*kron_r/ss**3 +  3*rirjrk/ss**5)*erfc2 \
        + ( 5*kron_r/ss**2 - 13*rirjrk/ss**4)*erfc3 \
        + (   kron_r/ss    +  4*rirjrk/ss**3)*erfc4 \
                           +    rirjrk/ss**2 *erfc5
