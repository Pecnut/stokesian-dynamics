def DDDDLap_rerfclr(double ri, double rj, double rk, double rl, double ss, double erfc0, double erfc1, double erfc2, double erfc3, double erfc4, double erfc5, double erfc6, int i, int j, int k, int l):
    cdef double kron_rr, rirjrlrm, a
    cdef int kron3
    kron_rr = (i==j)*rk*rl + (i==k)*rj*rl + (k==j)*ri*rl + (i==l)*rj*rk + (j==l)*ri*rk + (k==l)*ri*rj
    kron3 = (i==j)*(k==l) + (j==k)*(i==l) + (i==k)*(j==l)
    rirjrkrl = ri*rj*rk*rl
    a = 6*kron3/ss**5 - 30*kron_rr/ss**7 + 210*rirjrkrl/ss**9
    return                                                       a*erfc0 \
         + (                                                -a*ss)*erfc1 \
         + (-3*kron3/ss**3 +  3*kron_rr/ss**5 + 15*rirjrkrl/ss**7)*erfc2 \
         + ( 5*kron3/ss**2 - 13*kron_rr/ss**4 + 55*rirjrkrl/ss**6)*erfc3 \
         + (   kron3/ss    +  4*kron_rr/ss**3 - 25*rirjrkrl/ss**5)*erfc4 \
         + (                    kron_rr/ss**2 +  2*rirjrkrl/ss**4)*erfc5 \
         + (                                       rirjrkrl/ss**3)*erfc6
