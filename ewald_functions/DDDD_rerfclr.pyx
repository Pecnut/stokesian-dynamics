#import numpy as np
#cimport numpy as np

def DDDD_rerfclr(double ri, double rj, double rl, double rm, double ss, double erfc0, double erfc1, double erfc2, double erfc3, double erfc4, int i, int j, int l, int m):
    cdef double kron_rr, rirjrlrm, a
    cdef int kron3

    kron_rr = (i==j)*rl*rm + (i==l)*rj*rm + (l==j)*ri*rm + (i==m)*rj*rl + (j==m)*ri*rl + (l==m)*ri*rj
    kron3 = (i==j)*(l==m) + (j==l)*(i==m) + (i==l)*(j==m)
    rirjrlrm = ri*rj*rl*rm

    a = - kron3/ss**3 + 3*kron_rr/ss**5 - 15*rirjrlrm/ss**7

    return (                                                  a)*erfc0 \
         + (                                              -a*ss)*erfc1 \
         + (  kron3/ss                      -  3*rirjrlrm/ss**5)*erfc2 \
         + (                  kron_rr/ss**2 -  2*rirjrlrm/ss**4)*erfc3 \
                                            +    rirjrlrm/ss**3 *erfc4
