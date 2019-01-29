import numpy as np
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
from ewald_functions.DDDD_rerfclr import DDDD_rerfclr
from ewald_functions.DDLap_rerfclr import DDLap_rerfclr

def DD_J(np.ndarray r,double ss,int m,int l,int i,int j,tuple erfcs):
    cdef double rl, rm, ri, rj
    rl = r[l]
    rm = r[m]
    ri = r[i]
    rj = r[j]
    return (i==j)*DDLap_rerfclr(rl,rm,ss,erfcs[0],erfcs[1],erfcs[2],erfcs[3],erfcs[4],l,m) - DDDD_rerfclr(ri,rj,rl,rm,ss,erfcs[0],erfcs[1],erfcs[2],erfcs[3],erfcs[4],i,j,l,m)
