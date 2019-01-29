import numpy as np
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
from ewald_functions.DDDDLap_rerfclr import DDDDLap_rerfclr
from ewald_functions.DDLapLap_rerfclr import DDLapLap_rerfclr

def DLap_K(np.ndarray r, double ss, int l, int i, int j, int k, tuple erfcs):
    return (i==j)*DDLapLap_rerfclr(r[k],r[l],ss,erfcs[2],erfcs[3],erfcs[4],erfcs[5],erfcs[6],k,l) - DDDDLap_rerfclr(r[i],r[j],r[k],r[l],ss,erfcs[0],erfcs[1],erfcs[2],erfcs[3],erfcs[4],erfcs[5],erfcs[6],i,j,k,l)
