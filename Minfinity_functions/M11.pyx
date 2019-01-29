cpdef M11(double ri, double rj, double ss, double a1, double a2, int i, int j, double c, double mu):
    #from Minfinity_functions.J import J
    #from Minfinity_functions.Lap_J import Lap_J
    if ss > 1e-10:
        #return c*(               J(ri,rj,i,j,s)       +     (a1**2 + a2**2)/6. *        Lap_J(ri,rj,i,j,s))
        return c*(          (i==j)/ss + ri*rj/ss**3    +     (a1**2 + a2**2)/6. * (2*(i==j)/ss**3 - 6*ri*rj/ss**5)           )
    else:
        return (i==j)/(6*3.14159*mu*a1)
