from ewald_functions.Jtilde import Jtilde

cpdef DLap_Ktilde(double kkl, double kki, double kkj, double kkk, double ss, int l, int i, int j, int k, double RR):
    return 0.5 * ( kkk*kkl*ss**2 * Jtilde(kki,kkj,ss,i,j,RR) + kkl*kkj*ss**2 * Jtilde(kki,kkk,ss,i,k,RR) )
