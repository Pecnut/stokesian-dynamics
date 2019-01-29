from ewald_functions.DD_Jtilde import DD_Jtilde

cpdef D_Ktilde(double kkl, double kki, double kkj, double kkk, double ss, int l, int i, int j, int k, double RR):
    return 0.5*(DD_Jtilde(kkl,kkk,kki,kkj,ss,l,k,i,j,RR) + DD_Jtilde(kkl,kkj,kki,kkk,ss,l,j,i,k,RR))
