from ewald_functions.Jtilde import Jtilde

cpdef DD_Jtilde(double kkm, double kkl, double kki, double kkj, double ss, int m, int l, int i, int j, double RR):
  return -kkm*kkl * Jtilde(kki,kkj,ss,i,j,RR)
