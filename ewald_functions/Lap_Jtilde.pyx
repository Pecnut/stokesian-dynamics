from ewald_functions.Jtilde import Jtilde

cpdef Lap_Jtilde(double kki, double kkj, double ss, int i, int j, double RR):
  return -ss**2 * Jtilde(kki,kkj,ss,i,j,RR)
