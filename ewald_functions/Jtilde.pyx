cpdef Jtilde(double ki, double kj, double ss, int i, int j, double RR):
  return -((i==j)*ss**2 + ki*kj)*RR
