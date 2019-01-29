def DLap_rerfclr(double rl, double ss, double erfc0, double erfc1, double erfc2, double erfc3):
  return -2*rl/ss**3*erfc0 + 2*rl/ss**2*erfc1 + 5*rl/ss*erfc2 + rl*erfc3
