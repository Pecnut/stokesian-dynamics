def DLapLap_rerfclr(double rk, double ss, double erfc2, double erfc3, double erfc4, double erfc5):
   return -12*rk/ss**3*erfc2 \
        +  12*rk/ss**2*erfc3 \
        +  9*rk/ss*erfc4 \
        + rk*erfc5
