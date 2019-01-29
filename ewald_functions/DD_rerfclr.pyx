def DD_rerfclr(double ri, double rj, double ss, double erfc0, double erfc1, double erfc2, int i, int j):
    return ((i==j)/ss - ri*rj/ss**3)*erfc0 \
         + ((i==j) + ri*rj/ss**2)*erfc1 \
         + ri*rj/ss*erfc2
