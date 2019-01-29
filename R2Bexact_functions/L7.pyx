cpdef L7(int i, int j, int k, int l, double di, double dj, double dk, double dl):
  return 1.5*(di*dj - (i==j)/3.)*(dk*dl - (k==l)/3.)
