cpdef L8(int i, int j, int k, int l, double di, double dj, double dk, double dl):
  return 0.5*(di*(j==l)*dk + dj*(i==l)*dk + di*(j==k)*dl + dj*(i==k)*dl - 4*di*dj*dk*dl)
