cpdef L9(int i, int j, int k, int l, double di, double dj, double dk, double dl):
  return 0.5*((i==k)*(j==l) + (j==k)*(i==l) - (i==j)*(k==l) + di*dj*(k==l) + (i==j)*dk*dl - di*(j==l)*dk - dj*(i==l)*dk - di*(j==k)*dl - dj*(i==k)*dl + di*dj*dk*dl)
