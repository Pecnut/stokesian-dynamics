cpdef L5(int i, int j, int k, double di, double dj, double dk):
  return (di*(j==k) + dj*(i==k) - 2*di*dj*dk)
