cpdef L6(int i, int j, int k, int l, int m, double di, double dj, double dl, double dm):
    return (i-k)*(k-l)*(l-i)/2*dl*dj + (j-k)*(k-m)*(m-j)/2*dm*di
