def levi(int i, int j, int k):
  if i==j or j==k or k==i:
      return 0
  else:
    return (i-j)*(j-k)*(k-i)/2
