cpdef J(double ri, double rj, int i, int j, double ss):
  return (i==j)/ss + ri*rj/ss**3

#  def J(r,ss,i,j):
#      return kronmatrix[i][j]/ss + r[i]*r[j]/ss**3
