cpdef Lap_J(double ri, double rj, int i, int j, double ss):
  return 2*(i==j)/ss**3 - 6*ri*rj/ss**5

#  def Lap_J(r,ss,i,j):
#      return 2*kronmatrix[i][j]/ss**3 - 6*r[i]*r[j]/ss**5
