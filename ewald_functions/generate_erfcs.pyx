from math import erfc, pi, exp

def generate_erfcs(double s, double lamb):
  cdef double E, erfc0, erfc1, erfc2, erfc3, erfc4, erfc5, erfc6
  E = 2/pi**0.5*exp(-s**2*lamb**2)*lamb
  erfc0 = erfc(lamb*s)
  erfc1 = -E
  erfc2 = 2*lamb**2*s*E
  erfc3 = -2*lamb**2*E*(2*lamb**2*s**2-1)
  erfc4 = 4*lamb**4*s*E*(2*lamb**2*s**2-3)
  erfc5 = -4*lamb**4*E*(4*lamb**4*s**4-12*lamb**2*s**2+3)
  erfc6 = 8*lamb**6*s*E*(4*lamb**4*s**4-20*lamb**2*s**2+15)
  return erfc0,erfc1,erfc2,erfc3,erfc4,erfc5,erfc6
