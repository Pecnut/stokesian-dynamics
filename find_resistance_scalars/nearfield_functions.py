# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 12/07/2023.

"""
Functions file for generate_nearfield_scalars.py. 

Contains functions which generate X11A, X12A, Y11A, etc. 

Reference: Townsend 2018, 'Generating, from scratch, the near-field asymptotic
forms of scalar resistance functions for two unequal rigid spheres in
low-Reynolds-number flow.' arXiv:1802.08226[physics.flu-dyn]

Formulae are based on Jeffrey & Onishi 1984, Jeffrey 1992, and Ichiki et al.
2013 but have been corrected in the document referenced above.
Equation numbers relating to:
    Townsend 2018      are written (1),
    Jeffrey & Onishi      ,,       (J&O 1.1),
    Jeffrey               ,,       (J 1),
    Ichiki et al.         ,,       (I 1).

For two spheres of radius  a_1  and  a_2 , a distance  r  apart, the 
nondimensional particle centre-to-centre separation distance  s'  is defined as 
    s' = 2r / (a_1 + a+2). 
These functions are written in terms of the scaled surface separation distance, 
    xi = s' - 2.

The size ratio is  lambda = a_2/a_1.
    
"""

from math import log, comb
from functools import cache
from scipy.special import zeta


@cache
def kron(i, j):
    """Kronecker delta"""
    return int(i == j)


@cache
def binom(n, k):
    """Binomial coefficient n choose k"""
    return comb(n, k)


@cache
def intermediate_sum(f0, f1, f2, f3, parity, lam, N=100):
    """A form of sum which is found in many functions named e.g. AX12.

    Args:
        f0, f1, f2, f3: Given functions.
        parity: Sum over odd or even values.
        lam: Size ratio lambda.
        N: Max limit of the sum.
    """
    if parity == "odd":
        sumover = range(1, N, 2)
    else:
        sumover = range(2, N+1, 2)
    return sum([
        2**-m*(1+lam)**-m*f0(m, lam)
        - f1(lam)
        - 2*m**-1*f2(lam)
        + 4*m**-1*(m+2)**-1*f3(lam)
        for m in sumover
    ])


@cache
def intermediate_sum_no_f0_coeff(f0, f1, f2, f3, parity, lam, N=100):
    """Version of intermediate_sum but where the coefficient of f0 is instead
    assumed to already be in f0. 
    
    Note that, for example, this coefficient normally includes 2**-m, and yet
    every definition of the 'f' functions used in f0 includes 2**m, so it just
    cancels out. So why include it? This choice appears to have been made in
    the papers because it keeps the coefficients of the f functions integers,
    which makes them easier to type. But from a programming perspective it is
    wasteful, although harmless. More dangerous is keeping (1+lam)**-m on the
    outside, because by bring it inside f0, it can help to stop the terms in f0
    getting too large for double precision, which they do when lambda and N are
    both large. To keep the functions in this script closer to the original
    papers, we maintain the traditional form and only use this function for
    Z11M and Z12M because we need larger N on them.
    """
    if parity == "odd":
        sumover = range(1, N, 2)
    else:
        sumover = range(2, N+1, 2)
    return sum([
        f0(m, lam)
        - f1(lam)
        - 2*m**-1*f2(lam)
        + 4*m**-1*(m+2)**-1*f3(lam)
        for m in sumover
    ])


@cache
def Z(lam):
    """Zero function for use in intermediate_sum"""
    return 0


# XA --------------------------------------------------------------------------

@cache
def Pxa(n, p, q):
    """(5) and (8)"""
    if p == 0 and q == 0:
        return kron(1, n)
    o = sum([binom(n+s, n)
            * ((n*(2*n+1)*(2*n*s-n-s+2)/(2*(n+1)*(2*s-1)*(n+s))
                * Pxa(s, q-s, p-n+1)
                )
                - n*(2*n-1)/(2*(n+1)) * Pxa(s, q-s, p-n-1)
                - n*(4*n**2-1)/(2*(n+1)*(2*s+1)) * Vxa(s, q-s-2, p-n+1)
               )
             for s in range(1, q+1)
             ])
    return o


@cache
def Vxa(n, p, q):
    """(6) and (7)"""
    if p == 0 and q == 0:
        return kron(1, n)
    o = (Pxa(n, p, q)
         - 2*n/((n+1)*(2*n+3))
         * sum([binom(n+s, n)*Pxa(s, q-s, p-n-1) for s in range(1, q+1)])
         )
    return o


@cache
def Fxa(k, lam):
    """(9)"""
    return 2**k * sum([Pxa(1, k-q, q)*lam**q for q in range(0, k+1)])


@cache
def G1xa(lam):
    """(10)"""
    return 2*lam**2*(1+lam)**-3


@cache
def G2xa(lam):
    """(11)"""
    return 1/5*lam*(1+7*lam+lam**2)*(1+lam)**-3


@cache
def G3xa(lam):
    """(12)"""
    return 1/42*(1+18*lam-29*lam**2+18*lam**3+lam**4)*(1+lam)**-3


@cache
def AX11(lam):
    """(14a) / (J&O 3.22) but more conveniently written.

    See note next to (14a). The term `truncation_m1_match` is included to match
    original (using M1) definition (14) when truncating the infinite sum at 
    M. It is just truncation noise but is left in, in case you want to match
    with calculations using the original M1 version, truncated at some M."""
    M = 100
    truncation_m1_match = G3xa(lam)*4/M/(M+2)
    o = (1 - 1/4*G1xa(lam) - G3xa(lam) - truncation_m1_match
         + intermediate_sum(Fxa, G1xa, G2xa, G3xa, 'even', lam, M))
    return o


@cache
def AX12(lam):
    """(15) / (J&O 3.23)"""
    o = -2/(1+lam) * (
        1/4*G1xa(lam)
        + 2*G2xa(lam)*log(2)
        - 2*G3xa(lam)
        + intermediate_sum(Fxa, G1xa, G2xa, G3xa, 'odd', lam, 100)
    )
    return o


@cache
def X11A(xi, lam):
    """(16) / (J&O 3.17)"""
    xiinv = xi**-1
    return (G1xa(lam)*xiinv
            + G2xa(lam)*log(xiinv)
            + AX11(lam)
            + G3xa(lam)*xi*log(xiinv)
            )


@cache
def X12A(xi, lam):
    """(17) / (J&O 3.17) with different scaling"""
    xiinv = xi**-1
    return -(G1xa(lam)*xiinv
             + G2xa(lam)*log(xiinv)
             - 1/2*(1+lam)*AX12(lam)
             + G3xa(lam)*xi*log(xiinv)
             )


# YA --------------------------------------------------------------------------

@cache
def Pya(n, p, q):
    """(J&O 4.6 and 4.10)"""
    if p == 0 and q == 0:
        return kron(1, n)
    o = sum([binom(n+s, n+1)
            * (((2*n+1)/(2*(n+1))
                * (3*(n+s)-(n*s+1)*(2*n*s-s-n+2))/(s*(n+s)*(2*s-1))
                * Pya(s, q-s, p-n+1)
                )
                + n*(2*n-1)/(2*(n+1)) * Pya(s, q-s, p-n-1)
                + n*(4*n**2-1)/(2*(n+1)*(2*s+1)) * Vya(s, q-s-2, p-n+1)
                - 2*(4*n**2-1)/(3*(n+1)) * Qya(s, q-s-1, p-n+1)
               )
             for s in range(1, q+1)
             ])
    return o


@cache
def Qya(n, p, q):
    """(J&O 4.8 and 4.11)"""
    if p == 0 and q == 0:
        return 0
    o = sum([binom(n+s, n+1)
            * (s/(n+1)*Qya(s, q-s-1, p-n)
               - 3/(2*n*s*(n+1))*Pya(s, q-s, p-n)
               )
             for s in range(1, q+1)
             ])
    return o


@cache
def Vya(n, p, q):
    """(J&O 4.7) and (18)"""
    if p == 0 and q == 0:
        return kron(1, n)
    o = (Pya(n, p, q)
         + 2*n/((n+1)*(2*n+3))
         * sum([binom(n+s, n+1)*Pya(s, q-s, p-n-1) for s in range(1, q+1)])
         )
    return o


@cache
def Fya(k, lam):
    """(9) / (J&O 3.15)"""
    return 2**k * sum([Pya(1, k-q, q)*lam**q for q in range(0, k+1)])


@cache
def G2ya(lam):
    """(J&O 4.16.1)"""
    return 4/15*lam*(2+lam+2*lam**2)*(1+lam)**-3


@cache
def G3ya(lam):
    """(J&O 4.16.2)"""
    return 2/375*(16-45*lam+58*lam**2-45*lam**3+16*lam**4)*(1+lam)**-3


@cache
def AY11(lam):
    """(J&O 4.17) but more conveniently written. See docstring on AX11."""
    M = 100
    truncation_m1_match = G3ya(lam)*4/M/(M+2)
    o = (1 - G3ya(lam) - truncation_m1_match
         + intermediate_sum(Fya, Z, G2ya, G3ya, 'even', lam, M))
    return o


@cache
def AY12(lam):
    """(J&O 4.18) but more conveniently written. See docstring on AX11."""
    M = 100
    truncation_m1_match = G3ya(lam)*4/(M-1)/(M+1)
    o = (-2/(1+lam) *
         (2*G2ya(lam)*log(2) + 2*G3ya(lam) - 4*G3ya(lam)
          - truncation_m1_match
          + intermediate_sum(Fya, Z, G2ya, G3ya, 'odd', lam, M))
         )
    return o


@cache
def Y11A(xi, lam):
    """(19) / (J&O 4.15) with different scaling"""
    xiinv = xi**-1
    return (G2ya(lam)*log(xiinv)
            + AY11(lam)
            + G3ya(lam)*xi*log(xiinv)
            )


@cache
def Y12A(xi, lam):
    """(20) / (J&O 4.16)"""
    xiinv = xi**-1
    return -(G2ya(lam)*log(xiinv)
             - 1/2*(1+lam)*AY12(lam)
             + G3ya(lam)*xi*log(xiinv)
             )


# YB --------------------------------------------------------------------------

@cache
def Fyb(k, lam):
    """(21)"""
    return 2 * 2**k * sum([Qya(1, k-q, q)*lam**q for q in range(0, k+1)])


@cache
def G2yb(lam):
    """(J&O 5.6.1)"""
    return -1/5*lam*(4+lam)*(1+lam)**-2


@cache
def G3yb(lam):
    """(J&O 5.6.2)"""
    return -1/250*(32-33*lam+83*lam**2+43*lam**3)*(1+lam)**-2


@cache
def BY11(lam):
    """(22)"""
    o = (2*G2yb(lam)*log(2)
         - 2*G3yb(lam)
         + intermediate_sum(Fyb, Z, G2yb, G3yb, 'odd', lam, 100))
    return o


@cache
def BY12(lam):
    """(23)"""
    o = -4/(1+lam)**2 * (
        -G3yb(lam)
        + intermediate_sum(Fyb, Z, G2yb, G3yb, 'even', lam, 100)
    )
    return o


@cache
def Y11B(xi, lam):
    """(24)"""
    xiinv = xi**-1
    return (G2yb(lam)*log(xiinv)
            + BY11(lam)
            + G3yb(lam)*xi*log(xiinv)
            )


@cache
def Y12B(xi, lam):
    """(25)"""
    xiinv = xi**-1
    return -(G2yb(lam)*log(xiinv)
             - 1/4*(1+lam)**2*BY12(lam)
             + G3yb(lam)*xi*log(xiinv)
             )


# XC --------------------------------------------------------------------------

@cache
def X11C(xi, lam):
    """(26)"""
    xiinv = xi**-1
    return (lam**3/(1+lam)**3*zeta(3, lam/(1+lam))
            - lam**2/(4*(1+lam))*xi*log(xiinv))


@cache
def X12C(xi, lam):
    """(27)"""
    xiinv = xi**-1
    return (-lam**3/(1+lam)**3*zeta(3, 1)
            + lam**2/(4*(1+lam))*xi*log(xiinv))

# YC --------------------------------------------------------------------------


@cache
def Pyc(n, p, q):
    """(J&O 7.3 and 4.10)"""
    if p == 0 and q == 0:
        return 0
    o = sum([binom(n+s, n+1)
            * (((2*n+1)/(2*(n+1))
                * (3*(n+s)-(n*s+1)*(2*n*s-s-n+2))/(s*(n+s)*(2*s-1))
                * Pyc(s, q-s, p-n+1)
                )
                + n*(2*n-1)/(2*(n+1)) * Pyc(s, q-s, p-n-1)
                + n*(4*n**2-1)/(2*(n+1)*(2*s+1)) * Vyc(s, q-s-2, p-n+1)
                - 2*(4*n**2-1)/(3*(n+1)) * Qyc(s, q-s-1, p-n+1)
               )
             for s in range(1, q+1)
             ])
    return o


@cache
def Qyc(n, p, q):
    """(J&O 7.5 and 4.11)"""
    if p == 0 and q == 0:
        return kron(1, n)
    o = sum([binom(n+s, n+1)
            * (s/(n+1)*Qyc(s, q-s-1, p-n)
               - 3/(2*n*s*(n+1))*Pyc(s, q-s, p-n)
               )
             for s in range(1, q+1)
             ])
    return o


@cache
def Vyc(n, p, q):
    """(J&O 7.4) and (18)"""
    if p == 0 and q == 0:
        return 0
    o = (Pyc(n, p, q)
         + 2*n/((n+1)*(2*n+3))
         * sum([binom(n+s, n+1)*Pyc(s, q-s, p-n-1) for s in range(1, q+1)])
         )
    return o


@cache
def Fyc(k, lam):
    """(29)"""
    return 2**k * sum([Qyc(1, k-q, q)*lam**(q+k % 2) for q in range(0, k+1)])


@cache
def G2yc(lam):
    """(J&O 7.10.1)"""
    return 2/5*lam*(1+lam)**-1


@cache
def G3yc(lam):
    """(J&O 7.10.2)"""
    return 1/125*(8+6*lam+33*lam**2)*(1+lam)**-1


@cache
def G4yc(lam):
    """(J&O 7.10.3) * (1+lam)^3/8. See note around (34)."""
    return 1/10*lam**2*(1+lam)**-1


@cache
def G5yc(lam):
    """(30) * (1+lam)^3/8. See note around (34)."""
    return 1/500*lam*(43-24*lam+43*lam**2)*(1+lam)**-1


@cache
def CY11(lam):
    """(31)"""
    o = (1 - G3yc(lam)
         + intermediate_sum(Fyc, Z, G2yc, G3yc, 'even', lam, 100))
    return o


@cache
def CY12(lam):
    """(32). See note around (34) re scaling."""
    o = (8/(1+lam)**3 *
         (2*G4yc(lam)*log(2) - 2*G5yc(lam) +
          intermediate_sum(Fyc, Z, G4yc, G5yc, 'odd', lam, 100)
          ))
    return o


@cache
def Y11C(xi, lam):
    """(33)"""
    xiinv = xi**-1
    return (G2yc(lam)*log(xiinv)
            + CY11(lam)
            + G3yc(lam)*xi*log(xiinv)
            )


@cache
def Y12C(xi, lam):
    """(34a)"""
    xiinv = xi**-1
    return (G4yc(lam)*log(xiinv)
            + (1+lam)**3/8*CY12(lam)
            + G5yc(lam)*xi*log(xiinv)
            )


# XG --------------------------------------------------------------------------

@cache
def Fxg(k, lam):
    """(I 94)"""
    return 3/4 * 2**k * sum([Pxa(2, k-q, q)*lam**q for q in range(0, k+1)])


@cache
def G1xg(lam):
    """(J 19b.1)"""
    return 3*lam**2/(1+lam)**3


@cache
def G2xg(lam):
    """(J 19b.2)"""
    return 3/10*(lam+12*lam**2-4*lam**3)/(1+lam)**3


@cache
def G3xg(lam):
    """(J 19b.3)"""
    return 1/140*(5+181*lam-453*lam**2+566*lam**3-65*lam**4)/(1+lam)**3


@cache
def GX11(lam):
    """(J 21)"""
    o = (1/4*G1xg(lam) + 2*G2xg(lam)*log(2) - 2*G3xg(lam)
         + intermediate_sum(Fxg, G1xg, G2xg, G3xg, 'odd', lam, 100))
    return o


@cache
def GX12(lam):
    """(J 21)"""
    o = 4/((1+lam)**2) * (
        1/4*G1xg(lam) + G3xg(lam) -
        intermediate_sum(Fxg, G1xg, G2xg, G3xg, 'even', lam, 100))
    return o


@cache
def X11G(xi, lam):
    """(35)"""
    xiinv = xi**-1
    return (G1xg(lam)*xiinv
            + G2xg(lam)*log(xiinv)
            + GX11(lam)
            + G3xg(lam)*xi*log(xiinv)
            )


@cache
def X12G(xi, lam):
    """(36)"""
    xiinv = xi**-1
    return (-G1xg(lam)*xiinv
            - G2xg(lam)*log(xiinv)
            + 1/4*(1+lam)**2*GX12(lam)
            - G3xg(lam)*xi*log(xiinv)
            )


# YG --------------------------------------------------------------------------

@cache
def Fyg(k, lam):
    """(I 115)"""
    return 3/4 * 2**k * sum([Pya(2, k-q, q)*lam**q for q in range(0, k+1)])


@cache
def G2yg(lam):
    """(J 27b.1)"""
    return 1/10*(4*lam-lam**2+7*lam**3)/(1+lam)**3


@cache
def G3yg(lam):
    """(J 27b.2)"""
    return 1/500*(32-179*lam+532*lam**2-356*lam**3+221*lam**4)/(1+lam)**3


@cache
def GY11(lam):
    """(J 29)"""
    o = (2*G2yg(lam)*log(2)
         - 2*G3yg(lam)
         + intermediate_sum(Fyg, Z, G2yg, G3yg, 'odd', lam, 100)
         )
    return o


@cache
def GY12(lam):
    """(J 29)"""
    o = 4/(1+lam)**2 * (
        G3yg(lam) -
        intermediate_sum(Fyg, Z, G2yg, G3yg, 'even', lam, 100)
    )
    return o


@cache
def Y11G(xi, lam):
    """(37)"""
    xiinv = xi**-1
    return (G2yg(lam)*log(xiinv)
            + GY11(lam)
            + G3yg(lam)*xi*log(xiinv)
            )


@cache
def Y12G(xi, lam):
    """(38)"""
    xiinv = xi**-1
    return (-G2yg(lam)*log(xiinv)
            + 1/4*(1+lam)**2*GY12(lam)
            - G3yg(lam)*xi*log(xiinv)
            )

# YH --------------------------------------------------------------------------


@cache
def Fyh(k, lam):
    """(I 120)"""
    return -3/8 * 2**k * sum([Pyc(2, k-q, q)*lam**(q+k % 2) for q in range(0, k+1)])


@cache
def G2yh(lam):
    """(J 35b.1)"""
    return 1/10*(2*lam-lam**2)/(1+lam)**2


def G3yh(lam):
    """(J 35b.2)"""
    return 1/500*(16-61*lam+180*lam**2+2*lam**3)/(1+lam)**2


def G5yh(lam):
    """(J 35b.3)"""
    return 1/20*(lam**2+7*lam**3)/(1+lam)**2


def G6yh(lam):
    """(J 35b.4)"""
    return 1/1000*(43*lam+147*lam**2-185*lam**3+221*lam**4)/(1+lam)**2


@cache
def HY11(lam):
    """(J 37)"""
    o = (-G3yh(lam)
         + intermediate_sum(Fyh, Z, G2yh, G3yh, 'even', lam, 100))
    return o


@cache
def HY12(lam):
    """(J 37)"""
    o = 8/(1+lam)**3 * (
        2*G5yh(lam)*log(2)
        - 2*G6yh(lam) +
        intermediate_sum(Fyh, Z, G5yh, G6yh, 'odd', lam, 100)
    )
    return o


@cache
def Y11H(xi, lam):
    """(39)"""
    xiinv = xi**-1
    return (G2yh(lam)*log(xiinv)
            + HY11(lam)
            + G3yh(lam)*xi*log(xiinv)
            )


@cache
def Y12H(xi, lam):
    """(40)"""
    xiinv = xi**-1
    return (G5yh(lam)*log(xiinv)
            + (1+lam)**3/8*HY12(lam)
            + G6yh(lam)*xi*log(xiinv)
            )


# XM --------------------------------------------------------------------------
# Same recurrence relations as XA but different initial conditions

@cache
def Pxm(n, p, q):
    """(I 104) and (8)"""
    if p == 0 and q == 0:
        return kron(2, n)
    o = sum([binom(n+s, n)
            * ((n*(2*n+1)*(2*n*s-n-s+2)/(2*(n+1)*(2*s-1)*(n+s))
                * Pxm(s, q-s, p-n+1)
                )
                - n*(2*n-1)/(2*(n+1)) * Pxm(s, q-s, p-n-1)
                - n*(4*n**2-1)/(2*(n+1)*(2*s+1)) * Vxm(s, q-s-2, p-n+1)
               )
             for s in range(1, q+1)
             ])
    return o


@cache
def Vxm(n, p, q):
    """(I 104) and (7)"""
    if p == 0 and q == 0:
        return kron(2, n)
    o = (Pxm(n, p, q)
         - 2*n/((n+1)*(2*n+3))
         * sum([binom(n+s, n)*Pxm(s, q-s, p-n-1) for s in range(1, q+1)])
         )
    return o


@cache
def Fxm(k, lam):
    """(I 105)"""
    return 2**k * sum([Pxm(2, k-q, q)*lam**(q+k % 2) for q in range(0, k+1)])


@cache
def G1xm(lam):
    """(J 48b.1)"""
    return 6/5*lam**2/(1+lam)**3


@cache
def G2xm(lam):
    """(J 48b.2)"""
    return 3/25*(lam+17*lam**2-9*lam**3)/(1+lam)**3


@cache
def G3xm(lam):
    """(J 48b.3)"""
    return 1/350*(5+272*lam-831*lam**2+1322*lam**3-415*lam**4)/(1+lam)**3


@cache
def G4xm(lam):
    """(J 48b.4)"""
    return 6/5*lam**3/(1+lam)**3


@cache
def G5xm(lam):
    """(J 48b.5)"""
    return 3/25*(-4*lam**2+17*lam**3-4*lam**4)/(1+lam)**3


@cache
def G6xm(lam):
    """(J 48b.6)"""
    return 1/350*(-65*lam+832*lam**2-1041*lam**3+832*lam**4-65*lam**5)/(1+lam)**3


@cache
def MX11(lam):
    """(J 50)"""
    o = (-1/4*G1xm(lam)
         - G3xm(lam) + 1 +
         intermediate_sum(Fxm, G1xm, G2xm, G3xm, 'even', lam, 100))
    return o


@cache
def MX12(lam):
    """(J 50)"""
    o = 8/((1+lam)**3) * (
        1/4*G4xm(lam)
        + 2*G5xm(lam)*log(2)
        - 2*G6xm(lam)
        + intermediate_sum(Fxm, G4xm, G5xm, G6xm, 'odd', lam, 100)
    )
    return o


@cache
def X11M(xi, lam):
    """(41)"""
    xiinv = xi**-1
    return (G1xm(lam)*xiinv
            + G2xm(lam)*log(xiinv)
            + MX11(lam)
            + G3xm(lam)*xi*log(xiinv)
            )


@cache
def X12M(xi, lam):
    """(42)"""
    xiinv = xi**-1
    return (G4xm(lam)*xiinv
            + G5xm(lam)*log(xiinv)
            + 1/8*(1+lam)**3*MX12(lam)
            + G6xm(lam)*xi*log(xiinv)
            )

# YM --------------------------------------------------------------------------
# Same recurrence relations as YA but different initial conditions


@cache
def Pym(n, p, q):
    """(J 58) and (J&O 4.10)"""
    if p == 0 and q == 0:
        return kron(2, n)
    o = sum([binom(n+s, n+1)
            * (((2*n+1)/(2*(n+1))
                * (3*(n+s)-(n*s+1)*(2*n*s-s-n+2))/(s*(n+s)*(2*s-1))
                * Pym(s, q-s, p-n+1)
                )
                + n*(2*n-1)/(2*(n+1)) * Pym(s, q-s, p-n-1)
                + n*(4*n**2-1)/(2*(n+1)*(2*s+1)) * Vym(s, q-s-2, p-n+1)
                - 2*(4*n**2-1)/(3*(n+1)) * Qym(s, q-s-1, p-n+1)
               )
             for s in range(1, q+1)
             ])
    return o


@cache
def Qym(n, p, q):
    """(J 58) and (J&O 4.11)"""
    if p == 0 and q == 0:
        return 0
    o = sum([binom(n+s, n+1)
            * (s/(n+1)*Qym(s, q-s-1, p-n)
               - 3/(2*n*s*(n+1))*Pym(s, q-s, p-n)
               )
             for s in range(1, q+1)
             ])
    return o


@cache
def Vym(n, p, q):
    """(J 58) and (18)"""
    if p == 0 and q == 0:
        return kron(2, n)
    o = (Pym(n, p, q)
         + 2*n/((n+1)*(2*n+3))
         * sum([binom(n+s, n+1)*Pym(s, q-s, p-n-1) for s in range(1, q+1)])
         )
    return o


@cache
def Fym(k, lam):
    """(I 125)"""
    return 2**k * sum([Pym(2, k-q, q)*lam**(q+k % 2) for q in range(0, k+1)])


@cache
def G2ym(lam):
    """(J 64b.1)"""
    return 6/25*(lam-lam**2+4*lam**3)/(1+lam)**3


@cache
def G3ym(lam):
    """(J 64b.2)"""
    return 1/625*(24-201*lam+882*lam**2-1182*lam**3+591*lam**4)/(1+lam)**3


@cache
def G5ym(lam):
    """(J 64b.3)"""
    return 3/50*(7*lam**2-10*lam**3+7*lam**4)/(1+lam)**3


@cache
def G6ym(lam):
    """(J 64b.4)"""
    return 3/2500*lam*(221-728*lam+1902*lam**2-728*lam**3+221*lam**4)/(1+lam)**3


@cache
def MY11(lam):
    """(J 66)"""
    o = (-G3ym(lam) + 1
         + intermediate_sum(Fym, Z, G2ym, G3ym, 'even', lam, 130))
    return o


@cache
def MY12(lam):
    """(J 66)"""
    o = 8/(1+lam)**3 * (
        2*G5ym(lam)*log(2) - 2*G6ym(lam)
        + intermediate_sum(Fym, Z, G5ym, G6ym, 'odd', lam, 130)
    )
    return o


@cache
def Y11M(xi, lam):
    """(43)"""
    xiinv = xi**-1
    return (G2ym(lam)*log(xiinv)
            + MY11(lam)
            + G3ym(lam)*xi*log(xiinv)
            )


@cache
def Y12M(xi, lam):
    """(44)"""
    xiinv = xi**-1
    return (G5ym(lam)*log(xiinv)
            + 1/8*(1+lam)**3*MY12(lam)
            + G6ym(lam)*xi*log(xiinv)
            )


# ZM --------------------------------------------------------------------------

@cache
def Pzm(n, p, q):
    """(J 73 and 75)"""
    if p == 0 and q == 0:
        return kron(2, n)
    o = sum([binom(n+s, n+2)
            * (- 2*(2*n+1)*(2*n-1)/(n+1) * Qzm(s, q-s-1, p-n+1)
               + n*(2*n+1)*(2*n-1)/(2*(n+1)*(2*s+1)) * Vzm(s, q-s-2, p-n+1)
               + ((2*n+1)/(n+1)
                  * (n*s*(n+s-2*n*s-2)-4*(2*n*s-4*s-4*n+2))/(2*s*(n+s)*(2*s-1))
                  * Pzm(s, q-s, p-n+1)
                  )
                + n*(2*n-1)/(2*(n+1)) * Pzm(s, q-s, p-n-1)
               )
             for s in range(2, q+1)
             ])
    return o


@cache
def Qzm(n, p, q):
    """(J 73 and 76)"""
    if p == 0 and q == 0:
        return 0
    o = sum([binom(n+s, n+2)
            * (s/(n+1)*Qzm(s, q-s-1, p-n)
               - 2/(n*s*(n+1))*Pzm(s, q-s, p-n)
               )
             for s in range(2, q+1)
             ])
    return o


@cache
def Vzm(n, p, q):
    """(J 73 and 74)"""
    if p == 0 and q == 0:
        return kron(2, n)
    o = (Pzm(n, p, q)
         + 2*n/((n+1)*(2*n+3))
         * sum([binom(n+s, n+2)*Pzm(s, q-s, p-n-1) for s in range(2, q+1)])
         )
    return o


@cache
def Fzm(k, lam):
    """(I 131)"""
    return 2**k * sum([Pzm(2, k-q, q)*lam**(q + k % 2) for q in range(0, k+1)])


@cache
def Fzm_with_sum_coeff(k, lam):
    """(I 131) but multiplied by 2^(-k) (1+lambda)^(-k), which is the 
    coefficient of Fzm which is normally in MZ11 and MZ12. Putting the 
    coefficient inside this function means that we can avoid having lam^k,
    which can exceed the max double precision float size of 1e308 when
    lam=100 and k gets big, which you need as MZ11 and MZ12 converge slowly."""
    return sum([
        Pzm(2, k-q, q)
        * (lam/(1+lam))**q
        * lam**(k % 2)
        * (1+lam)**(q-k)
        for q in range(0, k+1)])


@cache
def G3zm(lam):
    """(J 79b.1)"""
    return -3/10*(lam**2+lam**4)/(1+lam)**3


@cache
def MZ11(lam):
    """(J 81). See docstring on Fzm_with_sum_coeff for reasoning on using
    intermediate_sum_no_f0_coeff."""
    o = (-G3zm(lam) + 1 +
         intermediate_sum_no_f0_coeff(Fzm_with_sum_coeff, Z, Z, G3zm,
                                      'even', lam, 150))
    return o


@cache
def MZ12(lam):
    """(J 81). See docstring on Fzm_with_sum_coeff for reasoning on using
    intermediate_sum_no_f0_coeff."""
    o = 8/(1+lam)**3 * (
        2*G3zm(lam) -
        intermediate_sum_no_f0_coeff(Fzm_with_sum_coeff, Z, Z, G3zm,
                                     'odd', lam, 200)
    )
    return o


@cache
def Z11M(xi, lam):
    """(45)"""
    xiinv = xi**-1
    return (MZ11(lam)
            + G3zm(lam)*xi*log(xiinv)
            )


@cache
def Z12M(xi, lam):
    """(46)"""
    xiinv = xi**-1
    return (1/8*(1+lam)**3*MZ12(lam)
            - G3zm(lam)*xi*log(xiinv)
            )