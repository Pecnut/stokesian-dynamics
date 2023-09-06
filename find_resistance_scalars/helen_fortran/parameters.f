*----------------------------------------------------------------------*
*
      implicit none
      integer Nmax
      parameter (Nmax=120)
      integer Nspheres
      parameter (Nspheres=2)
*
      real*8 pi
      parameter (pi = 3.141592653589793238462643d0)
*
*----------------------------------------------------------------------*
*
      integer NN(Nspheres,Nspheres), Neach(1:Nspheres), MainN
      common /dynamicN/ NN, Neach, MainN
      real*8 radius(1:Nspheres), r(Nspheres,Nspheres), th(Nspheres)
      real*8 scaling(3)
      common /geometry/ radius, r, th
*
*----------------------------------------------------------------------*
*
      real*8 A(1:2,1:Nspheres,1:3+Nspheres,0:Nmax,-Nmax:Nmax),
     +     B(1:2,1:Nspheres,1:3+Nspheres,0:Nmax,-Nmax:Nmax),
     +     C(1:2,1:Nspheres,1:3+Nspheres,0:Nmax,-Nmax:Nmax),
     +     D(1:2,1:Nspheres,1:3+Nspheres,0:Nmax,-Nmax:Nmax),
     +     E(1:2,1:Nspheres,1:3+Nspheres,0:Nmax,-Nmax:Nmax),
     +     F(1:2,1:Nspheres,1:3+Nspheres,0:Nmax,-Nmax:Nmax)
      common /coefficients/ A,B,C,D,E,F
*
*----------------------------------------------------------------------*
*
      real*8 s
      common /distance/ s
      real*8 globalerror
      common /error/ globalerror
*
*----------------------------------------------------------------------*
*
      real*8 g(0:Nmax,0:Nmax,-Nmax:Nmax)
      real*8 FracH(0:Nmax,0:Nmax,-Nmax:Nmax)
      common /Constants/ g, FracH
*
*----------------------------------------------------------------------*
*
      logical INALINE, NFUDGE
      common /Flags/ INALINE, NFUDGE
*
*----------------------------------------------------------------------*
*
      real*8 xg_sign
      parameter (xg_sign = -1.0d0)
*
*----------------------------------------------------------------------*