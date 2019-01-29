*======================================================================*
*
      SUBROUTINE Store_roots
*
      INCLUDE 'parameters.f'
*
************************************************************************
*                                                                      *
*     Stores the square root of the integers from 0 to 2*Nmax in       *
*     the array "root", and square roots of the factorials in "fact"   *
*                                                                      *
************************************************************************
*
      integer i
      real*8 root(0:2*Nmax), fact(0:2*Nmax)
      common /sqrts/ root, fact
*
      root(0) = 0.d0
      fact(0) = 1.d0
      do i=1,2*Nmax
         root(i) = dsqrt((0.d0+i))
         fact(i) = root(i)*fact(i-1)
      end do
*
      end
*
*======================================================================*
*
      SUBROUTINE Fill_G
*
      INCLUDE 'parameters.f'
*
************************************************************************
*                                                                      *
*     Fills the array G(nu,n,m) with the functions g_{\nu n}^m as      *
*     defined in equation (2.6) of the paper.                          *
*                                                                      *
************************************************************************
*
      integer nu,n,m,im
      real*8 root(0:2*Nmax), fact(0:2*Nmax)
      common /sqrts/ root, fact
*
      do nu=0,Nmax
         do n=0,Nmax
            do m=-Nmax,Nmax
               g(nu,n,m) = 0.d0
            end do
            g(nu,n,n) = 1.d0
            do m=n-1,1,-1
               g(nu,n,m) = (nu+m+1.d0)*g(nu,n,m+1)/(0.d0+n-m)
            end do
            g(nu,n,-n) = 1.d0
            if (n .NE. 0) then
               g(nu,n,-n+1) = (nu+n)*g(nu,n,-n)
               do m=-n+1,-2
                  g(nu,n,m+1) = (nu-m)*g(nu,n,m)/(n+m+1.d0)
               end do
               g(nu,n,0) = g(nu,n-1,0)*(nu+n)/(0.d0+n)
            else
               g(nu,n,0) = 1.d0
            end if
         end do
      end do
*
      do n=0,Nmax
         do m=-n,n
            if (m .LT. 0) then 
               im = -m
            else
               im = m
            end if
            FracH(n,n,m) = 1.d0
            do nu=n-1,im,-1
               FracH(n,nu,m) = root(2*nu+3)*root(nu-im+1)*
     +              FracH(n,nu+1,m)/(root(2*nu+1)*root(nu+im+1))
            end do
            do nu=n+1,Nmax
               FracH(n,nu,m) = root(2*nu-1)*root(nu+im)*FracH(n,nu-1,m)/
     +              (root(2*nu+1)*root(nu-im))
            end do
         end do
      end do
*
      end
*
*======================================================================*
*
      real*8 FUNCTION H(n,m)
*
      INCLUDE 'parameters.f'
*
************************************************************************
*                                                                      *
*     The function H(n,m) is the constant h_n^m as                     *
*     defined in equation (2.6) of the paper.                          *
*                                                                      *
************************************************************************
*
      integer n,m,im
      real*8 root(0:2*Nmax), fact(0:2*Nmax)
      common /sqrts/ root, fact      
*
      if (m .LT. 0) then 
         im = -m
      else
         im = m
      end if
      H = root(2*n+1)*fact(n-im)/fact(n+im)
      H = H/(2.d0*dsqrt(pi))
*
      end
*
*======================================================================*
*
      SUBROUTINE Set_N
*
      INCLUDE 'parameters.f'
*
************************************************************************
*                                                                      *
*     Fixes the following constants:                                   *
*        Neach(i): the number of n-terms of Lamb's solution to be used *
*                  for particle i                                      *
*        NN(i,j) : the N-value relevant to the separation between      *
*                  particles i and j (used for rotation matrices)      *
*        MainN   : the maximum of Neach(i) across all particles        *
*                                                                      *
*     The method is based on equation (2.33) of the paper.             *
*                                                                      *
*     If parameter NFUDGE is set to TRUE, the code will not abort if   *
*     the matrices are too small for optimal accuracy; a warning       *
*     message is printed out.                                          *
*                                                                      *
************************************************************************
*
      integer i,j
      real*8 zeta,q
*
      do i=1,Nspheres
         Neach(i) = 1
      end do
      MainN = 1
      do i=1,Nspheres
         do j=i+1,Nspheres
            zeta = 2.d0*dabs(r(i,j))/(radius(i)+radius(j))
            q = 0.5d0*(zeta - dsqrt(zeta**2 - 4.d0))
            NN(i,j) = int(dlog(globalerror)/dlog(q))+1
c-----c
            if (NN(i,j) .GT. Nmax) then
               if (NFUDGE) then
                  print *,'Using fudge for N'
                  NN(i,j) = Nmax
               else
                  print *,'Warning: N about to be too small',i,j,NN(i,j)
               end if
            end if
c-----c
            NN(j,i) = NN(i,j)
            if (NN(i,j) .GT. Neach(i)) then
               Neach(i) = NN(i,j)
            end if
            if (NN(i,j) .GT. Neach(j)) then
               Neach(j) = NN(i,j)
            end if
            if (NN(i,j) .GT. MainN) then
               MainN = NN(i,j)
            end if
         end do
      end do
*      
      if (MainN .GT. Nmax) then 
         print *,MainN,Nmax
         STOP 'Nmax is too small (or spheres too close)'
      end if
*
      end
*
*======================================================================*
*
      SUBROUTINE Update_Fields(i)
*
      INCLUDE 'parameters.f'
*
      integer i,j,m,n

***********************************************************************
*                                                                     *
*     This subroutine completes a reflection by:                      *
*     * Adding X(*,i,2,*,*) to X(*,i,1,*,*);                          *
*     * Moving X(*,i,3,*,*) to X(*,i,2,*,*); and                      *
*     * Setting X(*,i,3,*,*) to zero ready for the next reflection    * 
*     * Setting X(*,i,4,*,*) to zero ready for the next reflection    * 
*                                                                     *
***********************************************************************
*
      do j=1,2
         do n=0,Neach(i)
            do m=-n,n
               A(j,i,1,n,m) = A(j,i,1,n,m) + A(j,i,2,n,m)
               A(j,i,2,n,m) = A(j,i,3,n,m)
               A(j,i,3,n,m) = 0.d0
               B(j,i,1,n,m) = B(j,i,1,n,m) + B(j,i,2,n,m)
               B(j,i,2,n,m) = B(j,i,3,n,m)
               B(j,i,3,n,m) = 0.d0
               C(j,i,1,n,m) = C(j,i,1,n,m) + C(j,i,2,n,m)
               C(j,i,2,n,m) = C(j,i,3,n,m)
               C(j,i,3,n,m) = 0.d0
               D(j,i,1,n,m) = D(j,i,1,n,m) + D(j,i,2,n,m)
               D(j,i,2,n,m) = 0.d0
               E(j,i,1,n,m) = E(j,i,1,n,m) + E(j,i,2,n,m)
               E(j,i,2,n,m) = 0.d0
               F(j,i,1,n,m) = F(j,i,1,n,m) + F(j,i,2,n,m)
               F(j,i,2,n,m) = 0.d0
            end do
         end do
      end do
*
      end
*
*======================================================================*
*
      real*8 FUNCTION Find_force(k,p)
*
      INCLUDE 'parameters.f'
*
************************************************************************
*                                                                      *
*     Calculates the magnitude of the additional velocity contribution *
*     contained in the fields X(*,k,p,*,*). Result is the sum of the   *
*     magnitudes of velocity and angular velocity vectors. Used to     *
*     determine when to terminate the iterations; was also handy for   *
*     debugging.                                                       *
*                                                                      *
************************************************************************
*
*     Inputs
      integer k,p
*
*     Internal variables
      integer i,j
      real*8 Fxr, Fyr, Fzr, Fxi, Fyi, Fzi, H
      real*8 Txr, Tyr, Tzr, Txi, Tyi, Tzi, Torque
      real*8 Uxr, Uyr, Uzr, Uxi, Uyi, Uzi
      real*8 Oxr, Oyr, Ozr, Oxi, Oyi, Ozi
      real*8 V(1:2,-1:1), O(1:2,-1:1)
*
      do i=1,2
         do j=-1,1
            V(i,j) = H(1,j)*(-5.d0*A(i,k,p,1,j)/radius(k)**3 + 
     +           1.5d0*C(i,k,p,1,j)/radius(k) + 
     +           D(i,k,p,1,j))
            O(i,j) = H(1,j)*(B(i,k,p,1,j)/radius(k)**3+E(i,k,p,1,j))
         end do
      end do
      Uxr = V(1,0)
      Uxi = V(2,0)
      Uyr = -1.d0*(V(1,1)-V(1,-1))
      Uyi = -1.d0*(V(2,1)-V(2,-1))
      Uzr = -1.d0*(V(2,1)+V(2,-1))
      Uzi = V(1,1)+V(1,-1)
*
      Oxr = O(1,0)
      Oxi = O(2,0)
      Oyr = O(1,-1)-O(1,1)
      Oyi = O(2,-1)-O(2,1)
      Ozr = -1.d0*(O(2,1)+O(2,-1))
      Ozi = O(1,1)+O(1,-1)
*
      Find_force = Uxr**2.d0 + Uyr**2.d0 + Uzr**2.d0 + 
     +     Uxi**2.d0 + Uyi**2.d0 + Uzi**2.d0 
      Torque = Oxr**2.d0 + Oyr**2.d0 + Ozr**2.d0 + 
     +     Oxi**2.d0 + Oyi**2.d0 + Ozi**2.d0
*
      Find_force = 6.d0*pi*radius(k)*
     +     (dsqrt(Find_force) + dsqrt(Torque))
*     
      end
*
*======================================================================*
