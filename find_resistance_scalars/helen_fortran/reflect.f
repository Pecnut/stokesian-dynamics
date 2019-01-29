*======================================================================*
*
      SUBROUTINE Invert_field(k,i)
*
      INCLUDE 'parameters.f'
      integer k, i
*
************************************************************************
*                                                                      *
*     D,E,F(*,k,4,n,m) contain the D,E,F(n,m) coefficients of a        *
*     Lamb's solution valid inside particle k. This subroutine         *
*     calculates the coefficients A,B,C(n,m) of a series valid         *
*     outside the particle, which has the opposite velocity on the     *
*     surface of the particle. The new coefficients are added to       *
*     A,B,C(*,k,3,n,m).                                                *
*                                                                      *
************************************************************************
*
      integer j,n,m
*
      do j=1,2 
         do m=-1,1
            A(j,k,3,1,m) = A(j,k,3,1,m) - 
     +           radius(k)**5*F(j,k,3,1,m)/30.d0
            D(j,k,2,1,m) = D(j,k,2,1,m) + D(j,k,3,1,m)
            E(j,k,2,1,m) = E(j,k,2,1,m) + E(j,k,3,1,m)
            F(j,k,2,1,m) = F(j,k,2,1,m) + F(j,k,3,1,m)
         end do
      end do
*     
      do j=1,2
         do n=2,NN(k,i)
            do m=-n,n
               A(j,k,3,n,m) = A(j,k,3,n,m) - 
     +              n*radius(k)**(2.d0*n+1.d0)/(4.d0*(n+1.d0)*(2*n+3))*
     +              (2.d0*(2*n-1)*(2*n+3)*D(j,k,3,n,m) + 
     +              (2*n+1)*radius(k)**2.d0*F(j,k,3,n,m))
               B(j,k,3,n,m) = B(j,k,3,n,m) - 
     +              radius(k)**(2.d0*n+1.d0)*E(j,k,3,n,m)
               C(j,k,3,n,m) = C(j,k,3,n,m) - 
     +              n*(2*n-1)*radius(k)**(2.d0*n-1.d0)/(2.d0*(n+1))*(
     +              2.d0*(2*n+1)*D(j,k,3,n,m) + 
     +              radius(k)**2.d0*F(j,k,3,n,m))
               D(j,k,2,n,m) = D(j,k,2,n,m) + D(j,k,3,n,m)
               E(j,k,2,n,m) = E(j,k,2,n,m) + E(j,k,3,n,m)
               F(j,k,2,n,m) = F(j,k,2,n,m) + F(j,k,3,n,m)
            end do
         end do
      end do
*
      end
*
*======================================================================*
*
      SUBROUTINE Translate_field(i,j)
*
      INCLUDE 'parameters.f'
      integer i,j
*
************************************************************************
*                                                                      *
*     A field which is valid outside particle i is expressed in the    *
*     coefficients A,B,C(*,i,3+j,n,m). This subroutine converts it     *
*     into one valid inside particle j. The result of the process is   *
*     stored in the coefficients D,E,F(*,j,3+i,n,m).                   *
*                                                                      *
************************************************************************
*
      integer nu,n,m,ninit,p
      real*8 sign
*
      if (r(i,j) .LT. 0.d0) then
         sign = 1.d0
      else
         sign = -1.d0
      end if
*
      do nu=1,NN(i,j)
         do m=-nu,nu
            do p=1,2
               D(p,j,3+i,nu,m) = 0.d0
               E(p,j,3+i,nu,m) = 0.d0
               F(p,j,3+i,nu,m) = 0.d0
            end do
            if (m .EQ. 0) then
               ninit = 1
            else
               ninit = iabs(m)
            end if
            do n=ninit,NN(i,j)
               D(1,j,3+i,nu,m) = D(1,j,3+i,nu,m) + 
     +              (-1.d0)**(n+m)*sign**(n+nu)*(FracH(n,nu,m))/
     +              (dabs(r(i,j))**(n+nu+1))*(
     +              g(nu,n,m)*A(1,i,3+j,n,m) + 
     +              m*g(nu,n,m)*r(i,j)*B(2,i,3+j,n,m)/nu + 
     +              (g(nu-1,n,m)*(nu-iabs(m))*
     +              ((nu-1)*(n-2) - (n+1))/
     +              (nu*(2*nu-1)) - g(nu,n,m)*(n-2)/2.d0)*
     +              dabs(r(i,j))**2.d0*C(1,i,3+j,n,m)/
     +              (n*(2.d0*n-1.d0)))
               D(2,j,3+i,nu,m) = D(2,j,3+i,nu,m) + 
     +              (-1.d0)**(n+m)*sign**(n+nu)*(FracH(n,nu,m))/
     +              (dabs(r(i,j))**(n+nu+1))*(
     +              g(nu,n,m)*A(2,i,3+j,n,m) - 
     +              m*g(nu,n,m)*r(i,j)*B(1,i,3+j,n,m)/nu + 
     +              (g(nu-1,n,m)*(nu-iabs(m))*
     +              ((nu-1)*(n-2) - (n+1))/
     +              (nu*(2*nu-1)) - g(nu,n,m)*(n-2)/2.d0)*
     +              dabs(r(i,j))**2.d0*C(2,i,3+j,n,m)/
     +              (n*(2.d0*n-1.d0)))
               E(1,j,3+i,nu,m) = E(1,j,3+i,nu,m) + 
     +              (-1.d0)**(n+m)*sign**(n+nu)*
     +              g(nu,n,m)*(FracH(n,nu,m))/
     +              (dabs(r(i,j))**(n+nu+1))*(
     +              -(n/(1.d0+nu))*B(1,i,3+j,n,m) + 
     +              m*r(i,j)*C(2,i,3+j,n,m)/
     +              ((1.d0+nu)*n*nu))
               E(2,j,3+i,nu,m) = E(2,j,3+i,nu,m) + 
     +              (-1.d0)**(n+m)*sign**(n+nu)*
     +              g(nu,n,m)*(FracH(n,nu,m))/
     +              (dabs(r(i,j))**(n+nu+1))*(
     +              -(n/(1.d0+nu))*B(2,i,3+j,n,m) - 
     +              m*r(i,j)*C(1,i,3+j,n,m)/
     +              ((1.d0+nu)*n*nu))
               F(1,j,3+i,nu,m) = F(1,j,3+i,nu,m) + 
     +              (-1.d0)**(n+m)*sign**(n+nu)*(FracH(n,nu,m))/
     +              (dabs(r(i,j))**(1+n+nu))*
     +              g(nu,n,m)*C(1,i,3+j,n,m)
               F(2,j,3+i,nu,m) = F(2,j,3+i,nu,m) + 
     +              (-1.d0)**(n+m)*sign**(n+nu)*(FracH(n,nu,m))/
     +              (dabs(r(i,j))**(1+n+nu))*
     +              g(nu,n,m)*C(2,i,3+j,n,m)
            end do
         end do
      end do
*     
      end
*
*======================================================================*

