*======================================================================*
*
      PROGRAM Two_Sphere
*
      INCLUDE 'parameters.f'
*
      character(len=20)::casenumber
      character(len=20)::outputform
      character(len=20)::s_input
      character(len=20)::r1_input
      character(len=20)::r2_input
*
      integer i,j,k
      real*8 EPSILON, r1, r2
      real*8 Find_force
      real*8 Ff(1:Nspheres,1:3), Tt(1:Nspheres,1:3),
     >     Ee(1:Nspheres,1:3,1:3)
      real*8 factor
*
      do i=1,Nspheres
         do j=1,3
            Ff(i,j) = 0.d0
            Tt(i,j) = 0.d0
            do k=1,3
               Ee(i,j,k) = 0.d0
            end do
         end do
      end do
*
*=====* Set of parameters which may be modified
*
*-----* Numerical parameters
      globalerror = 1.d-6
*     - globalerror controls all error conditions.
      NFUDGE = .TRUE.
*     - If NFUDGE=.TRUE., a smaller than optimal N is allowed
*
*-----* Inputs
c      print *, "bullshit"

      call get_command_argument(1,s_input)
      call get_command_argument(2,r1_input)
      call get_command_argument(3,r2_input)
      call get_command_argument(4,casenumber)
      call get_command_argument(5,outputform)

      READ(s_input,*)s
      READ(r1_input,*)r1
      READ(r2_input,*)r2
*
*-----* Physical parameters
      radius(1) = r1
      radius(2) = r2
c	  print *, radius(1)
c	  print *, radius(2)

c
c     Scaling ("Conversion factors")
c     -Helen's originals
c      scaling(1)=6.d0*pi*radius(1)
c      scaling(2)=8.d0*pi*radius(1)**2.d0
c      scaling(3)=(20.d0/3.d0)*pi*radius(1)**3
c      scaling(3)=1
c     -To match Caltech
c      scaling(1)=6.d0*pi*radius(1)
c      scaling(2)=6.d0*pi*radius(1)**2.d0
c      scaling(3)=1
c     -To have it scaled on sqrt(r1r2)
c      scaling(1)=6.d0*pi*sqrt(radius(1)*radius(2))
c      scaling(2)=6.d0*pi*sqrt(radius(1)*radius(2))**2.d0
c      scaling(3)=6.d0*pi*sqrt(radius(1)*radius(2))**3
c     -Scale on 1
      scaling(1)=1.d0
      scaling(2)=1.d0
      scaling(3)=1.d0
c
c     The only essential cases, in order to get all the scalars, are
c     F11, F12, F21, F22,
c     T11, T12, T21, T22,
c     E1, E3, E4,

      if (outputform == "words") then
            print *,casenumber
      endif
      select case (casenumber)
      case ("F11")
            Ff(1,1) = scaling(1)
      case ("F12")
            Ff(1,2) = scaling(1)
      case ("F13")
            Ff(1,3) = scaling(1)
      case ("F1112")
            Ff(1,1) = scaling(1)
            Ff(1,2) = scaling(1)
      case ("F21")
            Ff(2,1) = scaling(1)
      case ("F22")
            Ff(2,2) = scaling(1)
      case ("F23")
            Ff(2,3) = scaling(1)
      case ("T11")
			Tt(1,1) = scaling(2)
      case ("T12")
            Tt(1,2) = scaling(2)
      case ("T13")
            Tt(1,3) = scaling(2)
      case ("T21")
            Tt(2,1) = scaling(2)
      case ("T22")
            Tt(2,2) = scaling(2)
      case ("T23")
            Tt(2,3) = scaling(2)
      case ("E1")
            Ee(1,1,1) = scaling(3)
            Ee(1,2,2) = -0.5d0*scaling(3)
            Ee(1,3,3) = -0.5d0*scaling(3)
            Ee(2,1,1) = scaling(3)
            Ee(2,2,2) = -0.5d0*scaling(3)
            Ee(2,3,3) = -0.5d0*scaling(3)
      case ("E3")
            Ee(1,2,3) = scaling(3)
            Ee(1,3,2) = scaling(3)
            Ee(2,2,3) = scaling(3)
            Ee(2,3,2) = scaling(3)
      case ("E4")
            Ee(1,1,2) = scaling(3)
            Ee(1,2,1) = scaling(3)
            Ee(2,1,2) = scaling(3)
            Ee(2,2,1) = scaling(3)
      case ("E11")
            Ee(1,1,1) = scaling(3)
            Ee(1,2,2) = -0.5d0*scaling(3)
            Ee(1,3,3) = -0.5d0*scaling(3)
      case ("E13")
            Ee(1,2,3) = scaling(3)
            Ee(1,3,2) = scaling(3)
      case ("E14")
            Ee(1,1,2) = scaling(3)
            Ee(1,2,1) = scaling(3)
      case ("E21")
            Ee(2,1,1) = scaling(3)
            Ee(2,2,2) = -0.5d0*scaling(3)
            Ee(2,3,3) = -0.5d0*scaling(3)
      case ("E23")
            Ee(2,2,3) = scaling(3)
            Ee(2,3,2) = scaling(3)
      case ("E24")
			Ee(2,1,2) = scaling(3)
            Ee(2,2,1) = scaling(3)
      end select
c
c
*     - Ff(i,j) is the force acting on particle i in the j direction
*     - Tt(i,j) is the torque acting on particle i in the j direction
*     - Ee(1,j,k) = Ee(2,j,k) is E^infinity and these two should be set
*       the same to make any physical sense.
*
*=====*
*
      call Store_Roots
      call Fill_G
      r(1,2) = s
      r(2,1) = -s
      call Set_N
      call Set_Initial_Conditions(Ff,Tt,Ee)
*
      EPSILON = 1.d0
      j=0
      do while ((j.LT.6) .OR. (EPSILON .GT. globalerror))
         j=j+1
*
c         print *,'*',j, EPSILON
         do i=1,Nspheres
            do k=1,Nspheres
               if (i.EQ.k) then
               else
                  call Rotate_field(i,k) ! Sets X(*,i,3+k,*,*)
                  call Translate_field(i,k) ! Sets X2(*,k,3+i,*,*)
                  call Rotate_arrival(k,i) ! Sets X2(*,k,3,*,*)
                  call Invert_field(k,i) ! Adds to X(*,k,3,*,*)
               end if
            end do
         end do
         EPSILON = 0.d0
         do i=1,Nspheres
            EPSILON = EPSILON + Find_force(i,3)
            call Update_Fields(i) ! Adds to X(*,i,1,*,*);
*                           Moves to X(*,i,2,*,*), Zeroes X(*,i,3,*,*)
         end do
      end do
*
*------* Use this code segment to see results on screen:
      if (outputform=="words") then
            print *,'The sphere separation is ',s
      endif
      call Output_Velocities
      if (outputform=="words") then
            print *,'It took',j,' iterations with eps=',globalerror
      endif
*------* Use this code segment to have results in a file:
c         open (10,file='results.dat',access='append')
c         call Output_Velocities_File(s, globalerror, j)
c         close (10)
*------* The file columns will be: lambda, s, error, k,
*     >     Ux(1), Uy(1), Uz(1), Ox(1), Oy(1), Oz(1),
*     >     Ux(2), Uy(2), Uz(2), Ox(2), Oy(2), Oz(2),
*     >     Sxx(1), Sxy(1), Syy(1), Sxz(1), Syz(1),
*     >     Sxx(2), Sxy(2), Syy(2), Sxz(2), Syz(2)
*------*
*
      end
*
*======================================================================*
*
      SUBROUTINE Rotate_field(i,k)
*
      INCLUDE 'parameters.f'
      integer i,j,k,n,m
*
      do j=1,2
         do n=0,NN(i,k)
            do m=-n,n
               A(j,i,3+k,n,m) = A(j,i,2,n,m)
               B(j,i,3+k,n,m) = B(j,i,2,n,m)
               C(j,i,3+k,n,m) = C(j,i,2,n,m)
            end do
         end do
      end do
*
      end
*
*======================================================================*
*
      SUBROUTINE Set_Initial_Conditions(Ff,Tt,Ee)
*
      INCLUDE 'parameters.f'
      real*8 Ff(1:Nspheres,1:3), Tt(1:Nspheres,1:3),
     >     Ee(1:Nspheres,1:3,1:3)
*
      integer i,j,k,l,m,n
      real*8 h, Hh, Symmetry
*
************************************************************************
*                                                                      *
*     forces Ff(i,1) are on particle i in the direction parallel to    *
*                    the line of centres of particles 1 and 2          *
*     forces Ff(i,2) are on particle i in the direction                *
*                    perpendicular to the plane of sphere centres      *
*     forces Ff(i,3) are on particle i in the direction                *
*                    perpendicular to the other two axes               *
*     torque Tt(i,j) acts on sphere i                                  *
*                    about the axis j described above in "forces"      *
*     background flow field has strain rate Ee                         *
*                                                                      *
************************************************************************
*
      do i=1,2
         do j=1,Nspheres
            do k=1,3+Nspheres
               do l=1,Nmax
                  do m=-l,l
                     A(i,j,k,l,m) = 0.d0
                     B(i,j,k,l,m) = 0.d0
                     C(i,j,k,l,m) = 0.d0
                     D(i,j,k,l,m) = 0.d0
                     E(i,j,k,l,m) = 0.d0
                     F(i,j,k,l,m) = 0.d0
                  end do
               end do
            end do
         end do
      end do
*
*-----* Velocities *---------------------------------------------------*
*
      do i=1,Nspheres
         Hh = H(1,0)
         C(1,i,2,1,0) = Ff(i,1)/(4.d0*Hh*pi)
         Hh = H(1,1)
         C(1,i,2,1,1)  = -Ff(i,2)/(8.d0*Hh*pi)
         C(1,i,2,1,-1) =  Ff(i,2)/(8.d0*Hh*pi)
         C(2,i,2,1,1)  = -Ff(i,3)/(8.d0*Hh*pi)
         C(2,i,2,1,-1) = -Ff(i,3)/(8.d0*Hh*pi)
*
         do j=1,2
            do k=-1,1
               A(j,i,2,1,k)  = C(j,i,2,1,k)*(radius(i)**2.d0)/6.d0
            end do
         end do
      end do
*
*-----* Angular Velocities *-------------------------------------------*
*
      do i=1,Nspheres
         Hh = H(1,0)
         B(1,i,2,1,0)  =  Tt(i,1)/(8.d0*Hh*pi)
         Hh = H(1,1)
         B(1,i,2,1,1)  = -Tt(i,2)/(16.d0*Hh*pi)
         B(1,i,2,1,-1) =  Tt(i,2)/(16.d0*Hh*pi)
         B(2,i,2,1,1)  = -Tt(i,3)/(16.d0*Hh*pi)
         B(2,i,2,1,-1) = -Tt(i,3)/(16.d0*Hh*pi)
      end do
*
*-----* Strain Rate *--------------------------------------------------*
*
      do i=1,Nspheres
         if (dabs(Ee(i,1,1)+Ee(i,2,2)+Ee(i,3,3)).GT.1.d-5) then
            print *,'On sphere ',i
            STOP 'Given value of Einf is not traceless'
         end if
         Symmetry = (Ee(i,1,2)-Ee(i,2,1))**2.d0 +
     >        (Ee(i,1,3)-Ee(i,3,1))**2.d0 + (Ee(i,3,2)-Ee(i,2,3))**2.d0
         if (dabs(Symmetry).GT.1.d-5) then
            print *,'On sphere ',i
            STOP 'Given value of Einf is not symmetric'
         end if
      end do

      do i=1,Nspheres
         Hh = H(2,0)
         D(1,i,3,2,0) = (Ee(i,1,1))/(Hh)
         Hh = H(2,1)
         D(1,i,3,2,1) = Ee(i,1,2)/(3.d0*Hh)
         D(1,i,3,2,-1) = -Ee(i,1,2)/(3.d0*Hh)
         D(2,i,3,2,1) = Ee(i,1,3)/(3.d0*Hh)
         D(2,i,3,2,-1) = Ee(i,1,3)/(3.d0*Hh)
         Hh = H(2,2)
         D(1,i,3,2,2) = (2.d0*Ee(i,2,2) + Ee(i,1,1))/(12.d0*Hh)
         D(1,i,3,2,-2) = (2.d0*Ee(i,2,2) + Ee(i,1,1))/(12.d0*Hh)
         D(2,i,3,2,2) = -Ee(i,2,3)/(6.d0*Hh)
         D(2,i,3,2,-2) = Ee(i,2,3)/(6.d0*Hh)
c-----------------------------------------------------c
         do j=1,2
            do m=-2,2
               D(j,i,3,2,m) = 0.5d0*D(j,i,3,2,m)
            end do
         end do
c--------c Added 24 June 2014 noon to account for the usual "G = 2D" error.
         call Invert_field(i,3-i)
         do j=1,2
            do m=-2,2
               A(j,i,2,2,m) = A(j,i,3,2,m)
               B(j,i,2,2,m) = B(j,i,3,2,m)
               C(j,i,2,2,m) = C(j,i,3,2,m)
               A(j,i,3,2,m) = 0.d0
               B(j,i,3,2,m) = 0.d0
               C(j,i,3,2,m) = 0.d0
            end do
         end do
      end do
*
      end
*
*======================================================================*
*
      SUBROUTINE Output_Velocities
*
      INCLUDE 'parameters.f'
      integer i
      real*8 Ux(Nspheres), Uy(Nspheres), Uz(Nspheres),
     +     Ox(Nspheres), Oy(Nspheres), Oz(Nspheres),
     +     Sxx(Nspheres), Sxy(Nspheres), Syy(Nspheres), Sxz(Nspheres),
     +     Syz(Nspheres)
      character(len=20)::outputform
      call get_command_argument(5,outputform)
*
      call Calculate_Velocities(Ux,Uy,Uz,Ox,Oy,Oz,Sxx,Sxy,Syy,Sxz,Syz)
*
      if (outputform=="words") then
		  print *,'The mobilities are:'
		  print *,'==================='
		  do i=1,Nspheres
			 print *,'For sphere ',i
			 print *,'Vely in the x-direction, ',Ux(i)
			 print *,'Vely in the y-direction, ',Uy(i)
			 print *,'Vely in the z-direction, ',Uz(i)
			 print *,'Rotn about the x-direction, ',Ox(i)
			 print *,'Rotn about the y-direction, ',Oy(i)
			 print *,'Rotn about the z-direction, ',Oz(i)
			 print '(A25,F21.17)',' Stresslet xx-component, ',Sxx(i)
			 print '(A25,F21.17)','Stresslet yy-component, ',Syy(i)
			 print '(A25,F21.17)','Stresslet zz-component, ',-Sxx(i)-Syy(i)
			 print '(A25,F21.17)','Stresslet xy-component, ',Sxy(i)
			 print '(A25,F21.17)','Stresslet xz-component, ',Sxz(i)
			 print '(A25,F21.17)','Stresslet yz-component, ',Syz(i)
			 print *,'-------------------------------------'
		  end do
      end if
*
      if (outputform=="mathematica") then
            write (*,"(A,F22.19,A,F22.19,A,F22.19,A,F22.19,A,F22.19,A,
     +      F22.19,A,F22.19,A,F22.19,A,F22.19,A,F22.19,A,F22.19,A,
     +      F22.19,A,F22.19,A,F22.19,A,F22.19,A,F22.19,A,F22.19,A,
     +      F22.19,A,F22.19,A,F22.19,A,F22.19,A,F22.19,A)"
     +      ,advance='no')
     + '{',Ux(1),'`20,', Uy(1),'`20,',Uz(1),
     +			  '`20,',Ox(1),'`20,',
     +	  		  Oy(1),'`20,',Oz(1),'`20,',Sxx(1),'`20,',Syy(1),'`20,',
     + 			  Sxy(1),'`20,',Sxz(1),'`20,',Syz(1),'`20,',
     +			  Ux(2),'`20,',Uy(2),'`20,',Uz(2),'`20,',Ox(2),'`20,',
     +			  Oy(2),'`20,',Oz(2),'`20,',Sxx(2),'`20,',Syy(2),'`20,',
     +			  Sxy(2),'`20,',Sxz(2),'`20,',Syz(2),'`20}'
         endif

      if(outputform=="short") then
            print *, Ux(1), Uy(1), Uz(1), Ox(1), Oy(1), Oz(1),
     >     Sxx(1),Syy(1),Sxy(1),Sxz(1),Syz(1),
     >     Ux(2), Uy(2), Uz(2), Ox(2), Oy(2), Oz(2),
     >     Sxx(2),Syy(2),Sxy(2),Sxz(2),Syz(2)
      endif
*
      end
*
*======================================================================*
*
      SUBROUTINE Output_Velocities_File(lambda, error, k)
*
      INCLUDE 'parameters.f'
      integer k
      real*8 lambda, error, Theta
      real*8 Ux(1:Nspheres),Uy(1:Nspheres),Uz(1:Nspheres),
     +     Ox(1:Nspheres),Oy(1:Nspheres),Oz(1:Nspheres),
     +     Sxx(Nspheres), Sxy(Nspheres), Syy(Nspheres), Sxz(Nspheres),
     +     Syz(Nspheres)
*
      call Calculate_Velocities(Ux,Uy,Uz,Ox,Oy,Oz,Sxx,Sxy,Syy,Sxz,Syz)
      write(10,900) lambda, s, error, k,
     >     Ux(1), Uy(1), Uz(1), Ox(1), Oy(1), Oz(1),
     >     Ux(2), Uy(2), Uz(2), Ox(2), Oy(2), Oz(2),
     >     Sxx(1), Sxy(1), Syy(1), Sxz(1), Syz(1),
     >     Sxx(2), Sxy(2), Syy(2), Sxz(2), Syz(2)
 900  format (3F12.6,I5,22E20.8)
*
      end
*
*======================================================================*
*
      SUBROUTINE Calculate_Velocities(Ux,Uy,Uz,Ox,Oy,Oz,Sxx,Sxy,Syy,
     >     Sxz,Syz)
*
      INCLUDE 'parameters.f'
*
************************************************************************
*                                                                      *
*     Calculates the velocities of all N particles, and stores them    *
*     in the arrays Ux, Uy, Uz; and the angular velocities stored in   *
*     arrays Ox, Oy, Oz.                                               *
*                                                                      *
************************************************************************
*
      real*8 Ux(1:Nspheres),Uy(1:Nspheres),Uz(1:Nspheres),
     +     Ox(1:Nspheres),Oy(1:Nspheres),Oz(1:Nspheres),
     +     Sxx(Nspheres), Sxy(Nspheres), Syy(Nspheres), Sxz(Nspheres),
     +     Syz(Nspheres)
*
      integer i,j,n,m
      real*8 root(0:2*Nmax), fact(0:2*Nmax)
      common /sqrts/ root, fact
      real*8 H, Uxr, Uxi, Uyr, Uyi, Uzr, Uzi,
     +     Oxr, Oxi, Oyr, Oyi, Ozr, Ozi,
     +     Sxxr, Sxxi, Sxyr, Sxyi, Syyr, Syyi, Sxzr, Sxzi, Syzr, Syzi
      real*8 V(1:2,1:Nspheres,-1:1), Om(1:2,1:Nspheres,-1:1),
     +     Sst(1:2,1:Nspheres,-2:2)
*
      do i=1,Nspheres
         do j=1,2
            do n=0,Neach(i)
               do m=-n,n
                  A(j,i,1,n,m) = A(j,i,1,n,m) + A(j,i,2,n,m)
                  B(j,i,1,n,m) = B(j,i,1,n,m) + B(j,i,2,n,m)
                  C(j,i,1,n,m) = C(j,i,1,n,m) + C(j,i,2,n,m)
                  D(j,i,1,n,m) = D(j,i,1,n,m) + D(j,i,2,n,m)
                  E(j,i,1,n,m) = E(j,i,1,n,m) + E(j,i,2,n,m)
                  F(j,i,1,n,m) = F(j,i,1,n,m) + F(j,i,2,n,m)
               end do
            end do
         end do
      end do
*
      do i=1,Nspheres
         do j=1,2
            do m=-1,1
               V(j,i,m) = H(1,m)*(-5.d0*A(j,i,1,1,m)/radius(i)**3 +
     +              1.5d0*C(j,i,1,1,m)/radius(i) +
     +              D(j,i,1,1,m))
               Om(j,i,m) = H(1,m)*(B(j,i,1,1,m)/radius(i)**3 +
     +              E(j,i,1,1,m))
            end do
            do m=-2,2
               Sst(j,i,m) = H(2,m)*(4.d0*pi*radius(i)**3.d0/15.d0)*(
     +              20.d0*D(j,i,1,2,m) -3.d0*C(j,i,1,2,m)/radius(i)**3 +
     +              2.d0*radius(i)**2.d0*F(j,i,1,2,m))
c--- \bS = \frac{4}{15}\pi\mu a^3\left(- 3a^{-3} \bC + 20\bD + 2a^2\bF\right)
            end do
         end do
      end do
*
      do i=1,Nspheres
         Uxr = V(1,i,0)
         Uxi = V(2,i,0)
         Uyr = V(1,i,-1)-V(1,i,1)
         Uyi = V(2,i,-1)-V(2,i,1)
         Uzr = -V(2,i,1)-V(2,i,-1)
         Uzi = V(1,i,1)+V(1,i,-1)
         Oxr = Om(1,i,0)
         Oxi = Om(2,i,0)
         Oyr = Om(1,i,-1)-Om(1,i,1)
         Oyi = Om(2,i,-1)-Om(2,i,1)
         Ozr = -Om(2,i,1)-Om(2,i,-1)
         Ozi = Om(1,i,1)+Om(1,i,-1)
c========c
c         Sst(1,i,3,2,0) = Stens(i,1,1)
c         Sst(1,i,3,2,1) = Stens(i,1,2)/(3.d0)
c         Sst(1,i,3,2,-1) = -Stens(i,1,2)/(3.d0)
c         Sst(2,i,3,2,1) = Stens(i,1,3)/(3.d0)
c         Sst(2,i,3,2,-1) = Stens(i,1,3)/(3.d0)
c         Sst(1,i,3,2,2) = (2.d0*Stens(i,2,2) + Stens(i,1,1))/(12.d0)
c         Sst(1,i,3,2,-2) = (2.d0*Stens(i,2,2) + Stens(i,1,1))/(12.d0)
c         Sst(2,i,3,2,2) = -Stens(i,2,3)/(6.d0)
c         Sst(2,i,3,2,-2) = Stens(i,2,3)/(6.d0)
c========c Old version
c         Sxxr = Sst(1,i,0)
c         Sxxi = Sst(2,i,0)
c         Syyr = -0.5d0*Sst(1,i,0) + 3.d0*(Sst(1,i,-2)+Sst(1,i,2))
c         Syyi = -0.5d0*Sst(2,i,0) + 3.d0*(Sst(2,i,-2)+Sst(2,i,2))
c         Sxyr = 3.d0*(Sst(1,i,-1)-Sst(1,i,1))
c         Sxyi = 3.d0*(Sst(2,i,-1)-Sst(2,i,1))
c         Sxzr = -3.d0*(Sst(2,i,-1)+Sst(2,i,1))
c         Sxzi = 3.d0*(Sst(1,i,-1)+Sst(1,i,1))
c         Syzr = -6.d0*(Sst(2,i,-2)-Sst(2,i,2))
c         Syzi = 6.d0*(Sst(1,i,-2)-Sst(1,i,2))
c========c New version
         Sxxr = Sst(1,i,0)
         Sxxi = Sst(2,i,0)
         Syyr = -0.5d0*Sst(1,i,0) + 3.d0*(Sst(1,i,-2)+Sst(1,i,2))
         Syyi = -0.5d0*Sst(2,i,0) + 3.d0*(Sst(2,i,-2)+Sst(2,i,2))
         Sxyr = 1.5d0*(Sst(1,i,1)-Sst(1,i,-1))
         Sxyi = 1.5d0*(Sst(2,i,1)-Sst(2,i,-1))
         Sxzr = 1.5d0*(Sst(2,i,1)+Sst(2,i,-1))
         Sxzi = -1.5d0*(Sst(1,i,1)+Sst(1,i,-1))
         Syzr = 3.d0*(Sst(2,i,-2)-Sst(2,i,2))
         Syzi = -3.d0*(Sst(1,i,-2)-Sst(1,i,2))
         if (dabs(Uxi) .GT. 1.d-4) STOP 'x-Vely is complex'
         if (dabs(Uyi) .GT. 1.d-4) STOP 'y-Vely is complex'
         if (dabs(Uzi) .GT. 1.d-4) STOP 'z-Vely is complex'
         if (dabs(Oxi) .GT. 1.d-4) STOP 'x-Rotn is complex'
         if (dabs(Oyi) .GT. 1.d-4) STOP 'y-Rotn is complex'
         if (dabs(Ozi) .GT. 1.d-4) STOP 'z-Rotn is complex'
         if (dabs(Sxxi) .GT. 1.d-4) STOP 'x-Vely is complex'
         if (dabs(Sxyi) .GT. 1.d-4) STOP 'x-Vely is complex'
         if (dabs(Syyi) .GT. 1.d-4) STOP 'x-Vely is complex'
         if (dabs(Sxzi) .GT. 1.d-4) STOP 'x-Vely is complex'
         if (dabs(Syzi) .GT. 1.d-4) STOP 'x-Vely is complex'
         Ux(i) = Uxr
         Uy(i) = Uyr
         Uz(i) = Uzr
         Ox(i) = Oxr
         Oy(i) = Oyr
         Oz(i) = Ozr
         Sxx(i) = Sxxr
         Sxy(i) = Sxyr
         Syy(i) = Syyr
         Sxz(i) = Sxzr
         Syz(i) = Syzr
      end do
*
      end
*
*======================================================================*
*
      SUBROUTINE Rotate_arrival(k,i)
*
      INCLUDE 'parameters.f'
*
      integer k,i,j,n,m
*
      do j=1,2
         do n=0,NN(k,i)
            do m=-n,n
               D(j,k,3,n,m) = D(j,k,3+i,n,m)
               E(j,k,3,n,m) = E(j,k,3+i,n,m)
               F(j,k,3,n,m) = F(j,k,3+i,n,m)
            end do
         end do
      end do
*
      end
*
*======================================================================*
