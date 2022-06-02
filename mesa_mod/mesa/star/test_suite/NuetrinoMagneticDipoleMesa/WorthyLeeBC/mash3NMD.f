      program mash3
C     Inquires of the user stellar parameters and returns colors, with errors

      integer      nind,nvk,nfe,ng
      parameter    (nind=9,nvk=86,nfe=7,ng=13)
C     The 8th nind is the temperature. 
C     There are actually only 75 vk points, but don't change the nvk parameter.
C     The 9th nind is BC_v
      real         a(nvk,nfe,ng,nind)
      real         grav,feh,teff,theta,logL,clrs(nind),cerr(nind)
      real         ete(8),eterr(8),cpres(9),bcefloor(8)
C     out-of-bounds flag. Nominal is zero.
      integer      error,iflag
      real         solarMag,magConst
      real         BCi,Mbol,Mi,Mierr

C      real         mass, helium, metal

C     theta and assumed percentage error arrays:
      data ete /0.1008,0.252,0.504,0.84,1.26,1.44,1.68,2.52 /
c         Teff= 50000, 20000,10000,6000,4000,3500,3000,2000
      data eterr / 4.0, 2.5, 1.0, 0.5, 0.5, 1.0, 1.5, 4.0 /
C     The cool errors probably ought to be a bit bigger for dwarfs, but that's
C       not implemented...

C     color (or BC) intrinsic precisions (entry 8, 9 are dummies)
C      data cpres / .014, .007, .007, .007, .007, .007, .007, .0, .05 /
      data cpres / .071, .017, .010, .010, .011, .004, .002, .0, .05 /
C     make bolometric correction floor error depend on theta
      data bcefloor / 0.2, 0.1, 0.07, 0.05, 0.05, 0.07, 0.10, 0.2 /
      integer :: numLines          ! Number of MESA models
      character(len=32) :: arg, inputFile, outputFile
C     set extrapolation flag to nominal value
      iflag = 0

C     load data table
      call readtable(a)

C     Begin MD edits:


C     arg: number of runs
C     inputFile: MESA output
C     outputFile: output file
      call get_command_argument(1, arg)
      call get_command_argument(2, inputFile)
      call get_command_argument(3, outputFile)
      
      read(arg, *) numLines
      
      open(42, file=inputFile)
      open(73, file=outputFile, status='new')

      solarMag=4.74
      magConst=2.5
      do j=1,numLines
         iflag=0
C     inquire as to input parameters
C     print*, 'enter log g '
C     read*, grav
C     print*, 'enter [Fe/H]'
C     read*, feh
C     print*, 'enter Teff '
C     read*, teff
C     print*, ' '
C     switch Teff to THETA
         read(42, *) error, grav, teff, feh, logL
         theta = 5040./teff
         if ( error .ne. 1 ) then
C     get colors for these parameters
            call teffinterp(a,grav,feh,theta,clrs,iflag)
C     iflag warns of extrapolations.
C     Nominally zero, it becomes -N if too hot or N if too cool,
C     where N is the number of point spacings extrapolated.
C     Obviously, abs(iflag) > 1 is of concern . . .
C            if ( iflag .ne. 0 ) print*, 'Extrapolation warning! Iflag =',iflag
C     and colors for 1% lower temperature to compute errors
            call teffinterp(a,grav,feh,1.01*theta,cerr,iflag)
            call linear(ete,eterr,8,theta,xerr,iOK)
            if (iOK.le.-2) print*, 'Panic! Interpolation error!'
            do i=1,7
               cerr(i)=sqrt(cpres(i)**2+(xerr**2)*(cerr(i)-clrs(i))**2)
            end do
C     BCv error
            call linear(ete,bcefloor,8,theta,xbcerr,iOK)
            cerr(9) = sqrt(xbcerr**2 + (xerr**2)*(cerr(9)-clrs(9))**2)

            BCi=clrs(4)+clrs(9)
            Mbol=solarMag-magConst*logL
            Mi=Mbol-BCi
            Mierr=sqrt(cerr(4)**2+cerr(9)**2)
C     write(73,'(a6,f7.3,a5,f5.3)') 'U-B = ',clrs(1),' +/- ',cerr(1)
C     write(73,'(a6,f7.3,a5,f5.3)') 'B-V = ',clrs(2),' +/- ',cerr(2)
C     write(73,'(a6,f7.3,a5,f5.3)') 'V-R = ',clrs(3),' +/- ',cerr(3)
C     write(73,'(a6,f7.3,a5,f5.3)') 'V-I = ',clrs(4),' +/- ',cerr(4)
C     write(73,'(a6,f7.3,a5,f5.3)') 'J-K = ',clrs(5),' +/- ',cerr(5)
C     write(73,'(a6,f7.3,a5,f5.3)') 'H-K = ',clrs(6),' +/- ',cerr(6)
C     write(73,'(a6,f7.3,a5,f5.3)') 'V-K = ',clrs(7),' +/- ',cerr(7)
C     write(73,'(a6,f7.3,a5,f5.3)') 'BCv = ',clrs(9),' +/- ',cerr(9)
            write(73, '(i1,a2,i3,a2,f7.3,a2,f5.3)') 
     &       error,', ',iflag,', ',Mi,', ',Mierr
         else
            write(73, '(a3,a3,a3,a1)') '1, ','1, ','1, ','1'
         end if
         
      end do
      stop
      end

C--------------------------------------------------------------------------
      subroutine readtable(a)

      integer      nind,nvk,nfe,ng,i
      parameter    (nind=9,nvk=86,nfe=7,ng=13)
C     the 8th nind is the temperature
C     the 9th nind is BC_v
      real         a(nvk,nfe,ng,nind),g(ng),fe(nfe)

      data g / -0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5 /
      data fe /  -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5  /

      open(unit=19,file='mash2.out',status='old')
      do ife = 1,nfe
         do ig = 1,ng
            do ivk=1,75
               read(19,*) x1,x2,xteff,x3,
     z                 (a(ivk,ife,ig,k),k=1,7),a(ivk,ife,ig,9)
               if ( abs(x1 - fe(ife)).gt. 0.01 ) print*, 'bad fe'
               if ( abs(x2 - g(ig)).gt. 0.01 ) print*, 'bad g'
C              transform to THETA rather than Teff . . . 
               a(ivk,ife,ig,8) = 5040./xteff
            end do
         end do
      end do

      return
      end
C----------------------------------------------------------------------------

C     subroutine teffinterp
      subroutine teffinterp(a,grav,feh,theta,clrs,iflag)
C     given a THETA (=5040/Teff), return colors
      integer      nind,nvk,nfe,ng,iflag
      parameter    (nind=9,nvk=86,nfe=7,ng=13)
      real         grav,feh,theta,clrs(nind)
      real         a(nvk,nfe,ng,nind),g(ng),fe(nfe),vk(nvk)
C     the 8th nind is the temperature, 9th BC_v
C     local variables
      integer      jg,jfe
      real         c(nvk,nind)

      data g / -0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5 /
      data fe /  -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5  /
C     clrs are  1 U-B  2 B-V  3 V-R  4 V-I  5 J-K  6 H-K  7 V-K  (8 Teff)
C     a() is organised: a(nvk,nfe,ng,nind)                       (9 BCv)

C     find jfe and jg interpolation corners
      call locate(g,ng,grav,jg)
      if (jg.eq.0) jg = 1
      if (jg.eq.ng) jg = ng-1
      call locate(fe,nfe,feh,jfe)
      if (jfe.eq.0) jfe=1
      if (jfe.eq.nfe) jfe=nfe-1
C     fill c array with bilinear-interp results
      ffe = ( feh - fe(jfe) )/( fe(jfe+1)-fe(jfe)  )
      gg  = ( grav - g(jg) )/( g(jg+1) - g(jg)  )
      do ivk=1,75
         do ind=1,nind
            c(ivk,ind) = (1.0-ffe)*(1.0-gg)*a(ivk,jfe,jg,ind)
     z                 +  ffe     *(1.0-gg)*a(ivk,jfe+1,jg,ind)
     z                 +  ffe     *gg      *a(ivk,jfe+1,jg+1,ind)
     z                 + (1.0-ffe)*gg      *a(ivk,jfe,jg+1,ind)
         end do
      end do

C     find temperature (it's really THETA) and interpolate colors
      call locate(c(1,8),75,theta,jt)
      jt0 = jt
      if (jt.eq.0) jt = 1
      if (jt.gt.70) jt = 71
      do i=1,7
         call polint(c(jt,8),c(jt,i),5,theta,clrs(i),dy)
      end do
      call polint(c(jt,8),c(jt,9),5,theta,clrs(9),dy)
C     if requested temperature is out-of-bounds, use linear interpolation to
C     extrapolate. Return iflag = int(number of segments beyond the tabulated)
      if ( jt0 .eq. 0 ) then
         frac = (theta - c(1,8))/(c(2,8)-c(1,8))
         iflag = -1 + int(frac)
         do i=1,7
            clrs(i) = (1.0-frac)*c(1,i) + frac*c(2,i)
         end do
            clrs(9) = (1.0-frac)*c(1,9) + frac*c(2,9)
      end if
      if ( jt0 .eq. 75 ) then
         frac = (theta - c(74,8))/(c(75,8)-c(74,8))
         iflag = 1 + int(frac)
         do i=1,7
            clrs(i) = (1.0-frac)*c(74,i) + frac*c(75,i)
         end do
            clrs(9) = (1.0-frac)*c(74,9) + frac*c(75,9)
      end if

      return
      end
C-----------------------------------------------------------------------

C     subroutine vkinterp
      subroutine vkinterp(a,grav,feh,theta,clrs)
C     given a V-K (in clrs(7)), return colors and THETA = 5040/Teff
      integer      nind,nvk,nfe,ng
      parameter    (nind=9,nvk=86,nfe=7,ng=13)
      real         grav,feh,theta,clrs(nind)
      real         a(nvk,nfe,ng,nind),g(ng),fe(nfe),vk(nvk)
C     the 8th nind is the temperature, 9th BC_v
C     local variables
      integer      jg,jfe
      real         c(nvk,nind)

      data g / -0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5 /
      data fe /  -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5  /
C     clrs are  1 U-B  2 B-V  3 V-R  4 V-I  5 J-K  6 H-K  7 V-K  (8 Teff)
C     a() is organised: a(nvk,nfe,ng,nind)                       (9 BCv)

C     find jfe and jg interpolation corners
      call locate(g,ng,grav,jg)
      if (jg.eq.0) jg = 1
      if (jg.eq.ng) jg = ng-1
      call locate(fe,nfe,feh,jfe)
      if (jfe.eq.0) jfe=1
      if (jfe.eq.nfe) jfe=nfe-1
C     fill c array with bilinear-interp results
      ffe = ( feh - fe(jfe) )/( fe(jfe+1)-fe(jfe)  )
      gg  = ( grav - g(jg) )/( g(jg+1) - g(jg)  )
      do ivk=1,75
         do ind=1,nind
            c(ivk,ind) = (1.0-ffe)*(1.0-gg)*a(ivk,jfe,jg,ind)
     z                 +  ffe     *(1.0-gg)*a(ivk,jfe+1,jg,ind)
     z                 +  ffe     *gg      *a(ivk,jfe+1,jg+1,ind)
     z                 + (1.0-ffe)*gg      *a(ivk,jfe,jg+1,ind)
         end do
      end do

C     find V-K and interpolate THETA and colors
      call locate(c(1,7),75,clrs(7),jt)
      if (jt.eq.0) jt = 1
      if (jt.gt.70) jt = 71
      do i=1,6
         call polint(c(jt,7),c(jt,i),5,clrs(7),clrs(i),dy)
      end do
      call polint(c(jt,7),c(jt,8),5,clrs(7),clrs(8),dy)
      call polint(c(jt,7),c(jt,9),5,clrs(7),clrs(9),dy)
      theta = clrs(8)

      return
      end
C-----------------------------------------------------------------------
C     subroutine viinterp
      subroutine viinterp(a,grav,feh,theta,clrs)
C     given a V-I (in clrs(4)), return colors and THETA = 5040/Teff
      integer      nind,nvk,nfe,ng
      parameter    (nind=9,nvk=86,nfe=7,ng=13)
      real         grav,feh,theta,clrs(nind)
      real         a(nvk,nfe,ng,nind),g(ng),fe(nfe),vk(nvk)
C     the 8th nind is the temperature, 9th BCv
C     local variables
      integer      jg,jfe
      real         c(nvk,nind)

      data g / -0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5 /
      data fe /  -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5  /
C     clrs are  1 U-B  2 B-V  3 V-R  4 V-I  5 J-K  6 H-K  7 V-K  (8 Teff)
C     a() is organised: a(nvk,nfe,ng,nind)

C     find jfe and jg interpolation corners
      call locate(g,ng,grav,jg)
      if (jg.eq.0) jg = 1
      if (jg.eq.ng) jg = ng-1
      call locate(fe,nfe,feh,jfe)
      if (jfe.eq.0) jfe=1
      if (jfe.eq.nfe) jfe=nfe-1
C     fill c array with bilinear-interp results
      ffe = ( feh - fe(jfe) )/( fe(jfe+1)-fe(jfe)  )
      gg  = ( grav - g(jg) )/( g(jg+1) - g(jg)  )
      do ivk=1,75
         do ind=1,nind
            c(ivk,ind) = (1.0-ffe)*(1.0-gg)*a(ivk,jfe,jg,ind)
     z                 +  ffe     *(1.0-gg)*a(ivk,jfe+1,jg,ind)
     z                 +  ffe     *gg      *a(ivk,jfe+1,jg+1,ind)
     z                 + (1.0-ffe)*gg      *a(ivk,jfe,jg+1,ind)
         end do
      end do

C     find V-I and interpolate THETA and colors
      call locate(c(1,4),75,clrs(4),jt)
      if (jt.eq.0) jt = 1
      if (jt.gt.70) jt = 71
      do i=1,3
         call polint(c(jt,4),c(jt,i),5,clrs(4),clrs(i),dy)
      end do
      do i=5,9
         call polint(c(jt,4),c(jt,i),5,clrs(4),clrs(i),dy)
      end do
      theta = clrs(8)

      return
      end
C-----------------------------------------------------------------------
C------------------------------------------------------------------------
C     subroutine linear
C     quick linear interpolation
      subroutine linear(x,y,nxy,xin,yout,iOK)
      integer    nxy,iOK
      real       x(nxy),y(nxy),xin,yout
C     x and y are nxy long
C     xin is the nontabulated input x value
C     yout is the interpolated y guess.
C     iOK is -3 if X is not in ascending order, -2 if input x is
C     out-of-bounds by more than 1 xpoint spacing, -1 if out-of-bounds
C     by less than 1 xpoint spacing, 0 if all is OK

C     local variables
      integer    j,jl,ju,jm
      real       frac

      iOK = 0
      if ( x(2).lt.x(1) ) then
         print*, 
     z   'Error. Sub LINEAR. Input X array must be in ascending order.'
         yout = 0.0
         iOK = -3
         return
      end if

C     locate correct array element by bisection
      jl=0
      ju=nxy+1
10    if (ju-jl.gt.1) then
         jm = (ju+jl)/2
         if ((x(nxy).gt.x(1)).eqv.(xin.gt.x(jm))) then
            jl=jm
         else
            ju=jm
         end if
         goto 10
      end if
      j = jl
C     j is 0 or nxy if xin is off the grid

C     if off-grid, reset j and set output flag iOK
      if ( j.eq.0) then
         if ( xin.lt.(x(1)-(x(2)-x(1))) ) then
            iOK = -2
         else
            iOK = -1
         end if
         j=1
      endif
      if ( j.eq.nxy) then
         if ( xin.gt.(x(nxy)+(x(nxy)-x(nxy-1))) ) then
            iOK = -2
         else
            iOK = -1
         end if
         j = nxy-1
      end if

C     now interpolate/extrapolate
      frac = (xin - x(j))/(x(j+1)-x(j))
      yout = (1.0-frac)*y(j) + frac*y(j+1)


      return
      end
C     end subroutine linear ---------------------------------------------

C     -----NUMERICAL RECIPES routines: locate and polint
      SUBROUTINE LOCATE(XX,N,X,J)
      DIMENSION XX(N)
      JL=0
      JU=N+1
10    IF(JU-JL.GT.1)THEN
        JM=(JU+JL)/2
        IF((XX(N).GT.XX(1)).EQV.(X.GT.XX(JM)))THEN
          JL=JM
        ELSE
          JU=JM
        ENDIF
      GO TO 10
      ENDIF
      J=JL
      RETURN
      END

      SUBROUTINE POLINT(XA,YA,N,X,Y,DY)
      PARAMETER (NMAX=10) 
      DIMENSION XA(N),YA(N),C(NMAX),D(NMAX)
      NS=1
      DIF=ABS(X-XA(1))
      DO 11 I=1,N 
        DIFT=ABS(X-XA(I))
        IF (DIFT.LT.DIF) THEN
          NS=I
          DIF=DIFT
        ENDIF
        C(I)=YA(I)
        D(I)=YA(I)
11    CONTINUE
      Y=YA(NS)
      NS=NS-1
      DO 13 M=1,N-1
        DO 12 I=1,N-M
          HO=XA(I)-X
          HP=XA(I+M)-X
          W=C(I+1)-D(I)
          DEN=HO-HP
          IF(DEN.EQ.0.)PAUSE
          DEN=W/DEN
          D(I)=HP*DEN
          C(I)=HO*DEN
12      CONTINUE
        IF (2*NS.LT.N-M)THEN
          DY=C(NS+1)
        ELSE
          DY=D(NS)
          NS=NS-1
        ENDIF
        Y=Y+DY
13    CONTINUE
      RETURN
      END
