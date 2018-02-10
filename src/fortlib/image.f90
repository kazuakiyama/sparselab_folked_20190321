module image
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc
  implicit none
  ! Epsiron for Zero judgement
  real(dp), parameter :: zeroeps=1d-10
contains
!-------------------------------------------------------------------------------
! Copy 1D image vector from/to 2D/3D image vector
!-------------------------------------------------------------------------------
subroutine I1d_I2d_fwd(xidx,yidx,I1d,I2d,N1d,Nx,Ny)
  !
  ! I1d --> I2d
  !
  implicit none
  !
  integer, intent(in) :: N1d,Nx,Ny
  integer, intent(in) :: xidx(N1d), yidx(N1d)
  real(dp),intent(in) :: I1d(N1d)
  real(dp),intent(inout) :: I2d(Nx,Ny)
  !
  integer :: i
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d,I1d,xidx,yidx) &
  !$OMP   PRIVATE(i)
  do i=1,N1d
    I2d(xidx(i),yidx(i))=I1d(i)
  end do
  !$OMP END PARALLEL DO
end subroutine

! I1d <-- I2d
subroutine I1d_I2d_inv(xidx,yidx,I1d,I2d,N1d,Nx,Ny)
  implicit none
  !
  integer, intent(in) :: N1d,Nx,Ny
  integer, intent(in) :: xidx(N1d), yidx(N1d)
  real(dp),intent(inout) :: I1d(N1d)
  real(dp),intent(in) :: I2d(Nx,Ny)
  !
  integer :: i
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d,I2d,xidx,yidx) &
  !$OMP   PRIVATE(i)
  do i=1,N1d
    I1d(i)=I2d(xidx(i),yidx(i))
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
!-------------------------------------------------------------------------------
! Transform Image (log-domain regularization)
!-------------------------------------------------------------------------------
subroutine log_fwd(thres,I1d,I1dout,N)
  implicit none
  !
  integer, intent(in) :: N
  real(dp), intent(in) :: thres
  real(dp), intent(in) :: I1d(N)
  real(dp), intent(out) :: I1dout(N)
  !
  integer :: i
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N,thres,I1d) &
  !$OMP   PRIVATE(i)
  do i=1,N
    if (abs(I1d(i)) > zeroeps) then
      I1dout(i)=(log(abs(I1d(i))+thres)-log(thres)) * sign(1d0,I1d(i))
    else
      I1dout(i) = 0d0
    end if
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine log_inv(thres,I1d,I1dout,N)
  implicit none
  !
  integer, intent(in) :: N
  real(dp), intent(in) :: thres
  real(dp), intent(in) :: I1d(N)
  real(dp), intent(out) :: I1dout(N)
  !
  integer :: i
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N,thres,I1d) &
  !$OMP   PRIVATE(i)
  do i=1,N
    if (abs(I1d(i)) > zeroeps) then
      I1dout(i) = thres*(exp(abs(I1d(i)))-1)*sign(1d0,I1d(i))
    else
      I1dout(i) = 0d0
    end if
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine log_grad(thres,I1d,gradreg,N)
  implicit none
  !
  integer, intent(in) :: N
  real(dp), intent(in) :: thres
  real(dp), intent(in) :: I1d(N)
  real(dp), intent(inout) :: gradreg(N)
  !
  integer :: i
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N,thres) &
  !$OMP   PRIVATE(i)
  do i=1,N
    if (abs(I1d(i)) > zeroeps) then
      gradreg(i) = gradreg(i)/(abs(I1d(i)) + thres)
    else
      gradreg(i) = 0d0
    end if
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine gamma_fwd(gamma,I1d,I1dout,N)
  implicit none
  !
  integer, intent(in) :: N
  real(dp), intent(in) :: gamma
  real(dp), intent(in) :: I1d(N)
  real(dp), intent(out) :: I1dout(N)
  !
  integer :: i
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N,gamma,I1d) &
  !$OMP   PRIVATE(i)
  do i=1,N
    if (abs(I1d(i)) > zeroeps) then
      I1dout(i)=abs(I1d(i))**gamma * sign(1d0,I1d(i))
    else
      I1dout(i) = 0d0
    end if
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine gamma_inv(gamma,I1d,I1dout,N)
  implicit none
  !
  integer, intent(in) :: N
  real(dp), intent(in) :: gamma
  real(dp), intent(in) :: I1d(N)
  real(dp), intent(out) :: I1dout(N)
  !
  integer :: i
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N,gamma,I1d) &
  !$OMP   PRIVATE(i)
  do i=1,N
    if (abs(I1d(i)) > zeroeps) then
      I1dout(i) = abs(I1d(i))**(1/gamma) * sign(1d0,I1d(i))
    else
      I1dout(i) = 0d0
    end if
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine gamma_grad(gamma,I1d,gradreg,N)
  implicit none
  !
  integer, intent(in) :: N
  real(dp), intent(in) :: gamma
  real(dp), intent(in) :: I1d(N)
  real(dp), intent(inout) :: gradreg(N)
  !
  integer :: i
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N,gamma) &
  !$OMP   PRIVATE(i)
  do i=1,N
    if (abs(I1d(i)) > zeroeps) then
      gradreg(i) = gradreg(i) * gamma * abs(I1d(i))**(gamma-1) * sign(1d0,I1d(i))
    else
      gradreg(i) = 0d0
    end if
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
!-------------------------------------------------------------------------------
! Regularization Function
!-------------------------------------------------------------------------------
!
! l1-norm
!
real(dp) function l1_e(I)
  implicit none
  real(dp),intent(in) :: I
  !
  l1_e = abs(I)
end function
!
! gradient of l1-norm
!
real(dp) function l1_grade(I)
  implicit none
  !
  real(dp),intent(in) :: I
  !
  if (abs(I) > zeroeps) then
    l1_grade = sign(1d0,I)
  else
    l1_grade = 0
  end if
end function
!
! MEM
!
real(dp) function mem_e(I)
  implicit none
  !
  real(dp),intent(in) :: I
  if (abs(I) > zeroeps) then
    mem_e = abs(I)*log(abs(I))
  else
    mem_e = 0d0
  end if
end function
!
! gradient of mem
!
real(dp) function mem_grade(I)
  implicit none
  !
  real(dp),intent(in) :: I
  if (abs(I) > zeroeps) then
    mem_grade = (log(abs(I))+1) * sign(1d0,I)
  else
    mem_grade = 0d0
  end if
end function
!
! Isotropic Total Variation
!
real(dp) function tv_e(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer,  intent(in)  ::Nx,Ny,xidx,yidx
  real(dp), intent(in)  ::I2d(Nx,Ny)
  !
  integer   ::i1,j1,i2,j2
  real(dp)  ::dIx,dIy
  !
  i1 = xidx
  j1 = yidx
  i2 = i1 + 1
  j2 = j1 + 1

  ! dIx = I(i+1,j) - I(i,j)
  if (i2 > Nx) then
    dIx = 0d0
  else
    dIx = I2d(i2,j1) - I2d(i1,j1)
  end if

  ! dIy = I(i,j+1) - I(i,j)
  if (j2 > Ny) then
    dIy = 0d0
  else
    dIy = I2d(i1,j2) - I2d(i1,j1)
  end if
  !
  tv_e = sqrt(dIx*dIx+dIy*dIy)
end function
!
! Gradient of Isotropic Total Variation
!
real(dp) function tv_grade(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in) :: Nx,Ny
  integer, intent(in) :: xidx, yidx
  real(dp),intent(in) :: I2d(Nx,Ny)
  !
  ! variables
  integer :: i0,j0,i1,j1,i2,j2
  real(dp):: dIx,dIy,tv_e
  !
  ! initialize tsv term
  tv_grade = 0d0
  !
  ! take indice
  i1 = xidx
  j1 = yidx
  i0 = i1 - 1
  j0 = j1 - 1
  i2 = i1 + 1
  j2 = j1 + 1
  !
  !-------------------------------------
  ! (i2,j1)-(i1,j1), (i1,j2)-(i1,j1)
  !-------------------------------------
  ! dIx = I(i+1,j) - I(i,j)
  if (i2 > Nx) then
    dIx = 0d0
  else
    dIx = I2d(i2,j1) - I2d(i1,j1)
  end if
  !
  ! dIy = I(i,j+1) - I(i,j)
  if (j2 > Ny) then
    dIy = 0d0
  else
    dIy = I2d(i1,j2) - I2d(i1,j1)
  end if
  !
  tv_e = sqrt(dIx*dIx+dIy*dIy)
  if (tv_e > zeroeps) then
    tv_grade = tv_grade - (dIx + dIy)/tv_e
  end if
  !
  !-------------------------------------
  ! (i1,j1)-(i0,j1), (i0,j2)-(i0,j1)
  !-------------------------------------
  if (i0 > 0) then
    ! dIx = I(i,j) - I(i-1,j)
    dIx = I2d(i1,j1) - I2d(i0,j1)

    ! dIy = I(i-1,j+1) - I(i,j)
    if (j2 > Ny) then
      dIy = 0d0
    else
      dIy = I2d(i0,j2) - I2d(i0,j1)
    end if

    tv_e = sqrt(dIx*dIx+dIy*dIy)
    if (tv_e > zeroeps) then
      tv_grade = tv_grade + dIx/tv_e
    end if
  end if
  !
  !-------------------------------------
  ! (i2,j0)-(i1,j0), (i1,j1)-(i1,j0)
  !-------------------------------------
  if (j0 > 0) then
    ! dIy = I(i,j) - I(i,j-1)
    dIy = I2d(i1,j1) - I2d(i1,j0)

    ! dIx = I(i+1,j-1) - I(i,j-1)
    if (i2 > Nx) then
      dIx = 0d0
    else
      dIx = I2d(i2,j0) - I2d(i1,j0)
    end if

    tv_e = sqrt(dIx*dIx+dIy*dIy)
    if (tv_e > zeroeps) then
      tv_grade = tv_grade + dIy/tv_e
    end if
  end if
  !
end function
!
! Total Squared Variation
!
real(dp) function tsv_e(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,xidx,yidx
  real(dp),intent(in)  :: I2d(Nx,Ny)
  !
  ! variables
  integer :: i1,j1,i2,j2
  real(dp):: dIx,dIy
  !
  ! initialize tsv term
  i1 = xidx
  j1 = yidx
  i2 = i1 + 1
  j2 = j1 + 1
  !
  ! dIx = I(i+1,j) - I(i,j)
  if (i2 > Nx) then
    dIx = 0d0
  else
    dIx  = I2d(i2,j1) - I2d(i1,j1)
  end if
  !
  ! dIy = I(i,j+1) - I(i,j)
  if (j2 > Ny) then
    dIy = 0d0
  else
    dIy  = I2d(i1,j2) - I2d(i1,j1)
  end if
  !
  tsv_e = dIx*dIx+dIy*dIy
end function
!
! Gradient of Total Squared Variation
!
real(dp) function tsv_grade(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny
  integer, intent(in)  :: xidx, yidx
  real(dp),intent(in)  :: I2d(Nx,Ny)
  !
  ! variables
  integer :: i0,j0,i1,j1,i2,j2
  !
  ! initialize tsv term
  tsv_grade = 0d0
  !
  ! take indice
  i1 = xidx
  j1 = yidx
  i0 = i1 - 1
  j0 = j1 - 1
  i2 = i1 + 1
  j2 = j1 + 1
  !
  ! dIx = I(i+1,j) - I(i,j)
  if (i2 <= Nx) then
    tsv_grade = tsv_grade - 2*(I2d(i2,j1) - I2d(i1,j1))
  end if
  !
  ! dIy = I(i,j+1) - I(i,j)
  if (j2 <= Ny) then
    tsv_grade = tsv_grade - 2*(I2d(i1,j2) - I2d(i1,j1))
  end if
  !
  ! dIx = I(i,j) - I(i-1,j)
  if (i0 > 0) then
    tsv_grade = tsv_grade + 2*(I2d(i1,j1) - I2d(i0,j1))
  end if
  !
  ! dIy = I(i,j) - I(i,j-1)
  if (j0 > 0) then
    tsv_grade = tsv_grade + 2*(I2d(i1,j1) - I2d(i1,j0))
  end if
  !
end function
!
! Centoroid Regularization
!
subroutine comreg(xidx,yidx,Nxref,Nyref,alpha,I1d,cost,gradcost,Npix)
  implicit none
  !
  integer, intent(in) :: Npix
  integer, intent(in) :: xidx(1:Npix), yidx(1:Npix)
  real(dp),intent(in) :: alpha
  real(dp),intent(in) :: Nxref, Nyref
  real(dp),intent(in) :: I1d(1:Npix)
  real(dp),intent(inout) :: cost
  real(dp),intent(inout) :: gradcost(1:Npix)
  !
  real(dp) :: dix, diy, Ip
  real(dp) :: sumx, sumy, sumI
  real(dp) :: gradsumx, gradsumy, gradsumI
  !
  integer :: ipix

  sumx = 0d0
  sumy = 0d0
  sumI = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,I1d,Npix) &
  !$OMP   PRIVATE(ipix, dix, diy, Ip) &
  !$OMP   REDUCTION(+: sumx, sumy, sumI)
  do ipix=1, Npix
    ! pixel from the reference pixel
    dix = xidx(ipix) - Nxref
    diy = yidx(ipix) - Nyref

    ! take a alpha
    if (abs(I1d(ipix)) > zeroeps) then
      Ip = abs(I1d(ipix))**alpha
    else
      Ip = 0
    end if

    ! calculate sum
    sumx = sumx + Ip * dix
    sumy = sumy + Ip * diy
    sumI = sumI + Ip
  end do
  !$OMP END PARALLEL DO

  if (abs(sumI) > zeroeps) then
    ! calculate cost function
    cost = cost + sqrt(sumx*sumx+sumy*sumy)/sumI

    ! calculate gradient of cost function
    !$OMP PARALLEL DO DEFAULT(SHARED) &
    !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,I1d,Npix,sumx,sumy,sumI,cost) &
    !$OMP   PRIVATE(ipix,dix,diy,gradsumI,gradsumx,gradsumy) &
    !$OMP   REDUCTION(+:gradcost)
    do ipix=1, Npix
      ! pixel from the reference pixel
      dix = xidx(ipix) - Nxref
      diy = yidx(ipix) - Nyref

      ! gradient of sums
      if (abs(I1d(ipix)) > zeroeps) then
        gradsumI = alpha*abs(I1d(ipix))**(alpha-1)*sign(1d0,I1d(ipix))
      else
        gradsumI = 0
      end if
      gradsumx = gradsumI*dix
      gradsumy = gradsumI*diy

      ! calculate gradint of cost function
      gradcost(ipix) = gradcost(ipix) + cost*gradsumI/sumI
      gradcost(ipix) = gradcost(ipix) + (sumx*gradsumx+sumy*gradsumy)/(cost*sumI**3)
    end do
    !$OMP END PARALLEL DO
  end if
end subroutine

end module
