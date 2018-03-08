module image3d
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc
  use image, only : ixiy2ixy, ixy2ixiy, comreg, zeroeps
  implicit none
contains
!
!
!-------------------------------------------------------------------------------
! Copy 1D image vector from/to 3D image vector
!-------------------------------------------------------------------------------
! I1d --> I3d
subroutine I1d_I3d_fwd(xidx,yidx,I1d,I3d,N1d,Nx,Ny,Nz)
  implicit none
  !
  integer, intent(in) :: N1d,Nx,Ny,Nz
  integer, intent(in) :: xidx(N1d), yidx(N1d)
  real(dp),intent(in) :: I1d(N1d)
  real(dp),intent(inout) :: I3d(Nx,Ny,Nz)
  !
  integer :: i, iz, Nxy
  !
  Nxy = Nx*Ny
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nxy,Nz,I1d,xidx,yidx) &
  !$OMP   PRIVATE(i,iz)
  do iz=1,Nz
    do i=1,Nxy !N1d
      I3d(xidx(i),yidx(i),iz)=I1d(i + Nxy*(iz - 1))
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
! I1d <-- I3d
subroutine I1d_I3d_inv(xidx,yidx,I1d,I3d,N1d,Nx,Ny,Nz)
  implicit none
  !
  integer, intent(in) :: N1d,Nx,Ny,Nz
  integer, intent(in) :: xidx(N1d), yidx(N1d)
  real(dp),intent(inout) :: I1d(N1d)
  real(dp),intent(in) :: I3d(Nx,Ny,Nz)
  !
  integer :: i,iz
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d,Nz,I3d,xidx,yidx) &
  !$OMP   PRIVATE(i,iz)
  do iz=1,Nz
    do i=1,N1d
      I1d(i)=I3d(xidx(i),yidx(i),iz)
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine


!-------------------------------------------------------------------------------
! Regularization Function
!-------------------------------------------------------------------------------
!
! Centoroid Regularization
!
! subroutine comreg3d(xidx,yidx,Nxref,Nyref,alpha,I1d,cost,gradcost,Npix,Nz)
!   implicit none
!   !
!   integer, intent(in) :: Npix,Nz
!   integer, intent(in) :: xidx(Npix), yidx(Npix)
!   real(dp),intent(in) :: alpha
!   real(dp),intent(in) :: Nxref, Nyref
!   real(dp),intent(in) :: I1d(Npix*Nz)
!   real(dp),intent(inout) :: cost
!   real(dp),intent(inout) :: gradcost(1:Npix)
!   !
!   real(dp) :: Isum(Npix)
!   !
!   real(dp) :: dix, diy, Ip
!   real(dp) :: sumx, sumy, sumI
!   real(dp) :: gradsumx, gradsumy, gradsumI
!   !
!   integer :: ipix
!
! end subroutine
!
!
!-------------------------------------------------------------------------------
! A convinient function to compute regularization functions
! for python interfaces
!-------------------------------------------------------------------------------
!
!
!-------------------------------------------------------------------------------
! A convinient function to compute regularization functions
! for python interfaces
!-------------------------------------------------------------------------------
subroutine ixyz2ixiyiz(ixyz,ix,iy,iz,Nx,Ny)
  implicit none

  ! arguments
  integer, intent(in):: ixyz,Nx,Ny
  integer, intent(out):: ix,iy,iz
  !
  integer :: ixy

  ! calc ix, iy
  ixy = mod(ixyz-1,Nx*Ny)+1
  call ixy2ixiy(ixy,ix,iy,Nx)

  ! calc iz
  iz = (ixyz-1)/(Nx*Ny)+1
end subroutine


subroutine ixiyiz2ixyz(ix,iy,iz,ixyz,Nx,Ny)
  implicit none

  ! arguments
  integer, intent(in):: ix,iy,iz,Nx,Ny
  integer, intent(out):: ixyz
  !
  integer :: ixy

  call ixiy2ixy(ix,iy,ixy,Nx)
  ixyz = ixy + (iz-1) * Nx * Ny
end subroutine
end module
