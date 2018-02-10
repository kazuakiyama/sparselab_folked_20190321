module fftlib
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc
  implicit none

  ! Parameters related to NuFFT
  !   FINUFFT's numerical accracy is around 1d-13
  real(dp), parameter :: ffteps=1d-12

  interface
    subroutine finufft2d1_f(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      integer :: nj, iflag, ms, mt, ier
      real(kind(1.0d0)) :: xj(nj), yj(nj), eps
      complex(kind((1.0d0,1.0d0))) :: cj(nj), fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2)
    end subroutine
  end interface

  interface
    subroutine finufft2d2_f(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      integer :: nj, iflag, ms, mt, ier
      real(kind(1.0d0)) :: xj(nj), yj(nj), eps
      complex(kind((1.0d0,1.0d0))) :: cj(nj), fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2)
    end subroutine
  end interface
contains
!-------------------------------------------------------------------------------
! NuFFT related functions
!-------------------------------------------------------------------------------
subroutine NUFFT_fwd(u,v,I2d,Vcmp,Nx,Ny,Nuv)
  !
  !  Forward Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in)  :: Nx, Ny, Nuv
  real(dp), intent(in)  :: u(Nuv),v(Nuv)  ! uv coordinates
                                          ! multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in)  :: I2d(Nx,Ny)     ! Two Dimensional Image
  complex(dpc), intent(out) :: Vcmp(Nuv)  ! Complex Visibility

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the forward Fourier Transformation
  !     0: positive (the standard in Radio Astronomy)
  !     1: negative (the textbook standard; e.g. TMS)
  integer,  parameter :: iflag=0
  !   Numerical Accuracy required for FINUFFT
  real(dp),  parameter :: eps=ffteps
  !   error log
  integer :: ier

  ! Call FINUFFT subroutine
  call finufft2d2_f(Nuv,u,v,Vcmp,iflag,eps,Nx,Ny,dcmplx(I2d),ier)

  ! debug
  !print *, ' ier = ',ier
end subroutine


subroutine NUFFT_adj(u,v,Vcmp,I2d,Nx,Ny,Nuv)
  !
  !  Adjoint Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in) :: Nx, Ny, Nuv
  real(dp), intent(in) :: u(Nuv),v(Nuv)  ! uv coordinates
                                         ! multiplied by 2*pi*dx, 2*pi*dy
  complex(dpc), intent(in) :: Vcmp(Nuv)  ! Complex Visibility
  complex(dpc), intent(out):: I2d(Nx,Ny) ! Two Dimensional Image

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the adjoint Fourier Transformation
  !     0: positive (the textbook standard TMS)
  !     1: negative (the standard in Radio Astronomy)
  integer, parameter:: iflag=1
  !   Numerical Accuracy required for FINUFFT
  real(dp),  parameter :: eps=ffteps
  !   error log
  integer :: ier

  ! Call FINUFFT subroutine
  call finufft2d1_f(Nuv,u,v,Vcmp,iflag,eps,Nx,Ny,I2d,ier)

  ! debug
  !print *, ' ier = ',ier
end subroutine


subroutine NUFFT_adj_resid(u,v,Vre,Vim,I2d,Nx,Ny,Nuv)
  !
  !  This function takes the adjoint non-uniform Fast Fourier Transform
  !  of input visibilities and then sum the real and imag parts of
  !  the transformed image.
  !
  implicit none

  integer,  intent(in):: Nx, Ny, Nuv
  real(dp), intent(in):: u(Nuv),v(Nuv)      ! uv coordinates
                                            ! multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in):: Vre(Nuv),Vim(Nuv)  ! Complex Visibility
  real(dp), intent(out):: I2d(Nx,Ny)        ! Two Dimensional Image

  complex(dpc):: I2dcmp1(Nx,Ny), I2dcmp2(Nx,Ny)

  ! Call adjoint NuFFT
  call NUFFT_adj(u,v,dcmplx(Vre),I2dcmp1,Nx,Ny,Nuv)
  call NUFFT_adj(u,v,dcmplx(Vim),I2dcmp2,Nx,Ny,Nuv)

  ! Take a sum of real part and imaginary part
  I2d = dreal(I2dcmp1)+dimag(I2dcmp2)
end subroutine


subroutine NUFFT_adjrea(u,v,Vcmp,I2d,Nx,Ny,Nuv)
  !
  !  This function takes the adjoint non-uniform Fast Fourier Transform
  !  of input visibilities and then take the real part of the transformed image.
  !
  implicit none

  integer,  intent(in):: Nx, Ny, Nuv
  real(dp), intent(in):: u(Nuv),v(Nuv)  ! uv coordinates
                                        ! multiplied by 2*pi*dx, 2*pi*dy
  complex(dpc), intent(in):: Vcmp(Nuv)  ! Complex Visibility
  real(dp), intent(out):: I2d(Nx,Ny)    ! Two Dimensional Image

  complex(dpc):: I2dcmp(Nx,Ny)

  ! Call adjoint NuFFT
  call NUFFT_adj(u,v,Vcmp,I2dcmp,Nx,Ny,Nuv)

  ! Take a sum of real part and imaginary part
  I2d = dreal(I2dcmp)
end subroutine


subroutine phashift_c2r(u,v,Nxref,Nyref,Nx,Ny,Vcmp_in,Vcmp_out)
  !
  !  This function shift the tracking center of the input full complex visibilities
  !  from the image center to the reference pixel
  !
  implicit none

  integer,  intent(in):: Nx, Ny
  real(dp), intent(in):: u,v            ! uv coordinates multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in):: Nxref, Nyref   ! x,y reference ppixels (1=the leftmost/lowermost pixel)
  complex(dpc), intent(in)  :: Vcmp_in  ! Complex Visibility
  complex(dpc), intent(out) :: Vcmp_out ! Complex Visibility

  real(dp) :: dix, diy

  ! pixels to be shifted
  dix = Nx/2 + 1 - Nxref
  diy = Ny/2 + 1 - Nyref

  Vcmp_out = Vcmp_in * exp(i_dpc * (u*dix + v*diy))
end subroutine


subroutine phashift_r2c(u,v,Nxref,Nyref,Nx,Ny,Vcmp_in,Vcmp_out)
  !
  !  This function shift the tracking center of the input full complex visibilities
  !  from the reference pixel to the image center
  !
  implicit none

  integer,  intent(in):: Nx, Ny
  real(dp), intent(in):: u,v            ! uv coordinates multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in):: Nxref, Nyref   ! x,y reference pixels
                                        ! (1=the leftmost/lowermost pixel)
  complex(dpc), intent(in)  :: Vcmp_in  ! Complex Visibility
  complex(dpc), intent(out) :: Vcmp_out ! Complex Visibility

  real(dp) :: dix, diy

  ! pixels to be shifted
  dix = Nxref - Nx/2 - 1
  diy = Nyref - Ny/2 - 1

  Vcmp_out = Vcmp_in * exp(i_dpc * (u*dix + v*diy))
end subroutine


!-------------------------------------------------------------------------------
! Functions to compute chisquares and also residual vectors
!-------------------------------------------------------------------------------
subroutine chisq_fcv(Vcmp,&
                     uvidxfcv,Vfcv,Varfcv,&
                     fnorm,&
                     chisq,Vresre,Vresim,&
                     Nuv,Nfcv)
  implicit none

  ! NuFFT-ed visibilities
  integer,      intent(in):: Nuv
  complex(dpc), intent(in):: Vcmp(Nuv)
  ! Data
  integer,  intent(in):: Nfcv             ! Number of data
  integer,  intent(in):: uvidxfcv(Nfcv)   ! UV Index of FCV data
  complex(dpc), intent(in):: Vfcv(Nfcv)   ! Full complex visibility (FCV) data
  real(dp), intent(in):: Varfcv(Nfcv)     ! variances of FCV data
  ! Normalization Factor of Chisquare
  real(dp), intent(in):: fnorm
  ! Outputs
  real(dp), intent(inout):: chisq           ! chisquare
  real(dp), intent(inout):: Vresre(Nuv), Vresim(Nuv) ! residual vector
                                            !   its adjoint FT provides
                                            !   the gradient of chisquare)

  complex(dpc):: resid
  real(dp):: factor
  integer:: uvidx, ifcv

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nfcv,fnorm,uvidxfcv,Vcmp,Vfcv,Varfcv) &
  !$OMP   PRIVATE(ifcv,uvidx,resid,factor),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim)
  do ifcv=1, Nfcv
    ! pick up uv index
    uvidx = abs(uvidxfcv(ifcv))

    ! take residual
    if (uvidxfcv(ifcv) > 0) then
      resid = Vfcv(ifcv) - Vcmp(uvidx)
    else
      resid = Vfcv(ifcv) - dconjg(Vcmp(uvidx))
    end if

    ! compute chisquare
    chisq = chisq + abs(resid)**2/Varfcv(ifcv)/fnorm

    ! compute residual vector
    factor = -2/Varfcv(ifcv)/fnorm
    Vresre(uvidx) = Vresre(uvidx) + factor*dreal(resid)
    Vresim(uvidx) = Vresim(uvidx) + factor*dimag(resid)*sign(1,uvidxfcv(ifcv))
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine chisq_amp(Vcmp,&
                     uvidxamp,Vamp,Varamp,&
                     fnorm,&
                     chisq,Vresre,Vresim,&
                     Nuv,Namp)
  implicit none

  ! NuFFT-ed visibilities
  integer,      intent(in):: Nuv
  complex(dpc), intent(in):: Vcmp(Nuv)
  ! Data
  integer,  intent(in):: Namp           ! Number of data
  integer,  intent(in):: uvidxamp(Namp) ! UV Index of Amp data
  real(dp), intent(in):: Vamp(Namp)     ! Amp data
  real(dp), intent(in):: Varamp(Namp)   ! variances of Amp data
  ! Normalization Factor of Chisquare
  real(dp), intent(in):: fnorm
  ! Outputs
  real(dp), intent(inout):: chisq           ! chisquare
  real(dp), intent(inout):: Vresre(Nuv), Vresim(Nuv) ! residual vector
                                            !   its adjoint FT provides
                                            !   the gradient of chisquare)

  real(dp):: resid, factor, model
  integer:: uvidx, iamp

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Namp,fnorm,uvidxamp,Vcmp,Vamp,Varamp) &
  !$OMP   PRIVATE(iamp,uvidx,resid,factor,model),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim)
  do iamp=1, Namp
    ! pick up uv index
    uvidx = abs(uvidxamp(iamp))

    ! take residual
    model = abs(Vcmp(uvidx))
    resid = Vamp(iamp) - model

    ! compute chisquare
    chisq = chisq + resid**2/Varamp(iamp)/fnorm

    ! compute residual vector
    factor = -2*resid/Varamp(iamp)/model/fnorm
    Vresre(uvidx) = Vresre(uvidx) + factor * dreal(Vcmp(uvidx))
    Vresim(uvidx) = Vresim(uvidx) + factor * dimag(Vcmp(uvidx))
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine chisq_ca(Vcmp,&
                    uvidxca,CA,Varca,&
                    fnorm,&
                    chisq,Vresre,Vresim,&
                    Nuv,Nca)
  implicit none

  ! NuFFT-ed visibilities
  integer,      intent(in):: Nuv
  complex(dpc), intent(in):: Vcmp(Nuv)
  ! Data
  integer,  intent(in):: Nca            ! Number of data
  integer,  intent(in):: uvidxca(4,Nca) ! UV Index of Amp data
  real(dp), intent(in):: CA(Nca)        ! Amp data
  real(dp), intent(in):: Varca(Nca)     ! variances of Amp data
  ! Normalization Factor of Chisquare
  real(dp), intent(in):: fnorm
  ! Outputs
  real(dp), intent(inout):: chisq           ! chisquare
  real(dp), intent(inout):: Vresre(Nuv), Vresim(Nuv) ! residual vector
                                            !   its adjoint FT provides
                                            !   the gradient of chisquare)

  real(dp):: resid, factor, model
  real(dp):: Vamp1, Vamp2, Vamp3, Vamp4
  complex(dpc):: Vcmp1, Vcmp2, Vcmp3, Vcmp4
  integer:: uvidx1, uvidx2, uvidx3, uvidx4
  integer:: ica

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nca,fnorm,uvidxca,Vcmp,CA,Varca) &
  !$OMP   PRIVATE(ica,model,resid,&
  !$OMP           uvidx1,uvidx2,uvidx3,uvidx4,&
  !$OMP           Vcmp1,Vcmp2,Vcmp3,Vcmp4,&
  !$OMP           Vamp1,Vamp2,Vamp3,Vamp4),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim)
  do ica=1, Nca
    ! pick up uv index
    uvidx1 = abs(uvidxca(1,ica))
    uvidx2 = abs(uvidxca(2,ica))
    uvidx3 = abs(uvidxca(3,ica))
    uvidx4 = abs(uvidxca(4,ica))

    ! pick up full complex visibilities
    Vcmp1 = Vcmp(uvidx1)
    Vcmp2 = Vcmp(uvidx2)
    Vcmp3 = Vcmp(uvidx3)
    Vcmp4 = Vcmp(uvidx4)
    Vamp1 = abs(Vcmp1)
    Vamp2 = abs(Vcmp2)
    Vamp3 = abs(Vcmp3)
    Vamp4 = abs(Vcmp4)

    ! calculate model log closure amplitude and residual
    model = log(Vamp1)+log(Vamp2)-log(Vamp3)-log(Vamp4)
    resid = CA(ica) - model

    ! compute chisquare
    chisq = chisq + resid**2/Varca(ica)/fnorm

    ! compute residual vectors
    factor = -2*resid/Varca(ica)/fnorm
    ! re
    Vresre(uvidx1) = Vresre(uvidx1) + factor / Vamp1**2 * dreal(Vcmp1)
    Vresre(uvidx2) = Vresre(uvidx2) + factor / Vamp2**2 * dreal(Vcmp2)
    Vresre(uvidx3) = Vresre(uvidx3) - factor / Vamp3**2 * dreal(Vcmp3)
    Vresre(uvidx4) = Vresre(uvidx4) - factor / Vamp4**2 * dreal(Vcmp4)
    ! im
    Vresim(uvidx1) = Vresim(uvidx1) + factor / Vamp1**2 * dimag(Vcmp1)
    Vresim(uvidx2) = Vresim(uvidx2) + factor / Vamp2**2 * dimag(Vcmp2)
    Vresim(uvidx3) = Vresim(uvidx3) - factor / Vamp3**2 * dimag(Vcmp3)
    Vresim(uvidx4) = Vresim(uvidx4) - factor / Vamp4**2 * dimag(Vcmp4)
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine chisq_cp(Vcmp,&
                    uvidxcp,CP,Varcp,&
                    fnorm,&
                    chisq,Vresre,Vresim,&
                    Nuv,Ncp)
  implicit none

  ! NuFFT-ed visibilities
  integer,      intent(in):: Nuv
  complex(dpc), intent(in):: Vcmp(Nuv)
  ! Data
  integer,  intent(in):: Ncp            ! Number of data
  integer,  intent(in):: uvidxcp(3,Ncp) ! UV Index of Amp data
  real(dp), intent(in):: CP(Ncp)        ! Amp data
  real(dp), intent(in):: Varcp(Ncp)     ! variances of Amp data
  ! Normalization Factor of Chisquare
  real(dp), intent(in):: fnorm
  ! Outputs
  real(dp), intent(inout):: chisq ! chisquare
  real(dp), intent(inout):: Vresre(Nuv), Vresim(Nuv) ! residual vector
                                            !   its adjoint FT provides
                                            !   the gradient of chisquare)

  real(dp):: resid, factor, model
  real(dp):: Vampsq1, Vampsq2, Vampsq3
  complex(dpc):: Vcmp1, Vcmp2, Vcmp3
  integer:: uvidx1, uvidx2, uvidx3
  integer:: icp
  integer:: sign1, sign2, sign3

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Ncp,fnorm,uvidxcp,Vcmp,CP,Varcp) &
  !$OMP   PRIVATE(icp,model,resid,&
  !$OMP           uvidx1,uvidx2,uvidx3,&
  !$OMP           Vcmp1,Vcmp2,Vcmp3,&
  !$OMP           Vampsq1,Vampsq2,Vampsq3,&
  !$OMP           sign1,sign2,sign3),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim)
  do icp=1, Ncp
    ! pick up uv index
    uvidx1 = abs(uvidxcp(1,icp))
    uvidx2 = abs(uvidxcp(2,icp))
    uvidx3 = abs(uvidxcp(3,icp))

    ! pick up full complex visibilities
    Vcmp1 = Vcmp(uvidx1)
    Vcmp2 = Vcmp(uvidx2)
    Vcmp3 = Vcmp(uvidx3)
    sign1 = sign(1,uvidxcp(1,icp))
    sign2 = sign(1,uvidxcp(2,icp))
    sign3 = sign(1,uvidxcp(3,icp))
    Vampsq1 = abs(Vcmp1)**2
    Vampsq2 = abs(Vcmp2)**2
    Vampsq3 = abs(Vcmp3)**2

    ! calculate model closure phases and residual
    model = atan2(dimag(Vcmp1),dreal(Vcmp1))*sign1
    model = atan2(dimag(Vcmp2),dreal(Vcmp2))*sign2 + model
    model = atan2(dimag(Vcmp3),dreal(Vcmp3))*sign3 + model
    resid = CP(icp) - model
    resid = atan2(sin(resid),cos(resid))

    ! compute chisquare
    chisq = chisq + resid**2/Varcp(icp)/fnorm

    ! compute residual vectors
    factor = -2*resid/Varcp(icp)/fnorm

    Vresre(uvidx1) = Vresre(uvidx1) - factor/Vampsq1*dimag(Vcmp1)*sign1
    Vresre(uvidx2) = Vresre(uvidx2) - factor/Vampsq2*dimag(Vcmp2)*sign2
    Vresre(uvidx3) = Vresre(uvidx3) - factor/Vampsq3*dimag(Vcmp3)*sign3

    Vresim(uvidx1) = Vresim(uvidx1) + factor/Vampsq1*dreal(Vcmp1)*sign1
    Vresim(uvidx2) = Vresim(uvidx2) + factor/Vampsq2*dreal(Vcmp2)*sign2
    Vresim(uvidx3) = Vresim(uvidx3) + factor/Vampsq3*dreal(Vcmp3)*sign3
  end do
  !$OMP END PARALLEL DO
end subroutine
end module
