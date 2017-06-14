module static_imaging_dft
  !$use omp_lib
  use nrtype, only: dp, tol, pi
  use static_imaging_lib, only: tv, tsv, gradtve, gradtsve, calc_I2d
  implicit none
  !
  ! External Functions (BLAS Functions)
  interface
    real(kind(1d0)) function dasum(n, x, incx)
      integer,         intent(in) :: n, incx
      real(kind(1d0)), intent(in) :: x(n)
    end function
  end interface

  interface
    real(kind(1d0)) function ddot(n, x, incx, y, incy)
      integer,         intent(in) :: n, incx, incy
      real(kind(1d0)), intent(in) :: x(n), y(n)
    end function
  end interface
contains
!
!-------------------------------------------------------------------------------
! Imaging Routine
!-------------------------------------------------------------------------------
!
subroutine imaging( &
  Iin, x, y, xidx, yidx, Nx, Ny, u, v, &
  lambl1, lambtv, lambtsv, nonneg, Niter, &
  isfcv, uvidxfcv, Vfcvr, Vfcvi, Varfcv, &
  isamp, uvidxamp, Vamp, Varamp, &
  iscp, uvidxcp, CP, Varcp, &
  isca, uvidxca, CA, Varca, &
  m, factr, pgtol, &
  Iout, &
  Npix, Nuv, Nfcv, Namp, Ncp, Nca &
)
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: x(Npix),y(Npix) ! xy coordinates in radian.
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! uv coordinate
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates in wavelength

  ! Imaging parameters
  real(dp), intent(in) :: lambl1  ! imaging parameter for L1-norm
  real(dp), intent(in) :: lambtv  ! imaging parameter for iso-TV
  real(dp), intent(in) :: lambtsv ! imaging parameter for TSV
  logical,  intent(in) :: nonneg  ! if nonneg > 0, the image will be solved
                                  ! with a non-negative condition
  integer,  intent(in) :: Niter   ! the number of iterations

  ! Parameters related to full complex visibilities
  logical,  intent(in) :: isfcv
  integer,  intent(in) :: Nfcv
  integer,  intent(in) :: uvidxfcv(Nfcv) ! uvidx
  real(dp), intent(in) :: Vfcvr(Nfcv), Vfcvi(Nfcv) ! Real and Imag parts of data
  real(dp), intent(in) :: Varfcv(Nfcv) ! Variance

  ! Parameters related to amplitude
  logical,  intent(in) :: isamp
  integer,  intent(in) :: Namp
  integer,  intent(in) :: uvidxamp(Namp)
  real(dp), intent(in) :: Vamp(Namp)
  real(dp), intent(in) :: Varamp(Namp)

  ! Parameters related to the closure phase
  logical,  intent(in) :: iscp
  integer,  intent(in) :: Ncp
  integer,  intent(in) :: uvidxcp(3,Ncp)
  real(dp), intent(in) :: CP(Ncp)
  real(dp), intent(in) :: Varcp(Ncp)

  ! Parameters related to the closure amplitude
  logical,  intent(in) :: isca
  integer,  intent(in) :: Nca
  integer,  intent(in) :: uvidxca(4,Nca)
  real(dp), intent(in) :: CA(Nca)
  real(dp), intent(in) :: Varca(Nca)

  ! Paramters related to the L-BFGS-B
  integer,  intent(in) :: m
  real(dp), intent(in) :: factr, pgtol
  !
  ! Output Image
  real(dp), intent(out) :: Iout(1:Npix)

  ! chisquare and grad chisq
  real(dp) :: cost
  real(dp) :: gradcost(1:Npix)

  ! Number of Data
  integer :: Ndata

  ! Fourier Matrix
  real(dp), allocatable :: Freal(:,:), Fimag(:,:)

  ! variables and parameters tuning L-BFGS-B
  integer,  parameter   :: iprint = 1
  character(len=60)     :: task, csave
  logical               :: lsave(4)
  integer               :: isave(44)
  real(dp)              :: dsave(29)
  integer,  allocatable :: nbd(:),iwa(:)
  real(dp), allocatable :: lower(:),upper(:),wa(:)

  ! loop variables
  integer :: i

  ! Check Ndata
  Ndata = 0
  if (isfcv .eqv. .True.) then
    Ndata = Ndata + Nfcv
  end if
  if (isamp .eqv. .True.) then
    Ndata = Ndata + Namp
  end if
  if (iscp .eqv. .True.) then
    Ndata = Ndata + Ncp
  end if
  if (isca .eqv. .True.) then
    Ndata = Ndata + Nca
  end if

  !-------------------------------------
  ! Preperation
  !-------------------------------------
  ! allocate vectors and arrays
  !write(*,*) 'static_imaging_dft.imaging: Allocate Freal, Fimag'
  allocate(Freal(Npix, Nuv), Fimag(Npix, Nuv))
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nuv) &
  !$OMP   PRIVATE(i)
  do i=1,Nuv
    Freal(1:Npix,i) = 0d0
    Fimag(1:Npix,i) = 0d0
  end do
  !$OMP END PARALLEL DO

  ! calculate Fourier Matrix
  !write(*,*) 'static_imaging_dft.imaging: Calc the Fourier Matrix'
  call calc_F(x,y,u,v,Freal,Fimag,Npix,Nuv)

  ! copy images
  !write(*,*) 'static_imaging_dft.imaging: Initialize the image'
  Iout(1:Npix) = Iin(1:Npix)
  !-------------------------------------
  ! L-BFGS-B
  !-------------------------------------
  !write(*,*) 'static_imaging_dft.imaging: Initialize the L-BFGS-B'
  ! initialise L-BFGS-B
  !   Allocate some arrays
  allocate(iwa(3*Npix))
  allocate(wa(2*m*Npix + 5*Npix + 11*m*m + 8*m))

  !   set boundary conditions
  allocate(lower(Npix),upper(Npix),nbd(Npix))
  if (nonneg .eqv. .True.) then
    nbd(:) = 1  ! put lower limit
    lower(:) = 0d0  ! put lower limit
  else
    nbd(:) = 0 ! no boundary conditions
  end if

  ! start L-BFGS-B
  !write(*,*) 'static_imaging_dft.imaging: start L-BFGS-B calculations'
  task = 'START'
  do while(task(1:2) == 'FG' &
          .or. task == 'NEW_X' &
          .or. task == 'START')
    ! This is the call to the L-BFGS-B code.
    call setulb ( Npix, m, Iout, lower, upper, nbd, cost, gradcost, &
                  factr, pgtol, wa, iwa, task, iprint,&
                  csave, lsave, isave, dsave )

    if (task(1:2) == 'FG') then
      ! Calculate cost function and gradcostent of cost function
      call calc_cost(&
          Iout, xidx, yidx, Npix, Nx, Ny, Nuv, Ndata, &
          lambl1, lambtv, lambtsv,&
          Freal,Fimag,&
          isfcv, Nfcv, uvidxfcv, Vfcvr, Vfcvi, Varfcv,&
          isamp, Namp, uvidxamp, Vamp, Varamp,&
          iscp,  Ncp, uvidxcp, CP, Varcp,&
          isca,  Nca, uvidxca, CA, Varca,&
          cost, gradcost &
      )
    else
      ! If iteration number exceeds the total iteration number, make a flag
      ! to STOP L-BFGS-B iterations
      if (isave(30) > Niter) then
        task='STOP: TOTAL ITERATION NUMBER EXCEEDS LIMIT'
      end if

      ! If we have a flag to STOP the L-BFGS-B algorithm, print it out.
      if (task(1:4) .eq. 'STOP') then
        write (6,*) task
      end if
    end if
  end do

  !-------------------------------------
  ! deallocate vectors
  !-------------------------------------
  deallocate(Freal,Fimag)
end subroutine
!
!-------------------------------------------------------------------------------
! calc cost functions
!-------------------------------------------------------------------------------
!
subroutine calc_cost(&
    Iin, xidx, yidx, Npix, Nx, Ny, Nuv, Ndata, &
    lambl1, lambtv, lambtsv,&
    Freal,Fimag,&
    isfcv, Nfcv, uvidxfcv, Vfcvr, Vfcvi, Varfcv,&
    isamp, Namp, uvidxamp, Vamp, Varamp,&
    iscp,  Ncp, uvidxcp, CP, Varcp,&
    isca,  Nca, uvidxca, CA, Varca,&
    cost, gradcost)
  !
  ! Calculate Cost Functions
  !
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! DFT matrix
  integer,  intent(in) :: Nuv
  integer,  intent(in) :: Ndata
  !real(dp), intent(in) :: Freal(1:Npix,1:Nuv), Fimag(1:Npix,1:Nuv)

  ! Imaging parameters
  real(dp), intent(in) :: lambl1,lambtv,lambtsv

  ! Allocatable arrays and vectors common in the module
  real(dp), intent(in) :: Freal(Npix,Nuv), Fimag(Npix,Nuv)

  ! Parameters related to full complex visibilities
  logical,  intent(in) :: isfcv
  integer,  intent(in) :: Nfcv
  integer,  intent(in) :: uvidxfcv(Nfcv)
  real(dp), intent(in) :: Vfcvr(Nfcv), Vfcvi(Nfcv)
  real(dp), intent(in) :: Varfcv(Nfcv)

  ! Parameters related to amplitude
  logical,  intent(in) :: isamp
  integer,  intent(in) :: Namp
  integer,  intent(in) :: uvidxamp(Namp)
  real(dp), intent(in) :: Vamp(Namp)
  real(dp), intent(in) :: Varamp(Namp)

  ! Parameters related to the closure phase
  logical,  intent(in) :: iscp
  integer,  intent(in) :: Ncp
  integer,  intent(in) :: uvidxcp(3,Ncp)
  real(dp), intent(in) :: CP(Ncp)
  real(dp), intent(in) :: Varcp(Ncp)

  ! Parameters related to the closure amplitude
  logical,  intent(in) :: isca
  integer,  intent(in) :: Nca
  integer,  intent(in) :: uvidxca(4,Nca)
  real(dp), intent(in) :: CA(Nca)
  real(dp), intent(in) :: Varca(Nca)

  ! Outputs
  real(dp), intent(out) :: cost
  real(dp), intent(out) :: gradcost(1:Npix)

  ! integer
  integer :: ipix, iuv

  ! chisquares
  real(dp) :: chisq, gradchisq(1:Npix)

  ! allocatable arrays
  real(dp), allocatable :: Vreal(:), Vimag(:)
  real(dp), allocatable :: gradVamp(:,:), gradVpha(:,:)
  real(dp), allocatable :: I2d(:,:)

  ! logicals
  logical :: needgradV

  ! Check if we need to calculate gradVamp and gradVphase
  if ((isamp .eqv. .True.) .or. (iscp .eqv. .True.)&
                           .or. (isca .eqv. .True.)) then
    needgradV = .True.
  else
    needgradV = .False.
  end if

  ! Initialize the chisquare and its gradient
  !write(*,*) 'static_imaging_dft.calc_cost: initialize cost and gradcost'
  cost = 0d0
  gradcost(1:Npix) = 0d0

  ! allocate arrays related with the DFT
  allocate(Vreal(Nuv), Vimag(Nuv))
  if (needgradV .eqv. .True.) then
    allocate(gradVamp(Npix, Nuv),gradVpha(Npix, Nuv))
  end if
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix, Nuv, needgradV) &
  !$OMP   PRIVATE(iuv)
  do iuv=1,Nuv
    Vreal(iuv) = 0d0
    Vimag(iuv) = 0d0
    if (needgradV .eqv. .True.) then
      gradVamp(1:Npix, iuv) = 0d0
      gradVpha(1:Npix, iuv) = 0d0
    end if
  end do
  !$OMP END PARALLEL DO

  if ((lambtv > 0) .or. (lambtsv > 0)) then
    allocate(I2d(Nx, Ny))
    I2d(1:Nx,1:Ny) = 0d0
    call calc_I2d(Iin,xidx,yidx,I2d,Npix,Nx,Ny)
  end if

  !-------------------------------------
  ! Calculate Chi Square and its gradient
  !-------------------------------------
  ! Do Fourier Transfromation
  !write(*,*) 'static_imaging_dft.calc_cost: run DFT'
  call DFT(Iin,Freal,Fimag,Vreal,Vimag,Npix,Nuv)

  !   Full Complex Vibisility
  if (isfcv .eqv. .True.) then
    !write(*,*) 'static_imaging_dft.calc_cost: calculate chisq and gradchisq on Full Complex Visibility'
    call chisq_fcv(Freal,Fimag,Vreal,Vimag,&
                   uvidxfcv,Vfcvr,Vfcvi,Varfcv,&
                   chisq,gradchisq,Npix,Nuv,Nfcv)
    !write(*,*) 'static_imaging_dft.calc_cost: chisq fcv',chisq
    cost = cost + chisq / Ndata
    call daxpy(Npix,1d0/Ndata,gradchisq(1:Npix),1,gradcost(1:Npix),1)
  end if

  !   Other quantities
  if (needgradV .eqv. .True.) then
    !write(*,*) 'static_imaging_dft.calc_cost: calculate gradients of visibility amplitudes and phases'
    call grad_amppha(Freal,Fimag,Vreal,Vimag,gradVamp,gradVpha,Npix,Nuv)

    if (isamp .eqv. .True.) then
      !write(*,*) 'static_imaging_dft.calc_cost: calculate chisq and gradchisq on Visibility Amplitudes'
      call chisq_amp(gradVamp,Vreal,Vimag,&
                     uvidxamp,Vamp,Varamp,&
                     chisq,gradchisq,Npix,Nuv,Namp)
      !write(*,*) 'static_imaging_dft.calc_cost: chisq amp',chisq
      cost = cost + chisq
      call daxpy(Npix,1d0/Ndata,gradchisq(1:Npix),1,gradcost(1:Npix),1)
    end if

    if (isca .eqv. .True.) then
      !write(*,*) 'static_imaging_dft.calc_cost: calculate chisq and gradchisq on Closure Amplitudes'
      call chisq_camp(gradVamp,Vreal,Vimag,&
                      uvidxca,CA,Varca,&
                      chisq,gradchisq,Npix,Nuv,Nca)
      !write(*,*) 'static_imaging_dft.calc_cost: chisq camp',chisq
      cost = cost + chisq / Ndata
      call daxpy(Npix,1d0/Ndata,gradchisq(1:Npix),1,gradcost(1:Npix),1)
    end if

    if (iscp .eqv. .True.) then
      !write(*,*) 'static_imaging_dft.calc_cost: calculate chisq and gradchisq on Closure Phases'
      call chisq_cphase(gradVpha,Vreal,Vimag,&
                        uvidxcp,CP,Varcp,&
                        chisq,gradchisq,Npix,Nuv,Ncp)
      !write(*,*) 'static_imaging_dft.calc_cost: chisq cphase',chisq
      cost = cost + chisq / Ndata
      call daxpy(Npix,1d0/Ndata,gradchisq(1:Npix),1,gradcost(1:Npix),1)
    end if
  end if

  !-------------------------------------
  ! Calculate regularization functions
  !-------------------------------------
  ! calc l1norm
  if (lambl1 > 0) then
    cost = cost + lambl1 * dasum(Npix, Iin, 1) !* sum(abs(Iin))
  end if
  !
  ! calc tv and tsv
  if ((lambtv > 0) .or. (lambtsv > 0)) then
    if (lambtv > 0) then
      cost = cost + lambtv * tv(I2d,Nx,Ny)
    end if

    if (lambtsv > 0) then
      cost = cost + lambtsv * tsv(I2d,Nx,Ny)
    end if
  end if

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nx,Ny,Iin,I2d, &
  !$OMP                lambl1,lambtv,lambtsv) &
  !$OMP   PRIVATE(ipix)
  do ipix = 1, Npix
    ! calc l1norm
    if (lambl1 > 0) then
      if (Iin(ipix) > tol) then
        gradcost(ipix) = gradcost(ipix) + lambl1
      elseif (Iin(ipix) < -tol) then
        gradcost(ipix) = gradcost(ipix) - lambl1
      end if
    end if
    !
    ! gradient for tsv term
    if (lambtv > 0) then
      gradcost(ipix) = gradcost(ipix) + &
                       lambtv * gradtve(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
    end if

    ! gradient for tsv term
    if (lambtsv > 0) then
      gradcost(ipix) = gradcost(ipix) + &
                       lambtsv * gradtsve(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
    end if
  end do
  !$OMP END PARALLEL DO


  ! deallocate arrays
  deallocate(Vreal, Vimag)

  if (needgradV .eqv. .True.) then
    deallocate(gradVamp,gradVpha)
  end if

  if ((lambtv > 0) .or. (lambtsv > 0)) then
    deallocate(I2d)
  end if
end subroutine
!
!-------------------------------------------------------------------------------
! calc the Fourier Matrix
!-------------------------------------------------------------------------------
!
subroutine calc_F(x,y,u,v,Freal,Fimag,Npix,Nuv)
  !
  ! This subroutine calculates the DFT matrix from the input data
  !
  implicit none

  integer,  intent(in)  :: Npix,Nuv
  real(dp), intent(in)  :: x(Npix),y(Npix)
  real(dp), intent(in)  :: u(Nuv),v(Nuv)
  real(dp), intent(out) :: Freal(Npix,Nuv), Fimag(Npix,Nuv)

  integer :: iuv
  real(dp):: phase(1:Npix)

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nuv,Npix,x,y,u,v) &
  !$OMP   PRIVATE(iuv, phase)
  do iuv = 1, Nuv
    ! Calculate phases first
    phase(1:Npix) = 0d0
    call daxpy(Npix,2*pi*u(iuv),x(1:Npix),1,phase(1:Npix),1)
    call daxpy(Npix,2*pi*v(iuv),y(1:Npix),1,phase(1:Npix),1)
    !write(*,*) maxval(v(iuv)*y), minval(v(iuv)*y)

    ! Calculate the matrix
    Freal(1:Npix, iuv) = cos(phase(1:Npix))
    Fimag(1:Npix, iuv) = sin(phase(1:Npix))
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine grad_amppha(Freal,Fimag,Vreal,Vimag,gradVamp,gradVpha,Npix,Nuv)
  implicit none

  integer,  intent(in)  :: Npix,Nuv
  real(dp), intent(in)  :: Freal(Npix,Nuv), Fimag(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv),Vimag(Nuv)
  real(dp), intent(out) :: gradVamp(Npix,Nuv),gradVpha(Npix,Nuv)

  real(dp) :: Vamp, Vampsq
  integer  :: iuv

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nuv,Npix,Vreal,Vimag) &
  !$OMP   PRIVATE(iuv, Vamp, Vampsq)
  do iuv=1, Nuv
    Vamp = sqrt(Vreal(iuv)*Vreal(iuv)+Vimag(iuv)*Vimag(iuv))
    Vampsq = Vamp*Vamp

    ! using BLAS functions
    gradVamp(1:Npix,iuv) = 0d0
    call daxpy(Npix,Vreal(iuv)/Vamp,Freal(1:Npix,iuv),1,gradVamp(1:Npix,iuv),1)
    call daxpy(Npix,Vimag(iuv)/Vamp,Fimag(1:Npix,iuv),1,gradVamp(1:Npix,iuv),1)
    gradVpha(1:Npix,iuv) = 0d0
    call daxpy(Npix, Vreal(iuv)/Vampsq,Fimag(1:Npix,iuv),1,gradVpha(1:Npix,iuv),1)
    call daxpy(Npix,-Vimag(iuv)/Vampsq,Freal(1:Npix,iuv),1,gradVpha(1:Npix,iuv),1)

    ! using Fortran functions
    !gradVamp(1:Npix,iuv)=Vreal(iuv)/Vamp*Freal(1:Npix, iuv)&
    !                    +Vimag(iuv)/Vamp*Fimag(1:Npix, iuv)
    !gradVpha(1:Npix,iuv)=Vreal(iuv)/Vampsq*Fimag(1:Npix, iuv)&
    !                    -Vimag(iuv)/Vampsq*Freal(1:Npix, iuv)
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
!-------------------------------------------------------------------------------
! DFT
!-------------------------------------------------------------------------------
!
subroutine DFT(I,Freal,Fimag,Vreal,Vimag,Npix,Nuv)
  ! This subroutine do DFT
  implicit none
  integer,  intent(in)  :: Npix,Nuv
  real(dp), intent(in)  :: Freal(Npix,Nuv), Fimag(Npix,Nuv)
  real(dp), intent(in)  :: I(Npix)
  real(dp), intent(out) :: Vreal(Nuv),Vimag(Nuv)

  integer  :: iuv

  !call dgemv('T',Nuv,Npix,1d0,Freal(1:Npix,1:Nuv),&
  !           Npix,I(1:Npix),1,0d0,Vreal(1:Nuv),1)
  !call dgemv('T',Nuv,Npix,1d0,Fimag(1:Npix,1:Nuv),&
  !           Npix,I(1:Npix),1,0d0,Vimag(1:Nuv),1)

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nuv,Npix,I) &
  !$OMP   PRIVATE(iuv)
  do iuv=1, Nuv
    Vreal(iuv) = ddot(Npix, I(1:Npix), 1, Freal(1:Npix,iuv), 1)
    Vimag(iuv) = ddot(Npix, I(1:Npix), 1, Fimag(1:Npix,iuv), 1)
  end do
  !$OMP END PARALLEL DO
  !write(*,*) maxval(I),maxval(Freal),maxval(Fimag),minval(Freal),minval(Fimag)
end subroutine
!
!
!-------------------------------------------------------------------------------
! calc chisquares
!-------------------------------------------------------------------------------
!
!
subroutine chisq_fcv(Freal,Fimag,Vreal,Vimag,&
                     uvidxfcv,Vfcvr,Vfcvi,Varfcv,&
                     chisq,gradchisq,Npix,Nuv,Nfcv)
  integer,  intent(in)  :: Npix,Nuv,Nfcv
  integer,  intent(in)  :: uvidxfcv(Nfcv)
  real(dp), intent(in)  :: Freal(Npix,Nuv), Fimag(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: Vfcvr(Nfcv),Vfcvi(Nfcv)
  real(dp), intent(in)  :: Varfcv(Nfcv)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid1,resid2,factor1,factor2
  integer   :: uvidx,ifcv

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Nfcv,Vreal,Vimag,uvidxfcv,Vfcvr,Vfcvi,Varfcv) &
  !$OMP   PRIVATE(ifcv,uvidx,resid1,resid2,factor1,factor2) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do ifcv=1, Nfcv
    ! calc model fcvlitude
    uvidx = abs(uvidxfcv(ifcv))
    resid1 = Vreal(uvidx)-Vfcvr(ifcv)
    resid2 = sign(1,uvidxfcv(ifcv))*Vimag(uvidx)-Vfcvi(ifcv)

    ! calc chi-square
    chisq = chisq + (resid1*resid1+resid2*resid2)/Varfcv(ifcv)

    ! calc gradient of chi-square
    factor1 = 2/Varfcv(ifcv)*resid1
    factor2 = 2/Varfcv(ifcv)*resid2*sign(1,uvidxfcv(ifcv))
    call daxpy(Npix,factor1,Freal(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    call daxpy(Npix,factor2,Fimag(1:Npix,uvidx),1,gradchisq(1:Npix),1)
  end do
  !$OMP END PARALLEL DO
end subroutine


subroutine chisq_amp(gradVamp,Vreal,Vimag,&
                     uvidxamp,Vamp,Varamp,&
                     chisq,gradchisq,Npix,Nuv,Namp)
  integer,  intent(in)  :: Npix, Nuv, Namp
  integer,  intent(in)  :: uvidxamp(Namp)
  real(dp), intent(in)  :: gradVamp(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: Vamp(Namp),Varamp(Namp)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid
  integer   :: uvidx,iamp

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Namp,Vreal,Vimag,uvidxamp,Vamp,Varamp) &
  !$OMP   PRIVATE(iamp,uvidx,resid)  &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do iamp=1, Namp
    ! calc model amplitude
    uvidx = abs(uvidxamp(iamp))
    resid = sqrt(Vreal(uvidx)*Vreal(uvidx)+Vimag(uvidx)*Vimag(uvidx))-Vamp(iamp)

    ! calc chi-square
    chisq = chisq + resid*resid/Varamp(iamp)

    ! calc gradient of chisquare
    !  using BLAS
    call daxpy(Npix,2/Varamp(iamp)*resid,&
               gradVamp(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    !  not using BLAS
    !gradchisq(1:Npix) = gradchisq(1:Npix) &
    !                  + 2/Varamp(iamp)*resid*gradVamp(1:Npix,uvidx)
  end do
  !$OMP END PARALLEL DO
end subroutine


subroutine chisq_cphase(gradVpha,Vreal,Vimag,&
                        uvidxcp,CP,Varcp,&
                        chisq,gradchisq,Npix,Nuv,Ncp)
  integer,  intent(in)  :: Npix, Nuv, Ncp
  integer,  intent(in)  :: uvidxcp(3,Ncp)
  real(dp), intent(in)  :: gradVpha(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: CP(Ncp),Varcp(Ncp)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid,factor
  integer   :: uvidx,icp,ibl

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Ncp,Vreal,Vimag,uvidxcp,CP,Varcp) &
  !$OMP   PRIVATE(icp,ibl,resid,uvidx,factor) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do icp=1, Ncp
    ! calc model closure phase
    resid = 0d0
    do ibl=1,3
      uvidx = abs(uvidxcp(ibl,icp))
      resid = resid + atan2(sign(1,uvidxcp(ibl,icp))*Vimag(uvidx),Vreal(uvidx))
    end do

    ! take a residual betweem model and data
    resid = resid - CP(icp)

    ! adjust a residual phase
    !do while (resid > pi)
    !  resid = resid - 2*pi
    !end do
    !do while (resid < -pi)
    !  resid = resid + 2*pi
    !end do
    resid = atan2(sin(resid),cos(resid))

    ! calc chi-square
    chisq = chisq + resid*resid/Varcp(icp)

    ! calc gradient of chi-square
    factor=2/Varcp(icp)*resid
    do ibl=1,3
      uvidx = abs(uvidxcp(ibl,icp))
      call daxpy(Npix,sign(1,uvidxcp(ibl,icp))*factor,&
                 gradVpha(1:Npix,uvidx),1,gradchisq(1:Npix),1)
      !gradchisq(1:Npix) = gradchisq(1:Npix)&
      !                  + sign(1,uvidxcp(ibl,icp))*factor*gradVpha(1:Npix,uvidx)
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine chisq_camp(gradVamp,Vreal,Vimag,&
                      uvidxca,CA,Varca,&
                      chisq,gradchisq,Npix,Nuv,Nca)
  integer,  intent(in)  :: Npix, Nuv, Nca
  integer,  intent(in)  :: uvidxca(4,Nca)
  real(dp), intent(in)  :: gradVamp(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: CA(Nca),Varca(Nca)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid,amp,factor1,factor2(4)
  integer   :: uvidx,ica,ibl

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Nca,Vreal,Vimag,uvidxca,CA,Varca) &
  !$OMP   PRIVATE(ica,ibl,resid,uvidx,amp,factor1,factor2) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do ica=1, Nca
    ! calc model closure phase
    resid = 0d0
    do ibl=1,4
      uvidx = abs(uvidxca(ibl,ica))
      amp = sqrt(Vreal(uvidx)*Vreal(uvidx)+Vimag(uvidx)*Vimag(uvidx))
      if (ibl < 3) then
        factor2(ibl) = 1/amp
        resid = resid + log(amp)
      else
        factor2(ibl) = -1/amp
        resid = resid - log(amp)
      end if
    end do

    ! take a residual betweem model and data
    resid = resid - CA(ica)

    ! calc chi-square
    chisq = chisq + resid*resid/Varca(ica)

    ! calc gradient of chisquare
    factor1=2/Varca(ica)*resid
    do ibl=1,4
      uvidx = abs(uvidxca(ibl,ica))
      call daxpy(Npix,factor1*factor2(ibl),&
                 gradVamp(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
!
!-------------------------------------------------------------------------------
! calc model and statisitcs
!-------------------------------------------------------------------------------
!
subroutine statistics( &
  Iin, x, y, xidx, yidx, Nx, Ny, u, v, &
  lambl1, lambtv, lambtsv, &
  isfcv, uvidxfcv, Vfcvr, Vfcvi, Varfcv, &
  isamp, uvidxamp, Vamp,  Varamp, &
  iscp,  uvidxcp,  CP,    Varcp, &
  isca,  uvidxca,  CA,    Varca, &
  cost, gradcost,&
  chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
  l1val, tvval, tsvval,&
  Vfcvrmod,Vfcvimod,Vfcvres,Vampmod,Vampres,CPmod,CPres,CAmod,CAres,&
  Npix, Nuv, Nfcv, Namp, Ncp, Nca &
)
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: x(Npix),y(Npix) ! xy coordinates in radian.
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! uv coordinate
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates in wavelength

  ! Imaging parameters
  real(dp), intent(in) :: lambl1  ! imaging parameter for L1-norm
  real(dp), intent(in) :: lambtv  ! imaging parameter for iso-TV
  real(dp), intent(in) :: lambtsv ! imaging parameter for TSV

  ! Parameters related to full complex visibilities
  logical,  intent(in) :: isfcv
  integer,  intent(in) :: Nfcv
  integer,  intent(in) :: uvidxfcv(Nfcv) ! uvidx
  real(dp), intent(in) :: Vfcvr(Nfcv), Vfcvi(Nfcv) ! Real and Imag parts of data
  real(dp), intent(in) :: Varfcv(Nfcv) ! Variance
  real(dp), intent(out):: Vfcvrmod(Nfcv),Vfcvimod(Nfcv),Vfcvres(Nfcv)

  ! Parameters related to amplitude
  logical,  intent(in) :: isamp
  integer,  intent(in) :: Namp
  integer,  intent(in) :: uvidxamp(Namp)
  real(dp), intent(in) :: Vamp(Namp)
  real(dp), intent(in) :: Varamp(Namp)
  real(dp), intent(out):: Vampmod(Namp),Vampres(Namp)

  ! Parameters related to the closure phase
  logical,  intent(in) :: iscp
  integer,  intent(in) :: Ncp
  integer,  intent(in) :: uvidxcp(3,Ncp)
  real(dp), intent(in) :: CP(Ncp)
  real(dp), intent(in) :: Varcp(Ncp)
  real(dp), intent(out):: CPmod(Ncp),CPres(Ncp)

  ! Parameters related to the closure amplitude
  logical,  intent(in) :: isca
  integer,  intent(in) :: Nca
  integer,  intent(in) :: uvidxca(4,Nca)
  real(dp), intent(in) :: CA(Nca)
  real(dp), intent(in) :: Varca(Nca)
  real(dp), intent(out):: CAmod(Nca),CAres(Nca)

  ! Output Image
  real(dp), intent(out) :: cost, chisq, chisqfcv, chisqamp, chisqcp, chisqca
  real(dp), intent(out) :: gradcost(1:Npix)
  real(dp), intent(out) :: l1val, tvval, tsvval
  real(dp) :: gradchisq(1:Npix)

  ! Fourier Matrix
  real(dp), allocatable :: Freal(:,:), Fimag(:,:)
  real(dp), allocatable :: Vreal(:), Vimag(:)
  real(dp), allocatable :: gradVamp(:,:), gradVpha(:,:)
  real(dp), allocatable :: I2d(:,:)

  ! integer
  integer :: ipix, iuv, Ndata

  ! logicals
  logical :: needgradV
  !
  ! Check Ndata
  Ndata = 0
  if (isfcv .eqv. .True.) then
    Ndata = Ndata + Nfcv
  end if
  if (isamp .eqv. .True.) then
    Ndata = Ndata + Namp
  end if
  if (iscp .eqv. .True.) then
    Ndata = Ndata + Ncp
  end if
  if (isca .eqv. .True.) then
    Ndata = Ndata + Nca
  end if

  !-------------------------------------
  ! Preperation
  !-------------------------------------
  ! Check if we need to calculate gradVamp and gradVphase
  if ((isamp .eqv. .True.) .or. (iscp .eqv. .True.)&
                           .or. (isca .eqv. .True.)) then
    needgradV = .True.
  else
    needgradV = .False.
  end if

  ! allocate vectors and arrays
  !write(*,*) 'static_imaging_dft.imaging: Allocate Freal, Fimag'
  allocate(Freal(Npix, Nuv), Fimag(Npix, Nuv))
  allocate(Vreal(Nuv), Vimag(Nuv))

  if (needgradV .eqv. .True.) then
    allocate(gradVamp(Npix, Nuv),gradVpha(Npix, Nuv))
  end if

  allocate(I2d(Nx, Ny))
  I2d(1:Nx,1:Ny) = 0d0
  call calc_I2d(Iin,xidx,yidx,I2d,Npix,Nx,Ny)

  ! First touch
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nuv, Npix) &
  !$OMP   PRIVATE(iuv)
  do iuv=1,Nuv
    Freal(1:Npix,iuv) = 0d0
    Fimag(1:Npix,iuv) = 0d0
    Vreal(iuv) = 0d0
    Vimag(iuv) = 0d0
    if (needgradV .eqv. .True.) then
      gradVamp(1:Npix, iuv) = 0d0
      gradVpha(1:Npix, iuv) = 0d0
    end if
  end do
  !$OMP END PARALLEL DO

  ! Initialize the chisquare and its gradient
  cost = 0d0
  chisq = 0d0
  chisqfcv = 0d0
  chisqamp = 0d0
  chisqca = 0d0
  chisqcp = 0d0
  gradcost(1:Npix) = 0d0
  gradchisq(1:Npix) = 0d0
  Vfcvrmod(1:Nfcv) = 0d0
  Vfcvimod(1:Nfcv) = 0d0
  Vfcvres(1:Nfcv) = 0d0
  Vampmod(1:Namp) = 0d0
  Vampres(1:Namp) = 0d0
  CPmod(1:Ncp) = 0d0
  CPres(1:Ncp) = 0d0
  CAmod(1:Nca) = 0d0
  CAres(1:Nca) = 0d0
  l1val = 0d0
  tvval = 0d0
  tsvval= 0d0

  ! calculate Fourier Matrix
  call calc_F(x,y,u,v,Freal,Fimag,Npix,Nuv)
  call DFT(Iin,Freal,Fimag,Vreal,Vimag,Npix,Nuv)

  !   Full Complex Vibisility
  if (isfcv .eqv. .True.) then
    call model_fcv(Freal,Fimag,Vreal,Vimag,&
                   uvidxfcv,Vfcvr,Vfcvi,Varfcv,&
                   Vfcvrmod,Vfcvimod,Vfcvres,&
                   chisqfcv,gradchisq,Npix,Nuv,Nfcv)
    cost  = cost + chisqfcv/Ndata
    chisq = chisq+ chisqfcv
    call daxpy(Npix,1d0/Ndata,gradchisq(1:Npix),1,gradcost(1:Npix),1)
  end if

  !   Other quantities
  if (needgradV .eqv. .True.) then
    call grad_amppha(Freal,Fimag,Vreal,Vimag,gradVamp,gradVpha,Npix,Nuv)

    if (isamp .eqv. .True.) then
      call model_amp(gradVamp,Vreal,Vimag,&
                     uvidxamp,Vamp,Varamp,&
                     Vampmod,Vampres,&
                     chisqamp,gradchisq,Npix,Nuv,Namp)
      cost = cost  + chisqamp/Ndata
      chisq= chisq + chisqamp
      call daxpy(Npix,1d0/Ndata,gradchisq(1:Npix),1,gradcost(1:Npix),1)
    end if

    if (isca .eqv. .True.) then
      call model_camp(gradVamp,Vreal,Vimag,&
                      uvidxca,CA,Varca,&
                      CAmod,CAres,&
                      chisqca,gradchisq,Npix,Nuv,Nca)
      cost = cost  + chisqca/Ndata
      chisq= chisq + chisqca
      call daxpy(Npix,1d0/Ndata,gradchisq(1:Npix),1,gradcost(1:Npix),1)
    end if

    if (iscp .eqv. .True.) then
      call model_cphase(gradVpha,Vreal,Vimag,&
                        uvidxcp,CP,Varcp,&
                        CPmod,CPres,&
                        chisqcp,gradchisq,Npix,Nuv,Ncp)
      cost = cost  + chisqcp/Ndata
      chisq= chisq + chisqcp
      call daxpy(Npix,1d0/Ndata,gradchisq(1:Npix),1,gradcost(1:Npix),1)
    end if
  end if

  !-------------------------------------
  ! Calculate regularization functions
  !-------------------------------------
  l1val = dasum(Npix, Iin, 1)
  tvval = tv(I2d,Nx,Ny)
  tsvval = tsv(I2d,Nx,Ny)
  ! calc l1norm
  if (lambl1 > 0) then
    cost = cost + lambl1 * l1val !* sum(abs(Iin))
  end if
  !
  ! calc tv and tsv
  if (lambtv > 0) then
    cost = cost + lambtv * tvval
  end if

  if (lambtsv > 0) then
    cost = cost + lambtsv * tsvval
  end if

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nx,Ny,Iin,I2d, &
  !$OMP                lambl1,lambtv,lambtsv) &
  !$OMP   PRIVATE(ipix)
  do ipix = 1, Npix
    ! calc l1norm
    if (lambl1 > 0) then
      if (Iin(ipix) > tol) then
        gradcost(ipix) = gradcost(ipix) + lambl1
      elseif (Iin(ipix) < -tol) then
        gradcost(ipix) = gradcost(ipix) - lambl1
      end if
    end if
    !
    ! gradient for tsv term
    if (lambtv > 0) then
      gradcost(ipix) = gradcost(ipix) + &
                       lambtv * gradtve(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
    end if

    ! gradient for tsv term
    if (lambtsv > 0) then
      gradcost(ipix) = gradcost(ipix) + &
                       lambtsv * gradtsve(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
    end if
  end do
  !$OMP END PARALLEL DO

  ! deallocate arrays
  deallocate(Vreal, Vimag)
  deallocate(I2d)
  deallocate(Freal,Fimag)

  if (needgradV .eqv. .True.) then
    deallocate(gradVamp,gradVpha)
  end if
end subroutine
!
!
subroutine model_fcv(Freal,Fimag,Vreal,Vimag,&
                     uvidxfcv,Vfcvr,Vfcvi,Varfcv,&
                     Vfcvrmod,Vfcvimod,Vfcvres,&
                     chisq,gradchisq,Npix,Nuv,Nfcv)
  integer,  intent(in)  :: Npix,Nuv,Nfcv
  integer,  intent(in)  :: uvidxfcv(Nfcv)
  real(dp), intent(in)  :: Freal(Npix,Nuv), Fimag(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: Vfcvr(Nfcv),Vfcvi(Nfcv)
  real(dp), intent(in)  :: Varfcv(Nfcv)
  real(dp), intent(out) :: Vfcvrmod(Nfcv),Vfcvimod(Nfcv),Vfcvres(Nfcv)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid1,resid2,factor1,factor2
  integer   :: uvidx,ifcv

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  Vfcvrmod(1:Nfcv) = 0d0
  Vfcvimod(1:Nfcv) = 0d0
  Vfcvres(1:Nfcv) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Nfcv,Vreal,Vimag,uvidxfcv,Vfcvr,Vfcvi,Varfcv) &
  !$OMP   PRIVATE(ifcv,uvidx,resid1,resid2,factor1,factor2) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do ifcv=1, Nfcv
    ! calc model fcvlitude
    uvidx = abs(uvidxfcv(ifcv))
    Vfcvrmod(ifcv) = Vreal(uvidx)
    Vfcvimod(ifcv) = sign(1,uvidxfcv(ifcv))*Vimag(uvidx)
    resid1 = Vfcvrmod(ifcv)-Vfcvr(ifcv)
    resid2 = Vfcvimod(ifcv)-Vfcvi(ifcv)

    ! calc chi-square
    Vfcvres(ifcv)=sqrt(resid1*resid1+resid2*resid2)
    chisq = chisq + (resid1*resid1+resid2*resid2)/Varfcv(ifcv)

    ! calc gradient of chi-square
    factor1 = 2/Varfcv(ifcv)*resid1
    factor2 = 2/Varfcv(ifcv)*resid2*sign(1,uvidxfcv(ifcv))
    call daxpy(Npix,factor1,Freal(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    call daxpy(Npix,factor2,Fimag(1:Npix,uvidx),1,gradchisq(1:Npix),1)
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine model_amp(gradVamp,Vreal,Vimag,&
                     uvidxamp,Vamp,Varamp,&
                     Vampmod,Vampres,&
                     chisq,gradchisq,Npix,Nuv,Namp)
  integer,  intent(in)  :: Npix, Nuv, Namp
  integer,  intent(in)  :: uvidxamp(Namp)
  real(dp), intent(in)  :: gradVamp(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: Vamp(Namp), Varamp(Namp)
  real(dp), intent(out) :: Vampmod(Namp),Vampres(Namp)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  !real(dp)  :: resid
  integer   :: uvidx,iamp

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  Vampmod(1:Namp) = 0d0
  Vampres(1:Namp) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Namp,Vreal,Vimag,uvidxamp,Vamp,Varamp) &
  !$OMP   PRIVATE(iamp,uvidx)  &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do iamp=1, Namp
    ! calc model amplitude
    uvidx = abs(uvidxamp(iamp))
    Vampmod(iamp) = sqrt(Vreal(uvidx)*Vreal(uvidx)+Vimag(uvidx)*Vimag(uvidx))
    Vampres(iamp) = Vampmod(iamp)-Vamp(iamp)

    ! calc chi-square
    chisq = chisq + Vampres(iamp)*Vampres(iamp)/Varamp(iamp)

    ! calc gradient of chisquare
    !  using BLAS
    call daxpy(Npix,2/Varamp(iamp)*Vampres(iamp),&
               gradVamp(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    !  not using BLAS
    !gradchisq(1:Npix) = gradchisq(1:Npix) &
    !                  + 2/Varamp(iamp)*resid*gradVamp(1:Npix,uvidx)
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine model_cphase(gradVpha,Vreal,Vimag,&
                        uvidxcp,CP,Varcp,&
                        CPmod,CPres,&
                        chisq,gradchisq,Npix,Nuv,Ncp)
  integer,  intent(in)  :: Npix, Nuv, Ncp
  integer,  intent(in)  :: uvidxcp(3,Ncp)
  real(dp), intent(in)  :: gradVpha(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: CP(Ncp),Varcp(Ncp)
  real(dp), intent(out) :: CPmod(Ncp),CPres(Ncp)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid,factor
  integer   :: uvidx,icp,ibl

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  CPmod(1:Ncp) = 0d0
  CPres(1:Ncp) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Ncp,Vreal,Vimag,uvidxcp,CP,Varcp) &
  !$OMP   PRIVATE(icp,ibl,resid,uvidx,factor) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do icp=1, Ncp
    ! calc model closure phase
    resid = 0d0
    do ibl=1,3
      uvidx = abs(uvidxcp(ibl,icp))
      resid = resid + atan2(sign(1,uvidxcp(ibl,icp))*Vimag(uvidx),Vreal(uvidx))
    end do
    CPmod(icp)=atan2(sin(resid),cos(resid))

    ! take a residual betweem model and data
    resid = resid - CP(icp)

    ! adjust a residual phase
    resid = atan2(sin(resid),cos(resid))
    CPres(icp) = resid

    ! calc chi-square
    chisq = chisq + resid*resid/Varcp(icp)

    ! calc gradient of chi-square
    factor=2/Varcp(icp)*resid
    do ibl=1,3
      uvidx = abs(uvidxcp(ibl,icp))
      call daxpy(Npix,sign(1,uvidxcp(ibl,icp))*factor,&
                 gradVpha(1:Npix,uvidx),1,gradchisq(1:Npix),1)
      !gradchisq(1:Npix) = gradchisq(1:Npix)&
      !                  + sign(1,uvidxcp(ibl,icp))*factor*gradVpha(1:Npix,uvidx)
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine model_camp(gradVamp,Vreal,Vimag,&
                      uvidxca,CA,Varca,&
                      CAmod,CAres,&
                      chisq,gradchisq,Npix,Nuv,Nca)
  integer,  intent(in)  :: Npix, Nuv, Nca
  integer,  intent(in)  :: uvidxca(4,Nca)
  real(dp), intent(in)  :: gradVamp(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: CA(Nca),Varca(Nca)
  real(dp), intent(out) :: CAmod(Nca),CAres(Nca)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid,amp,factor1,factor2(4)
  integer   :: uvidx,ica,ibl

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  CAmod(1:Nca) = 0d0
  CAres(1:Nca) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Nca,Vreal,Vimag,uvidxca,CA,Varca) &
  !$OMP   PRIVATE(ica,ibl,resid,uvidx,amp,factor1,factor2) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do ica=1, Nca
    ! calc model closure phase
    resid = 0d0
    do ibl=1,4
      uvidx = abs(uvidxca(ibl,ica))
      amp = sqrt(Vreal(uvidx)*Vreal(uvidx)+Vimag(uvidx)*Vimag(uvidx))
      if (ibl < 3) then
        factor2(ibl) = 1/amp
        resid = resid + log(amp)
      else
        factor2(ibl) = -1/amp
        resid = resid - log(amp)
      end if
    end do
    CAmod(ica) = resid

    ! take a residual betweem model and data
    resid = resid - CA(ica)
    CAres(ica) = resid

    ! calc chi-square
    chisq = chisq + resid*resid/Varca(ica)

    ! calc gradient of chisquare
    factor1=2/Varca(ica)*resid
    do ibl=1,4
      uvidx = abs(uvidxca(ibl,ica))
      call daxpy(Npix,factor1*factor2(ibl),&
                 gradVamp(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
end module
