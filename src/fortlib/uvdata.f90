module uvdata
  !$use omp_lib
  use param, only: dp, sp
  implicit none
contains
!
! average
!
subroutine average(uvdata,coord1,coord2,ant,subarray,tsec,solint,minpoint, &
                   uvdataout,coord1out,coord2out,isdata,&
                   Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata,Nt,Nant,Narr)
  integer, intent(in) :: Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata
  integer, intent(in) :: Nt,Nant,Narr
  real(sp), intent(in) :: uvdata(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  integer, intent(in) :: coord1(3,Ndata),ant(Nant),subarray(Narr)
  real(dp), intent(in) :: coord2(2,Ndata)
  integer, intent(in) :: minpoint
  real(dp), intent(in) :: solint,tsec(Nt)
  integer, intent(out) :: coord1out(3,Nant*(Nant-1)/2*Narr*Nt)
  real(dp), intent(out) :: coord2out(2,Nant*(Nant-1)/2*Narr*Nt)
  real(sp), intent(out) :: uvdataout(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Nant*(Nant-1)/2*Narr*Nt)
  logical, intent(out) :: isdata(Nant*(Nant-1)/2*Narr*Nt)

  integer :: i1,i2,i3,i4,i5,i6,i7
  integer :: Nbl, Ndata2, dammy
  integer :: iant1(Nant*(Nant-1)/2), iant2(Nant*(Nant-1)/2), ibl, iarr, it
  integer :: count
  logical :: flag

  Nbl = Nant*(Nant-1)/2
  Ndata2 = Nbl*Narr*Nt

  ! initialize arrays
  uvdataout(:,:,:,:,:,:,:) = 0.0
  coord1out(:,:) = 0d0
  coord2out(:,:) = 0d0
  isdata(:) = .False.

  ! initialize iant1, iant2
  i2 = 1
  i3 = 2
  do i1 = 1, Nbl
    iant1(i1) = i2
    iant2(i1) = i3
    i2 = i2 + 1
    if (i2 > Nant) then
      i2 = i3
      i3 = i3+1
    end if
  end do

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata,Nbl,Ndata2, &
  !$OMP                iant1,iant2,ant,subarray,tsec,solint,minpoint, &
  !$OMP                coord1,coord2) &
  !$OMP   PRIVATE(i1,i2,i3,i4,i5,i6,i7,count,dammy,ibl,iarr,flag)
  do i1=1,Ndata2
    ! get indexes
    ibl = mod(i1, Nbl) + 1
    dammy = (i1-ibl)/Nbl
    iarr = mod(dammy, Narr) + 1
    dammy = (dammy - iarr)/Narr
    it = mod(dammy, Nt) + 1

    coord1out(1,i1) = ant(iant1(ibl))
    coord1out(2,i1) = ant(iant2(ibl))
    coord1out(3,i1) = subarray(iarr)
    coord2out(1,i1) = tsec(it)

    count = 0
    do i2=1,Ndata
      ! check this segment has on the same baseline, subarray and within solint.
      flag = coord1(i2,1) == coord1out(1,i1)
      flag = flag .and. (coord1(2,i2) == coord1out(2,i1))
      flag = flag .and. (coord1(3,i2) == coord1out(3,i1))
      flag = flag .and. (abs(coord2(1,i2)-coord2out(1,i1)) < solint)
      if (flag .eqv. .False.) then
        cycle
      end if

      ! Take sum of visibilities and weights
      isdata(i1) = .True.  ! Note that data are averaged
      count = count + 1    ! number of points
      coord2out(2,i1) = coord2out(2,i1) + coord2(2,i2)
      do i3=1, Ndec
        do i4=1, Nra
          do i5=1, Nif
            do i6=1, Nch
              do i7=1, Nstokes
                if (uvdata(3,i7,i6,i5,i4,i3,i2) < epsilon(1.0)) then
                  cycle
                end if
                if (uvdata(3,i7,i6,i5,i4,i3,i2) > huge(1.0)) then
                  cycle
                end if
                if (uvdata(3,i7,i6,i5,i4,i3,i2) .ne. uvdata(3,i7,i6,i5,i4,i3,i2)) then
                  cycle
                end if
                uvdataout(1:2,i7,i6,i5,i4,i3,i1) &
                  = uvdata(1:2,i7,i6,i5,i4,i3,i2) * uvdata(3,i7,i6,i5,i4,i3,i2) &
                  + uvdataout(1:2,i7,i6,i5,i4,i3,i1)
                uvdataout(3,i7,i6,i5,i4,i3,i1) &
                  = uvdataout(3,i7,i6,i5,i4,i3,i1) + uvdata(3,i7,i6,i5,i4,i3,i2)
              end do
            end do
          end do
        end do
      end do
    end do
    if (isdata(i1) .eqv. .True.) then
      uvdataout(1,:,:,:,:,:,i1) = uvdataout(1,:,:,:,:,:,i1)/uvdataout(3,:,:,:,:,:,i1)
      uvdataout(2,:,:,:,:,:,i1) = uvdataout(2,:,:,:,:,:,i1)/uvdataout(3,:,:,:,:,:,i1)
    end if
    if (count < minpoint) then
      uvdataout(3,:,:,:,:,:,i1) = 0
      isdata(i1) = .False.
    end if
  end do
  !$OMP END PARALLEL DO
end subroutine
!
! weightcal
!
subroutine weightcal(uvdata,tsec,ant1,ant2,subarray,source,&
                     solint,dofreq,minpoint,uvdataout,&
                     Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  implicit none

  integer,  intent(in) :: Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata
  integer, intent(in) :: source(Ndata),ant1(Ndata),ant2(Ndata),subarray(Ndata)
  integer, intent(in) :: dofreq,minpoint
  real(sp), intent(in) :: uvdata(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  real(dp), intent(in) :: solint,tsec(Ndata)
  real(sp), intent(out) :: uvdataout(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)

  integer :: i1,i2,i3,i4,i5,i6,i7,N
  logical :: flag
  real(dp) :: ave, msq, var
  real(dp) :: vm(Nstokes,Nch,Nif,Nra,Ndec), vr(Nstokes,Nch,Nif,Nra,Ndec)
  integer :: cnt(Nstokes,Nch,Nif,Nra,Ndec)

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(source,ant1,ant2,subarray,tsec,solint,dofreq,minpoint, &
  !$OMP                Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata) &
  !$OMP   PRIVATE(i1,i2,i3,i4,i5,i6,i7,N,ave,msq,var,vm,vr,cnt,flag)
  do i1=1,Ndata
    uvdataout(:,:,:,:,:,:,i1) = uvdata(:,:,:,:,:,:,i1)
    vm(:,:,:,:,:) = 0d0
    vr(:,:,:,:,:) = 0d0
    cnt(:,:,:,:,:) = 0
    do i2=1,Ndata
      ! check source and stokes
      flag = (source(i1) == source(i2))
      flag = flag .and. (ant1(i1) == ant1(i2))
      flag = flag .and. (ant2(i1) == ant2(i2))
      flag = flag .and. (subarray(i1) == subarray(i2))
      flag = flag .and. (abs(tsec(i2)-tsec(i1)) < solint)
      if (flag .eqv. .False.) then
        cycle
      end if
      ! calc sum and squared-sum of vreal, vimag
      do i3=1, Ndec
        do i4=1, Nra
          do i5=1, Nif
            do i6=1, Nch
              do i7=1, Nstokes
                if (uvdataout(3,i7,i6,i5,i4,i3,i2) < epsilon(1.0)) then
                  cycle
                end if
                if (uvdataout(3,i7,i6,i5,i4,i3,i2) > huge(1.0)) then
                  cycle
                end if
                if (uvdataout(3,i7,i6,i5,i4,i3,i2) .ne. uvdataout(3,i7,i6,i5,i4,i3,i2)) then
                  cycle
                end if
                vm(i7,i6,i5,i4,i3) = uvdata(1,i7,i6,i5,i4,i3,i2) &
                                   + vm(i7,i6,i5,i4,i3)
                vm(i7,i6,i5,i4,i3) = uvdata(2,i7,i6,i5,i4,i3,i2) &
                                   + vm(i7,i6,i5,i4,i3)
                vr(i7,i6,i5,i4,i3) = uvdata(1,i7,i6,i5,i4,i3,i2) &
                                   * uvdata(1,i7,i6,i5,i4,i3,i2) &
                                   + vr(i7,i6,i5,i4,i3)
                vr(i7,i6,i5,i4,i3) = uvdata(2,i7,i6,i5,i4,i3,i2) &
                                   * uvdata(2,i7,i6,i5,i4,i3,i2) &
                                   + vr(i7,i6,i5,i4,i3)
                cnt(i7,i6,i5,i4,i3) = cnt(i7,i6,i5,i4,i3)+2
              end do
            end do
          end do
        end do
      end do
    end do

    ! calc weight
    do i2=1, Ndec
      do i3=1, Nra
        if (dofreq .eq. 0) then
          do i4=1, Nstokes
            N = sum(cnt(i4,:,:,i3,i2))
            if (N <= 2*minpoint) then
              uvdataout(3,i4,:,:,i3,i2,i1) = 0.0
            end if
            ave = sum(vm(i4,:,:,i3,i2))/N
            msq = sum(vr(i4,:,:,i3,i2))/N
            var = msq - ave*ave
            uvdataout(3,i4,:,:,i3,i2,i1) = sngl(1d0/var)
          end do
        else if (dofreq .eq. 1) then
          do i4=1, Nif
            do i5=1, Nstokes
              N = sum(cnt(i5,:,i4,i3,i2))
              if (N <= 2*minpoint) then
                uvdataout(3,i5,:,i4,i3,i2,i1) = 0.0
              end if
              ave = sum(vm(i5,:,i4,i3,i2))/N
              msq = sum(vr(i5,:,i4,i3,i2))/N
              var = msq - ave*ave
              uvdataout(3,i5,:,i4,i3,i2,i1) = sngl(1d0/var)
            end do
          end do
        else
          do i4=1, Nif
            do i5=1, Nch
              do i6=1, Nstokes
                N = cnt(i6,i5,i4,i3,i2)
                if (N <= 2*minpoint) then
                  uvdataout(3,i6,i5,i4,i3,i2,i1) = 0.0
                end if
                ave = vm(i6,i5,i4,i3,i2)/N
                msq = vr(i6,i5,i4,i3,i2)/N
                var = msq - ave*ave
                uvdataout(3,i6,i5,i4,i3,i2,i1) = sngl(1d0/var)
              end do
            end do
          end do
        end if
        do i4=1, Nif
          do i5=1, Nch
            do i6=1, Nstokes
              if (uvdata(3,i6,i5,i4,i3,i2,i1) < epsilon(1.0)) then
                uvdataout(3,i6,i5,i4,i3,i2,i1) = 0
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) > huge(1.0)) then
                uvdataout(3,i6,i5,i4,i3,i2,i1) = 0
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) .ne. uvdata(3,i6,i5,i4,i3,i2,i1)) then
                uvdataout(3,i6,i5,i4,i3,i2,i1) = 0
              end if
            end do
          end do
        end do
      end do
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
! average spectrum
!
subroutine avspc_dofreq0(uvdata,uvdataout,&
                         Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  implicit none

  integer,  intent(in) :: Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata
  real(sp), intent(in) :: uvdata(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  real(sp), intent(out) :: uvdataout(Ncomp,Nstokes,1,1,Nra,Ndec,Ndata)

  real(dp) :: weigsum(1:Nstokes), realsum(1:Nstokes), imagsum(1:Nstokes)
  integer :: i1,i2,i3,i4,i5,i6

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata) &
  !$OMP   PRIVATE(i1,i2,i3,i4,i5,i6,weigsum,realsum,imagsum)
  do i1=1,Ndata
    do i2=1,Ndec
      do i3=1,Nra
        weigsum(:) = 0
        realsum(:) = 0
        imagsum(:) = 0
        do i4=1,Nif
          do i5=1,Nch
            do i6=1,Nstokes
              if (uvdata(3,i6,i5,i4,i3,i2,i1) < epsilon(1.0)) then
                cycle
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) > huge(1.0)) then
                cycle
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) .ne. uvdata(3,i6,i5,i4,i3,i2,i1)) then
                cycle
              end if
              weigsum(i6) = weigsum(i6) + uvdata(3,i6,i5,i4,i3,i2,i1)
              realsum(i6) = realsum(i6) + uvdata(1,i6,i5,i4,i3,i2,i1) * uvdata(3,i6,i5,i4,i3,i2,i1)
              imagsum(i6) = imagsum(i6) + uvdata(2,i6,i5,i4,i3,i2,i1) * uvdata(3,i6,i5,i4,i3,i2,i1)
            end do
          end do
        end do
        do i4=1,Nstokes
          uvdataout(1,i4,1,1,i3,i2,i1) = sngl(realsum(i4)/weigsum(i4))
          uvdataout(2,i4,1,1,i3,i2,i1) = sngl(imagsum(i4)/weigsum(i4))
          uvdataout(3,i4,1,1,i3,i2,i1) = sngl(weigsum(i4))
        end do
      end do
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
! average spectrum
!
subroutine avspc_dofreq1(uvdata,uvdataout,&
                         Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  implicit none

  integer,  intent(in) :: Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata
  real(sp), intent(in) :: uvdata(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  real(sp), intent(out) :: uvdataout(Ncomp,Nstokes,1,Nif,Nra,Ndec,Ndata)

  real(dp) :: weigsum(1:Nstokes), realsum(1:Nstokes), imagsum(1:Nstokes)
  integer :: i1,i2,i3,i4,i5,i6

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata) &
  !$OMP   PRIVATE(i1,i2,i3,i4,i5,i6,weigsum,realsum,imagsum)
  do i1=1,Ndata
    do i2=1,Ndec
      do i3=1,Nra
        do i4=1,Nif
          weigsum(:) = 0
          realsum(:) = 0
          imagsum(:) = 0
          do i5=1,Nch
            do i6=1,Nstokes
              if (uvdata(3,i6,i5,i4,i3,i2,i1) < epsilon(1.0)) then
                cycle
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) > huge(1.0)) then
                cycle
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) .ne. uvdata(3,i6,i5,i4,i3,i2,i1)) then
                cycle
              end if
              weigsum(i6) = weigsum(i6) + uvdata(3,i6,i5,i4,i3,i2,i1)
              realsum(i6) = realsum(i6) + uvdata(1,i6,i5,i4,i3,i2,i1) * uvdata(3,i6,i5,i4,i3,i2,i1)
              imagsum(i6) = imagsum(i6) + uvdata(2,i6,i5,i4,i3,i2,i1) * uvdata(3,i6,i5,i4,i3,i2,i1)
            end do
          end do
          do i5=1,Nstokes
            uvdataout(1,i5,1,i4,i3,i2,i1) = sngl(realsum(i5)/weigsum(i5))
            uvdataout(2,i5,1,i4,i3,i2,i1) = sngl(imagsum(i5)/weigsum(i5))
            uvdataout(3,i5,1,i4,i3,i2,i1) = sngl(weigsum(i5))
          end do
        end do
      end do
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
end module
