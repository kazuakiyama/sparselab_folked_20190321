#include "common.h"
#include <fftw3.h>

#ifdef NEED_EXTERN_C
extern "C" {
  #include "../contrib/legendre_rule_fast.h"
}
#else
  #include "../contrib/legendre_rule_fast.h"
#endif
#include <fftw3.h>
#include <math.h>
#include <stdio.h>

void finufft_default_opts(nufft_opts &o)
// set default nufft opts. See finufft.h for definition of opts.
// This was created to avoid uncertainty about C++11 style static initialization
// when called from MEX. Barnett 10/30/17
{
  o.R = (FLT)2.0;
  o.debug = 0;
  o.spread_debug = 0;
  o.spread_sort = 1;       
  o.fftw = FFTW_ESTIMATE;  // use FFTW_MEASURE for slow first call, fast rerun
  o.modeord = 0;
  o.chkbnds = 1;
}

int setup_spreader_for_nufft(spread_opts &spopts, FLT eps, nufft_opts opts)
// Set up the spreader parameters given eps, and pass across various nufft
// options. Report status of setup_spreader.  Barnett 10/30/17
{
  int ier = setup_spreader(spopts, eps, opts.R);
  spopts.debug = opts.spread_debug;
  spopts.sort = opts.spread_sort;
  spopts.chkbnds = opts.chkbnds;
  spopts.pirange = 1;             // could allow user control?
  return ier;
} 

void set_nf_type12(BIGINT ms, nufft_opts opts, spread_opts spopts, INT64 *nf)
// type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms.
{
  *nf = (INT64)(opts.R*ms);
  if (*nf<2*spopts.nspread) *nf=2*spopts.nspread; // otherwise spread fails
  if (*nf<MAX_NF)                                 // otherwise will fail anyway
    *nf = next235even(*nf);                       // expensive at huge nf
}

void set_nhg_type3(FLT S, FLT X, nufft_opts opts, spread_opts spopts,
		     INT64 *nf, FLT *h, FLT *gam)
/* sets nf, h (upsampled grid spacing), and gamma (x_j rescaling factor),
   for type 3 only.
   Inputs:
   X and S are the xj and sk interval half-widths respectively.
   opts and spopts are the NUFFT and spreader opts strucs, respectively.
   Outputs:
   nf is the size of upsampled grid for a given single dimension.
   h is the grid spacing = 2pi/nf
   gam is the x rescale factor, ie x'_j = x_j/gam  (modulo shifts).
   Barnett 2/13/17. Caught inf/nan 3/14/17. io int types changed 3/28/17
   New logic 6/12/17
*/
{
  int nss = spopts.nspread + 1;      // since ns may be odd
  FLT Xsafe=X, Ssafe=S;              // may be tweaked locally
  if (X==0.0)                        // logic ensures XS>=1, handle X=0 a/o S=0
    if (S==0.0) {
      Xsafe=1.0;
      Ssafe=1.0;
    } else Xsafe = max(Xsafe, 1/S);
  else
    Ssafe = max(Ssafe, 1/X);
  // use the safe X and S...
  FLT nfd = 2.0*opts.R*Ssafe*Xsafe/PI + nss;
  if (!isfinite(nfd)) nfd=0.0;                // use FLT to catch inf
  *nf = (INT64)nfd;
  //printf("initial nf=%ld, ns=%d\n",*nf,spopts.nspread);
  // catch too small nf, and nan or +-inf, otherwise spread fails...
  if (*nf<2*spopts.nspread) *nf=2*spopts.nspread;
  if (*nf<MAX_NF)                             // otherwise will fail anyway
    *nf = next235even(*nf);                   // expensive at huge nf
  *h = 2*PI / *nf;                            // upsampled grid spacing
  *gam = (FLT)*nf / (2.0*opts.R*Ssafe);       // x scale fac to x'
}

void onedim_dct_kernel(BIGINT nf, FLT *fwkerhalf, spread_opts opts)
/*
  Computes DCT coeffs of cnufftspread's real symmetric kernel, directly,
  exploiting narrowness of kernel. Uses phase winding for cheap eval on the
  regular freq grid.
  Note: obsolete, superceded by onedim_fseries_kernel.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier coeffs from indices 0 to nf/2 inclusive.
              (should be allocated for at least nf/2+1 FLTs)

  Single thread only. Barnett 1/24/17
 */
{
  int m=ceil(opts.nspread/2.0);        // how many "modes" (ker pts) to include
  FLT f[MAX_NSPREAD/2];
  for (int n=0;n<=m;++n)    // actual freq index will be nf/2-n, for cosines
    f[n] = evaluate_kernel((FLT)n, opts);  // center at nf/2
  for (int n=1;n<=m;++n)               //  convert from exp to cosine ampls
    f[n] *= 2.0;
  dcomplex a[MAX_NSPREAD/2],aj[MAX_NSPREAD/2];
  for (int n=0;n<=m;++n) {             // set up our rotating phase array...
    a[n] = exp(2*PI*ima*(FLT)(nf/2-n)/(FLT)nf);   // phase differences
    aj[n] = dcomplex(1.0,0.0);         // init phase factors
  }
  for (BIGINT j=0;j<=nf/2;++j) {       // loop along output array
    FLT x = 0.0;                    // register
    for (int n=0;n<=m;++n) {
      x += f[n] * real(aj[n]);         // only want cosine part
      aj[n] *= a[n];                   // wind the phases
    }
    fwkerhalf[j] = x;
  }
}

void onedim_fseries_kernel(BIGINT nf, FLT *fwkerhalf, spread_opts opts)
/*
  Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  grid. Note that this is also the Fourier transform of the non-periodized
  kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
  overall prefactor of 1/h, which is needed anyway for the correction, and
  arises because the quadrature weights are scaled for grid units not x units.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive,
              divided by h = 2pi/n.
              (should be allocated for at least nf/2+1 FLTs)

  Compare onedim_dct_kernel which has same interface, but computes DFT of
  sampled kernel, not quite the same object.

  todo: understand how to openmp it? - subtle since private aj's. Want to break
        up fwkerhalf into contiguous pieces, one per thread. Low priority.
  Barnett 2/7/17
 */
{
  FLT J2 = opts.nspread/2.0;         // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q=(int)(2 + 3.0*J2);  // not sure why so large? cannot exceed MAX_NQUAD
  FLT f[MAX_NQUAD]; double z[2*MAX_NQUAD],w[2*MAX_NQUAD];
  legendre_compute_glr(2*q,z,w);        // only half the nodes used, eg on (0,1)
  dcomplex a[MAX_NQUAD],aj[MAX_NQUAD];  // phase rotators
  for (int n=0;n<q;++n) {
    z[n] *= J2;                 // rescale nodes
    f[n] = J2*(FLT)w[n] * evaluate_kernel((FLT)z[n], opts);  // w/ quadr weights
    a[n] = exp(2*PI*ima*(FLT)(nf/2-z[n])/(FLT)nf);  // phase windings
    aj[n] = dcomplex(1.0,0.0);         // init phase factors
  }
  for (BIGINT j=0;j<=nf/2;++j) {       // loop along output array
    FLT x = 0.0;                    // register
    for (int n=0;n<q;++n) {
      x += f[n] * 2*real(aj[n]);       // include the negative freq
      aj[n] *= a[n];                   // wind the phases
    }
    fwkerhalf[j] = x;
  }
}

void onedim_nuft_kernel(BIGINT nk, FLT *k, FLT *phihat, spread_opts opts)
/*
  Approximates exact 1D Fourier transform of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Evaluates at set of arbitrary freqs k in [-pi,pi],
  for a kernel with x measured in grid-spacings. (See previous routine for
  FT definition).

  Inputs:
  nk - number of freqs
  k - frequencies, dual to the kernel's natural argument, ie exp(i.k.z)
       Note, z is in grid-point units, and k values must be in [-pi,pi] for
       accuracy.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  phihat - real Fourier transform evaluated at freqs (alloc for nk FLTs)

  Barnett 2/8/17. openmp since cos slow 2/9/17
 */
{
  FLT J2 = opts.nspread/2.0;        // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q=(int)(2 + 2.0*J2);     // > pi/2 ratio.  cannot exceed MAX_NQUAD
  if (opts.debug) printf("q (# ker FT quadr pts) = %d\n",q);
  FLT f[MAX_NQUAD]; double z[2*MAX_NQUAD],w[2*MAX_NQUAD];
  legendre_compute_glr(2*q,z,w);        // only half the nodes used, eg on (0,1)
  for (int n=0;n<q;++n) {
    z[n] *= J2;                                    // quadr nodes for [0,J/2]
    f[n] = J2*(FLT)w[n] * evaluate_kernel((FLT)z[n], opts);  // w/ quadr weights
    //    printf("f[%d] = %.3g\n",n,f[n]);
  }
  #pragma omp parallel for schedule(dynamic)
  for (BIGINT j=0;j<nk;++j) {          // loop along output array
    FLT x = 0.0;                    // register
    for (int n=0;n<q;++n) x += f[n] * 2*cos(k[j]*z[n]);  // pos & neg freq pair
    phihat[j] = x;
  }
}  

void deconvolveshuffle1d(int dir,FLT prefac,FLT* ker, BIGINT ms,
			 FLT *fk, BIGINT nf1, FFTW_CPX* fw, int modeord)
/*
  if dir==1: copies fw to fk with amplification by prefac/ker
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  modeord=0: use CMCL-compatible mode ordering in fk (from -N/2 up to N/2-1)
          1: use FFT-style (from 0 to N/2-1, then -N/2 up to -1).

  fk is size-ms FLT complex array (2*ms FLTs alternating re,im parts)
  fw is a FFTW style complex array, ie FLT [nf1][2], essentially FLTs
       alternating re,im parts.
  ker is real-valued FLT array of length nf1/2+1.

  Single thread only, but shouldn't matter since mostly data movement.

  It has been tested that the repeated floating division in this inner loop
  only contributes at the <3% level in 3D relative to the fftw cost (8 threads).
  This could be removed by passing in an inverse kernel and doing mults.

  todo: rewrite w/ native dcomplex I/O, check complex divide not slower than
        real divide, or is there a way to force a real divide?

  Barnett 1/25/17. Fixed ms=0 case 3/14/17. modeord flag & clean 10/25/17
*/
{
  BIGINT kmin = -ms/2, kmax = (ms-1)/2;    // inclusive range of k indices
  if (ms==0) kmax=-1;           // fixes zero-pad for trivial no-mode case
  // set up pp & pn as ptrs to start of pos(ie nonneg) & neg chunks of fk array
  BIGINT pp = -2*kmin, pn = 0;       // CMCL mode-ordering case (2* since cmplx)
  if (modeord==1) { pp = 0; pn = 2*(kmax+1); }   // or, instead, FFT ordering
  if (dir==1) {    // read fw, write out to fk...
    for (BIGINT k=0;k<=kmax;++k) {                    // non-neg freqs k
      fk[pp++] = prefac * fw[k][0] / ker[k];          // re
      fk[pp++] = prefac * fw[k][1] / ker[k];          // im
    }
    for (BIGINT k=kmin;k<0;++k) {                     // neg freqs k
      fk[pn++] = prefac * fw[nf1+k][0] / ker[-k];     // re
      fk[pn++] = prefac * fw[nf1+k][1] / ker[-k];     // im
    }
  } else {    // read fk, write out to fw w/ zero padding...
    for (BIGINT k=kmax+1; k<nf1+kmin; ++k) {  // zero pad precisely where needed
      fw[k][0] = fw[k][1] = 0.0; }
    for (BIGINT k=0;k<=kmax;++k) {                    // non-neg freqs k
      fw[k][0] = prefac * fk[pp++] / ker[k];          // re
      fw[k][1] = prefac * fk[pp++] / ker[k];          // im
    }
    for (BIGINT k=kmin;k<0;++k) {                     // neg freqs k
      fw[nf1+k][0] = prefac * fk[pn++] / ker[-k];     // re
      fw[nf1+k][1] = prefac * fk[pn++] / ker[-k];     // im
    }
  }
}

void deconvolveshuffle2d(int dir,FLT prefac,FLT *ker1, FLT *ker2,
			 BIGINT ms, BIGINT mt,
			 FLT *fk, BIGINT nf1, BIGINT nf2, FFTW_CPX* fw,
			 int modeord)
/*
  2D version of deconvolveshuffle1d, calls it on each x-line using 1/ker2 fac.

  if dir==1: copies fw to fk with amplification by prefac/(ker1(k1)*ker2(k2)).
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  modeord=0: use CMCL-compatible mode ordering in fk (each dim increasing)
          1: use FFT-style (pos then negative, on each dim)

  fk is complex array stored as 2*ms*mt FLTs alternating re,im parts, with
    ms looped over fast and mt slow.
  fw is a FFTW style complex array, ie FLT [nf1*nf2][2], essentially FLTs
       alternating re,im parts; again nf1 is fast and nf2 slow.
  ker1, ker2 are real-valued FLT arrays of lengths nf1/2+1, nf2/2+1
       respectively.

  Barnett 2/1/17, Fixed mt=0 case 3/14/17. modeord 10/25/17
*/
{
  BIGINT k2min = -mt/2, k2max = (mt-1)/2;    // inclusive range of k2 indices
  if (mt==0) k2max=-1;           // fixes zero-pad for trivial no-mode case
  // set up pp & pn as ptrs to start of pos(ie nonneg) & neg chunks of fk array
  BIGINT pp = -2*k2min*ms, pn = 0;   // CMCL mode-ordering case (2* since cmplx)
  if (modeord==1) { pp = 0; pn = 2*(k2max+1)*ms; }  // or, instead, FFT ordering
  if (dir==2)               // zero pad needed x-lines (contiguous in memory)
    for (BIGINT j=nf1*(k2max+1); j<nf1*(nf2+k2min); ++j)  // sweeps all dims
      fw[j][0] = fw[j][1] = 0.0;
  for (BIGINT k2=0;k2<=k2max;++k2, pp+=2*ms)          // non-neg y-freqs
    // point fk and fw to the start of this y value's row (2* is for complex):
    deconvolveshuffle1d(dir,prefac/ker2[k2],ker1,ms,fk + pp,nf1,&fw[nf1*k2],modeord);
  for (BIGINT k2=k2min;k2<0;++k2, pn+=2*ms)           // neg y-freqs
    deconvolveshuffle1d(dir,prefac/ker2[-k2],ker1,ms,fk + pn,nf1,&fw[nf1*(nf2+k2)],modeord);
}

void deconvolveshuffle3d(int dir,FLT prefac,FLT *ker1, FLT *ker2,
			 FLT *ker3, BIGINT ms, BIGINT mt, BIGINT mu,
			 FLT *fk, BIGINT nf1, BIGINT nf2, BIGINT nf3,
			 FFTW_CPX* fw, int modeord)
/*
  3D version of deconvolveshuffle2d, calls it on each xy-plane using 1/ker3 fac.

  if dir==1: copies fw to fk with ampl by prefac/(ker1(k1)*ker2(k2)*ker3(k3)).
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  modeord=0: use CMCL-compatible mode ordering in fk (each dim increasing)
          1: use FFT-style (pos then negative, on each dim)

  fk is complex array stored as 2*ms*mt*mu FLTs alternating re,im parts, with
    ms looped over fastest and mu slowest.
  fw is a FFTW style complex array, ie FLT [nf1*nf2*nf3][2], effectively
       FLTs alternating re,im parts; again nf1 is fastest and nf3 slowest.
  ker1, ker2, ker3 are real-valued FLT arrays of lengths nf1/2+1, nf2/2+1,
       and nf3/2+1 respectively.

  Barnett 2/1/17, Fixed mu=0 case 3/14/17. modeord 10/25/17
*/
{
  BIGINT k3min = -mu/2, k3max = (mu-1)/2;    // inclusive range of k3 indices
  if (mu==0) k3max=-1;           // fixes zero-pad for trivial no-mode case
  // set up pp & pn as ptrs to start of pos(ie nonneg) & neg chunks of fk array
  BIGINT pp = -2*k3min*ms*mt, pn = 0; // CMCL mode-ordering (2* since cmplx)
  if (modeord==1) { pp = 0; pn = 2*(k3max+1)*ms*mt; }  // or FFT ordering
  
  BIGINT k03 = mu/2;    // z-index shift in fk's = magnitude of most neg z-freq
  BIGINT k13 = (mu-1)/2;  // most pos freq
  if (mu==0) k13=-1;      // correct the rounding down for no-mode case

  BIGINT np = nf1*nf2;  // # pts in an upsampled Fourier xy-plane
  if (dir==2)           // zero pad needed xy-planes (contiguous in memory)
    for (BIGINT j=np*(k3max+1);j<np*(nf3+k3min);++j)  // sweeps all dims
      fw[j][0] = fw[j][1] = 0.0;
  for (BIGINT k3=0;k3<=k3max;++k3, pp+=2*ms*mt)      // non-neg z-freqs
    // point fk and fw to the start of this z value's plane (2* is for complex):
    deconvolveshuffle2d(dir,prefac/ker3[k3],ker1,ker2,ms,mt,
			fk + pp,nf1,nf2,&fw[np*k3],modeord);
  for (BIGINT k3=k3min;k3<0;++k3, pn+=2*ms*mt)       // neg z-freqs
    deconvolveshuffle2d(dir,prefac/ker3[-k3],ker1,ker2,ms,mt,
			fk + pn,nf1,nf2,&fw[np*(nf3+k3)],modeord);
}
