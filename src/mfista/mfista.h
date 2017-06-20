/* 
   Copyright (C) 2015   Shiro Ikeda <shiro@ism.ac.jp>

   This is file 'mfista.h'. An optimization algorithm for imaging of
   interferometry. The idea of the algorithm was from the following
   two papers,

   Beck and Teboulle (2009) SIAM J. Imaging Sciences, 
   Beck and Teboulle (2009) IEEE trans. on Image Processing 


   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "blas.h"
#include "lapack.h"

#define CINIT     10000 
#define MAXITER   50000
#define MINITER   100
#define FGPITER   100
#define TD         50 
#define ETA        1.1
#define EPS        1.0e-5

/* result format */

struct RESULT{
  int M;
  int N;
  int NX;
  int NY;
  int N_active;
  int maxiter;
  int ITER;
  int nonneg;
  double lambda_l1;
  double lambda_tv;
  double lambda_tsv;
  double sq_error;
  double mean_sq_error;
  double l1cost;
  double tvcost;
  double tsvcost;
  double looe;
  double Hessian_positive;
  double finalcost;
};

/* memory allocation of matrix and vectors */

extern double *alloc_vector(int length);
extern double *alloc_matrix(int height, int width);
extern void    clear_matrix(double *matrix, int height, int width);

/* file in-out */

extern FILE* fopenr(char* fn);
extern FILE* fopenw(char* fn);

/* simple matrix operations */

extern void transpose_matrix(double *matrix, int origheight, int origwidth);

extern void calc_yAz(int *M, int *N,
                     double *yvec, double *Amat, double *zvec,
                     int *inc, double *yAz);

extern double calc_F_part(int *M, int *N,
			  double *yvec, double *Amatrix,
			  double *xvec, int *inc, double *buffvec);

extern double calc_Q_part(int *N, 
			  double *xvec1, double *xvec2,
			  double c, int *inc,
			  double *AyAz, double *buffxvec1);
			  
/* thresholding */

extern void soft_threshold(double *vector, int length, double eta, 
			   double *newvec);

extern void soft_threshold_nonneg(double *vector, int length, double eta, 
				  double *newvec);

extern int find_active_set(int N, double *xvec, int *indx_list);

/* index transform */

extern int i2r(int i, int NX);

extern int i2c(int i, int NX);

extern int rc2i(int r, int c, int NX);

/* Some routines for computing LOOE */

extern double *shrink_A(int M, int N, int N_active, int *indx_list,
			double *Amat);

extern int solve_lin_looe(int *NA, int *NB, double *Hessian, double *B);

extern double compute_LOOE_core(int *M, int N_active, 
				double *yvec, double *Amat, double *xvec,
				double *yAx,  double *Amat_s, double *Hessian);

/* subroutines for mfista_L1 */

extern void mfista_L1_core(double *yvec, double *Amat, int *M, int *N, 
			   double lambda, double cinit,
			   double *xvec, int nonneg_flag, int looe_flag,
			   struct RESULT *mfista_result);

/* subroutines for mfista_L1_TV_nonneg */

extern void mfista_L1_TV_core(double *yvec, double *Amat, 
			      int *M, int *N, int NX, int NY,
			      double lambda, double lambda_tv, double cinit,
			      double *xvec,
			      struct RESULT *mfista_result);

/* subroutines for mfista_L1_TV_nonneg */

extern void mfista_L1_TV_core_nonneg(double *yvec, double *Amat, 
				     int *M, int *N, int NX, int NY,
				     double lambda, double lambda_tv, double cinit,
				     double *xvec,
				     struct RESULT *mfista_result);

/* subroutines for mfista_L1_sqTV_nonneg */

extern void mfista_L1_TSV_core(double *yvec, double *Amat, 
				int *M, int *N, int NX, int NY,
				double lambda, double lambda_tv, double cinit,
				double *xvec, int nonneg_flag, int looe_flag,
				struct RESULT *mfista_result);

/* subroutines for calculating A matrix */
extern void calc_A(int M,int N,int dftsign,
             double *u,double *v,double *x,double *y,double *A,double *Verr);

/* subroutines for mfista */
extern void mfista(double* u,double* v,double* x,double* y,
            double* Iin,double* Iout,double* V,double* Vsigma,
            int dftsign,int M,int N,int NX,int NY,
            double cinit,double lambda_l1,double lambda_tv,double lambda_tsv,
            int nonneg_flag,int rec_flag,int looe_flag,int log_flag,
            char *log_fname,struct RESULT *mfista_result);

/* output */

extern void show_result(FILE *fid, char *fname, struct RESULT *mfista_result);

