#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
#-------------------------------------------------------------------------------
AC_PREREQ([2.69])
AC_INIT([Sparselab], [0.0.1], [kakiyama@mit.edu])
AC_CONFIG_SRCDIR([src/mfista/blas.h])


#-------------------------------------------------------------------------------
# Checks for programs.
#-------------------------------------------------------------------------------
# C Compiler
AC_PROG_CC

# Fortran Compiler
AC_PROG_FC

# CPP Compiler
AC_PROG_CXX

# f2py in numpy
AC_CHECK_PROGS([F2PY],
               [f2py],
               [f2py])


#-------------------------------------------------------------------------------
# Check FORTRAN libraries.
#-------------------------------------------------------------------------------
AC_LANG([Fortran])
AC_LANG_PUSH(Fortran)

# BLAS
AC_CHECK_LIB(openblas, dgemm,
             [blaslib="-lopenblas"],
             [AC_CHECK_LIB(blas, dgemm,
                           [blaslib="-lblas"],
                           [AC_MSG_ERROR([blas was not found.])]
             )]
)
AC_SUBST([blaslib])

# LAPACK
AC_CHECK_LIB(lapack, dgesv,
             [lapacklib="-llapack"],
             [AC_MSG_ERROR([lapack was not found.])])
AC_SUBST([lapacklib])
AC_LANG_POP(Fortran)


#-------------------------------------------------------------------------------
# Check C libraries.
#-------------------------------------------------------------------------------
AC_LANG_PUSH(C)

# FFTW 3
AC_CHECK_LIB(fftw3, fftw_malloc,
             [fftw3lib="-lfftw3"],
             [AC_MSG_ERROR([fftw3 was not found.])]
)
AC_SUBST([fftw3lib])

AC_CHECK_LIB(fftw3_omp, fftw_init_threads,
  [fftw3thlib="-lfftw3_omp"],
  [AC_CHECK_LIB(fftw3_threads, fftw_init_threads,
    [fftw3thlib="-lfftw3_threads"],
    [AC_MSG_ERROR([fftw3_threads nor fftw3_omp was not found.])]
  )]
)

AC_SUBST([fftw3thlib])

AC_LANG_POP(C)


#-------------------------------------------------------------------------------
# Check CPP libraries.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Checks for header files.
#-------------------------------------------------------------------------------
# AC_CHECK_HEADERS([limits.h stdlib.h string.h sys/time.h])


#-------------------------------------------------------------------------------
# Checks for typedefs, structures, and compiler characteristics.
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Checks for library functions.
#-------------------------------------------------------------------------------
#AC_FUNC_MALLOC
#AC_CHECK_FUNCS([clock_gettime floor gettimeofday memset pow sqrt])


#-------------------------------------------------------------------------------
# Command Options
#-------------------------------------------------------------------------------
#  debug options
AC_ARG_ENABLE([dbg],
              [AC_HELP_STRING([--enable-dbg],
                              [Compile source files with DEBUG options])],
              [dbg_flag="True"],
              [dbg_flag="False"])
AC_SUBST([dbg_flag])


#-------------------------------------------------------------------------------
# Makefiles
AC_CONFIG_FILES([makefile
                 src/lbfgsb/makefile
                 src/slsqp/makefile
                 src/finufft/makefile
                 src/fortlib/makefile
                 src/mfista/makefile])
AC_OUTPUT
