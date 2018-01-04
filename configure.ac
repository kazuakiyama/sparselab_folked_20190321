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
AC_PROG_FC([gfortran], [95])

# f2py in numpy
AC_CHECK_PROGS([F2PY],
               [f2py],
               [AC_MSG_ERROR([f2py was not found.])])


#-------------------------------------------------------------------------------
# Checks for FORTRAN libraries.
#-------------------------------------------------------------------------------
AC_LANG([Fortran])
AC_LANG_PUSH(Fortran)

# Blas
AC_CHECK_LIB(openblas, dgemm,
             [blaslib="-lopenblas"],
             [AC_CHECK_LIB(blas, dgemm,
                           [blaslib="-lblas"],
                           [AC_MSG_ERROR([blas was not found.])]
             )]
)
AC_SUBST([blaslib])

# lapack
AC_CHECK_LIB(lapack, dgesv,
             [lapacklib="-llapack"],
             [AC_MSG_ERROR([lapack was not found.])])
AC_SUBST([lapacklib])
AC_LANG_POP(Fortran)


#-------------------------------------------------------------------------------
# Checks for FORTRAN libraries.
#-------------------------------------------------------------------------------
AC_LANG_PUSH(C)
AC_CHECK_LIB(fftw3, fftw_malloc,
             [fftw3lib="-lfftw3"],
             [AC_MSG_ERROR([fftw3 was not found.])]
)
AC_SUBST([fftw3lib])
AC_LANG_POP(C)


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
                 src/fortlib/makefile
                 src/lbfgsb/makefile
                 src/mfista/makefile
                 src/slsqp/makefile])
AC_OUTPUT
