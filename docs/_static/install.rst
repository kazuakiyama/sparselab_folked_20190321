============
Installation
============

Requirements
===============

Sparselab consists of python modules and Fortran/C internal libraries called from python modules.
Here, we summarize required python packages and external packages for Sparselab.

You will also need **autoconf** and `ds9`_ for compiling the library.

.. _ds9: http://ds9.si.edu/site/Home.html

Python Packages and Modules
---------------------------

Sparselab uses **numpy**, **scipy**, **matplotlib**, **pandas**, **astropy**, **xarray**, **pyds9**, **tqdm**.
Sparselab has been tested and developped in Python 2.7 environments provided by the `Anaconda`_ package that
includes required packages except **xarray** and **pyds9**. We recommend using Anaconda for Sparselab.

.. _Anaconda: https://www.continuum.io/anaconda-overview

You can install **xarray**, **tqdm** and **pyds9** with conda and/or pip as follows
(see the official website of `pyds9`_ for its installation).

.. code-block:: Bash

  # if you have conda
  conda install xarray
  conda install tqdm
  # You may use pip, if you do not have or want to use conda
  pip install xarray
  pip install tqdm

  # to install pyds9, you can use pip command.
  pip install git+https://github.com/ericmandel/pyds9.git#egg=pyds9

.. _xarray: http://xarray.pydata.org/en/stable/
.. _pyds9: https://github.com/ericmandel/pyds9


External Libraries
------------------

Fortran/C internal libraries of Sparselab use following external libraries.

1) BLAS
  **We strongly recommend using OpenBLAS**, which is the fastest library among publicly-available BLAS implementations.
  Our recommendation is to build up `OpenBLAS`_ by yourself with a compile option USE_OPENMP=1 and use it for our library.
  The option USE_OPENMP=1 enables OpenBLAS to perform paralleled multi-threads calculations, which will accelerate our library.

  .. _OpenBLAS: https://github.com/xianyi/OpenBLAS

  Some Tips:
    If you are using Ubuntu (at least after 14.04 LTS), the default OpenBLAS package,
    which is installable with `apt-get` or `aptitude`, seems compiled with
    this option (USE_OPENMP=1), so you do not have to compile it by yourself.

    It seems that RedHat or its variant (Cent OS, Scientific Linux, etc) do not have
    the standard package compiled with this option, so we recommend compiling OpenBLAS by yourself.

    If you are using Mac OS X, unfortunately, this option is not available so far.
    You may use a package available in a popular package system (e.g. MacPort, Fink, Homebrew).

2) LAPACK
  LAPACK does not have a big impact on computational costs of imaging.
  The default LAPACK package in your Linux/OS X package system would be acceptable for Spareselab.
  Of course, you may build up `LAPACK`_ by yourself.

  .. _LAPACK: https://github.com/Reference-LAPACK/lapack-release

3) FFTW3
  Some module uses fftw3. The default FFTW 3 package in your Linux/OS X package
  system should be acceptable for Sparselab.
  Of course, you may build up `FFTW3`_ by yourself.

  .. _FFTW3: http://www.fftw.org


Download, Install and Update
============================

Downloading Sparselab
---------------------
You can download the code from github.

.. code-block:: Bash

  # Clone the repository
  git clone https://github.com/eht-jp/sparselab

Installing Sparselab
--------------------

For compiling the whole library, you need to work in your Sparselab directory.

.. code-block:: Bash

  cd (Your Sparselab Directory)

A configure file can be generated with `autoconf`.

.. code-block:: Bash

  autoconf

Generate Makefiles with `./configure`. You might need `LDFLAGS` for links to BLAS and LAPACK.

.. code-block:: Bash

  # If you already have a library path to both BLAS and LAPACK.
  ./configure

  # If you don't have a PATH to BLAS and LAPACK, you can add links to them as follows
  ./configure LDFLAGS="-L(path-to-your-BLAS) -L(path-to-your-LAPACK) -L(path-to-your-FFTW3)"

If you are a Mac OS X user using MacPort, Fink, or Homebrew,
`LDFLAGS="-L/opt/local/lib"`, `LDFLAGS="-L/sw/lib"` or `LDFLAGS="-L/usr/local/lib"`
would work, respectively.

Make and compile the library.
The internal C/Fortran Library will be compiled into python modules,
and then the whole python modules will be added to the package list of
your Python environment.

.. code-block:: Bash

  make install

If you can load following modules in your python interpretator,
Sparselab is probably installed successfully.

.. code-block:: Python

  # import sparselab
  from sparselab import imdata, uvdata, imaging

**(IMPORTANT NOTE; 2018/01/04)**
Previously, you needed to add a PYTHONPATH to your Sparselab Directory.
This is no longer required, because the `make` command will run setup.py and install
sparselab into the package list of your Python environment.

Updating Sparselab
------------------

**We strongly recommend cleaning up the entire library before updating.**

.. code-block:: Bash

  cd (Your Sparselab Directory)
  make uninstall

Then, you can update the repository with `git pull`.

.. code-block:: Bash

  git pull

Now, the repository has updated. You can follow the above section `Installing Sparselab`_ for recompiling your Sparselab.
