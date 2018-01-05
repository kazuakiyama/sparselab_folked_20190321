from setuptools import setup, find_packages
from codecs     import open
from os         import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding='utf-8') as f:
    long_description = f.read()

if path.isfile(path.join(here, "sparselab/libmfista.so")):
    errmsg="Apparently, you have not compiled C/Fortran libraries with make."
    errmsg+=" Please install this library by 'make install' not by 'python setup.py install'"
    raise RuntimeError(errmsg)

setup(
    name="sparselab",
    version = "0.0.1",
    description = "A Library for Interferometric Imaging and related analysis using Sparse Modeling",
    long_description = long_description,
    url = "https://eht-jp.github.io/sparselab",
    author = "Kazu Akiyama",
    author_email = "kakiyama@mit.edu",
    license = "MIT",
    keywords = "imaging astronomy EHT",
    packages = find_packages(exclude=["doc*", "test*"]),
    package_data={'sparselab': ['*.so']},
    install_requires = [
        "numpy","scipy","matplotlib","pandas","xarray",
        "scikit-image", "astropy", "tqdm", "future"]
)
