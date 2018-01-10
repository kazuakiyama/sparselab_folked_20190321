#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of sparselab.modelfit, containing functions to
calculate visibilities and images of some geometric models
'''
import numpy as np
import astropy.constants as ac
import theano
import theano.tensor as tt

from .. import util

#-------------------------------------------------------------------------------
# Phase for symmetric source
#-------------------------------------------------------------------------------
def phase(u,v,x0=0,y0=0,angunit="mas"):
    return 2*np.pi*(u*x0+v*y0)*util.angconv(angunit, "rad")


#-------------------------------------------------------------------------------
# Circular Gaussian
#-------------------------------------------------------------------------------
def cgauss_cmp(u,v,x0=0.0,y0=0.0,totalflux=1.0,size=1.0,angunit="mas"):
    Vamp = cgauss_amp(u,v,totalflux,size,angunit)
    Vpha = tt.exp(1j*phase(u,v,x0,y0,angunit))
    return Vamp * Vpha


def cgauss_amp(u,v,totalflux=1.0,size=1.0,angunit="mas"):
    gamma = cgauss_lognamp(u,v,size,angunit)
    return totalflux * tt.exp(gamma)


def cgauss_lognamp(u,v,size=1.0,angunit="mas"):
    # conversion factors
    angunit2rad = util.angconv(angunit, "rad")
    deg2rad = np.pi/180.
    fwhm2sigrad = angunit2rad/np.sqrt(8*np.log(2))

    # maj/min axis size in std
    sig = majsize*fwhm2sigrad # sigma in radian

    #  Exponent
    gamma = -2*np.pi*np.pi*(tt.square(u)+tt.square(v))*sig*sig

    return gamma


#-------------------------------------------------------------------------------
# Elliptical Gaussian
#-------------------------------------------------------------------------------
def egauss_cmp(
        u,v,x0=0.0,y0=0.0,totalflux=1.0,majsize=1.0,minsize=1.0,pa=0.0,angunit="mas"):
    Vamp = egauss_amp(u,v,totalflux,majsize,minsize,pa,angunit)
    Vpha = tt.exp(1j*phase(u,v,x0,y0,angunit))
    return Vamp * Vpha


def egauss_amp(u,v,totalflux=1.0,majsize=1.0,minsize=1.0,pa=0.0,angunit="mas"):
    #  Exponent
    gamma = egauss_lognamp(u,v,majsize,minsize,pa,angunit)
    return totalflux * tt.exp(gamma)


def egauss_lognamp(u,v,majsize=1.0,minsize=1.0,pa=0.0,angunit="mas"):
    # conversion factors
    angunit2rad = util.angconv(angunit, "rad")
    deg2rad = np.pi/180.
    fwhm2sigrad = angunit2rad/np.sqrt(8*np.log(2))

    # maj/min axis size in std
    sigmaj = majsize*fwhm2sigrad # sigma_major in radian
    sigmin = minsize*fwhm2sigrad # sigma_minor in radian

    # Calculate Gaussian
    if np.isscalar(pa):
        parad = pa*deg2rad
        cospa = np.cos(parad)
        sinpa = np.sin(parad)
    else:
        parad = pa*deg2rad
        cospa = tt.cos(parad)
        sinpa = tt.sin(parad)

    #  Rotation
    urot = u*cospa - v*sinpa
    vrot = u*sinpa + v*cospa

    #  Exponent
    gamma = -2*np.pi*np.pi*(tt.square(urot*sigmin)+tt.square(vrot*sigmaj))

    return gamma

def egauss_camp(
        u1,v1,u2,v2,u3,v3,u4,v4,
        majsize=1.0,minsize=1.0,pa=0.0,angunit="mas"):
    logcamp = egauss_logcamp(u1,v1,u2,v2,u3,v3,u4,v4,majsize,minsize,pa,angunit)
    return tt.exp(logcamp)

def egauss_logcamp(
        u1,v1,u2,v2,u3,v3,u4,v4,
        majsize=1.0,minsize=1.0,pa=0.0,angunit="mas"):
    logcamp = egauss_lognamp(u1,v1,majsize,minsize,pa,angunit)
    logcamp+= egauss_lognamp(u2,v2,majsize,minsize,pa,angunit)
    logcamp-= egauss_lognamp(u3,v3,majsize,minsize,pa,angunit)
    logcamp-= egauss_lognamp(u4,v4,majsize,minsize,pa,angunit)
    return logcamp
