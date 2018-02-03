#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a submodule of sparselab.modelfit, containing functions to
calculate visibilities and images of some geometric models.

Here, we note that the whole design of this module is imspired by
Lindy Blackburn's python module 'modmod', and
we would like to thank Lindy to share his idea.
'''
import numpy as np
import astropy.constants as ac
import sympy as S
from .. import util

# initial variables
x_S, y_S = S.symbols("x y", real=True)
u_S, v_S = S.symbols("u v", real=True)
u1_S, v1_S = S.symbols("u1 v1", real=True)
u2_S, v2_S = S.symbols("u2 v2", real=True)
u3_S, v3_S = S.symbols("u3 v3", real=True)
u4_S, v4_S = S.symbols("u4 v4", real=True)

def A_cos(x):
    if hasattr(x, "applyfunc"):
        return x.applyfunc(S.cos)
    else:
        return S.cos(x)

def A_sin(x):
    if hasattr(x, "applyfunc"):
        return x.applyfunc(S.sin)
    else:
        return S.sin(x)

def A_exp(x):
    if hasattr(x, "applyfunc"):
        return x.applyfunc(S.exp)
    else:
        return S.exp(x)

def A_sqrt(x):
    if hasattr(x, "applyfunc"):
        return x.applyfunc(S.sqrt)
    else:
        return S.sqrt(x)

def A_log(x):
    if hasattr(x, "applyfunc"):
        return x.applyfunc(S.log)
    else:
        return S.log(x)

def A_square(x):
    if hasattr(x, "applyfunc"):
        return x.applyfunc(lambda x: x*x)
    else:
        return x*x

def A_mul(x,y):
    if hasattr(x, "applyfunc") and hasattr(y, "applyfunc"):
        return S.Array([xi * yi for xi, yi in zip(x, y)])
    else:
        return x*y

def A_div(x,y):
    if hasattr(x, "applyfunc") and hasattr(y, "applyfunc"):
        return S.Array([xi/yi for xi, yi in zip(x, y)])
    else:
        return x/y

class GeoModel(object):
    def __init__(self, Vreal=None, Vimag=None, I=None):
        if Vreal is None:
            self.Vreal = lambda u=u_S, v=v_S: 0
        else:
            self.Vreal = Vreal

        if Vimag is None:
            self.Vimag = lambda u=u_S, v=v_S: 0
        else:
            self.Vimag = Vimag

        if I is None:
            self.I = lambda x=x_S, y=y_S: 0
        else:
            self.I = I

    def __add__(self, other):
        if type(self) == type(other):
            Vreal = lambda u=u_S, v=v_S: self.Vreal(u,v) + other.Vreal(u,v)
            Vimag = lambda u=u_S, v=v_S: self.Vimag(u,v) + other.Vimag(u,v)
            I = lambda x=x_S, y=y_S: self.I(x,y) + other.I(x,y)
            return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)
        else:
            raise ValueError("Addition can be calculated only between the same type of objects")

    def __sub__(self, other):
        if type(self) == type(other):
            Vreal = lambda u=u_S, v=v_S: self.Vreal(u,v) - other.Vreal(u,v)
            Vimag = lambda u=u_S, v=v_S: self.Vimag(u,v) - other.Vimag(u,v)
            I = lambda x=x_S, y=y_S: self.I(x,y) - other.I(x,y)
            return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)
        else:
            raise ValueError("Subtraction can be calculated only between the same type of objects")


    def __mul__(self, other):
        if np.isscalar(other):
            Vreal = lambda u=u_S, v=v_S: A_mul(self.Vreal(u,v), other)
            Vimag = lambda u=u_S, v=v_S: A_mul(self.Vimag(u,v), other)
            I = lambda x=x_S, y=y_S: A_mul(self.I(x,y), other)
            return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)
        else:
            raise ValueError("Multiplication can be calculated only for scalar-like objects")

    def __truediv__(self, other):
        if np.isscalar(other):
            Vreal = lambda u=u_S, v=v_S: A_div(self.Vreal(u,v), other)
            Vimag = lambda u=u_S, v=v_S: A_div(self.Vimag(u,v), other)
            I = lambda x=x_S, y=y_S: A_div(self.I(x,y), other)
            return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)
        else:
            raise ValueError("Division can be calculated only for scalar-like objects")


    def shift(self, deltax=0., deltay=0., angunit="mas"):
        angunit2rad = angconv(angunit, "rad")
        dx = deltax * angunit2rad
        dy = deltay * angunit2rad
        phase = lambda u=u_S, v=v_S: 2*S.S.Pi*(A_mul(u,dx)+A_mul(v,dy))
        cosp  = lambda u=u_S, v=v_S: A_cos(phase(u,v))
        sinp  = lambda u=u_S, v=v_S: A_sin(phase(u,v))
        func1 = lambda u=u_S, v=v_S: A_mul(self.Vreal(u,v),cosp(u,v))
        func2 = lambda u=u_S, v=v_S: A_mul(self.Vimag(u,v),sinp(u,v))
        func3 = lambda u=u_S, v=v_S: A_mul(self.Vreal(u,v),sinp(u,v))
        func4 = lambda u=u_S, v=v_S: A_mul(self.Vimag(u,v),cosp(u,v))
        Vreal = lambda u=u_S, v=v_S: func1(u,v) - func2(u,v)
        Vimag = lambda u=u_S, v=v_S: func3(u,v) + func4(u,v)
        I = lambda x=x_S, y=y_S: self.I(x-dx,y-dy)
        return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)


    def rotate(self, deltaPA=0., deg=True):
        if deg:
            dPA = deltaPA * S.S.Pi / 180
        else:
            dPA = deltaPA
        cosdpa = A_cos(dPA)
        sindpa = A_sin(dPA)
        x1 = lambda x=x_S, y=y_S: A_mul(x, cosdpa) - A_mul(y, sindpa)
        y1 = lambda x=x_S, y=y_S: A_mul(x, sindpa) + A_mul(y, cosdpa)
        Vreal = lambda u=u_S, v=v_S: self.Vreal(x1(u,v),y1(u,v))
        Vimag = lambda u=u_S, v=v_S: self.Vimag(x1(u,v),y1(u,v))
        I = lambda x=x_S, y=y_S: self.I(x1(x,y),y1(x,y))
        return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)


    def scale(self, hx=1., hy=None):
        if hy is None:
            hy = hx
        Vreal = lambda u=u_S, v=v_S: self.Vreal(A_mul(u,hx), A_mul(v,hy))
        Vimag = lambda u=u_S, v=v_S: self.Vimag(A_mul(u,hx), A_mul(v,hy))
        I = lambda x=x_S, y=y_S: self.I(A_div(u,hx), A_div(v,hy))/hx/hy
        return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)


    # Full complex visibilities
    def Vamp(self, u=u_S, v=v_S):
        '''
        Return theano symbolic represenation of the visibility amplitude

        Args:
            u, v: uv-coordinates
        Return:
            amplitude
        '''
        Vre = self.Vreal(u,v)
        Vim = self.Vimag(u,v)
        return A_sqrt(A_square(Vre) + A_square(Vim))


    def logVamp(self, u=u_S, v=v_S):
        '''
        Return theano symbolic represenation of the visibility amplitude

        Args:
            u, v: uv-coordinates
        Return:
            amplitude
        '''
        return A_log(self.Vamp(u,v))


    def Vphase(self, u=u_S, v=v_S):
        '''
        Return theano symbolic represenation of the visibility phase

        Args:
            u, v: uv-coordinates
        Return:
            phase in radian
        '''
        return S.arctan2(self.Vimag(u,v), self.Vreal(u,v))


    # Bi-spectrum
    def Bre(self, u1=u1_S, v1=v1_S, u2=u2_S, v2=v2_S, u3=u3_S, v3=v3_S):
        '''
        Return theano symbolic represenation of the real part of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            real part of the bi-spectrum
        '''
        Vre1 = self.Vreal(u1,v1)
        Vim1 = self.Vimag(u1,v1)
        Vre2 = self.Vreal(u2,v2)
        Vim2 = self.Vimag(u2,v2)
        Vre3 = self.Vreal(u3,v3)
        Vim3 = self.Vimag(u3,v3)
        term1 = A_mul(A_mul(-Vim1,Vim2),Vre3)
        term2 = A_mul(A_mul(-Vim1,Vim3),Vre2)
        term3 = A_mul(A_mul(-Vim2,Vim3),Vre1)
        term4 = A_mul(A_mul(Vre1,Vre2),Vre3)
        #Bre =  -Vim1*Vim2*Vre3 - Vim1*Vim3*Vre2 - Vim2*Vim3*Vre1 + Vre1*Vre2*Vre3
        Bre = term1 + term2 + term3 + term4
        return Bre


    def Bim(self, u1=u1_S, v1=v1_S, u2=u2_S, v2=v2_S, u3=u3_S, v3=v3_S):
        '''
        Return theano symbolic represenation of the imaginary part of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            imaginary part of the bi-spectrum
        '''
        Vre1 = self.Vreal(u1,v1)
        Vim1 = self.Vimag(u1,v1)
        Vre2 = self.Vreal(u2,v2)
        Vim2 = self.Vimag(u2,v2)
        Vre3 = self.Vreal(u3,v3)
        Vim3 = self.Vimag(u3,v3)
        term1 = A_mul(A_mul(-Vim1,Vim2),Vim3)
        term2 = A_mul(A_mul(Vim1,Vre2),Vre3)
        term3 = A_mul(A_mul(Vim2,Vre1),Vre3)
        term4 = A_mul(A_mul(Vim3,Vre1),Vre2)
        #Bim = -Vim1*Vim2*Vim3 + Vim1*Vre2*Vre3 + Vim2*Vre1*Vre3 + Vim3*Vre1*Vre2
        Bim = term1 + term2 + term3 + term4
        return Bim


    def Bamp(self, u1=u1_S, v1=v1_S, u2=u2_S, v2=v2_S, u3=u3_S, v3=v3_S):
        '''
        Return theano symbolic represenation of the amplitude of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            amplitude of the bi-spectrum
        '''
        Bre = self.Bre(u1, v1, u2, v2, u3, v3)
        Bim = self.Bim(u1, v1, u2, v2, u3, v3)
        Bamp = A_sqrt(A_square(Bre)+A_square(Bim))
        return Bamp


    def Bphase(self, u1=u1_S, v1=v1_S, u2=u2_S, v2=v2_S, u3=u3_S, v3=v3_S):
        '''
        Return theano symbolic represenation of the phase of the Bi-spectrum.
        if given uv-coodinates are closed, this will be the closure phase.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            phase of the bi-spectrum
        '''
        Bre = self.Bre(u1, v1, u2, v2, u3, v3)
        Bim = self.Bim(u1, v1, u2, v2, u3, v3)
        Bphase = S.arctan2(Bim, Bre)
        return Bphase


    # Closure Amplitudes
    def Camp(self, u1=u1_S, v1=v1_S, u2=u2_S, v2=v2_S, u3=u3_S, v3=v3_S, u4=u4_S, v4=v4_S):
        '''
        Return theano symbolic represenation of the phase of the Bi-spectrum.
        if given uv-coodinates are closed, this will be the closure phase.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            phase of the bi-spectrum
        '''
        Vamp1 = self.Vamp(u1, v1)
        Vamp2 = self.Vamp(u2, v2)
        Vamp3 = self.Vamp(u3, v3)
        Vamp4 = self.Vamp(u4, v4)
        return A_div(A_div(A_mul(Vamp1,Vamp2),Vamp3),Vamp4)


    def logCamp(self, u1=u1_S, v1=v1_S, u2=u2_S, v2=v2_S, u3=u3_S, v3=v3_S, u4=u4_S, v4=v4_S):
        '''
        Return theano symbolic represenation of the phase of the Bi-spectrum.
        if given uv-coodinates are closed, this will be the closure phase.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            phase of the bi-spectrum
        '''
        logVamp1 = self.logVamp(u1, v1)
        logVamp2 = self.logVamp(u2, v2)
        logVamp3 = self.logVamp(u3, v3)
        logVamp4 = self.logVamp(u4, v4)
        return logVamp1+logVamp2-logVamp3-logVamp4


#-------------------------------------------------------------------------------
# Some calculations
#-------------------------------------------------------------------------------
def dphase(phase1, phase2):
    dphase = phase2 - phase1
    return S.arctan2(A_sin(dphase), A_cos(dphase))

def angconv(unit1="deg", unit2="deg"):
    '''
    return a conversion factor from unit1 to unit2
    Available angular units are uas, mas, asec or arcsec, amin or arcmin and degree.
    '''
    if unit1 == unit2:
        return 1

    # Convert from unit1 to "arcsec"
    if unit1 == "deg":
        conv = 3600
    elif unit1 == "rad":
        conv = 180 * 3600 / S.S.Pi
    elif unit1 == "arcmin" or unit1 == "amin":
        conv = 60
    elif unit1 == "arcsec" or unit1 == "asec":
        conv = 1
    elif unit1 == "mas":
        conv = 1e-3
    elif unit1 == "uas":
        conv = 1e-6
    else:
        print("Error: unit1=%s is not supported" % (unit1))
        return -1

    # Convert from "arcsec" to unit2
    if unit2 == "deg":
        conv /= 3600
    elif unit2 == "rad":
        conv /= (180 * 3600 / S.S.Pi)
    elif unit2 == "arcmin" or unit2 == "amin":
        conv /= 60
    elif unit2 == "arcsec" or unit2 == "asec":
        pass
    elif unit2 == "mas":
        conv *= 1000
    elif unit2 == "uas":
        conv *= 1000000
    else:
        print("Error: unit2=%s is not supported" % (unit2))
        return -1

    return conv

#-------------------------------------------------------------------------------
# Phase for symmetric sources
#-------------------------------------------------------------------------------
def phaseshift(u,v,x0=0,y0=0,angunit="mas"):
    '''
    Phase of a symmetric object (Gaussians, Point Sources, etc).
    This function also can be used to compute a phase shift due to positional shift.
    Args:
        u, v (mandatory): uv coordinates in lambda
        x0=0, y0=0: position of centorid or positional shift in angunit.
        angunit="mas": angular unit of x0, y0 (uas, mas, asec, amin, deg, rad)
    return:
        phase in rad
    '''
    return 2*S.S.Pi*(A_mul(u,x0)+A_mul(v,y0))*angconv(angunit, "rad")


#-------------------------------------------------------------------------------
# Point Source for symmetric sources
#-------------------------------------------------------------------------------
def Gaussian(x0=0,y0=0,totalflux=1,majsize=1,minsize=None,pa=0,angunit="mas"):
    '''
    Create modelfit.geomodel.GeoModel Object for the specified Gaussian
    Args:
        x0=0.0, y0=0.0: position of the centorid
        totalflux=1.0: total flux of the Gaussian
        majsize, minsize: Major/Minor-axis FWHM size of the Gaussian
        pa: Position Angle of the Gaussian in degree
        angunit="mas": angular unit of x0, y0, majsize, minsize (uas, mas, asec, amin, deg, rad)
    Returns:
        modelfit.geomodel.GeoModel Object for the specified Gaussian
    '''
    if minsize is None:
        minsize = majsize

    # define a Gaussian with F=1 jy, size = 1 (angunit)
    sigma = angconv(angunit, "rad")/np.sqrt(8*np.log(2))
    Vreal = lambda u=u_S, v=v_S: A_exp(-2*S.S.Pi*S.S.Pi*(A_square(u)+A_square(v))*sigma*sigma)
    I = lambda x=x_S, y=y_S: 1/2/S.S.Pi/sigma/sigma*A_exp(-(A_square(x)+A_square(y))/2/sigma/sigma)
    output = GeoModel(Vreal=Vreal, I=I)

    # transform Gaussian, so that it will be elliptical Gaussian
    output = output * totalflux
    output = output.scale(hx=minsize, hy=majsize)
    output = output.rotate(deltaPA=pa, deg=True)
    output = output.shift(deltax=x0, deltay=y0, angunit=angunit)
    return output
