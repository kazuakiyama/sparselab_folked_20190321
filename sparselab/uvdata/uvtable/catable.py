#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
This module describes uv data table for closure amplitudes.
'''
__author__ = "Sparselab Developer Team"
# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
import copy
import itertools

import numpy as np
import pandas as pd
import theano.tensor as T

import matplotlib.pyplot as plt

# internal
from .uvtable import UVTable, UVSeries
from .tools import get_uvlist
from ... import fortlib


# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class CATable(UVTable):
    uvunit = "lambda"

    catable_columns = ["utc", "gsthour",
                       "freq", "stokesid", "ifid", "chid", "ch",
                       "u1", "v1", "w1", "uvdist1",
                       "u2", "v2", "w2", "uvdist2",
                       "u3", "v3", "w3", "uvdist3",
                       "u4", "v4", "w4", "uvdist4",
                       "uvdistmin", "uvdistmax", "uvdistave",
                       "st1", "st2", "st3", "st4",
                       #"st1name", "st2name", "st3name", "st4name",
                       "amp", "sigma", "logamp", "logsigma"]
    catable_types = [np.asarray, np.float64,
                     np.float64, np.int32, np.int32, np.int32, np.int32,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64,
                     np.int32, np.int32, np.int32, np.int32,
                     #np.asarray, np.asarray, np.asarray, np.asarray,
                     np.float64, np.float64, np.float64, np.float64]

    @property
    def _constructor(self):
        return CATable

    @property
    def _constructor_sliced(self):
        return CASeries

    def eval_image(self, imfits, mask=None, istokes=0, ifreq=0):
        #uvdata.BSTable object (storing model closure phase)
        model = self._call_fftlib(imfits=imfits,mask=mask,
                                  istokes=istokes, ifreq=ifreq)
        Ndata = model[1]
        camodel = model[0][2]
        catable = self.copy()
        catable["phase"] = np.zeros(Ndata)
        catable["logamp"] = camodel
        return catable
    
    def residual_image(self, imfits, mask=None, istokes=0, ifreq=0):
        #uvdata BSTable object (storing residual closure phase)
        model = self._call_fftlib(imfits=imfits,mask=mask,
                                  istokes=istokes, ifreq=ifreq)
        Ndata = model[1]
        resida = model[0][3]
        residtable = self.copy()
        residtable["logamp"] = resida
        residtable["phase"] = np.zeros(Ndata)
        return residtable

    def chisq_image(self, imfits, mask=None, istokes=0, ifreq=0):
        # calcurate chisqared and reduced chisqred.
        model = self._call_fftlib(imfits=imfits,mask=mask,
                                  istokes=istokes, ifreq=ifreq)
        chisq = model[0][0]
        Ndata = model[1]
        rchisq = chisq/Ndata

        return chisq,rchisq

    def _call_fftlib(self, imfits, mask, istokes=0, ifreq=0):
        # get initial images
        istokes = istokes
        ifreq = ifreq
        
        # size of images
        Iin = np.float64(imfits.data[istokes, ifreq])
        Nx = imfits.header["nx"]
        Ny = imfits.header["ny"]
        Nyx = Nx * Ny
        
        # pixel coordinates
        x, y = imfits.get_xygrid(twodim=True, angunit="rad")
        xidx = np.arange(Nx) + 1
        yidx = np.arange(Ny) + 1
        xidx, yidx = np.meshgrid(xidx, yidx)
        Nxref = imfits.header["nxref"]
        Nyref = imfits.header["nyref"]
        dx_rad = np.deg2rad(imfits.header["dx"])
        dy_rad = np.deg2rad(imfits.header["dy"])
        
        # apply the imaging area
        if mask is None:
            print("Imaging Window: Not Specified. We calcurate the image on all the pixels.")
            Iin = Iin.reshape(Nyx)
            x = x.reshape(Nyx)
            y = y.reshape(Nyx)
            xidx = xidx.reshape(Nyx)
            yidx = yidx.reshape(Nyx)
        else:
            print("Imaging Window: Specified. Images will be calcurated on specified pixels.")
            idx = np.where(mask)
            Iin = Iin[idx]
            x = x[idx]
            y = y[idx]
            xidx = xidx[idx]
            yidx = yidx[idx]
        
        # Closure Phase
        Ndata = 0
        catable = self.copy()
        ca = np.array(catable["logamp"], dtype=np.float64)
        varca = np.square(np.array(catable["logsigma"], dtype=np.float64))
        Ndata += len(ca)
        
        # get uv coordinates and uv indice
        u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = get_uvlist(
                fcvtable=None, amptable=None, bstable=None, catable=catable
                )

        # normalize u, v coordinates
        u *= 2*np.pi*dx_rad
        v *= 2*np.pi*dy_rad
    
        # run model_cp
        model = fortlib.fftlib.model_ca(
                # Images
                iin=np.float64(Iin),
                xidx=np.int32(xidx),
                yidx=np.int32(yidx),
                nxref=np.float64(Nxref),
                nyref=np.float64(Nyref),
                nx=np.int32(Nx),
                ny=np.int32(Ny),
                # UV coordinates,
                u=u,
                v=v,
                # Closure Phase
                uvidxca=np.int32(uvidxca),
                ca=np.float64(ca),
                varca=np.float64(varca)
                )
            
        return model,Ndata

    def eval_geomodel(self, geomodel, evalargs={}):
        '''
        Evaluate model values and output them to a new table

        Args:
            geomodel (geomodel.geomodel.GeoModel) object
        Returns:
            uvdata.VisTable object
        '''
        # create a table to be output
        outtable = copy.deepcopy(self)

        # u,v coordinates
        u1 = outtable.u1.values
        v1 = outtable.v1.values
        u2 = outtable.u2.values
        v2 = outtable.v2.values
        u3 = outtable.u3.values
        v3 = outtable.v3.values
        u4 = outtable.u4.values
        v4 = outtable.v4.values
        outtable["amp"] = geomodel.Camp(u1,v1,u2,v2,u3,v3,u4,v4).eval(**evalargs)
        outtable["logamp"] = geomodel.logCamp(u1,v1,u2,v2,u3,v3,u4,v4).eval(**evalargs)
        return outtable


    def residual_geomodel(self, geomodel, normed=True, doeval=False, evalargs={}):
        '''
        Calculate residuals of log closure amplitudes
        for an input geometric model

        Args:
            geomodel (geomodel.geomodel.GeoModel object):
                input model
            normed (boolean, default=True):
                if True, residuals will be normalized by 1 sigma error
            eval (boolean, default=False):
                if True, actual residual values will be calculated.
                Otherwise, resduals will be given as a theano graph.
        Returns:
            ndarray (if doeval=True) or theano object (otherwise)
        '''
        # u,v coordinates
        u1 = self.u1.values
        v1 = self.v1.values
        u2 = self.u2.values
        v2 = self.v2.values
        u3 = self.u3.values
        v3 = self.v3.values
        u4 = self.u4.values
        v4 = self.v4.values
        logCA = self.logamp.values
        logsigma = self.logsigma.values

        modlogCA = geomodel.logCamp(u1,v1,u2,v2,u3,v3,u4,v4)
        residual = logCA - modlogCA
        if normed:
            residual /= logsigma

        if doeval:
            return residual.eval(**evalargs)
        else:
            return residual


    #-------------------------------------------------------------------------
    # Plot Functions
    #-------------------------------------------------------------------------
    def uvplot(self, uvunit=None, conj=True,
               ls="none", marker=".", **plotargs):
        '''
        Plot uv-plot on the current axes.
        This method uses matplotlib.pyplot.plot().

        Args:
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.
          conj (boolean, default = True):
            if conj=True, it will plot complex conjugate components (i.e. (-u, -v)).
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot.
            Defaults are {'ls': "none", 'marker': "."}
        '''

        # Set Unit
        if uvunit is None:
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # Label
        unitlabel = self.get_unitlabel(uvunit)

        plotargs2 = copy.deepcopy(plotargs)
        plotargs2["label"] = ""

        # plotting
        plt.plot(self["u1"] * conv, self["v1"] * conv,
                 ls=ls, marker=marker, **plotargs)
        plt.plot(self["u2"] * conv, self["v2"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        plt.plot(self["u3"] * conv, self["v3"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        plt.plot(self["u4"] * conv, self["v4"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        if conj:
            plt.plot(-self["u1"] * conv, -self["v1"] * conv,
                     ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u2"] * conv, -self["v2"] * conv,
                     ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u3"] * conv, -self["v3"] * conv,
                     ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u4"] * conv, -self["v4"] * conv,
                     ls=ls, marker=marker, **plotargs2)
        plt.xlabel(r"Baseline Length $u$ (%s)" % (unitlabel))
        plt.ylabel(r"Baseline Length $v$ (%s)" % (unitlabel))

        ax = plt.gca()
        ax.set_aspect("equal")
        xlim = np.asarray(ax.get_xlim())
        ylim = np.asarray(ax.get_ylim())
        ax.set_xlim(-np.sort(-xlim))
        ax.set_ylim(np.sort(ylim))

    def radplot(self, uvdtype="ave", uvunit=None, errorbar=True, model=None,
                ls="none", marker=".", **plotargs):
        '''
        Plot log(closure amplitudes) as a function of baseline lengths
        on the current axes. This method uses matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbar().

        Args:
          uvdtype (str, default = "ave"):
            The type of the baseline length plotted along the horizontal axis.
              "max": maximum of four baselines (=self["uvdistmax"])
              "min": minimum of four baselines (=self["uvdistmin"])
              "ave": average of four baselines (=self["uvdistave"])
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.
          errorbar (boolean, default = True):
            If errorbar is True, it will plot data with errorbars using
            matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
            errorbars using matplotlib.pyplot.plot().

            If you plot model closure phases (i.e. model is not None),
            it will plot without errobars regardless of this parameter.
          model (dict-like such as pd.DataFrame, pd.Series, default is None):
            Model data sets. Model closure amplitudes must be given by model["camod"].
            Otherwise, it will plot closure amplitudes in the table (i.e. self["logamp"]).
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        # Set Unit
        if uvunit is None:
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # uvdistance
        if uvdtype.lower().find("ave") * uvdtype.lower().find("mean") == 0:
            uvdist = self["uvdistave"] * conv
            head = "Mean"
        elif uvdtype.lower().find("min") == 0:
            uvdist = self["uvdistmin"] * conv
            head = "Minimum"
        elif uvdtype.lower().find("max") == 0:
            uvdist = self["uvdistmax"] * conv
            head = "Maximum"
        else:
            print("[Error] uvdtype=%s is not available." % (uvdtype))
            return -1

        # Label
        unitlabel = self.get_unitlabel(uvunit)

        # plotting data
        if model is not None:
            plt.plot(uvdist, model["camod"], ls=ls, marker=marker, **plotargs)
        elif errorbar:
            plt.errorbar(uvdist, self["logamp"], self["logsigma"],
                         ls=ls, marker=marker, **plotargs)
        else:
            plt.plot(uvdist, self["logamp"], ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"%s Baseline Length (%s)" % (head,unitlabel))
        plt.ylabel(r"Log Closure Amplitude")
        plt.xlim(0,)


class CASeries(UVSeries):

    @property
    def _constructor(self):
        return CASeries

    @property
    def _constructor_expanddim(self):
        return CATable


# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def read_catable(filename, uvunit=None, **args):
    '''
    This fuction loads uvdata.CATable from an input csv file using pd.read_csv().

    Args:
      filename:
        str, pathlib.Path, py._path.local.LocalPath or any object with a read()
        method (such as a file handle or StringIO)
      uvunit (str, default is None):
        units of uvdistance for plotting. If uvunit is None, uvunit will be
        inferred from the maximum baseline length. Availables are ["l[ambda]",
        "kl[ambda]", "ml[ambda]", "gl[ambda]", "m", "km"].

    Returns:
      uvdata.CATable object
    '''
    table = CATable(pd.read_csv(filename, **args))

    maxuvd = np.max(table["uvdistmax"])

    if uvunit is None:
        if maxuvd < 1e3:
            table.uvunit = "lambda"
        elif maxuvd < 1e6:
            table.uvunit = "klambda"
        elif maxuvd < 1e9:
            table.uvunit = "mlambda"
        else:
            table.uvunit = "glambda"
    else:
        table.uvunit = uvunit

    return table
