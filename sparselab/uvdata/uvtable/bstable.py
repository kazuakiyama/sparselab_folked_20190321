#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
This module describes uv data table for bi-spectrum.
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


# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class BSTable(UVTable):
    '''
    This class is for handling two dimentional tables of Bi-spectrua of
    visibilities. The class inherits pandas.DataFrame class, so you can use this
    class like pandas.DataFrame. The class also has additional methods to edit,
    visualize and convert data.
    '''
    uvunit = "lambda"

    bstable_columns = ["utc", "gsthour",
                       "freq", "stokesid", "chid", "ifid", "ch",
                       "u12", "v12", "w12", "uvdist12",
                       "u23", "v23", "w23", "uvdist23",
                       "u31", "v31", "w31", "uvdist31",
                       "uvdistmin", "uvdistmax", "uvdistave",
                       #"st1name","st2name","st3name",
                       "st1", "st2", "st3", "ch",
                       "amp", "phase", "sigma"]
    bstable_types = [np.asarray, np.float64,
                     np.float64, np.int32, np.int32, np.int32, np.int32,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64,
                     #np.asarray, np.asarray, np.asarray,
                     np.int32, np.int32, np.int32, np.int32,
                     np.float64, np.float64, np.float64]

    @property
    def _constructor(self):
        return BSTable

    @property
    def _constructor_sliced(self):
        return BSSeries

    def eval_geomodel(self, geomodel, evalargs={}):
        '''
        Evaluate model values and output them to a new table

        Args:
            geomodel (modelfit.geomodel.GeoModel) object
        Returns:
            uvdata.BSTable object
        '''
        # create a table to be output
        outtable = copy.deepcopy(self)

        # u,v coordinates
        u1 = outtable.u12.values
        v1 = outtable.v12.values
        u2 = outtable.u23.values
        v2 = outtable.v23.values
        u3 = outtable.u31.values
        v3 = outtable.v31.values
        outtable["amp"] = geomodel.Bamp(u1,v1,u2,v2,u3,v3).eval(**evalargs)
        outtable["phase"] = geomodel.Bphase(u1,v1,u2,v2,u3,v3).eval(**evalargs) * 180./np.pi
        return outtable

    def residual_geomodel(self, geomodel, normed=True, doeval=False, evalargs={}):
        '''
        Calculate residuals of closure phases in radian
        for an input geometric model

        Args:
            geomodel (modelfit.geomodel.GeoModel object):
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
        u1 = self.u12.values
        v1 = self.v12.values
        u2 = self.u23.values
        v2 = self.v23.values
        u3 = self.u31.values
        v3 = self.v31.values
        CP = self.phase.values * np.pi / 180.
        sigma = self.sigma.values/self.amp.values

        modCP = geomodel.Bphase(u1,v1,u2,v2,u3,v3)
        residual = CP - modCP
        residual = T.arctan2(T.sin(residual), T.cos(residual))

        if normed:
            residual /= sigma

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
        plt.plot(self["u12"] * conv, self["v12"] *
                 conv, ls=ls, marker=marker, **plotargs)
        plt.plot(self["u23"] * conv, self["v23"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        plt.plot(self["u31"] * conv, self["v31"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        if conj:
            plt.plot(-self["u12"] * conv, -self["v12"] *
                     conv, ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u23"] * conv, -self["v23"] *
                     conv, ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u31"] * conv, -self["v31"] *
                     conv, ls=ls, marker=marker, **plotargs2)
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
        Plot closure phases as a function of baseline lengths on the current axes.
        This method uses matplotlib.pyplot.plot() or matplotlib.pyplot.errorbar().

        Args:
          uvdtype (str, default = "ave"):
            The type of the baseline length plotted along the horizontal axis.
              "max": maximum of three baselines (=self["uvdistmax"])
              "min": minimum of three baselines (=self["uvdistmin"])
              "ave": average of three baselines (=self["uvdistave"])
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
            Model data sets. Model closure phases must be given by model["cpmod"].
            Otherwise, it will plot closure phases in the table (i.e. self["phase"]).
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
            plt.plot(uvdist, model["cpmod"],
                     ls=ls, marker=marker, **plotargs)
        elif errorbar:
            pherr = np.rad2deg(self["sigma"] / self["amp"])
            plt.errorbar(uvdist, self["phase"], pherr,
                         ls=ls, marker=marker, **plotargs)
        else:
            plt.plot(uvdist, self["phase"],
                     ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"%s Baseline Length (%s)" % (head, unitlabel))
        plt.ylabel(r"Closure Phase ($^\circ$)")
        plt.xlim(0,)
        plt.ylim(-180, 180)


class BSSeries(UVSeries):

    @property
    def _constructor(self):
        return BSSeries

    @property
    def _constructor_expanddim(self):
        return BSTable


# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def read_bstable(filename, uvunit=None, **args):
    '''
    This fuction loads uvdata.BSTable from an input csv file using pd.read_csv().

    Args:
      filename:
        str, pathlib.Path, py._path.local.LocalPath or any object with a read()
        method (such as a file handle or StringIO)
      uvunit (str, default is None):
        units of uvdistance for plotting. If uvunit is None, uvunit will be
        inferred from the maximum baseline length. Availables are ["l[ambda]",
        "kl[ambda]", "ml[ambda]", "gl[ambda]", "m", "km"].

    Returns:
      uvdata.BSTable object
    '''
    table = BSTable(pd.read_csv(filename, **args))

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
