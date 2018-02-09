#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
This module describes uv data table for uv-gridded visibilities.
'''
__author__ = "Sparselab Developer Team"
# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
import copy
import tqdm

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

# internal
from .uvtable import UVTable, UVSeries
from sparselab import imdata


# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class GVisTable(UVTable):
    '''
    This class is for handling two dimentional tables of full complex visibilities
    and amplitudes. The class inherits pandas.DataFrame class, so you can use this
    class like pandas.DataFrame. The class also has additional methods to edit,
    visualize and convert data.
    '''
    @property
    def _constructor(self):
        return GVisTable

    @property
    def _constructor_sliced(self):
        return GVisSeries

    def recalc_uvdist(self):
        '''
        Re-calculate the baseline length from self["u"] and self["v"].
        '''
        self["uvdist"] = np.sqrt(self["u"] * self["u"] + self["v"] * self["v"])

    def fit_beam(self, angunit="mas", errweight=0., ftsign=+1):
        '''
        This function estimates the synthesized beam size at natural weighting.

        Args:
          angunit (string):
            Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
          errweight (float; experimental):
            index for errer weighting
          ftsign (integer):
            a sign for fourier matrix
        '''
        # infer the parameters of clean beam
        parm0 = _calc_bparms(self)

        # generate a small image 4 times larger than the expected major axis
        # size of the beam
        fitsdata = imdata.IMFITS(fov=[parm0[0], -parm0[0], -parm0[0], parm0[0]],
                                 nx=20, ny=20, angunit="deg")

        # create output fits
        dbfitsdata, dbflux = _calc_dbeam(
            fitsdata, self, errweight=errweight, ftsign=ftsign)

        X, Y = fitsdata.get_xygrid(angunit="deg", twodim=True)
        dbeam = dbfitsdata.data[0, 0]
        dbeam /= np.max(dbeam)

        parms = optimize.leastsq(_fit_chisq, parm0, args=(X, Y, dbeam))

        (maja, mina, PA) = parms[0]
        maja = np.abs(maja)
        mina = np.abs(mina)

        # adjust these parameters
        if maja < mina:
            maja, mina = mina, maja
            PA += 90
        while np.abs(PA) > 90:
            if PA > 90:
                PA -= 90
            elif PA < -90:
                PA += 90

        # return as parameters of gauss_convolve
        factor = fitsdata.angconv("deg", angunit)
        cb_parms = ({'majsize': maja * factor, 'minsize': mina *
                     factor, 'angunit': angunit, 'pa': PA})
        return cb_parms

    def fftshift(self, fitsdata, fgfov=1):
        '''
        Arguments:
          vistable (pandas.Dataframe object):
            input visibility table

          fitsdata (imdata.IMFITS object):
            input imdata.IMFITS object

          fgfov (int)
            a number of gridded FOV/original FOV

        Output: pandas.Dataframe object
        '''
        # Copy vistable for edit
        vistable = self.copy()

        # Calculate du and dv
        Nupix = fitsdata.header["nx"] * fgfov
        Nvpix = fitsdata.header["ny"] * fgfov
        du = 1 / np.radians(np.abs(fitsdata.header["dx"]) * Nupix)
        dv = 1 / np.radians(fitsdata.header["dy"] * Nvpix)

        # Shift vistable
        vistable.loc[vistable["vgidx"] < 0, "vgidx"] += Nvpix

        # Create new list for shift
        outlist = {
            "ugidx": [],
            "vgidx": [],
            "u": [],
            "v": [],
            "orgu": [],
            "orgv": [],
            "uvdist": [],
            "amp": [],
            "phase": [],
            "weight": [],
            "sigma": []
        }

        # Save shifted data
        outlist["ugidx"] = vistable["ugidx"]
        outlist["vgidx"] = vistable["vgidx"]
        outlist["u"] = vistable["ugidx"] * du
        outlist["v"] = vistable["vgidx"] * dv
        outlist["orgu"] = self["u"]
        outlist["orgv"] = self["v"]
        outlist["uvdist"] = np.sqrt(self["u"]*self["u"]+self["v"]*self["v"])
        outlist["amp"] = self["amp"]
        outlist["phase"] = self["phase"]
        outlist["weight"] = self["weight"]
        outlist["sigma"] = self["sigma"]

        # Output as pandas.DataFrame
        outtable = GVisTable(outlist)
        return outtable

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

        # plotting
        plt.plot(self["u"] * conv, self["v"] * conv,
                 ls=ls, marker=marker, **plotargs)
        if conj:
            plotargs2 = copy.deepcopy(plotargs)
            plotargs2["label"] = ""
            plt.plot(-self["u"] * conv, -self["v"] * conv,
                     ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"Baseline Length $u$ (%s)" % (unitlabel))
        plt.ylabel(r"Baseline Length $v$ (%s)" % (unitlabel))

        ax = plt.gca()
        ax.set_aspect("equal")
        xlim = np.asarray(ax.get_xlim())
        ylim = np.asarray(ax.get_ylim())
        ax.set_xlim(-np.sort(-xlim))
        ax.set_ylim(np.sort(ylim))

    def radplot_amp(self, uvunit=None, errorbar=True, model=None, modeltype="amp",
                    ls="none", marker=".", **plotargs):
        '''
        Plot visibility amplitudes as a function of baseline lengths
        on the current axes. This method uses matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbar().

        Args:
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
            Model data sets. Model amplitudes must be given by model["fcvampmod"]
            for full complex visibilities (modeltype="fcv") or model["ampmod"]
            for visibility amplitudes (modeltype="amp").
            Otherwise, it will plot amplitudes in the table (i.e. self["amp"]).

          modeltype (string, default = "amp"):
            The type of models. If you would plot model amplitudes, set modeltype="amp".
            Else if you would plot model full complex visibilities, set modeltype="fcv".

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

        # Label
        unitlabel = self.get_unitlabel(uvunit)

        # plotting data
        if model is not None:
            if modeltype.lower().find("amp") == 0:
                plt.plot(self["uvdist"] * conv, model["ampmod"],
                         ls=ls, marker=marker, **plotargs)
            elif modeltype.lower().find("fcv") == 0:
                plt.plot(self["uvdist"] * conv, model["fcvampmod"],
                         ls=ls, marker=marker, **plotargs)
        elif errorbar:
            plt.errorbar(self["uvdist"] * conv, self["amp"], self["sigma"],
                         ls=ls, marker=marker, **plotargs)
        else:
            plt.plot(self["uvdist"] * conv, self["amp"],
                     ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
        plt.ylabel(r"Visibility Amplitude (Jy)")
        plt.xlim(0,)
        plt.ylim(0,)

    def radplot_phase(self, uvunit=None, errorbar=True, model=None,
                      ls="none", marker=".", **plotargs):
        '''
        Plot visibility phases as a function of baseline lengths
        on the current axes. This method uses matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbar().

        Args:
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
            Model data sets. Model phases must be given by model["fcvphamod"].
            Otherwise, it will plot amplitudes in the table (i.e. self["phase"]).

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

        # Label
        unitlabel = self.get_unitlabel(uvunit)

        # plotting data
        if model is not None:
            plt.plot(self["uvdist"] * conv, model["fcvphamod"],
                     ls=ls, marker=marker, **plotargs)
        elif errorbar:
            pherr = np.rad2deg(self["sigma"] / self["amp"])
            plt.errorbar(self["uvdist"] * conv, self["phase"], pherr,
                         ls=ls, marker=marker, **plotargs)
        else:
            plt.plot(self["uvdist"] * conv, self["phase"],
                     ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
        plt.ylabel(r"Visibility Phase ($^\circ$)")
        plt.xlim(0,)
        plt.ylim(-180, 180)

class GVisSeries(UVSeries):

    @property
    def _constructor(self):
        return GVisSeries

    @property
    def _constructor_expanddim(self):
        return GVisTable
