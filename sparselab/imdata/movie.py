#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a sub-module of sparselab handling dynamical imaging.
'''
__author__ = "Sparselab Developer Team"
#-------------------------------------------------------------------------
# Modules
#-------------------------------------------------------------------------
# standard modules
import os
import copy
import datetime as dt

# numerical packages
import numpy as np
import pandas as pd
import scipy.ndimage as sn
import astropy.time as at
import astropy.coordinates as coord
import astropy.io.fits as pyfits
from astropy.convolution import convolve_fft

# matplotlib
import matplotlib.pyplot as plt

# internal
from .. import fortlib, util

#-------------------------------------------------------------------------
# IMAGEFITS (Manupulating FITS FILES)
#-------------------------------------------------------------------------
class MOVIE(object):
    def __init__(self, tstart='2000-01-01T00:00:00',
                 tint=60, tintunit="sec", Nf=1, initimage=None,
                 dtable=None, **args):
        '''
        Args:
            tstart (datetime):
                start time
            tint (float):
                constant time span of each frame (sec)
            tintunit (string):
                unit of time difference (sec, min, hrs, day)
            Nf (integer):
                number of frames
            initimage (imdata.IMFITS object):
                initial image
        Returns:
            imdata.MOVIE object
        '''
        # formatting the input tstart
        self.tstart = at.Time(tstart)
        #self.tstart = at.Time(np.datetime_as_string(tstart))
        # formatting the input tint
        if tintunit == "sec":
            self.tint = at.TimeDelta(tint, format='sec')
        elif tintunit == "min":
            self.tint = at.TimeDelta(tint*60, format='sec')
        elif tintunit == "hrs":
            self.tint = at.TimeDelta(tint*3600, format='sec')
        elif tintunit == "day":
            self.tint = at.TimeDelta(tint*3600*24, format='sec')
        # assigning the input Nf
        self.Nf = Nf
        # dataframe tables
        self.dtable = dtable

    def timetable(self):
        tmtable = pd.DataFrame()
        tmtable["frame"] = np.zeros(self.Nf, dtype='int32')
        tmtable["utc"] = np.zeros(self.Nf)
        tmtable["gsthour"] = np.zeros(self.Nf)
        tmtable["tint(sec)"] = np.zeros(self.Nf)
        for i in np.arange(self.Nf):
            tmtable.loc[i, "frame"] = i
            centime = self.tstart + (self.tint/2) + self.tint*i
            utctime = centime.datetime
            gsthour = centime.sidereal_time("apparent", "greenwich").hour
            tmtable.loc[i, "utc"] = utctime
            tmtable.loc[i, "gsthour"] = gsthour
            tmtable.loc[i, "tint(sec)"] = self.tint
        return tmtable

    def frmidx(self):
        if (self.dtable is None):
            print("DataFrame table is not given.")
            return -1
        # time of input table which DataFrame
        #attime = at.Time(np.datetime_as_string(self.dtable["utc"]))
        attime = np.asarray(self.dtable["utc"], np.str)
        attime = at.Time(attime)
        utctime = attime.datetime
        # call timetable
        tmtable = self.timetable()
        idx = tmtable["frame"]
        tmframe = tmtable["utc"]
        # assigning
        self.dtable["frmidx"] = np.zeros(len(self.dtable), dtype='int32')
        for i in range(len(utctime)):
            for j in range(len(idx)-1):
                if (utctime[i] >= tmframe[j]) and (utctime[i] < tmframe[j+1]):
                    self.dtable.loc[i, "frmidx"] = idx[j]
            if utctime[i] > tmframe[-1]:
                self.dtable.loc[i, "frmidx"] = idx[-1] + 1
        # concatenate the indexed tables


    def initimlist(self):
        pass
