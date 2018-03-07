#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
This module describes gain calibration table for full complex visibilities.
'''
__author__ = "Sparselab Developer Team"
# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
import copy
import itertools
import tqdm

# numerical packages
import numpy as np
import pandas as pd
from scipy import optimize
#import matplotlib.pyplot as plt
import astropy.time as at

# internal
#from ..uvtable   import UVTable, UVSeries
#from ..gvistable import GVisTable, GVisSeries
#from ..catable   import CATable, CASeries
#from ..bstable   import BSTable, BSSeries
#from ... import imdata


# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class CLTable(pd.DataFrame):
    '''
    This is a class describing gain calibrations.
    '''
    @property
    def _constructor(self):
        return CLTable

    @property
    def _constructor_sliced(self):
        return CLSeries

    def __init__(self, uvfits):
        '''
        '''
        # get UTC
        utc = np.datetime_as_string(uvfits.visdata.coord["utc"])
        utc = sorted(set(utc)) # remove duplicated
        self.utc = at.Time(utc, scale="utc")

        #self.gain = np.zeros([Ntime,Nif,Nch,Nstokes,Nant,3])

class CLSeries(pd.Series):
    @property
    def _constructor(self):
        return CLSeries

    @property
    def _constructor_expanddim(self):
        return CLTable
