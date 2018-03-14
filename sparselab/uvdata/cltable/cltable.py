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
class CLTable(object):
    '''
    This is a class describing gain calibrations.
    '''

    def __init__(self, uvfits):
        '''
        '''
        self.gaintabs = {} #空dictionaryの作成

        # サブアレイの考慮
        subarrids = uvfits.subarrays.keys()
        for subarrid in subarrids:

            self.gaintabs[subarrid] = {} #空dictionaryの作成

            # get UTC
            utc = np.datetime_as_string(uvfits.visdata.coord["utc"])
            utc = sorted(set(utc)) # remove duplicated
            #print(utc)

            utc = at.Time(utc, scale="utc")

            # dictionaryの"utc"の名前にutcの値を格納
            self.gaintabs[subarrid]["utc"]=utc
            #print(self.utc)

            #def gain(self,uvfits):
            # gainの配列の初期化
            Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp=uvfits.visdata.data.shape
            Ntime = len(utc)
            #print(Ntime)
            #print(utc)
            # test Ntime
            #Ntime =200

            # Nantの計算 (uvfits.pyを参考にする)
            arraydata = uvfits.subarrays[subarrid]
            Nant = arraydata.antable["name"].shape[0]
            #print("Ntime,Nif,Nch,Nstokes,Nant,3(=greal,gimag,sigma)=")
            #print(Ntime,Nif,Nch,Nstokes,Nant,3)
            #self.gain = np.zeros([Ntime,Nif,Nch,Nstokes,Nant,3])

            gain = np.zeros([Ntime,Nif,Nch,Nstokes,Nant,3])
            gain[:,:,:,:,:,0]=1
            # dictionaryの"gain"にgainを格納
            self.gaintabs[subarrid]["gain"]=gain
