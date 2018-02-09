#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
This module describes data classes and related functions to handle UVFITS data.
'''
__author__ = "Sparselab Developer Team"
# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
# standard modules
import copy
import itertools
import collections
import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import astropy.time as at
import astropy.io.fits as pyfits

# internal
from ..uvtable import VisTable


# ------------------------------------------------------------------------------
# Classes for UVFITS FILE
# ------------------------------------------------------------------------------
class UVFITS():
    '''
    This is a class to load uvfits data and edit data sets before making tables
    for imaging.

    The instance of this class can be initialized by loading an uvfits file.
    Currently, this function can read only single-source uvfits file with
    AIPS AN, FQ tables. The data will be uv-sorted after data loading.

    Args:
      infile (string): input uvfits file

    Returns:
      uvdata.UVFITS object
    '''
    def __init__(self, infile):
        '''
        This is a class to load uvfits data and edit data sets before making tables
        for imaging.

        The instance of this class can be initialized by loading an uvfits file.
        Currently, this function can read only single-source uvfits file with
        AIPS AN, FQ tables. The data will be uv-sorted after data loading.

        Args:
          infile (string): input uvfits file

        Returns:
          uvdata.UVFITS object
        '''
        self.read_uvfits(infile)
        self.uvsort()

    def read_uvfits(self, infile):
        '''
        Read the uvfits file. Currently, this function can read only
        single-source uvfits file.

        Args:
          infile (string): input uvfits file

        Returns:
          uvdata.UVFITS object
        '''

        #----------------------------------------------------------------------
        # open uvfits file
        #----------------------------------------------------------------------
        hdulist = pyfits.open(infile)
        print('CONTENTS OF INPUT FITS FILE:')
        hdulist.info()

        hduinfos = hdulist.info(output=False)
        for hduinfo in hduinfos:
            idx = hduinfo[0]
            if hduinfo[1] == "PRIMARY":
                grouphdu = hdulist[idx]
            elif hduinfo[1] == "AIPS NX":
                aipsnx = hdulist[idx]
            elif hduinfo[1] == "AIPS FQ":
                aipsfq = hdulist[idx]
            elif hduinfo[1] == "AIPS AN":
                aipsan = hdulist[idx]

        if 'grouphdu' not in locals():
            print("[Error]   %s does not contain the Primary HDU" % (infile))

        if 'aipsfq' not in locals():
            print("[Error]   %s does not contain AIPS FQ table" % (infile))
        else:
            self.aipsfq = aipsfq.copy()

        if 'aipsan' not in locals():
            print("[WARNING] %s does not have any AIPS AN tables" % (infile))
        else:
            self.aipsan = aipsan.copy()

        # Save Group HDU Header
        self.header = grouphdu.header.copy()

        #----------------------------------------------------------------------
        # read random parameters
        #----------------------------------------------------------------------
        pars = grouphdu.data.parnames
        firstdate = 0
        Npars = len(pars)
        for ipar in xrange(Npars):
            par = pars[ipar]
            if par.find('UU') == 0:
                # FITS stores uvw coordinates in sec.
                usec = np.float64(grouphdu.data.par(ipar))
            elif par.find('VV') == 0:
                vsec = np.float64(grouphdu.data.par(ipar))
            elif par.find('WW') == 0:
                wsec = np.float64(grouphdu.data.par(ipar))
            elif par.find('BASELINE') == 0:
                bl = grouphdu.data.par(ipar)  # st1 * 256 + st2
                st1 = np.int64(bl // 256)
                st2 = np.int64(bl % 256)
            elif par.find('DATE') == 0:
                if firstdate == 0:
                    jd = np.float64(grouphdu.data.par(ipar))
                    timeobj = at.Time(np.float64(jd), format='jd', scale='utc')
                    firstdate += 1
                elif firstdate == 1:
                    jd = np.float64(grouphdu.data.par(ipar))
                    timeobj+= at.TimeDelta(np.float64(jd), format='jd')
                    firstdate += 1
            elif par.find('INTTIM') == 0:
                integ = grouphdu.data.par(ipar)  # integration time
        if (not 'usec' in locals()) or (not 'vsec' in locals()) or (not 'wsec' in locals()) or \
           (not 'bl' in locals()) or (not 'jd' in locals()):
            print("[Error] %s does not contain required random parameters in the Primary HDU" % (infile))

        #----------------------------------------------------------------------
        # Make Coordinates
        #----------------------------------------------------------------------
        coord = {}

        # UV Coordinate, etc
        uvsec = np.sqrt(usec * usec + vsec * vsec)
        coord["usec"] = ("data", np.asarray(
            usec, dtype=np.float64))  # U coordinate in sec
        coord["vsec"] = ("data", np.asarray(
            vsec, dtype=np.float64))  # V coordinate in sec
        coord["wsec"] = ("data", np.asarray(
            wsec, dtype=np.float64))  # W coordinate in sec
        coord["uvsec"] = ("data", np.asarray(
            uvsec, dtype=np.float64))  # UV Distance in sec
        coord["st1"] = ("data", np.asarray(
            st1, dtype=np.int64))  # Station ID 1
        coord["st2"] = ("data", np.asarray(
            st2, dtype=np.int64))  # Station ID 2

        # Original Baseline ID in FITS: st1 * 256 + st2
        coord["baseline"] = ("data", np.asarray(bl, dtype=np.int64))

        if "integ" in locals():
            coord["integ"] = ("data", np.float64(integ))  # integration time

        # Time Tag
        gsthour = timeobj.sidereal_time(
            kind="mean", longitude="greenwich", model=None).hour
        coord["jd"] = ("data", np.asarray(timeobj.jd, dtype=np.float64))
        coord["utc"] = ("data", timeobj.datetime)
        coord["gsthour"] = ("data", np.asarray(gsthour, dtype=np.float64))

        # Stokes parameter
        stokes = (np.arange(grouphdu.header['NAXIS3']) - grouphdu.header['CRPIX3'] +
                  1) * grouphdu.header['CDELT3'] + grouphdu.header['CRVAL3']
        coord["stokes"] = ("stokes", np.asarray(
            np.around(stokes), dtype=np.float64))

        # Frequency Parameter
        freqch = (np.arange(grouphdu.header['NAXIS4']) - grouphdu.header['CRPIX4'] +
                  1) * grouphdu.header['CDELT4'] + grouphdu.header['CRVAL4']
        #   Get IF Freq
        freqifdata = aipsfq.data['IF FREQ']
        if len(freqifdata.shape) == 1:
            freqif = aipsfq.data['IF FREQ']
        else:
            freqif = aipsfq.data['IF FREQ'][0]
        #   Get Obs Frequency and calc UVW
        freq = np.zeros([grouphdu.header['NAXIS5'], grouphdu.header['NAXIS4']])
        u = np.zeros([grouphdu.header['GCOUNT'],
                      grouphdu.header['NAXIS5'], grouphdu.header['NAXIS4']])
        v = np.zeros([grouphdu.header['GCOUNT'],
                      grouphdu.header['NAXIS5'], grouphdu.header['NAXIS4']])
        w = np.zeros([grouphdu.header['GCOUNT'],
                      grouphdu.header['NAXIS5'], grouphdu.header['NAXIS4']])
        uv = np.zeros([grouphdu.header['GCOUNT'],
                       grouphdu.header['NAXIS5'], grouphdu.header['NAXIS4']])
        for iif, ich in itertools.product(np.arange(grouphdu.header['NAXIS5']),
                                          np.arange(grouphdu.header['NAXIS4'])):
            freq[iif, ich] = freqif[iif] + freqch[ich]
            u[:, iif, ich] = freq[iif, ich] * usec[:]
            v[:, iif, ich] = freq[iif, ich] * vsec[:]
            w[:, iif, ich] = freq[iif, ich] * wsec[:]
            uv[:, iif, ich] = freq[iif, ich] * uvsec[:]
        coord["freqch"] = ("freqch", np.asarray(freqch, dtype=np.float64))
        coord["freqif"] = ("freqif", np.asarray(freqif, dtype=np.float64))
        coord["freq"] = (("freqif", "freqch"),
                         np.asarray(freq, dtype=np.float64))
        coord["u"] = (("data", "freqif", "freqch"),
                      np.asarray(u, dtype=np.float64))
        coord["v"] = (("data", "freqif", "freqch"),
                      np.asarray(v, dtype=np.float64))
        coord["w"] = (("data", "freqif", "freqch"),
                      np.asarray(w, dtype=np.float64))
        coord["uv"] = (("data", "freqif", "freqch"),
                       np.asarray(uv, dtype=np.float64))

        # RA and Dec
        ra = (np.arange(grouphdu.header['NAXIS6']) - grouphdu.header['CRPIX6'] +
              1) * grouphdu.header['CDELT6'] + grouphdu.header['CRVAL6']
        dec = (np.arange(grouphdu.header['NAXIS7']) - grouphdu.header['CRPIX7'] +
               1) * grouphdu.header['CDELT7'] + grouphdu.header['CRVAL7']
        coord["ra"] = ("ra",  np.asarray(ra, dtype=np.float64))
        coord["dec"] = ("dec", np.asarray(dec, dtype=np.float64))

        # Reset Index
        self.data = xr.DataArray(grouphdu.data.data,
                                 coords=coord,
                                 dims=["data", "dec", "ra", "freqif", "freqch", "stokes", "complex"])

        # Close hdu
        hdulist.close()
        print("")

    def uvsort(self):
        '''
        Check station IDs of each visibility and switch its order if "st1" > "st2".
        Then, data will be TB-sorted.
        '''
        # check station IDs
        select = np.asarray(self.data["st1"] > self.data["st2"])
        if True in select:
            self.data.usec.loc[select] *= -1
            self.data.vsec.loc[select] *= -1
            self.data.wsec.loc[select] *= -1
            self.data.u.loc[select, :, :] *= -1
            self.data.v.loc[select, :, :] *= -1
            self.data.w.loc[select, :, :] *= -1
            self.data.baseline.loc[select] = 256 * \
                self.data.st2.loc[select] + self.data.st1.loc[select]
            dammy = self.data.st2.loc[select]
            self.data.st2.loc[select] = self.data.st1.loc[select]
            self.data.st1.loc[select] = dammy
            self.data.loc[select, :, :, :, :, :, 1] *= - \
                1  # frip imaginary part of visibilities

            Nselect = len(np.where(select)[0])
            print("Station IDs of %d data points are flipped due to their wrong orders (st1 > st2)." % (
                Nselect))
        else:
            print("Station IDs have correct orders (st1 < st2). ")

        # TB-sord data
        idx1 = np.argsort(self.data["baseline"])
        idx2 = np.argsort(self.data["jd"][idx1])
        idx = idx1[idx2]
        check = idx1 == np.arange(self.data.shape[1])
        if False in check:
            print("Data are not TB sorted. Sorting data....")
            self.data = self.data.loc[idx, :, :, :, :, :, :]
            print("Data sort was finished!")
        else:
            print("Data are TB sorted correctly.")
        print("")

    def make_vistable(self, flag=True):
        '''
        Convert visibility data to a two dimentional table.

        Args:
          flag (boolean):
            if flag=True, data with weights <= 0 or sigma <=0 will be ignored.

        Returns:
          uvdata.VisTable object
        '''
        outdata = VisTable()

        # Get size of data
        Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp = self.data.shape

        # Get time
        # DOY, HH, MM, SS
        attime = at.Time(np.datetime_as_string(self.data["utc"]))
        utctime = attime.datetime
        gsthour = attime.sidereal_time("apparent", "greenwich").hour
        for idec, ira, iif, ich, istokes in itertools.product(np.arange(Ndec),
                                                              np.arange(Nra),
                                                              np.arange(Nif),
                                                              np.arange(Nch),
                                                              np.arange(Nstokes)):
            tmpdata = VisTable()

            # Time
            tmpdata["utc"] = utctime
            tmpdata["gsthour"] = gsthour

            # Frequecny
            tmpdata["freq"] = np.zeros(Ndata, dtype=np.float32)
            tmpdata.loc[:, "freq"] = np.float64(self.data["freq"][iif, ich])

            # Stokes ID
            tmpdata["stokesid"] = np.zeros(Ndata, dtype=np.int32)
            if Nstokes == 1:
                tmpdata.loc[:, "stokesid"] = np.int32(self.data["stokes"])
            else:
                tmpdata.loc[:, "stokesid"] = np.int32(
                    self.data["stokes"][istokes])

            # ch/if id, frequency
            tmpdata["ifid"] = np.zeros(Ndata, dtype=np.int32)
            tmpdata.loc[:, "ifid"] = np.int32(iif)
            tmpdata["chid"] = np.zeros(Ndata, dtype=np.int32)
            tmpdata.loc[:, "chid"] = np.int32(ich)
            tmpdata["ch"] = tmpdata["ifid"] + tmpdata["chid"] * Nif

            # uvw
            tmpdata["u"] = np.float64(self.data["u"][:, iif, ich])
            tmpdata["v"] = np.float64(self.data["v"][:, iif, ich])
            tmpdata["w"] = np.float64(self.data["w"][:, iif, ich])
            tmpdata["uvdist"] = np.float64(self.data["uv"][:, iif, ich])

            # station number
            tmpdata["st1"] = np.int32(self.data["st1"])
            tmpdata["st2"] = np.int32(self.data["st2"])

            visreal = np.float64(
                self.data.data[:, idec, ira, iif, ich, istokes, 0])
            visimag = np.float64(
                self.data.data[:, idec, ira, iif, ich, istokes, 1])
            visweig = np.float64(
                self.data.data[:, idec, ira, iif, ich, istokes, 2])
            tmpdata["amp"] = np.sqrt(visreal * visreal + visimag * visimag)
            tmpdata["phase"] = np.rad2deg(np.arctan2(visimag, visreal))
            tmpdata["weight"] = visweig
            tmpdata["sigma"] = np.sqrt(1. / visweig)

            outdata = pd.concat([outdata, tmpdata])

        if flag:
            select = outdata["weight"] > 0
            select *= outdata["sigma"] > 0
            select *= np.isnan(outdata["weight"]) == False
            select *= np.isnan(outdata["sigma"]) == False
            select *= np.isinf(outdata["weight"]) == False
            select *= np.isinf(outdata["sigma"]) == False
            outdata = outdata.loc[select, :].reset_index(drop=True)

        return outdata

    #-------------------------------------------------------------------------
    # Edit UV fits files
    #-------------------------------------------------------------------------
    def select_stokes(self, stokes="I"):
        '''
        Pick up single polarization data

        Args:
          stokes (string; default="I"):
            Output stokes parameters.
            Availables are ["I", "Q", "U", "V", "LL", "RR", "RL", "LR"].

        Output: uvdata.UVFITS object
        '''
        # get stokes data
        stokesids = np.asarray(self.data["stokes"], dtype=np.int64)

        # create output data
        outfits = copy.deepcopy(self)
        if stokes == "I":
            if (1 in stokesids):  # I <- I
                print("Stokes I data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == 1, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 1
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-1 in stokesids) and (-2 in stokesids):  # I <- (RR + LL)/2
                print("Stokes I data will be calculated from input RR and LL data")
                outfits.data = bindstokes(
                    self.data, stokes=1, stokes1=-1, stokes2=-2, factr1=0.5, factr2=0.5)
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 1
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-1 in stokesids):  # I <- RR
                print("Stokes I data will be copied from input RR data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -1, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -1
                outfits.header["CDELT3"] = -1
                outfits.header["CROTA3"] = 0
            elif (-2 in stokesids):  # I <- LL
                print("Stokes I data will be copied from input LL data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -2, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -2
                outfits.header["CDELT3"] = -1
                outfits.header["CROTA3"] = 0
            elif (-5 in stokesids) and (-6 in stokesids):  # I <- (XX + YY)/2
                print("Stokes I data will be calculated from input XX and YY data")
                outfits.data = bindstokes(
                    self.data, stokes=1, stokes1=-5, stokes2=-6, factr1=0.5, factr2=0.5)
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 1
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-5 in stokesids):  # I <- XX
                print("Stokes I data will be copied from input XX data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -5, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -5
                outfits.header["CDELT3"] = -1
                outfits.header["CROTA3"] = 0
            elif (-6 in stokesids):  # I <- YY
                print("Stokes I data will be copied from input YY data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -6, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -6
                outfits.header["CDELT3"] = -1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "Q":
            if (2 in stokesids):  # Q <- Q
                print("Stokes Q data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == 2, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 2
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-3 in stokesids) and (-4 in stokesids):  # Q <- (RL + LR)/2
                print("Stokes Q data will be calculated from input RL and LR data")
                outfits.data = bindstokes(
                    self.data, stokes=2, stokes1=-3, stokes2=-4, factr1=0.5, factr2=0.5)
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 2
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-5 in stokesids) and (-6 in stokesids):  # Q <- (XX - YY)/2
                print("Stokes Q data will be calculated from input XX and YY data")
                outfits.data = bindstokes(
                    self.data, stokes=2, stokes1=-5, stokes2=-6, factr1=0.5, factr2=-0.5)
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 2
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "U":
            if (3 in stokesids):  # V <- V
                print("Stokes U data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == 2, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 3
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-3 in stokesids) and (-4 in stokesids):  # U <- (RL - LR)/2i = (- RL + LR)i/2
                print("Stokes U data will be calculated from input RL and LR data")
                outfits.data = bindstokes(
                    self.data, stokes=3, stokes1=-3, stokes2=-4, factr1=-0.5j, factr2=0.5j)
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 3
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-7 in stokesids) and (-8 in stokesids):  # U <- (XY + YX)/2
                print("Stokes U data will be calculated from input XX and YY data")
                outfits.data = bindstokes(
                    self.data, stokes=3, stokes1=-7, stokes2=-8, factr1=0.5, factr2=0.5)
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 3
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "V":
            if (4 in stokesids):  # V <- V
                print("Stokes V data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == 4, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 4
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-1 in stokesids) and (-2 in stokesids):  # V <- (RR - LL)/2
                print("Stokes V data will be calculated from input RR and LL data")
                outfits.data = bindstokes(
                    self.data, stokes=4, stokes1=-1, stokes2=-2, factr1=0.5, factr2=-0.5)
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 4
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            elif (-7 in stokesids) and (-8 in stokesids):  # V <- (XY - YX)/2i = (-XY + YX)/2
                print("Stokes V data will be calculated from input XX and YY data")
                outfits.data = bindstokes(
                    self.data, stokes=4, stokes1=-7, stokes2=-8, factr1=-0.5j, factr2=0.5j)
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = 4
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "RR":
            if (-1 in stokesids):  # V <- V
                print("Stokes RR data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -1, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -1
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "LL":
            if (-2 in stokesids):  # V <- V
                print("Stokes LL data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -2, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -2
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "RL":
            if (-3 in stokesids):  # V <- V
                print("Stokes RL data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -3, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -3
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        elif stokes == "LR":
            if (-4 in stokesids):  # V <- V
                print("Stokes LR data will be copied from the input data")
                outfits.data = self.data[:, :, :, :, :, stokesids == -4, :]
                outfits.header["CRPIX3"] = 1
                outfits.header["CRVAL3"] = -4
                outfits.header["CDELT3"] = 1
                outfits.header["CROTA3"] = 0
            else:
                print(
                    "[WARNING] No data are available to calculate Stokes %s" % (stokes))
        else:
            print(
                "[WARNING] Currently Stokes %s is not supported in this function." % (stokes))

        return outfits

    def weightcal(self, dofreq=0, solint=120., minpoint=2):
        '''
        This method will recalculate sigmas and weights of data from scatter
        in full complex visibilities over specified frequency and time segments.

        Args:
          dofreq (int; default = 0):
            Parameter for multi-frequency data sets.
              dofreq = 0: calculate weights and sigmas over IFs and channels
              dofreq = 1: calculate weights and sigmas over channels at each IF
              dofreq = 2: calculate weights and sigmas at each IF and Channel

          solint (float; default = 120.):
            solution interval in sec

          minpoint (int; default=2):
            A minimum number of points that weight will be estimated.
            For data points that do not have surrounding data points more than
            this value, data will be flagged.

        Returns: uvdata.UVFITS object
        '''
        # Default Averaging alldata
        doif = True
        doch = True
        if np.int64(dofreq) > 0:
            doif = False
        if np.int64(dofreq) > 1:
            doch = False

        # Save and Return re-weighted uv-data
        outfits = copy.deepcopy(self)

        # Get size of data
        Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp = self.data.shape

        # Get unix
        unix = at.Time(self.data["jd"], format="jd", scale="utc").unix
        baseline = np.asarray(self.data["baseline"], dtype=np.int64)

        if doif == True and doch == True:
            for idata in xrange(Ndata):
                #if idata%1000 == 0: print("%d / %d"%(idata,Ndata))
                dataidx1 = np.where(np.abs(unix - unix[idata]) < solint)
                dataidx2 = np.where(baseline[dataidx1] == baseline[idata])
                seldata = self.data.data[dataidx1][dataidx2]
                for idec, ira, istokes in itertools.product(np.arange(Ndec),
                                                            np.arange(Nra),
                                                            np.arange(Nstokes)):
                    vreal = seldata[:, idec, ira, :, :, istokes, 0]
                    vimaj = seldata[:, idec, ira, :, :, istokes, 1]
                    vweig = seldata[:, idec, ira, :, :, istokes, 2]

                    Ntime = np.prod(vreal.shape)
                    vreal = vreal.reshape(Ntime)
                    vimaj = vimaj.reshape(Ntime)
                    vweig = vweig.reshape(Ntime)

                    select = vweig > 0
                    select *= (np.isnan(vweig) == False)
                    select *= (np.isinf(vweig) == False)
                    select = np.where(select)
                    vreal = vreal[select]
                    vimaj = vimaj[select]
                    vweig = vweig[select]
                    Nsel = len(vreal)

                    if Nsel < minpoint:
                        outfits.data[idata, idec, ira, :, :, istokes, 2] = 0.0
                        continue

                    vcomp = vreal + 1j * vimaj
                    vweig = 1. / np.var(vcomp)
                    outfits.data[idata, idec, ira, :, :, istokes, 2] = vweig
            select = self.data.data[:, :, :, :, :, :, 2] <= 0.0
            select += np.isnan(self.data.data[:, :, :, :, :, :, 2])
            select += np.isinf(self.data.data[:, :, :, :, :, :, 2])
            outfits.data.data[:, :, :, :, :, :, 2][np.where(select)] = 0.0
        elif doif == True:
            for idata in xrange(Ndata):
                #if idata%1000 == 0: print("%d / %d"%(idata,Ndata))
                dataidx1 = np.where(np.abs(unix - unix[idata]) < solint)
                dataidx2 = np.where(baseline[dataidx1] == baseline[idata])
                seldata = self.data.data[dataidx1][dataidx2]
                for idec, ira, iif, istokes in itertools.product(np.arange(Ndec),
                                                                 np.arange(
                                                                     Nra),
                                                                 np.arange(
                                                                     Nif),
                                                                 np.arange(Nstokes)):
                    vreal = seldata[:, idec, ira, iif, :, istokes, 0]
                    vimaj = seldata[:, idec, ira, iif, :, istokes, 1]
                    vweig = seldata[:, idec, ira, iif, :, istokes, 2]

                    Ntime = np.prod(vreal.shape)
                    vreal = vreal.reshape(Ntime)
                    vimaj = vimaj.reshape(Ntime)
                    vweig = vweig.reshape(Ntime)

                    select = vweig > 0
                    select *= (np.isnan(vweig) == False)
                    select *= (np.isinf(vweig) == False)
                    select = np.where(select)
                    vreal = vreal[select]
                    vimaj = vimaj[select]
                    vweig = vweig[select]
                    Nsel = len(vreal)

                    if Nsel < minpoint:
                        outfits.data[idata, idec, ira, iif, :, istokes, 2] = 0
                        continue

                    vcomp = vreal + 1j * vimaj
                    vweig = 1. / np.var(vcomp)
                    outfits.data[idata, idec, ira, :, :, istokes, 2] = vweig
            select = self.data.data[:, :, :, :, :, :, 2] <= 0.0
            select += np.isnan(self.data.data[:, :, :, :, :, :, 2])
            select += np.isinf(self.data.data[:, :, :, :, :, :, 2])
            outfits.data.data[:, :, :, :, :, :, 2][np.where(select)] = 0.0
        else:
            for idata in xrange(Ndata):
                #if idata%1000 == 0: print("%d / %d"%(idata,Ndata))
                dataidx1 = np.where(np.abs(unix - unix[idata]) < solint)
                dataidx2 = np.where(baseline[dataidx1] == baseline[idata])
                seldata = self.data.data[dataidx1][dataidx2]
                for idec, ira, iif, ich, istokes in itertools.product(np.arange(Ndec),
                                                                      np.arange(
                                                                          Nra),
                                                                      np.arange(
                                                                          Nif),
                                                                      np.arange(
                                                                          Nch),
                                                                      np.arange(Nstokes)):
                    weight = outfits.data[idata, idec,
                                          ira, iif, ich, istokes, 2]
                    if weight <= 0.0 or np.isnan(weight) or np.isinf(weight):
                        outfits.data[idata, idec, ira,
                                     iif, ich, istokes, 2] = 0.0
                        continue

                    vreal = seldata[:, idec, ira, iif, ich, istokes, 0]
                    vimaj = seldata[:, idec, ira, iif, ich, istokes, 1]
                    vweig = seldata[:, idec, ira, iif, ich, istokes, 2]

                    Ntime = np.prod(vreal.shape)
                    vreal = vreal.reshape(Ntime)
                    vimaj = vimaj.reshape(Ntime)
                    vweig = vweig.reshape(Ntime)

                    select = vweig > 0
                    select *= (np.isnan(vweig) == False)
                    select *= (np.isinf(vweig) == False)
                    select = np.where(select)
                    vreal = vreal[select]
                    vimaj = vimaj[select]
                    vweig = vweig[select]
                    Nsel = len(vreal)

                    if Nsel < minpoint:
                        outfits.data[idata, idec, ira,
                                     iif, ich, istokes, 2] = 0
                        continue

                    vcomp = vreal + 1j * vimaj
                    outfits.data[idata, idec, ira, iif, ich,
                                 istokes, 2] = 1. / np.var(vcomp)
        return outfits

#-------------------------------------------------------------------------
# Subfunctions for UVFITS
#-------------------------------------------------------------------------
def bindstokes(dataarray, stokes, stokes1, stokes2, factr1, factr2):
    '''
    This is a subfunction for uvdata.UVFITS.
    '''
    stokesids = np.asarray(dataarray["stokes"], dtype=np.int64)
    istokes1 = np.where(stokesids == stokes1)[0][0]
    istokes2 = np.where(stokesids == stokes2)[0][0]
    coords = copy.deepcopy(dataarray.coords)
    coords["stokes"] = np.asarray([stokes], dtype=np.float64)
    outdata = xr.DataArray(dataarray.data[:, :, :, :, :, istokes1:istokes1 + 1, :],
                           coords=coords,
                           dims=dataarray.dims)

    vcomp1 = dataarray.data[:, :, :, :, :, istokes1, 0] + \
        1j * dataarray.data[:, :, :, :, :, istokes1, 1]
    vweig1 = dataarray.data[:, :, :, :, :, istokes1, 2]
    vcomp2 = dataarray.data[:, :, :, :, :, istokes2, 0] + \
        1j * dataarray.data[:, :, :, :, :, istokes2, 1]
    vweig2 = dataarray.data[:, :, :, :, :, istokes2, 2]

    vcomp = factr1 * vcomp1 + factr2 * vcomp2
    vweig = np.power(np.abs(factr1)**2 / vweig1 +
                     np.abs(factr2)**2 / vweig2, -1)

    select = vweig1 <= 0
    select += vweig2 <= 0
    select += vweig <= 0
    select += np.isnan(vweig1)
    select += np.isnan(vweig2)
    select += np.isnan(vweig)
    select += np.isinf(vweig1)
    select += np.isinf(vweig2)
    select += np.isinf(vweig)
    vweig[np.where(select)] = 0.0

    outdata.data[:, :, :, :, :, 0, 0] = np.real(vcomp)
    outdata.data[:, :, :, :, :, 0, 1] = np.imag(vcomp)
    outdata.data[:, :, :, :, :, 0, 2] = vweig

    return outdata
