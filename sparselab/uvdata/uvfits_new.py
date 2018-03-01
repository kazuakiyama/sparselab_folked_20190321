#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a submodule of sparselab handling various types of Visibility data sets.
'''
__author__ = "Sparselab Developer Team"

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
# standard modules
import copy
import itertools
import collections
import datetime
import tqdm

# numerical packages
import numpy as np
import pandas as pd
import astropy.constants as ac
import astropy.coordinates as acd
import astropy.time as at
import astropy.io.fits as pf

# matplotlib
import matplotlib.pyplot as plt

# internal
from .. import imdata, fortlib
from ..util import prt

indent = "  "

# ------------------------------------------------------------------------------
# Classes for UVFITS FILE
# ------------------------------------------------------------------------------
stokesDict = {
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    -1: "RR",
    -2: "LL",
    -3: "RL",
    -4: "LR",
    -5: "XX",
    -6: "YY",
    -7: "XY",
    -8: "YX"
}
stokesDictinv = {}
for key in stokesDict.keys():
    val = stokesDict[key]
    stokesDictinv[val] = key

# ------------------------------------------------------------------------------
# Classes for UVFITS FILE
# ------------------------------------------------------------------------------
class UVFITS(object):
    '''
    This is a class to load, edit, write uvfits data
    '''
    def __init__(self, uvfits):
        '''
        Load an uvfits file. Currently, this function can read only
        single-source uvfits file. The data will be uv-sorted.

        Args:
          infile (string or pyfits.HDUList object): input uvfits data

        Returns:
          uvdata.UVFITS object
        '''
        # check input files
        if type(uvfits) == type(""):
            hdulist = pf.open(uvfits)
        else:
            hdulist = uvfits
        hdulist.info()
        print("")
        hduinfo = hdulist.info(output=False)
        Nhdu = len(hduinfo)

        # Read hdus
        FQtab = None
        ANtabs = {}
        SUtabs = {}
        ghdu = None
        prt("Loading HDUs in the input UVFITS files.")
        for ihdu in xrange(Nhdu):
            hduname = hduinfo[ihdu][1]
            if hduname == "PRIMARY":
                if ghdu is not None:
                    prt("[WARNING] This UVFITS has more than two Primary HDUs.",indent)
                    prt("          The later one will be taken.",indent)
                else:
                    prt("Primary HDU was loaded.", indent)
                ghdu = hdulist[ihdu]
            if hduname == "AIPS FQ":
                if FQtab is not None:
                    prt("[WARNING] This UVFITS has more than two AIPS FQ tables.",indent)
                    prt("          The later one will be taken.",indent)
                else:
                    prt("AIPS FQ Table was loaded.", indent)
                FQtab = hdulist[ihdu]
            if hduname == "AIPS AN":
                subarrid = np.int64(hdulist[ihdu].header.get("EXTVER"))
                if subarrid == -1:
                    subarrid = 1
                if subarrid in ANtabs.keys():
                    prt("[WARNING] There are duplicated subarrays with subarray ID=%d."%(subarrid),indent)
                    pri("          The later one will be adopted.", indent)
                else:
                    prt("Subarray %d was found in an AIPS AN table"%(subarrid), indent)
                ANtabs[subarrid] = hdulist[ihdu]
            if hduname == "AIPS SU":
                suid = np.int64(hdulist[ihdu].header.get("FREQID"))
                if suid in SUtabs.keys():
                    prt("[WARNING] There are more than two SU tables for the same Frequency setup frqselid=%d."%(suid),indent)
                    prt("          The later one will be adopted.",indent)
                else:
                    prt("A SU Table for a frequency Setup %d was found"%(suid),indent)
                SUtabs[suid] = hdulist[ihdu]

        # Check number of AIPS FQ/AN tables loaded.
        print("")
        prt("Checking loaded HDUs.")
        # Group HDU
        if ghdu is None:
            errmsg = "No GroupHDUs are included in the input UVFITS data."
            raise ValueError(errmsg)
        # AIPS FQ Table
        if FQtab is None:
            errmsg = "No FQ tables are included in the input UVFITS data."
            raise ValueError(errmsg)
        # AN Table
        if len(ANtabs)==0:
            errmsg = "No AN tables are included in the input UVFITS data."
            raise ValueError(errmsg)
        else:
            prt("%d Subarray settings are found."%(len(ANtabs)),indent)
        # AIPS SU Table
        if len(SUtabs)==0:
            prt("No AIPS SU tables were found.",indent)
            prt("  Assuming that this is a single source UVFITS file.",indent)
            self.ismultisrc = False
        else:
            prt("AIPS SU tables were found.",indent)
            prt("  Assuming that this is a multi source UVFITS file.",indent)
            self.ismultisrc = True

        #if self.ismultisrc:
        #    raise ImportError("Sorry, this library currently can read only single-source UVFITS files.")

        print("")
        prt("Reading FQ Tables")
        self._read_freqdata(FQtab)

        # Read AN Tables
        print("")
        prt("Reading AN Tables")
        self._read_arraydata(ANtabs)

        # Read SU Tables
        print("")
        if self.ismultisrc:
            prt("Reading SU Tables")
            self._read_srcdata_multi(SUtabs)
        else:
            prt("Reading Source Information from Primary HDU")
            self._read_srcdata_single(ghdu)

        # Load GroupHDU
        print("")
        prt("Reading Primary HDU data")
        self._read_grouphdu(ghdu, timescale="utc")

        # Load Stokes Parameters
        stokesid = np.arange(ghdu.header["NAXIS3"])+1-ghdu.header["CRPIX3"]
        stokesid*= ghdu.header["CDELT3"]
        stokesid+= ghdu.header["CRVAL3"]
        self.stokes = [stokesDict[int(sid)] for sid in stokesid]

        # Load OBSERVER
        self.observer = ghdu.header["OBSERVER"]

    def write_fits(self, filename, overwrite=True):
        '''
        save to uvfits file. The data will be uv-sorted.

        Args:
          infile (string or pyfits.HDUList object): input uvfits data

        Returns:
          uvdata.UVFITS object
        '''
        if self.ismultisrc:
            raise ValueError("Sorry, this library currently can not create multi-source UVFITS data.")

        hdulist = []
        hdulist.append(self._create_ghdu_single())
        hdulist.append(self._create_fqtab())
        hdulist += self._create_antab()
        hdulist = pf.HDUList(hdulist)
        hdulist.writeto(filename, overwrite=overwrite)

    def _read_arraydata(self, ANtabs):
        subarrays = {}
        subarrids = ANtabs.keys()
        for subarrid in subarrids:
            # Create Array data
            arrdata = ArrayData()

            # AN Table
            ANtab = ANtabs[subarrid]

            # Read AN Table Header
            ANheadkeys = ANtab.header.keys()
            headkeys = arrdata.header.keys()
            for key in ANheadkeys:
                if key in headkeys:
                    arrdata.header[key] = ANtab.header.get(key)
            arrdata.frqsel = ANtab.header.get("FREQID")
            if arrdata.frqsel is None:
                arrdata.frqsel=1
            elif arrdata.frqsel == -1:
                arrdata.frqsel=1
            arrdata.subarray = subarrid

            # Read AN Table Data
            arrdata.antable["name"] = ANtab.data["ANNAME"]
            arrdata.antable["x"] = ANtab.data["STABXYZ"][:,0]
            arrdata.antable["y"] = ANtab.data["STABXYZ"][:,1]
            arrdata.antable["z"] = ANtab.data["STABXYZ"][:,2]
            arrdata.antable["id"] = ANtab.data["NOSTA"]
            arrdata.antable["mnttype"] = ANtab.data["MNTSTA"]
            arrdata.antable["axisoffset"] = ANtab.data["STAXOF"]
            arrdata.antable["poltypeA"] = ANtab.data["POLTYA"]
            arrdata.antable["polangA"] = ANtab.data["POLAA"]
            arrdata.antable["poltypeB"] = ANtab.data["POLTYB"]
            arrdata.antable["polangB"] = ANtab.data["POLAB"]
            arrdata.anorbparm = ANtab.data["ORBPARM"]
            arrdata.anpolcalA = ANtab.data["POLCALA"]
            arrdata.anpolcalB = ANtab.data["POLCALB"]
            subarrays[subarrid]=arrdata
            prt(arrdata, indent)
        self.subarrays = subarrays

    def _read_freqdata(self, FQtab):
        freqdata = FrequencyData()

        # Load Data
        freqdata.frqsels = FQtab.data["FRQSEL"]
        Nfrqsel = len(freqdata.frqsels)
        for i in xrange(Nfrqsel):
            frqsel = freqdata.frqsels[i]
            fqtable = pd.DataFrame(columns=freqdata.fqtable_cols)
            fqtable["if_freq_offset"]=FQtab.data["IF FREQ"][i]
            fqtable["ch_bandwidth"]=FQtab.data["CH WIDTH"][i]
            fqtable["if_bandwidth"]=FQtab.data["TOTAL BANDWIDTH"][i]
            fqtable["sideband"]=FQtab.data["SIDEBAND"][i]
            freqdata.fqtables[frqsel] = fqtable
        prt(freqdata, indent)

        self.fqsetup = freqdata
        if Nfrqsel==1:
            self.ismultifrq = False
        else:
            self.ismultifrq = True

    def _read_srcdata_single(self, ghdu):
        # Create Array data
        srcdata = SourceData()

        # Read Header
        srcdata.frqsel = 1
        srcdata.header["VELDEF"] = "RADIO"
        srcdata.header["VELTYP"] = "GEOCENTR"
        srcdata.header["NO_IF"] = ghdu.header.get("NAXIS5")

        # Load Data
        srcdata.sutable["id"] = np.asarray([1], dtype=np.int64)
        srcdata.sutable["source"] = np.asarray([ghdu.header.get("OBJECT")])
        srcdata.sutable["qual"] = np.asarray([0], dtype=np.int64)
        srcdata.sutable["calcode"] = np.asarray([""])
        srcdata.sutable["bandwidth"] = np.asarray([ghdu.header.get("CDELT4")], dtype=np.float64)
        if "EQUINOX" in ghdu.header.keys():
            srcdata.sutable["equinox"] = np.asarray([ghdu.header.get("EQUINOX")], dtype=np.float64)
        elif "EPOCH" in ghdu.header.keys():
            srcdata.sutable["equinox"] = np.asarray([ghdu.header.get("EPOCH")], dtype=np.float64)
        else:
            srcdata.sutable["equinox"] = np.asarray([2000.0], dtype=np.float64)
        srcdata.sutable["ra_app"] = np.asarray([ghdu.header.get("CRVAL6")], dtype=np.float64)
        srcdata.sutable["dec_app"] = np.asarray([ghdu.header.get("CRVAL7")], dtype=np.float64)
        srcdata.sutable["pmra"] = np.asarray([0.0], dtype=np.float64)
        srcdata.sutable["pmdec"] = np.asarray([0.0], dtype=np.float64)
        srcdata.suiflux = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.suqflux = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.suuflux = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.suvflux = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.sufreqoff = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.sulsrvel = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.surestfreq = np.zeros([1, srcdata.header["NO_IF"]])

        # Here I note that we assume equinox of coordinates will be J2000.0
        # which is the currend default of acd.SkyCoord (Feb 28 2018)
        radec = acd.SkyCoord(ra=[ghdu.header.get("CRVAL6")], dec=[ghdu.header.get("CRVAL7")],
                             #equinox="J%d"%(srcdata.sutable.loc[0,"equinox"]),
                             unit="deg",
                             frame="icrs")
        srcdata.sutable["radec"] = radec.to_string("hmsdms")

        prt(srcdata, indent)
        sources = {}
        sources[1]=srcdata
        self.sources = sources

    def _read_srcdata_multi(self, SUtabs):
        sources = {}
        srclist = []
        frqselids = SUtabs.keys()
        for frqselid in frqselids:
            # Create Array data
            srcdata = SourceData()

            # FQ Table
            SUtab = SUtabs[frqselid]

            # Read Header
            SUheadkeys = SUtab.header.keys()
            headkeys = srcdata.header.keys()
            for key in SUheadkeys:
                if key in headkeys:
                    srcdata.header[key] = SUtab.header.get(key)
            srcdata.frqsel = SUtab.header.get("FREQID")

            # Load Data
            srcdata.sutable["id"] = SUtab.data["ID. NO."]
            srcdata.sutable["source"] = SUtab.data["SOURCE"]
            srcdata.sutable["qual"] = SUtab.data["QUAL"]
            srcdata.sutable["calcode"] = SUtab.data["CALCODE"]
            srcdata.sutable["bandwidth"] = SUtab.data["BANDWIDTH"]
            srcdata.sutable["equinox"] = SUtab.data["EPOCH"]
            srcdata.sutable["ra_app"] = SUtab.data["RAAPP"]
            srcdata.sutable["dec_app"] = SUtab.data["DECAPP"]
            srcdata.sutable["pmra"] = SUtab.data["PMRA"]
            srcdata.sutable["pmdec"] = SUtab.data["PMDEC"]
            srcdata.iflux = SUtab.data["IFLUX"]
            srcdata.qflux = SUtab.data["QFLUX"]
            srcdata.uflux = SUtab.data["UFLUX"]
            srcdata.vflux = SUtab.data["VFLUX"]
            srcdata.freqoff = SUtab.data["FREQOFF"]
            srcdata.lsrvel = SUtab.data["LSRVEL"]
            srcdata.restfreq = SUtab.data["RESTFREQ"]
            radec = acd.SkyCoord(ra=SUtab.data["RAEPO"], dec=SUtab.data["DECEPO"],
                                 equinox=srcdata.sutable["equinox"], unit="deg",
                                 frame="icrs")
            srcdata.sutable["radec"] = radec.to_string("hmsdms")

            srclist += list(set(srcdata.sutable["source"]))
            sources[frqselid]=srcdata
            prt(srcdata, indent)
        srclist = list(set(srclist))
        if len(srclist) == 1:
            self.ismultisrc = False
        self.sources = sources

    def _read_grouphdu(self, hdu, timescale="utc"):
        visdata = VisibilityData()

        # Read data
        visdata.data = hdu.data.data
        Ndata = visdata.data.shape[0]

        # Read Random Parameters
        paridxes = [None for i in xrange(9)]
        parnames = hdu.data.parnames
        Npar = len(parnames)
        visdata.coord = pd.DataFrame()
        for i in xrange(Npar):
            parname = parnames[i]
            if "UU" in parname:
                paridxes[0] = i+1
                visdata.coord["usec"] = np.float64(hdu.data.par(i))
            if "VV" in parname:
                paridxes[1] = i+1
                visdata.coord["vsec"] = np.float64(hdu.data.par(i))
            if "WW" in parname:
                paridxes[2] = i+1
                visdata.coord["wsec"] = np.float64(hdu.data.par(i))
            if "DATE" in parname:
                if paridxes[3] is None:
                    paridxes[3] = i+1
                    jd1 = np.float64(hdu.data.par(i))
                elif paridxes[4] is None:
                    paridxes[4] = i+1
                    jd2 = np.float64(hdu.data.par(i))
                else:
                    errmsg = "Random Parameters have too many 'DATE' columns."
                    raise ValueError(errmsg)
            if "BASELINE" in parname:
                paridxes[5] = i+1
                bl = np.float64(hdu.data.par(i))
            if "SOURCE" in parname:
                paridxes[6] = i+1
                visdata.coord["source"] = np.int64(hdu.data.par(i))
            if "INTTIM" in parname:
                paridxes[7] = i+1
                visdata.coord["inttim"] = np.float64(hdu.data.par(i))
            if "FREQSEL" in parname:
                paridxes[8] = i+1
                visdata.coord["freqsel"] = np.int64(hdu.data.par(i))

        # Check Loaded Random Parameters
        if self.ismultisrc and (paridxes[6] is None):
            errmsg = "Random Parameters do not have 'SOURCE' although UVFITS is for multi sources."
            raise ValueError(errmsg)
        elif (self.ismultisrc is False) and (paridxes[6] is None):
            visdata.coord["source"] = np.asarray([1 for i in xrange(Ndata)])
            paridxes[6] = -1
        if self.ismultifrq and (paridxes[8] is None):
            errmsg = "Random Parameters do not have 'FREQSEL' although UVFITS have multi frequency setups."
            raise ValueError(errmsg)
        elif (self.ismultifrq is False) and (paridxes[8] is None):
            visdata.coord["freqsel"] = np.asarray([1 for i in xrange(Ndata)])
            paridxes[8] = -1
        if None in paridxes:
            errmsg = "Random Parameters do not have mandatory columns."
            raise ValueError(errmsg)

        # Time
        timeobj = at.Time(val=jd1, val2=jd2, format="jd", scale=timescale)
        timeobj = timeobj.utc
        visdata.coord["utc"] = timeobj.datetime

        # Baseline
        subarr, bl = np.modf(bl)
        visdata.coord["subarray"] = np.int64(100*(subarr)+1)
        visdata.coord["ant1"] = np.int64(bl//256)
        visdata.coord["ant2"] = np.int64(bl%256)

        # Sort Columns
        visdata.coord = visdata.coord[visdata.coord_cols]
        visdata.sort()
        visdata.check()

        self.visdata = visdata

    def _create_ghdu_single(self):
        # Generate Randam Group
        parnames = []
        pardata = []

        # Get some information
        # Baseline
        bl = self.visdata.coord["ant1"] * 256 + self.visdata.coord["ant2"]
        bl += (self.visdata.coord["subarray"] - 1) * 0.01
        # First FREQ SETUP ID
        frqsel = self.subarrays[1].frqsel
        #   Ref Freq
        reffreq = self.subarrays[1].header["FREQ"]
        chwidth = self.fqsetup.fqtables[frqsel].loc[0, "ch_bandwidth"]
        #   RADEC & Equinox
        srcname = self.sources[frqsel].sutable.loc[0, "source"]
        radec = self.sources[frqsel].sutable.loc[0, "radec"]
        equinox = self.sources[frqsel].sutable.loc[0, "equinox"]
        radec = acd.SkyCoord(radec, equinox=equinox, frame="icrs")
        if len(self.sources[frqsel].surestfreq) != 0:
            restfreq = self.sources[frqsel].surestfreq[0, 0]
        else:
            restfreq = 0.
        #   first date of year
        utc = np.datetime_as_string(np.asarray(self.visdata.coord["utc"]))
        utc = at.Time(utc, scale="utc")
        utc_ref = at.Time(datetime.datetime(utc[0].datetime.year,1,1), scale="utc")
        # U
        parnames.append("UU")
        pardata.append(np.asarray(self.visdata.coord["usec"], dtype=np.float32))
        # V
        parnames.append("VV")
        pardata.append(np.asarray(self.visdata.coord["vsec"], dtype=np.float32))
        # W
        parnames.append("WW")
        pardata.append(np.asarray(self.visdata.coord["wsec"], dtype=np.float32))
        # Baseline
        parnames.append("BASELINE")
        pardata.append(np.asarray(bl, dtype=np.float32))
        # DATE
        parnames.append("DATE")
        parnames.append("DATE")
        pardata.append(np.asarray(utc.jd1-utc_ref.jd1, dtype=np.float64))
        pardata.append(np.asarray(utc.jd2-utc_ref.jd2, dtype=np.float64))
        # inttime
        parnames.append("INTTIM")
        pardata.append(np.asarray(self.visdata.coord["inttim"], dtype=np.float32))
        # Frequency Setup Data
        if self.ismultifrq:
            parnames.append("FREQSEL")
            pardata.append(np.asarray(self.visdata.coord["freqsel"], dtype=np.float32))
        # Group HDU
        gdata = pf.GroupData(
            input=np.float32(self.visdata.data),
            parnames=parnames,
            pardata=pardata,
            bscale=1.0,
            bzero=0.0,
            bitpix=-32)
        ghdu = pf.GroupsHDU(gdata)


        # CTYPE HEADER
        cards = []
        # Complex
        cards.append(("CTYPE2","COMPLEX",""))
        cards.append(("CRPIX2",1.0,""))
        cards.append(("CRVAL2",1.0,""))
        cards.append(("CDELT2",1.0,""))
        cards.append(("CROTA2",0.0,""))
        # Stokes
        cards.append(("CTYPE3","STOKES",""))
        cards.append(("CRPIX3",1.0,""))
        cards.append(("CRVAL3",np.float32(stokesDictinv[self.stokes[0]]),""))
        cards.append(("CDELT3",np.float32(np.sign(stokesDictinv[self.stokes[0]])),""))
        cards.append(("CROTA3",0.0,""))
        # FREQ
        self.subarrays[1].header["FREQ"]
        cards.append(("CTYPE4","FREQ",""))
        cards.append(("CRPIX4",1.0,""))
        cards.append(("CRVAL4",reffreq,""))
        cards.append(("CDELT4",chwidth,""))
        cards.append(("CROTA4",0.0,""))
        # Complex
        cards.append(("CTYPE5","IF",""))
        cards.append(("CRPIX5",1.0,""))
        cards.append(("CRVAL5",1.0,""))
        cards.append(("CDELT5",1.0,""))
        cards.append(("CROTA5",0.0,""))
        # RA & Dec
        cards.append(("CTYPE6","RA",""))
        cards.append(("CRPIX6",1.0,""))
        cards.append(("CRVAL6",radec.ra.deg,""))
        cards.append(("CDELT6",1.0,""))
        cards.append(("CROTA6",0.0,""))
        cards.append(("CTYPE7","DEC",""))
        cards.append(("CRPIX7",1.0,""))
        cards.append(("CRVAL7",radec.dec.deg,""))
        cards.append(("CDELT7",1.0,""))
        cards.append(("CROTA7",0.0,""))
        for card in cards:
            ghdu.header.insert("PTYPE1", card)

        # PTYPE HEADER
        for i in xrange(len(parnames)):
            if i == 4:
                pzero = utc_ref.jd1 - 0.5
            elif i == 5:
                pzero = utc_ref.jd2 + 0.5
            else:
                pzero = 0.0
            card = ("PZERO%d"%(i+1), pzero)
            ghdu.header.insert("PTYPE%d"%(i+1), card, after=True)
            card = ("PSCAL%d"%(i+1), 1.0)
            ghdu.header.insert("PTYPE%d"%(i+1), card, after=True)

        # Other Header
        cards = []
        cards.append(("DATE-OBS", utc[0].isot[0:10]))
        cards.append(("TELESCOP", self.subarrays[frqsel].header["ARRNAM"]))
        cards.append(("INSTRUME", self.subarrays[frqsel].header["ARRNAM"]))
        cards.append(("OBSERVER", self.subarrays[frqsel].header["ARRNAM"]))
        cards.append(("OBJECT", srcname))
        cards.append(("EPOCH", equinox))
        cards.append(("BSCALE", 1.0))
        cards.append(("BSZERO", 0.0))
        cards.append(("BUNIT", "UNCALIB"))
        cards.append(("VELREF", 3))
        cards.append(("ALTRVAL", 0.0))
        cards.append(("ALTRPIX", 1.0))
        cards.append(("RESTFREQ", restfreq))
        cards.append(("OBSRA", radec.ra.deg))
        cards.append(("OBSDEC", radec.dec.deg))
        for card in cards:
            ghdu.header.append(card)

        ghdu.header.append()
        return ghdu

    def _create_fqtab(self):
        '''
        Generate FQ Table
        '''
        freqdata = self.fqsetup
        Nfrqsel = len(freqdata.frqsels)

        # Columns
        tables = []
        for i in xrange(Nfrqsel):
            fqtable = freqdata.fqtables[freqdata.frqsels[i]]
            Nif = fqtable.shape[0]
            tables.append(np.asarray(fqtable).transpose().reshape([4,1,Nif]))
        tables = np.concatenate(tables, axis=1)
        c1=pf.Column(
            name="FRQSEL", format='1J', unit=" ",
            array=np.asarray(freqdata.frqsels,dtype=np.int32))
        c2=pf.Column(
            name="IF FREQ", format='%dD'%(Nif), unit="HZ",
            array=np.float64(tables[0]))
        c3=pf.Column(
            name="CH WIDTH", format='%dE'%(Nif), unit="HZ",
            array=np.float32(tables[1]))
        c4=pf.Column(
            name="TOTAL BANDWIDTH", format='%dE'%(Nif), unit="HZ",
            array=np.float32(tables[2]))
        c5=pf.Column(
            name="SIDEBAND", format='%dJ'%(Nif), unit=" ",
            array=np.int16(tables[3]))
        cols = pf.ColDefs([c1, c2, c3, c4, c5])
        hdu = pf.BinTableHDU.from_columns(cols)

        # header for columns
        '''
        hdu.header.comments["TTYPE1"] = "frequency setup ID number"
        hdu.header.comments["TTYPE2"] = "frequency offset"
        hdu.header.comments["TTYPE3"] = "spectral channel separation"
        hdu.header.comments["TTYPE4"] = "total width of spectral wndow"
        hdu.header.comments["TTYPE5"] = "sideband"
        '''

        # keywords
        card = ("EXTNAME","AIPS FQ","")
        hdu.header.insert("TTYPE1", card)
        card = ("EXTVER",1,"")
        hdu.header.insert("TTYPE1", card)
        card = ("EXTLEVEL",1,"")
        hdu.header.insert("TTYPE1", card)
        card = ("NO_IF",np.int32(Nif),"Number IFs")
        hdu.header.append(card=card)

        return hdu

    def _create_antab(self):
        '''
        Generate AN Table
        '''
        hdus = []
        subarrids = self.subarrays.keys()
        for subarrid in subarrids:
            arraydata = self.subarrays[subarrid]
            Nant = arraydata.antable["name"].shape[0]

            # Number of IFs
            if arraydata.header["NO_IF"] is None:
                noif=1
            else:
                noif=arraydata.header["NO_IF"]

            # Columns
            #   ANNAME
            c1=pf.Column(
                name="ANNAME", format='8A', unit=" ",
                array=np.asarray(arraydata.antable["name"],dtype="|S8"))
            #   STABXYZ
            stabxyz = np.zeros([Nant,3],dtype=np.float64)
            stabxyz[:,0] = arraydata.antable["x"]
            stabxyz[:,1] = arraydata.antable["y"]
            stabxyz[:,2] = arraydata.antable["z"]
            c2=pf.Column(
                name="STABXYZ", format='3D', unit="METERS",
                array=stabxyz)
            #   ORBPARM
            c3=pf.Column(
                name="ORBPARM", format='%dD'%(arraydata.header["NUMORB"]), unit=" ",
                array=np.asarray(arraydata.anorbparm,dtype=np.float64))
            #   NOSTA
            c4=pf.Column(
                name="NOSTA", format='1J', unit=" ",
                array=np.asarray(arraydata.antable["id"],dtype=np.int16))
            #   MNTSTA
            c5=pf.Column(
                name="MNTSTA", format='1J', unit=" ",
                array=np.asarray(arraydata.antable["mnttype"],dtype=np.int16))
            #   MNTSTA
            c6=pf.Column(
                name="STAXOF", format='1E', unit="METERS",
                array=np.asarray(arraydata.antable["axisoffset"],dtype=np.float32))
            #   POLTYA
            c7=pf.Column(
                name="POLTYA", format='1A', unit=" ",
                array=np.asarray(arraydata.antable["poltypeA"],dtype="|S1"))
            #   POLTYA
            c8=pf.Column(
                name="POLAA", format='1E', unit="DEGREES",
                array=np.asarray(arraydata.antable["polangA"],dtype=np.float32))
            #   POLTYA
            c9=pf.Column(
                name="POLCALA", format='%dE'%(arraydata.header["NOPCAL"]*noif), unit=" ",
                array=np.asarray(arraydata.anpolcalA,dtype=np.float32))
            #   POLTYA
            c10=pf.Column(
                name="POLTYB", format='1A', unit=" ",
                array=np.asarray(arraydata.antable["poltypeB"],dtype="|S1"))
            #   POLTYA
            c11=pf.Column(
                name="POLAB", format='1E', unit="DEGREES",
                array=np.asarray(arraydata.antable["polangB"],dtype=np.float32))
            #   POLTYA
            c12=pf.Column(
                name="POLCALB", format='%dE'%(arraydata.header["NOPCAL"]*noif), unit=" ",
                array=np.asarray(arraydata.anpolcalB,dtype=np.float32))
            cols = pf.ColDefs([c1, c2, c3, c4, c5, c6,
                            c7, c8, c9, c10, c11, c12])
            hdu = pf.BinTableHDU.from_columns(cols)

            # header for columns
            '''
            hdu.header.comments["TTYPE1"] = "antenna name"
            hdu.header.comments["TTYPE2"] = "antenna station coordinates"
            hdu.header.comments["TTYPE3"] = "orbital parameters"
            hdu.header.comments["TTYPE4"] = "antenna number"
            hdu.header.comments["TTYPE5"] = "mount type"
            hdu.header.comments["TTYPE6"] = "axis offset"
            hdu.header.comments["TTYPE7"] = "feed A: 'R', 'L'"
            hdu.header.comments["TTYPE8"] = "feed A: position angle"
            hdu.header.comments["TTYPE9"] = "feed A: calibration parameters"
            hdu.header.comments["TTYPE10"] = "feed B: 'R', 'L'"
            hdu.header.comments["TTYPE11"] = "feed B: position angle"
            hdu.header.comments["TTYPE12"] = "feed B: calibration parameters"
            '''

            # keywords
            card = ("EXTNAME","AIPS AN","")
            hdu.header.insert("TTYPE1", card)
            card = ("EXTVER",np.int32(arraydata.subarray),"")
            hdu.header.insert("TTYPE1", card)
            card = ("EXTLEVEL",np.int32(arraydata.subarray),"")
            hdu.header.insert("TTYPE1", card)
            #
            keys = "ARRAYX,ARRAYY,ARRAYZ,GSTIA0,DEGPDY,FREQ,RDATE,"
            keys+= "POLARX,POLARY,UT1UTC,DATUTC,TIMSYS,ARRNAM,XYZHAND,FRAME,"
            keys+= "NUMORB,NO_IF,NOPCAL,POLTYPE"
            keys = keys.split(",")
            types = [
                np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,
                str,np.float64,np.float64,np.float64,np.float64,
                str,str,str,str,np.int64,np.int64,np.int64,str,
            ]
            comments = [
                "x coodinates of array center (meters)",
                "y coodinates of array center (meters)",
                "z coodinates of array center (meters)",
                "GST at 0h on reference date (degrees)",
                "Earth's rotation rate (degrees/day)",
                "reference frequency (Hz)",
                "reference date",
                "x coodinates of North Pole (arcseconds)",
                "y coodinates of North Pole (arcseconds)",
                "UT1 - UTC (sec)",
                "time system - UTC (sec)",
                "time system",
                "array name",
                "handedness of station coordinates",
                "coordinate frame",
                "number of orbital parameters in table",
                "number IFs (=Nif)",
                "number of polarization valibration values / Nif",
                "type of polarization calibration",
            ]
            for i in xrange(len(keys)):
                key = keys[i]
                if arraydata.header[key] is not None:
                    #card = (key,types[i](arraydata.header[key]),
                    #        comments[i])
                    card = (key,types[i](arraydata.header[key]))
                    hdu.header.append(card=card)
            #
            if self.ismultifrq:
                card = ("FREQID",np.int32(arraydata.frqsel),"frequency setup number")
                hdu.header.append(card=card)

            # append HDU
            hdus.append(hdu)
        return hdus

    def _create_sutab(self):
        '''
        Generate SU Table
        '''
        hdus = []
        subarrids = self.subarrays.keys()
        for subarrid in subarrids:
            arraydata = self.subarrays[subarrid]
            Nant = arraydata.antable["name"].shape[0]

            # Number of IFs
            if arraydata.header["NO_IF"] is None:
                noif=1
            else:
                noif=arraydata.header["NO_IF"]

            # Columns

            # Create Columns
            cols = []
            for i in xrange(Ncol):
                args = {}
                args["name"] = names[i]
                args["format"] = formats[i]
                if units[i] is not None:
                    args["unit"] = units[i]
                args["array"] = np.asarray(coldata[i],dtype=dtypes[i])
                cols.append(pf.Column(**args))
            cols = fits.ColDefs(cols)

            # create HDU
            hdu = pf.BinTableHDU.from_columns(cols)

            # header for columns
            hdu.header.comments["TTYPE1"] = "source number"
            hdu.header.comments["TTYPE2"] = "source name"
            hdu.header.comments["TTYPE3"] = "source qualifier number"
            hdu.header.comments["TTYPE4"] = "calibration code"
            hdu.header.comments["TTYPE5"] = "Stokes I flux"
            hdu.header.comments["TTYPE6"] = "Stokes Q flux"
            hdu.header.comments["TTYPE7"] = "Stokes U flux"
            hdu.header.comments["TTYPE8"] = "Stokes V flux"
            hdu.header.comments["TTYPE9"] = "frequency offset"
            hdu.header.comments["TTYPE10"] = "spectral channel sepration"
            hdu.header.comments["TTYPE11"] = "RA of equinox"
            hdu.header.comments["TTYPE12"] = "Dec of equinox"
            hdu.header.comments["TTYPE13"] = "equinox"
            hdu.header.comments["TTYPE14"] = "ra of date"
            hdu.header.comments["TTYPE15"] = "dec of date"
            hdu.header.comments["TTYPE16"] = "velocity"
            hdu.header.comments["TTYPE17"] = "rest frequency"
            hdu.header.comments["TTYPE18"] = "proper motion in RA"
            hdu.header.comments["TTYPE19"] = "proper motion in Dec"

            # keywords
            card = ("EXTNAME","AIPS SU","")
            hdu.header.insert("TTYPE1", card)
            card = ("EXTVER",np.int32(srcdata.frqsel),"")
            hdu.header.insert("TTYPE1", card)
            card = ("FREQID",np.int32(arraydata.frqsel),"frequency setup ID number")
            hdu.header.append(card=card)
            card = ("VELDEF",str(arraydata.header["VELDEF"]),"'RADIO' or 'OPTICAL'")
            hdu.header.append(card=card)
            card = ("VELTYP",str(arraydata.header["VELTYP"]),"velocity coordinate reference")
            hdu.header.append(card=card)

            # append HDU
            hdus.append(hdu)
        return hdus


    def avspc(self, dofreq=0, minpoint=2):
        '''
        This method will recalculate sigmas and weights of data from scatter
        in full complex visibilities over specified frequency and time segments.

        Arguments:
          self (uvarray.uvfits object):
            input uvfits data

          dofreq (int; default = 0):
            Parameter for multi-frequency data.
              dofreq = 0: calculate weights and sigmas over IFs and channels
              dofreq = 1: calculate weights and sigmas over channels at each IF

          solint (float; default = 120.):
            solution interval in sec

        Output: uvfits.UVFITS object
        '''
        # Area Settigns
        outfits = copy.deepcopy(self)
        if np.int32(dofreq) > 0.5:
            outfits.visdata.data = np.ascontiguousarray(fortlib.uvdata.avspc_dofreq1(
                    uvdata=np.asarray(self.visdata.data.T, dtype=np.float32))).T
        else:
            outfits.visdata.data = np.ascontiguousarray(fortlib.uvdata.avspc_dofreq0(
                    uvdata=np.asarray(self.visdata.data.T, dtype=np.float32))).T
        return outfits


    def weightcal(self, dofreq=0, solint=120., minpoint=2):
        '''
        This method will recalculate sigmas and weights of data from scatter
        in full complex visibilities over specified frequency and time segments.

        Arguments:
          self (uvarray.uvfits object):
            input uvfits data

          dofreq (int; default = 0):
            Parameter for multi-frequency data.
              dofreq = 0: calculate weights and sigmas over IFs and channels
              dofreq = 1: calculate weights and sigmas over channels at each IF
              dofreq = 2: calculate weights and sigmas at each IF and Channel

          solint (float; default = 120.):
            solution interval in sec

        Output: uvfits.UVFITS object
        '''
        # Save and Return re-weighted uv-data
        outfits = copy.deepcopy(self)
        tsec = at.Time(np.datetime_as_string(outfits.visdata.coord["utc"]), scale="utc").cxcsec
        outfits.visdata.data = np.ascontiguousarray(fortlib.uvdata.weightcal(
                uvdata=np.asarray(self.visdata.data.T, dtype=np.float32),
                tsec=np.array(tsec, dtype=np.float64),
                ant1=np.asarray(self.visdata.coord["ant1"], dtype=np.int32),
                ant2=np.asarray(self.visdata.coord["ant2"], dtype=np.int32),
                subarray=np.asarray(self.visdata.coord["subarray"], dtype=np.int32),
                source=np.asarray(self.visdata.coord["source"], dtype=np.int32),
                solint=np.float64(solint),
                dofreq=np.int32(dofreq),
                minpoint=np.int32(minpoint)).T)
        return outfits


    def weightcal_slow(self, dofreq=0, solint=120., minpoint=2):
        '''
        This method will recalculate sigmas and weights of data from scatter
        in full complex visibilities over specified frequency and time segments.

        Arguments:
          self (uvarray.uvfits object):
            input uvfits data

          dofreq (int; default = 0):
            Parameter for multi-frequency data.
              dofreq = 0: calculate weights and sigmas over IFs and channels
              dofreq = 1: calculate weights and sigmas over channels at each IF
              dofreq = 2: calculate weights and sigmas at each IF and Channel

          solint (float; default = 120.):
            solution interval in sec

        Output: uvfits.UVFITS object
        '''
        # Area Settigns
        doif = True
        doch = True
        if np.int64(dofreq) > 0:
            doif = False
        if np.int64(dofreq) > 1:
            doch = False

        # Sortdata
        #self.visdata.sort(by=["ant1","ant2","subarray", "utc"])

        # Save and Return re-weighted uv-data
        outfits = copy.deepcopy(self)
        # Get size of data
        Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp = outfits.visdata.data.shape
        # Get Timearray
        tsec = at.Time(np.datetime_as_string(outfits.visdata.coord["utc"]), scale="utc").cxcsec
        # Get baseline id
        bl = outfits.visdata.coord["ant1"] * 256
        bl += outfits.visdata.coord["ant2"]
        bl += (outfits.visdata.coord["subarray"] - 1)*0.01

        if doif and doch:
            for idata in tqdm.tqdm(xrange(Ndata)):
                idx = bl - bl[idata] < 0.002      # Baseline
                idx&= tsec - tsec[idata] < solint # Time
                idx = np.where(idx)[0]
                iproduct = itertools.product(xrange(Ndec),
                                             xrange(Nra),
                                             xrange(Nstokes))
                for idec, ira, istokes in iproduct:
                    vreal = outfits.visdata.data[idx, idec, ira, :, :, istokes, 0]
                    vimaj = outfits.visdata.data[idx, idec, ira, :, :, istokes, 1]
                    vweig = outfits.visdata.data[idx, idec, ira, :, :, istokes, 2]

                    Ntime = np.prod(vreal.shape)
                    vreal = vreal.reshape(Ntime)
                    vimaj = vimaj.reshape(Ntime)
                    vweig = vweig.reshape(Ntime)

                    select = vweig > 0
                    select &= (np.isnan(vweig) == False)
                    select &= (np.isinf(vweig) == False)
                    select = np.where(select)
                    vreal = vreal[select]
                    vimaj = vimaj[select]
                    vweig = vweig[select]
                    Nsel = len(vreal)

                    if Nsel < minpoint:
                        outfits.visdata.data[idata, idec, ira, :, :, istokes, 2] = 0.0
                        continue

                    vstack = np.hstack([vreal-vreal.mean(),vimaj-vimaj.mean()])
                    vweig = 1. / np.mean(vstack*vstack)
                    outfits.visdata.data[idata, idec, ira, :, :, istokes, 2] = vweig
        elif doch:
            for idata in tqdm.tqdm(xrange(Ndata)):
                idx = bl - bl[idata] < 0.002      # Baseline
                idx&= tsec - tsec[idata] < solint # Time
                idx = np.where(idx)[0]
                iproduct = itertools.product(xrange(Ndec),
                                             xrange(Nra),
                                             xrange(Nif),
                                             xrange(Nstokes))
                for idec, ira, iif, istokes in iproduct:
                    vreal = outfits.visdata.data[idx, idec, ira, iif, :, istokes, 0]
                    vimaj = outfits.visdata.data[idx, idec, ira, iif, :, istokes, 1]
                    vweig = outfits.visdata.data[idx, idec, ira, iif, :, istokes, 2]

                    Ntime = np.prod(vreal.shape)
                    vreal = vreal.reshape(Ntime)
                    vimaj = vimaj.reshape(Ntime)
                    vweig = vweig.reshape(Ntime)

                    select = vweig > 0
                    select &= (np.isnan(vweig) == False)
                    select &= (np.isinf(vweig) == False)
                    select = np.where(select)
                    vreal = vreal[select]
                    vimaj = vimaj[select]
                    vweig = vweig[select]
                    Nsel = len(vreal)

                    if Nsel < minpoint:
                        outfits.visdata.data[idata, idec, ira, iif, :, istokes, 2] = 0.0
                        continue

                    vstack = np.hstack([vreal-vreal.mean(),vimaj-vimaj.mean()])
                    vweig = 1. / np.mean(vstack*vstack)
                    outfits.visdata.data[idata, idec, ira, iif, :, istokes, 2] = vweig
        else:
            for idata in tqdm.tqdm(xrange(Ndata)):
                idx = bl - bl[idata] < 0.002      # Baseline
                idx&= tsec - tsec[idata] < solint # Time
                idx = np.where(idx)[0]
                iproduct = itertools.product(xrange(Ndec),
                                             xrange(Nra),
                                             xrange(Nif),
                                             xrange(Nch),
                                             xrange(Nstokes))
                for idec, ira, iif, ichan, istokes in iproduct:
                    vreal = outfits.visdata.data[idx, idec, ira, iif, ichan, istokes, 0]
                    vimaj = outfits.visdata.data[idx, idec, ira, iif, ichan, istokes, 1]
                    vweig = outfits.visdata.data[idx, idec, ira, iif, ichan, istokes, 2]

                    Ntime = np.prod(vreal.shape)
                    vreal = vreal.reshape(Ntime)
                    vimaj = vimaj.reshape(Ntime)
                    vweig = vweig.reshape(Ntime)

                    select = vweig > 0
                    select &= (np.isnan(vweig) == False)
                    select &= (np.isinf(vweig) == False)
                    select = np.where(select)
                    vreal = vreal[select]
                    vimaj = vimaj[select]
                    vweig = vweig[select]
                    Nsel = len(vreal)

                    if Nsel < minpoint:
                        outfits.visdata.data[idata, idec, ira, iif, ichan, istokes, 2] = 0.0
                        continue

                    vstack = np.hstack([vreal-vreal.mean(),vimaj-vimaj.mean()])
                    vweig = 1. / np.mean(vstack*vstack)
                    outfits.visdata.data[idata, idec, ira, iif, ichan, istokes, 2] = vweig
        select = self.visdata.data[:, :, :, :, :, :, 2] <= 0.0
        select += np.isnan(self.visdata.data[:, :, :, :, :, :, 2])
        select += np.isinf(self.visdata.data[:, :, :, :, :, :, 2])
        outfits.visdata.data[:, :, :, :, :, :, 2][np.where(select)] = 0.0

        #outfits.visdata.sort()
        #self.visdata.sort()

        return outfits


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
        stokesorg = self.stokes

        # create output data
        outfits = copy.deepcopy(self)
        dshape = list(outfits.visdata.data.shape)
        dshape[5] = 1

        if stokes == "I":
            outfits.stokes = ["I"]
            if ("I" in stokesorg):  # I <- I
                print("Stokes I data will be copied from the input data")
                idx = stokesorg.index("I")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            elif ("RR" in stokesorg) and ("LL" in stokesorg):  # I <- (RR + LL)/2
                print("Stokes I data will be calculated from input RR and LL data")
                idx1 = stokesorg.index("RR")
                idx2 = stokesorg.index("LL")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=0.5)
            elif ("RR" in stokesorg):  # I <- RR
                print("Stokes I data will be copied from input RR data")
                idx = stokesorg.index("RR")
                outfits.visdata.data = self.visdata.data[:, :, :, :, :, idx, :]
            elif ("LL" in stokesorg):  # I <- LL
                print("Stokes I data will be copied from input LL data")
                idx = stokesorg.index("LL")
                outfits.visdata.data = self.visdata.data[:, :, :, :, :, idx, :]
            elif ("XX" in stokesorg) and ("YY" in stokesorg):  # I <- (XX + YY)/2
                print("Stokes I data will be calculated from input XX and YY data")
                idx1 = stokesorg.index("XX")
                idx2 = stokesorg.index("YY")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=0.5)
            elif ("XX" in stokesorg):  # I <- XX
                print("Stokes I data will be copied from input XX data")
                idx = stokesorg.index("XX")
                outfits.visdata.data = self.visdata.data[:, :, :, :, :, idx, :]
            elif ("YY" in stokesorg):  # I <- YY
                print("Stokes I data will be copied from input YY data")
                idx = stokesorg.index("YY")
                outfits.visdata.data = self.visdata.data[:, :, :, :, :, idx, :]
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "Q":
            outfits.stokes = ["Q"]
            if ("Q" in stokesorg):  # Q <- Q
                print("Stokes Q data will be copied from the input data")
                idx = stokesorg.index("Q")
                outfits.visdata.data = self.visdata.data[:, :, :, :, :, idx, :]
            elif ("RL" in stokesorg) and ("LR" in stokesorg):  # Q <- (RL + LR)/2
                print("Stokes Q data will be calculated from input RL and LR data")
                idx1 = stokesorg.index("RL")
                idx2 = stokesorg.index("LR")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=0.5)
            elif ("XX" in stokesorg) and ("YY" in stokesorg):  # Q <- (XX - YY)/2
                print("Stokes Q data will be calculated from input XX and YY data")
                idx1 = stokesorg.index("XX")
                idx2 = stokesorg.index("YY")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=-0.5)
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "U":
            outfits.stokes = ["U"]
            if ("U" in stokesorg):  # U <- U
                print("Stokes U data will be copied from the input data")
                idx = stokesorg.index("U")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            elif ("RL" in stokesorg) and ("LR" in stokesorg):  # U <- (RL - LR)/2i = (- RL + LR)i/2
                print("Stokes U data will be calculated from input RL and LR data")
                idx1 = stokesorg.index("RL")
                idx2 = stokesorg.index("LR")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=-0.5j, factr2=0.5j)
            elif ("XY" in stokesorg) and ("YX" in stokesorg):  # U <- (XY + YX)/2
                print("Stokes U data will be calculated from input XX and YY data")
                idx1 = stokesorg.index("XY")
                idx2 = stokesorg.index("YX")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=0.5)
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "V":
            outfits.stokes = ["V"]
            if ("V" in stokesorg):  # V <- V
                print("Stokes V data will be copied from the input data")
                idx = stokesorg.index("V")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            elif ("RR" in stokesorg) and ("LL" in stokesorg):  # V <- (RR - LL)/2
                print("Stokes V data will be calculated from input RR and LL data")
                idx1 = stokesorg.index("RR")
                idx2 = stokesorg.index("LL")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=-0.5)
            elif ("XY" in stokesorg) and ("YX" in stokesorg):  # V <- (XY - YX)/2i = (-XY + YX)i/2
                print("Stokes V data will be calculated from input XX and YY data")
                idx1 = stokesorg.index("XY")
                idx2 = stokesorg.index("YX")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=-0.5j, factr2=0.5j)
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "RR":
            outfits.stokes = ["RR"]
            if ("RR" in stokesorg):
                print("Stokes RR data will be copied from the input data")
                idx = stokesorg.index("RR")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "LL":
            outfits.stokes = ["LL"]
            if ("LL" in stokesorg):
                print("Stokes LL data will be copied from the input data")
                idx = stokesorg.index("LL")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "RL":
            outfits.stokes = ["RL"]
            if ("RL" in stokesorg):
                print("Stokes RL data will be copied from the input data")
                idx = stokesorg.index("RL")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "LR":
            outfits.stokes = ["LR"]
            if ("LR" in stokesorg):
                print("Stokes LR data will be copied from the input data")
                idx = stokesorg.index("LR")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        else:
            errmsg="[WARNING] Currently Stokes %s is not supported in this function."%(stokes)
            raise ValueError(errmsg)
        outfits.visdata.data = outfits.visdata.data.reshape(dshape)
        return outfits


#-------------------------------------------------------------------------
# Subfunctions for UVFITS
#-------------------------------------------------------------------------
def _bindstokes(data, stokes1, stokes2, factr1, factr2):
    '''
    This is a subfunction for uvdata.UVFITS.
    '''
    vcomp1 = data[:, :, :, :, :, stokes1, 0] + \
        1j * data[:, :, :, :, :, stokes1, 1]
    vweig1 = data[:, :, :, :, :, stokes1, 2]
    vcomp2 = data[:, :, :, :, :, stokes2, 0] + \
        1j * data[:, :, :, :, :, stokes2, 1]
    vweig2 = data[:, :, :, :, :, stokes2, 2]

    vcomp = factr1 * vcomp1 + factr2 * vcomp2
    vweig = np.power(np.abs(factr1)**2 / vweig1 +
                     np.abs(factr2)**2 / vweig2, -1)

    select  = vweig1 <= 0
    select |= vweig2 <= 0
    select |= vweig <= 0
    select |= np.isnan(vweig1)
    select |= np.isnan(vweig2)
    select |= np.isnan(vweig)
    select |= np.isinf(vweig1)
    select |= np.isinf(vweig2)
    select |= np.isinf(vweig)
    vweig[np.where(select)] = 0.0

    outdata = data[:, :, :, :, :, stokes1, :]
    outdata[:, :, :, :, :, 0] = np.real(vcomp)
    outdata[:, :, :, :, :, 1] = np.imag(vcomp)
    outdata[:, :, :, :, :, 2] = vweig
    return outdata

class VisibilityData(object):
    # Default Variables
    def __init__(self):
        self.data = np.zeros([0,1,1,1,1,1,3])

        columns = "utc,usec,vsec,wsec,subarray,ant1,ant2,source,inttim,freqsel"
        self.coord_cols = columns.split(",")
        self.coord = pd.DataFrame(columns=self.coord_cols)

    def check(self):
        # Check Dimension
        Ndim = len(self.data.shape)
        if Ndim != 7:
            errmsg = "VisData.check: Dimension %d is not available (must be 7)."%(Ndim)
            raise ValueError(errmsg)
        if self.data.shape[6]!=3:
            errmsg = "VisData.check: COMPLEX should have NAXIS=3 (currently NAXIS=%d)."%(self.data.shape[6])
            raise ValueError(errmsg)

    def sort(self, by=["utc","ant1","ant2","subarray"]):
        # Check if ant1 > ant2
        self.coord.reset_index(drop=True, inplace=True)
        idx = self.coord["ant1"] > self.coord["ant2"]
        if True in idx:
            self.coord.loc[idx, ["usec", "vsec", "wsec"]] *= -1
            ant2 = self.coord.loc[idx, "ant2"]
            self.coord.loc[idx, "ant2"] = self.coord.loc[idx, "ant1"]
            self.coord.loc[idx, "ant1"] = ant2
            where = np.where(idx)[0]
            self.data[where,:,:,:,:,:,0:2] *= -1
            prt("VisData.sort: %d indexes have wrong station orders (ant1 > ant2)."%(len(where)),indent)
        else:
            prt("VisData.sort: Data have correct station orders (ant1 < ant2).",indent)

        # Sort Data
        self.coord.reset_index(drop=True, inplace=True)
        self.coord = self.coord.sort_values(by=by)
        rows = np.asarray(self.coord.index)
        self.data = self.data[rows,:,:,:,:,:,:]
        self.coord.reset_index(drop=True, inplace=True)
        prt("VisData.sort: Data have been sorted by %s"%(", ".join(by)),indent)


class FrequencyData(object):
    def __init__(self):
        # Frequency Setup Number
        self.frqsels=[]

        # Table
        fqtable_cols="if_freq_offset,ch_bandwidth,if_bandwidth,sideband"
        fqtable_cols=fqtable_cols.split(",")
        self.fqtables={}
        self.fqtable_cols=fqtable_cols

    def __repr__(self):
        lines = []
        for i in xrange(len(self.frqsels)):
            frqsel = self.frqsels[i]
            fqtable = self.fqtables[frqsel]
            lines.append("Frequency Setup ID: %d"%(frqsel))
            lines.append("  IF Freq setups (Hz):")
            lines.append(prt(fqtable,indent*2,output=True))
        lines.append("  Note: Central Frequency of ch=i at IF=j (where i,j=1,2,3...)")
        lines.append("     freq(i,j) = reffreq + (i-1) * ch_bandwidth(j) + if_freq_offset(j)")
        return "\n".join(lines)


class ArrayData(object):
    def __init__(self):
        self.subarray = 1
        self.frqsel = 1     #
        # Initialize Header
        #   Keywords
        keys = ""
        #     Originaly from AN Table Header
        keys+= "SUBARRAY,ARRAYX,ARRAYY,ARRAYZ,GSTIA0,DEGPDY,FREQ,RDATE,"
        keys+= "POLARX,POLARY,UT1UTC,DATUTC,TIMSYS,ARRNAM,XYZHAND,FRAME,"
        keys+= "NUMORB,NO_IF,NOPCAL,POLTYPE"
        keys = keys.split(",")
        #   Initialize
        header = collections.OrderedDict()
        for key in keys:
            header[key] = None
        self.header = header

        # Antenna Information
        self.antable_cols = "id,name,x,y,z,mnttype,axisoffset,"
        self.antable_cols+= "poltypeA,polangA,poltypeB,polangB"
        self.antable_cols = self.antable_cols.split(",")
        self.antable = pd.DataFrame(columns=self.antable_cols)
        self.anorbparm = np.zeros([0,0]) # Nant, Orbital Parmeters
        self.anpolcalA = np.zeros([0,0,0]) # Nant, Npcal, NO_IF
        self.anpolcalB = np.zeros([0,0,0]) # Nant, Npcal, NO_IF

    def __repr__(self):
        lines = []
        lines.append("Sub Array ID: %d"%(self.frqsel))
        lines.append("  Frequency Setup ID: %d"%(self.frqsel))
        lines.append("  Reference Frequency: %.0f Hz"%(self.header["FREQ"]))
        lines.append("  Reference Date: %s"%(self.header["RDATE"]))
        lines.append("  AN Table Contents:")
        lines.append(prt(self.antable["id,name,x,y,z,mnttype".split(",")],indent*2,output=True))
        return "\n".join(lines)

class SourceData(object):
    def __init__(self):
        # Frequency Setup Number
        self.frqsel=1

        # Initialize Header
        #   Keywords
        keys = "NO_IF,VELDEF,VELTYP"
        keys = keys.split(",")
        #   Initialize
        header = collections.OrderedDict()
        for key in keys:
            header[key] = None
        self.header = header

        # Table
        sutable_cols ="id,source,qual,calcode,bandwidth,radec,equinox,"
        sutable_cols+="raapp,decapp,pmra,pmdec"
        sutable_cols=sutable_cols.split(",")
        self.sutable=pd.DataFrame(columns=sutable_cols)
        self.sutable_cols=sutable_cols
        self.suiflux=np.zeros([0,0])
        self.suqflux=np.zeros([0,0])
        self.suuflux=np.zeros([0,0])
        self.suvflux=np.zeros([0,0])
        self.sufreqoff=np.zeros([0,0])
        self.sulsrvel=np.zeros([0,0])
        self.surestfreq=np.zeros([0,0])

    def __repr__(self):
        lines = []
        lines.append("Frequency Setup ID: %d"%(self.frqsel))
        lines.append("  Sources:")
        lines.append(prt(self.sutable["id,source,radec,equinox".split(",")],indent*2,output=True))
        return "\n".join(lines)
