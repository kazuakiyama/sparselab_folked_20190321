#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a submodule of sparselab for imaging static images.
'''
__author__ = "Sparselab Developer Team"
# -------------------------------------------------------------------------
# Modules
# -------------------------------------------------------------------------
# standard modules
import os
import copy
import collections
import itertools

# numerical packages
import numpy as np
import pandas as pd

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# internal modules
from .. import util, imdata, fortlib

#-------------------------------------------------------------------------
# Default Parameters
#-------------------------------------------------------------------------
lbfgsbprms = {
    "m": 5,
    "factr": 1e1,
    "pgtol": 0.
}


#-------------------------------------------------------------------------
# Reconstract static imaging
#-------------------------------------------------------------------------
def imaging3d(
        initimage,
        #Nuvs,
        #u, v,
        #uvidxfcv, uvidxamp, uvidxcp, uvidxca,
        Nf=1,
        imagewin=None,
        vistable=None,amptable=None, bstable=None, catable=None,
        lambl1=-1.,lambtv=-1.,lambtsv=-1.,lambmem=-1.,lambcom=-1.,normlambda=True,
        niter=1000,
        nonneg=True,
        transform=None, transprm=None,
        compower=1.,
        totalflux=None, fluxconst=False,
        istokes=0, ifreq=0):
    '''

    '''
    # Sanity Check: Data
    if ((vistable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # Sanity Check: Total Flux constraint
    dofluxconst = False
    if ((vistable is None) and (amptable is None) and (totalflux is None)):
        print("Error: No absolute amplitude information in the input data.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1
    elif ((vistable is None) and (amptable is None) and
          (totalflux is not None) and (fluxconst is False)):
        print("Warning: No absolute amplitude information in the input data.")
        print("         The total flux will be constrained, although you do not set fluxconst=True.")
        dofluxconst = True
    elif fluxconst is True:
        dofluxconst = True

    # Sanity Check: Transform
    if transform is None:
        print("No transform will be applied to regularization functions.")
        transtype = np.int32(0)
        transprm = np.float64(0)
    elif transform == "log":
        print("log transform will be applied to regularization functions.")
        transtype = np.int32(1)
        if transprm is None:
            transprm = 1e-10
        elif transprm <= 0:
            raise ValueError("transprm must be positive.")
        else:
            transprm = np.float64(transprm)
        print("  threshold of log transform: %g"%(transprm))
    elif transform == "gamma":
        print("Gamma transform will be applied to regularization functions.")
        transtype = np.int32(2)
        if transprm is None:
            transprm = 1/2.2
        elif transprm <= 0:
            raise ValueError("transprm must be positive.")
        else:
            transprm = np.float64(transprm)
        print("  Power of Gamma correction: %g"%(transprm))

    # get initial images
    Iin = np.float64(initimage.data[istokes, ifreq])

    # size of images
    Nx = initimage.header["nx"]
    Ny = initimage.header["ny"]
    Nyx = Nx * Ny

    # pixel coordinates
    x, y = initimage.get_xygrid(twodim=True, angunit="rad")
    xidx = np.arange(Nx) + 1
    yidx = np.arange(Ny) + 1
    xidx, yidx = np.meshgrid(xidx, yidx)
    Nxref = initimage.header["nxref"]
    Nyref = initimage.header["nyref"]
    dx_rad = np.deg2rad(initimage.header["dx"])
    dy_rad = np.deg2rad(initimage.header["dy"])

    # apply the imaging area
    if imagewin is None:
        print("Imaging Window: Not Specified. We solve the image on all the pixels.")
        Iin = Iin.reshape(Nyx)
        x = x.reshape(Nyx)
        y = y.reshape(Nyx)
        xidx = xidx.reshape(Nyx)
        yidx = yidx.reshape(Nyx)
    else:
        print("Imaging Window: Specified. Images will be solved on specified pixels.")
        idx = np.where(imagewin)
        Iin = Iin[idx]
        x = x[idx]
        y = y[idx]
        xidx = xidx[idx]
        yidx = yidx[idx]

    # dammy array
    dammyreal = np.zeros(1, dtype=np.float64)

    if totalflux is None:
        totalflux = []
        if vistable is not None:
            totalflux.append(vistable["amp"].max())
        if amptable is not None:
            totalflux.append(amptable["amp"].max())
        totalflux = np.max(totalflux)

    # Full Complex Visibility
    Ndata = 0
    if dofluxconst:
        print("Total Flux Constraint: set to %g" % (totalflux))
        totalfluxdata = {
            'u': [0.],
            'v': [0.],
            'amp': [totalflux],
            'phase': [0.],
            'sigma': [1.]
        }
        totalfluxdata = pd.DataFrame(totalfluxdata)
        fcvtable = pd.concat([totalfluxdata, vistable], ignore_index=True)
    else:
        print("Total Flux Constraint: disabled.")
        if vistable is None:
            fcvtable = None
        else:
            fcvtable = vistable.copy()

    if fcvtable is None:
        isfcv = False
        vfcvr = dammyreal
        vfcvi = dammyreal
        varfcv = dammyreal
    else:
        isfcv = True
        phase = np.deg2rad(np.array(fcvtable["phase"], dtype=np.float64))
        amp = np.array(fcvtable["amp"], dtype=np.float64)
        vfcvr = np.float64(amp*np.cos(phase))
        vfcvi = np.float64(amp*np.sin(phase))
        varfcv = np.square(np.array(fcvtable["sigma"], dtype=np.float64))
        Ndata += len(varfcv)
        del phase, amp

    # Visibility Amplitude
    if amptable is None:
        isamp = False
        vamp = dammyreal
        varamp = dammyreal
    else:
        isamp = True
        vamp = np.array(amptable["amp"], dtype=np.float64)
        varamp = np.square(np.array(amptable["sigma"], dtype=np.float64))
        Ndata += len(vamp)

    # Closure Phase
    if bstable is None:
        iscp = False
        cp = dammyreal
        varcp = dammyreal
    else:
        iscp = True
        cp = np.deg2rad(np.array(bstable["phase"], dtype=np.float64))
        varcp = np.square(
            np.array(bstable["sigma"] / bstable["amp"], dtype=np.float64))
        Ndata += len(cp)

    # Closure Amplitude
    if catable is None:
        isca = False
        ca = dammyreal
        varca = dammyreal
    else:
        isca = True
        ca = np.array(catable["logamp"], dtype=np.float64)
        varca = np.square(np.array(catable["logsigma"], dtype=np.float64))
        Ndata += len(ca)

    # Sigma for the total flux
    if dofluxconst:
        varfcv[0] = np.square(fcvtable.loc[0, "amp"] / (Ndata - 1.))

    # Normalize Lambda
    if normlambda:
        fluxscale = np.float64(totalflux)

        # convert Flux Scaling Factor
        fluxscale = np.abs(fluxscale) / Nyx
        if   transform=="log":   # log correction
            fluxscale = np.log(fluxscale+transprm)-np.log(transprm)
        elif transform=="gamma": # gamma correction
            fluxscale = (fluxscale)**transprm

        lambl1_sim = lambl1 / (fluxscale * Nyx)
        lambtv_sim = lambtv / (4 * fluxscale * Nyx)
        lambtsv_sim = lambtsv / (4 *fluxscale**2 * Nyx)
        lambmem_sim = lambmem / (fluxscale*np.log(fluxscale) * Nyx)
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv
        lambmem_sim = lambmem

    # Center of Mass regularization
    lambcom_sim = lambcom # No normalization for COM regularization

    # get uv coordinates and uv indice
    u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca, Nuvs = get_uvlist_loop(Nf=Nf,
        fcvconcat=fcvtable, ampconcat=amptable, bsconcat=bstable, caconcat=catable
    )

    # normalize u, v coordinates
    u *= 2*np.pi*dx_rad
    v *= 2*np.pi*dy_rad

    # copy the initimage to the number of frames
    Iin = np.concatenate([Iin]*Nf)
    xidx = np.concatenate([xidx]*Nf)
    yidx = np.concatenate([yidx]*Nf)

    # run imaging
    Iout = fortlib.fftim3d.imaging(
        # Images
        iin=np.float64(Iin),
        xidx=np.int32(xidx),
        yidx=np.int32(yidx),
        nxref=np.float64(Nxref),
        nyref=np.float64(Nyref),
        nx=np.int32(Nx),
        ny=np.int32(Ny),
        # 3D frames
        nz=np.int32(Nf),
        # UV coordinates,
        u=u,
        v=v,
        nuvs=np.int32(Nuvs),
        # Regularization Parameters
        lambl1=np.float64(lambl1_sim),
        lambtv=np.float64(lambtv_sim),
        lambtsv=np.float64(lambtsv_sim),
        lambmem=np.float64(lambmem_sim),
        lambcom=np.float64(lambcom_sim),
        # Imaging Parameter
        niter=np.int32(niter),
        nonneg=nonneg,
        transtype=np.int32(transtype),
        transprm=np.float64(transprm),
        pcom=np.float64(compower),
        # Full Complex Visibilities
        isfcv=isfcv,
        uvidxfcv=np.int32(uvidxfcv),
        vfcvr=np.float64(vfcvr),
        vfcvi=np.float64(vfcvi),
        varfcv=np.float64(varfcv),
        # Visibility Ampltiudes
        isamp=isamp,
        uvidxamp=np.int32(uvidxamp),
        vamp=np.float64(vamp),
        varamp=np.float64(varamp),
        # Closure Phase
        iscp=iscp,
        uvidxcp=np.int32(uvidxcp),
        cp=np.float64(cp),
        varcp=np.float64(varcp),
        # Closure Amplituds
        isca=isca,
        uvidxca=np.int32(uvidxca),
        ca=np.float64(ca),
        varca=np.float64(varca),
        # Following 3 parameters are for L-BFGS-B
        m=np.int32(lbfgsbprms["m"]), factr=np.float64(lbfgsbprms["factr"]),
        pgtol=np.float64(lbfgsbprms["pgtol"])
    )
    '''
    outimage = copy.deepcopy(initimage)
    outimage.data[istokes, ifreq] = 0.
    for i in np.arange(len(xidx)):
        outimage.data[istokes, ifreq, yidx[i] - 1, xidx[i] - 1] = Iout[i]
    outimage.update_fits()
    '''
    # multiple outimage
    outimlist = []
    ipix = 0
    iz = 0
    while iz < Nf:
        outimage = copy.deepcopy(initimage)
        outimage.data[istokes, ifreq] = 0.
        for i in np.arange(Nyx):
            outimage.data[istokes, ifreq, yidx[i]-1, xidx[i]-1] = Iout[ipix+i]
        outimage.update_fits()
        outimlist.append(outimage)
        ipix += Nyx-1
        iz += 1

    return outimlist


# ------------------------------------------------------------------------------
# Subfunctions
# ------------------------------------------------------------------------------
def get_uvlist_loop(Nf, fcvconcat=None, ampconcat=None, bsconcat=None, caconcat=None):
    '''
    '''
    if ((fcvconcat is None) and (ampconcat is None) and
            (bsconcat is None) and (caconcat is None)):
        print("Error: No data are input.")
        return -1

    u, v = [], []
    uvidxfcv, uvidxamp, uvidxcp, uvidxca = [], [], [], []
    Nuvs = []
    for i in np.arange(Nf):
        fcvsingle, ampsingle, bssingle, casingle = None, None, None, None
        if fcvconcat is not None:
            frmidx = fcvconcat["frmidx"]
            idx = np.where(frmidx == i) #tuple
            idx = idx[0].tolist() #list
            if idx != []:
                fcvsingle = fcvconcat.loc[idx, :]

        if ampconcat is not None:
            frmidx = ampconcat["frmidx"]
            idx = np.where(frmidx == i) #tuple
            idx = idx[0].tolist() #list
            if idx != []:
                ampsingle = ampconcat.loc[idx, :]

        if bsconcat is not None:
            frmidx = bsconcat["frmidx"]
            idx = np.where(frmidx == i) #tuple
            idx = idx[0].tolist() #list
            if idx != []:
                bssingle = bsconcat.loc[idx, :]

        if caconcat is not None:
            frmidx = caconcat["frmidx"]
            idx = np.where(frmidx == i) #tuple
            idx = idx[0].tolist() #list
            if idx != []:
                casingle = caconcat.loc[idx, :]

        if ((fcvsingle is not None) or (ampsingle is not None) or
                (bssingle is not None) or (casingle is not None)):
            u0, v0, uvidxfcv0, uvidxamp0, uvidxcp0, uvidxca0 = get_uvlist(
                fcvtable=fcvsingle, amptable=ampsingle, bstable=bssingle, catable=casingle)
            u.append(u0)
            v.append(v0)
            Nuvs.append(len(u0))
            if i == 0:
                uvidxfcv.append(uvidxfcv0)
                uvidxamp.append(uvidxamp0)
                uvidxcp.append(uvidxcp0)
                uvidxca.append(uvidxca0)
            else:
                uvidxfcv.append(uvidxfcv0+idxcon)
                uvidxamp.append(uvidxamp0+idxcon)
                uvidxcp.append(uvidxcp0+idxcon)
                uvidxca.append(uvidxca0+idxcon)
            idxcon = len(u0)
        if ((fcvsingle is None) and (ampsingle is None) and
                (bssingle is None) and (casingle is None)):
            Nuvs.append(0)

    u = np.concatenate(u)
    v = np.concatenate(v)
    uvidxfcv = np.concatenate(uvidxfcv)
    uvidxamp = np.concatenate(uvidxamp)
    uvidxcp = np.hstack(uvidxcp)
    uvidxca = np.hstack(uvidxca)

    if fcvconcat is None:
        uvidxfcv = np.zeros(1, dtype=np.int32)
    if ampconcat is None:
        uvidxamp = np.zeros(1, dtype=np.int32)
    if bsconcat is None:
        uvidxcp = np.zeros([3, 1], dtype=np.int32, order="F")
    if caconcat is None:
        uvidxca = np.zeros([4, 1], dtype=np.int32, order="F")

    #print("uvidxfcv: ", type(uvidxfcv), uvidxfcv.shape)
    #print("uvidxamp: ", type(uvidxamp), uvidxamp.shape)
    #print("uvidxcp: ", type(uvidxcp), uvidxcp.shape)
    #print("uvidxca: ", type(uvidxca), uvidxca.shape)

    return (u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca, Nuvs)

def get_uvlist(fcvtable=None, amptable=None, bstable=None, catable=None, thres=1e-2):
    '''

    '''
    if ((fcvtable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # Stack uv coordinates
    ustack = None
    vstack = None
    if fcvtable is not None:
        ustack = np.array(fcvtable["u"], dtype=np.float64)
        vstack = np.array(fcvtable["v"], dtype=np.float64)
        Nfcv = len(ustack)
    else:
        Nfcv = 0

    if amptable is not None:
        utmp = np.array(amptable["u"], dtype=np.float64)
        vtmp = np.array(amptable["v"], dtype=np.float64)
        Namp = len(utmp)
        if ustack is None:
            ustack = utmp
            vstack = vtmp
        else:
            ustack = np.concatenate((ustack, utmp))
            vstack = np.concatenate((vstack, vtmp))
    else:
        Namp = 0

    if bstable is not None:
        utmp1 = np.array(bstable["u12"], dtype=np.float64)
        vtmp1 = np.array(bstable["v12"], dtype=np.float64)
        utmp2 = np.array(bstable["u23"], dtype=np.float64)
        vtmp2 = np.array(bstable["v23"], dtype=np.float64)
        utmp3 = np.array(bstable["u31"], dtype=np.float64)
        vtmp3 = np.array(bstable["v31"], dtype=np.float64)
        Ncp = len(utmp1)
        if ustack is None:
            ustack = np.concatenate((utmp1, utmp2, utmp3))
            vstack = np.concatenate((vtmp1, vtmp2, vtmp3))
        else:
            ustack = np.concatenate((ustack, utmp1, utmp2, utmp3))
            vstack = np.concatenate((vstack, vtmp1, vtmp2, vtmp3))
    else:
        Ncp = 0

    if catable is not None:
        utmp1 = np.array(catable["u1"], dtype=np.float64)
        vtmp1 = np.array(catable["v1"], dtype=np.float64)
        utmp2 = np.array(catable["u2"], dtype=np.float64)
        vtmp2 = np.array(catable["v2"], dtype=np.float64)
        utmp3 = np.array(catable["u3"], dtype=np.float64)
        vtmp3 = np.array(catable["v3"], dtype=np.float64)
        utmp4 = np.array(catable["u4"], dtype=np.float64)
        vtmp4 = np.array(catable["v4"], dtype=np.float64)
        Nca = len(utmp1)
        if ustack is None:
            ustack = np.concatenate((utmp1, utmp2, utmp3, utmp4))
            vstack = np.concatenate((vtmp1, vtmp2, vtmp3, vtmp4))
        else:
            ustack = np.concatenate((ustack, utmp1, utmp2, utmp3, utmp4))
            vstack = np.concatenate((vstack, vtmp1, vtmp2, vtmp3, vtmp4))
    else:
        Nca = 0

    # make non-redundant u,v lists and index arrays for uv coordinates.
    Nstack = Nfcv + Namp + 3 * Ncp + 4 * Nca
    uvidx = np.zeros(Nstack, dtype=np.int32)
    maxidx = 1
    u = []
    v = []
    uvstack = np.sqrt(np.square(ustack) + np.square(vstack))
    uvthres = np.max(uvstack) * thres
    for i in np.arange(Nstack):
        if uvidx[i] == 0:
            dist1 = np.sqrt(
                np.square(ustack - ustack[i]) + np.square(vstack - vstack[i]))
            dist2 = np.sqrt(
                np.square(ustack + ustack[i]) + np.square(vstack + vstack[i]))
            #uvdist = np.sqrt(np.square(ustack[i])+np.square(vstack[i]))

            #t = np.where(dist1<uvthres)
            t = np.where(dist1 < thres * (uvstack[i] + 1))
            uvidx[t] = maxidx
            #t = np.where(dist2<uvthres)
            t = np.where(dist2 < thres * (uvstack[i] + 1))
            uvidx[t] = -maxidx
            u.append(ustack[i])
            v.append(vstack[i])
            maxidx += 1
    u = np.asarray(u)  # Non redundant u coordinates
    v = np.asarray(v)  # Non redundant v coordinates

    # distribute index information into each data
    if fcvtable is None:
        uvidxfcv = np.zeros(1, dtype=np.int32)
    else:
        uvidxfcv = uvidx[0:Nfcv]

    if amptable is None:
        uvidxamp = np.zeros(1, dtype=np.int32)
    else:
        uvidxamp = uvidx[Nfcv:Nfcv + Namp]

    if bstable is None:
        uvidxcp = np.zeros([3, 1], dtype=np.int32, order="F")
    else:
        uvidxcp = uvidx[Nfcv + Namp:Nfcv + Namp + 3 *
                        Ncp].reshape([Ncp, 3], order="F").transpose()

    if catable is None:
        uvidxca = np.zeros([4, 1], dtype=np.int32, order="F")
    else:
        uvidxca = uvidx[Nfcv + Namp + 3 * Ncp:Nfcv + Namp + 3 *
                        Ncp + 4 * Nca].reshape([Nca, 4], order="F").transpose()
    #print("u: ", type(u), u.shape)
    #print("v: ", type(v), v.shape)
    #print("uvidxamp: ", type(uvidxamp), uvidxamp.shape)
    #print("uvidxcp: ", type(uvidxcp), uvidxcp.shape)
    return (u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca)
