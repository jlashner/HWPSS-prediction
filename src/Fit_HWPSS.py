# -*- coding: utf-8 -*-

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:45:24 2018

This script fits the demodulated HWP signal to a sum of HWP harmonics. 
It figures out how much (un)polarized signal is modulated into the A2 and A4 
bands for each frequency and saves this to the HWPSS folder.

This data is required for the HWPSS prediction.

@author: jlashner
"""

import numpy as np
import thermo as th
from scipy import interpolate
from scipy import optimize
from scipy import integrate as intg
import matplotlib.pyplot as plt
import transfer_matrix as tm
from HWP_model import loadMaterials, loadStack
import Telescope as tp
import os
import json
import logging as log

try:
    from tqdm import *
except:
    tqdm = lambda x : x
#==============================================================================
# Constants
#==============================================================================
GHz = 1.0e9    # HZ -> GHz
c = 2.99792e8  # [m/s]

        
def demodFit(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, p1, p2, p3, p4, p5, p6, p7, p8):
    """
        Function with amplitudes and phases which are fit to the demodulated signal.
    """
    d  = a0 / 2
    d += a1 * np.cos(1 * x + p1)
    d += a2 * np.cos(2 * x + p2)
    d += a3 * np.cos(3 * x + p3)
    d += a4 * np.cos(4 * x + p4)
    d += a5 * np.cos(5 * x + p5)
    d += a6 * np.cos(6 * x + p6)
    d += a7 * np.cos(7 * x + p7)
    d += a8 * np.cos(8 * x + p8)
    return d


def fitAmplitudes(stack, freq, theta, stokes = np.array([1, 0, 0, 0]), reflected = False,  p = None):
    """
        Given a stack, frequency, and incident angle, this fits amplitudes and phases to the demodulated signal.
        Returns A2 and A4.
    """
    det = .5 * np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])    
    chis = np.linspace(0, 2 * np.pi, 100)
    demod = []
    for chi in chis:
        M = tm.Mueller(stack, freq, theta, chi, reflected = reflected)
        demod.append(det.dot(M).dot(stokes)[0])
    if type(p) != None:
        # Initial guess for fit
        p = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        
    popt, pcov = optimize.curve_fit(demodFit, chis, demod, p)    
    return popt

def fitAmplitudesBand(stack, freqs, theta, stokes, reflected = False):
    """
        Fits A2 and A4 coefficients for a range of frequencies.
    """
    popt = None
    A2 = []
    A4 = []
    print("Fitting amplitude band: theta = %.1f, reflected=%s, stokes = %s"%(theta, reflected, str(stokes)))
    for f in tqdm(freqs):
#        print(f/ GHz)
        popt = fitAmplitudes(stack, f, theta, stokes = stokes, reflected = reflected, p = popt)
        A2.append(popt[2])
        A4.append(popt[4])
    return np.array(A2), np.array(A4)

def calcHWPSSCoeffs(theta = 0.0, reflected = False, band = "MF"):
    labels = np.array(["freqs", "A2up", "A4up", "A2pp", "A4pp"])
    
    if band == "LF":
        freqs = np.linspace(5 * GHz, 60 * GHz, 200)
    elif band == "MF":
        freqs= np.linspace(50 * GHz, 200 * GHz, 200)
    elif band == "UHF":
        freqs = np.linspace(150 * GHz, 330 * GHz, 200)
    else:
        print("Band must be LF, MF, or UHF")
        raise ValueError
    
    A2up, A4up = fitAmplitudesBand(stack, freqs, theta, stokes = np.array([1,0,0,0]), reflected = reflected)
    A2pp, A4pp = fitAmplitudesBand(stack, freqs, theta, stokes = np.array([1,1,0,0]), reflected = reflected)
    
    return np.array([labels, freqs, A2up, A4up, A2pp, A4pp])


if __name__ == "__main__":
    HWP_dir = "/Users/jacoblashner/so/HWPSS-prediction/HWP/4LayerSapphire/MF/"
    datadir = os.path.join(HWP_dir, "HWPSS")
    band = "MF"
    
    mats = loadMaterials(os.path.join(HWP_dir, "materials.txt"))
    stack = loadStack(mats, os.path.join(HWP_dir, "stack.txt"))
    
    print(stack)
    
    freqs = np.linspace(50*GHz, 200*GHz, 100)
    A2, A4 = fitAmplitudesBand(stack, freqs, 0, np.array([1,0,0,0]), reflected = False)
    
    plt.plot(freqs, np.abs(A2))    
    
    
#    
#    for theta in [0, 20]:#range(21):
#        path = os.path.join(datadir, "{}_deg".format(theta))
#        
#        if (os.path.exists(os.path.join(path, "Refl.npy"))):
#            print("Skipping")
#            continue
#        os.makedirs(path, exist_ok = True)
#   
#        data = calcHWPSSCoeffs(theta = np.deg2rad(theta), reflected = False, band = band)
#        trans_fname = os.path.join(path, "Trans")ls
    
#        np.save(trans_fname, data)
#        
#        data = calcHWPSSCoeffs(theta = np.deg2rad(theta), reflected = True, band = band)
#        refl_fname = os.path.join(path, "Refl")
#        np.save(refl_fname, data)