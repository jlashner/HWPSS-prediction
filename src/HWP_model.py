#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:45:24 2018

@author: jlashner
"""

import numpy as np
import thermo as th
from scipy import interpolate
from scipy import optimize
from scipy import integrate as intg
import matplotlib.pyplot as plt
import transfer_matrix as tm
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

def loadMaterials(matFile):
    """
        Loads materials into Tom's code from external file of all applicable materials.
        These are returned as a dictionary.
    """
    mats = {}
    name, no, ne, lto, lte, mtype = np.loadtxt(matFile, dtype=np.str, unpack=True)
    no = np.array(list(map(np.float, no)))
    ne = np.array(list(map(np.float, ne)))
    lto = 1.0e-4 * np.array(list(map(np.float, lto)))
    lte = 1.0e-4 * np.array(list(map(np.float, lte)))
    for (i,n) in enumerate(name):
        mats[n] = tm.material(no[i], ne[i], lto[i], lte[i], n, mtype[i])
    return mats

def loadStack(materials, stackFile):
    """
        Loads a layer stack using materials from input dictionary.
    """
    name, thick, angle = np.loadtxt(stackFile, dtype=np.str, unpack=True)
    mats = [materials[n] for n in name]
    
    thick = np.array(list(map(np.float, thick))) * 1.e-3
    angle = np.array(list(map(np.float, angle)))
    angle = np.deg2rad(angle)
    return tm.Stack(thick, mats, angle)

class HWP:
    def __init__(self, hwpDir, temp, theta, det):
        self.name = "HWP"
        self.temp = temp 
        self.freqs = det.freqs
        
        if det.band_center < 60 * GHz:
            band = "LF"
        elif det.band_center < 200 * GHz:
            band = "MF"
        elif det.band_center < 300 * GHz:
            band = "UHF"
        else:
            print("Can't determine band")
            raise ValueError
            
        hwpDir = os.path.join(hwpDir, band)
        
        matFile = os.path.join(hwpDir, "materials.txt")
        stackFile = os.path.join(hwpDir, "stack.txt")
        
        
        # =============================================================================
        #   Find correct HWP file
        # =============================================================================
        
        theta_deg = int(5 * round(np.rad2deg(theta) / 5))
        datadir = os.path.join(hwpDir, "HWPSS_data/%d_deg/"%(np.rad2deg(theta)))
        
        HWPSS_transFile = os.path.join(datadir, "Trans.npy")
        HWPSS_reflFile = os.path.join(datadir, "Refl.npy")

        # =============================================================================
        #    Initialize materials, stack, and A2/4 fit files.     
        # =============================================================================
        self.theta = theta
        self.det = det
        self.materials = loadMaterials(matFile)
        self.stack = loadStack(self.materials, stackFile)
        
        self.thickness = 0.003
        self.thickness = sum(self.stack.thicknesses[2:-2] )
        
        _, fs, A2upT, A4upT, A2ppT, A4ppT = np.load(HWPSS_transFile)
        _, _, A2upR, A4upR, A2ppR, A4ppR = np.load(HWPSS_reflFile)
        

        # Resample fns to have same length as detector frequency
        self.A2upT, self.A4upT, self.A2ppT, self.A4ppT = [], [], [], []
        self.A2upR, self.A4upR, self.A2ppR, self.A4ppR = [], [], [], []

        for f in det.freqs:
            self.A2upT.append(abs(np.interp(f, fs, A2upT)))
            self.A4upT.append(abs(np.interp(f, fs, A4upT)))
            self.A2ppT.append(abs(np.interp(f, fs, A2ppT)))
            self.A4ppT.append(abs(np.interp(f, fs, A4ppT)))
            self.A2upR.append(abs(np.interp(f, fs, A2upR)))
            self.A4upR.append(abs(np.interp(f, fs, A4upR)))
            self.A2ppR.append(abs(np.interp(f, fs, A2ppR)))
            self.A4ppR.append(abs(np.interp(f, fs, A4ppR)))
        
        self.A2upT, self.A4upT, self.A2ppT, self.A4ppT = map(np.array,  [self.A2upT, self.A4upT, self.A2ppT, self.A4ppT])
        self.A2upR, self.A4upR, self.A2ppR, self.A4ppR = map(np.array,  [self.A2upR, self.A4upR, self.A2ppR, self.A4ppR])
        self.p = None
        

        # =============================================================================
        #    Calculates average trans and reflection Mueller matrices 
        # =============================================================================
        self.MTave = np.zeros((4,4))
        self.MRave = np.zeros((4,4))
        
        for f in det.freqs:
            self.MTave += self.Mueller(f, reflected = False)
            self.MRave += self.Mueller(f, reflected = False)
            
        self.MTave /= len(det.freqs)
        self.MRave /= len(det.freqs)
        
        #==============================================================================
        #   Calculates Efficiency and emission curves
        #==============================================================================
        self.eff = []
        self.emis = []
        self.pemis = []
        for f in self.freqs:
#            JT = tm.Jones(self.stack, f, self.theta, 0.0, reflected = False)
#            JR = tm.Jones(self.stack, f, self.theta, 0.0, reflected = True)

            JT = tm.Jones(self.stack, f, 0.0, 0.0, reflected = False)
            JR = tm.Jones(self.stack, f, 0.0, 0.0, reflected = True)
            
            Sp = np.array([1, 0])
            Ss = np.array([0, 1])
            
            SpT = JT.dot(Sp)
            SpR = JR.dot(Sp)
            SsT = JT.dot(Ss)
            SsR = JR.dot(Ss)
            
            Tp = SpT.dot(SpT.conj().T)
            Rp = SpR.dot(SpR.conj().T)
            Ts = SsT.dot(SsT.conj().T)
            Rs = SsR.dot(SsR.conj().T)    
            Ap = 1 - Tp - Rp
            As = 1 - Ts - Rs
            
            self.eff.append(abs(.5 * (Tp + Ts)))
            self.emis.append(abs(.5 * (Ap + As)))
            self.pemis.append(abs(.5 * (Ap - As)))
        
        self.eff = np.array(self.eff)
        self.emis = np.array(self.emis)
        self.pemis = np.array(self.pemis)
        
    def Eff(self, freq):
        return np.interp(freq, self.freqs, self.eff)

    def pEff(self, freq):
        return self.Eff(freq)
    
    def Emis(self, freq):
        return np.interp(freq, self.freqs, self.emis)
    
    def pEmis(self, freq):
        
        tdo = 2e-6
        tde = 1e-6
        no = 3.05
        ne = 3.38
        
        ao = 2 * np.pi  * no * tdo * freq / (3e8) #1/m
        ae = 2 * np.pi  * ne * tde * freq / (3e8) #1/m
        
        Potrans =  1 - np.exp(-ao * self.thickness) 
        Petrans =  1 - np.exp(-ae * self.thickness)
        pemis = abs(Petrans - Potrans) / 2            
        return pemis

        
    def Ip(self, freq):
        return 0
    
    def Refl(self, freq):
        return 0
    
    def pRefl(self, freq):
        return 0
        
    def Mueller(self, freq, reflected = False):
        """
            Returns the Mueller matrix for the HWP at the specified frequency 
        """
        return tm.Mueller(self.stack, freq, self.theta, 0.0, reflected = reflected)            
        
    def getHWPSS(self, freq, stokes, reflected = False, fit = False):
        """
            Returns A2 and A4 amplitudes generated by given incident (or reflected) stokes vector.
            
            If fit is true, returns fit from file.
        """
        if not fit:
            if not reflected:
                A4 = abs(stokes[1] * np.interp(freq, self.freqs, self.A4ppT)) + abs((stokes[0]-stokes[1]) * np.interp(freq, self.freqs, self.A4upT))
                A2 = abs(stokes[1] * np.interp(freq, self.freqs, self.A2upT)) + abs((stokes[0]-stokes[1]) * np.interp(freq, self.freqs, self.A2upT))
            else:
                A4 = abs(stokes[1] * np.interp(freq, self.freqs, self.A4ppR)) + abs((stokes[0]-stokes[1]) * np.interp(freq, self.freqs, self.A4upR))
                A2 = abs(stokes[1] * np.interp(freq, self.freqs, self.A2upR)) + abs((stokes[0]-stokes[1]) * np.interp(freq, self.freqs, self.A2upR))

        else:
            self.p = fitAmplitudes(self.stack, freq, self.theta, stokes, reflected = reflected, p = self.p)
            A2, A4 = abs(self.p[2]), abs(self.p[4])

        return A2, A4
    
    def fit1 (self, freq, stokes, reflected = False):
        p= fitAmplitudes(self.stack, freq, self.theta, stokes, reflected = reflected)
        return abs(p[2]), abs(p[4])
        
