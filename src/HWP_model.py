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


class HWP:
    def __init__(self, hwpDir, temp, theta, det):
        self.name = "HWP"
        self.temp = temp 
        self.freqs = det.freqs
        matFile = os.path.join(hwpDir, "materials.txt")
        stackFile = os.path.join(hwpDir, "stack.txt")
        
        
        # =============================================================================
        #   Find correct HWP file
        # =============================================================================
        
        #Right now we only have theta = 0,5,10,15,20 deg
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
        return np.interp(freq, self.freqs, self.pemis)
        
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
            I = stokes[0]
            Q = stokes[1]
            
            M = self.Mueller(freq, reflected = reflected)
            
            
            A2 = .5 * ((I * M[1,0] + Q * M[0,1])**2 + (I * M[2,0] + Q * M[0,2])**2)**.5
            A4 = .25 * (Q * Q * ((M[1,2] + M[2,1])**2 + (M[1,1] - M[2,2])**2))**.5
            
        else:           
#            p= fitAmplitudes(self.stack, freq, self.theta, stokes, reflected = reflected)
#            A2, A4 = abs(p[2]), abs(p[4])
            if not reflected:
                A4 = abs(stokes[1] * np.interp(freq, self.freqs, self.A4ppT)) + abs((stokes[0]-stokes[1]) * np.interp(freq, self.freqs, self.A4upT))
                A2 = abs(stokes[1] * np.interp(freq, self.freqs, self.A2upT)) + abs((stokes[0]-stokes[1]) * np.interp(freq, self.freqs, self.A2upT))
            else:
                A4 = abs(stokes[1] * np.interp(freq, self.freqs, self.A4ppR)) + abs((stokes[0]-stokes[1]) * np.interp(freq, self.freqs, self.A4upR))
                A2 = abs(stokes[1] * np.interp(freq, self.freqs, self.A2upR)) + abs((stokes[0]-stokes[1]) * np.interp(freq, self.freqs, self.A2upR))
            
        return A2, A4
    
    def fit1 (self, freq, stokes, reflected = False):
        p= fitAmplitudes(self.stack, freq, self.theta, stokes, reflected = reflected)
        return abs(p[2]), abs(p[4])
        

        
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
        p = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        
    popt, pcov = optimize.curve_fit(demodFit, chis, demod, p)    
    return popt

def fitAmplitudesBand(stack, freqs, theta, stokes, reflected = False):
    """
    
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

def calcHWPSSCoeffs(theta = 0.0, reflected = False):
    
    labels = np.array(["freqs", "A2up", "A4up", "A2pp", "A4pp"])
    freqs= np.linspace(50 * GHz, 200 * GHz, 200)
    
    A2up, A4up = fitAmplitudesBand(stack, freqs, theta, stokes = np.array([1,0,0,0]), reflected = reflected)
    A2pp, A4pp = fitAmplitudesBand(stack, freqs, theta, stokes = np.array([1,1,0,0]), reflected = reflected)
    
    return np.array([labels, freqs, A2up, A4up, A2pp, A4pp])



if __name__ == "__main__":

#    for theta in [0, 5, 10, 15, 20]:
#        data = np.load("../HWP/3LayerSapphire/HWPSS_data/%s_deg/Trans.npy"%theta)  
#        print(np.mean(data[3]))
#        plt.plot(data[1], data[5])
#
#    plt.show()


    hwpdir = "../HWP/3LayerSapphire/"
    datadir = os.path.join(hwpdir + "HWPSS_data/")
    materials = loadMaterials(os.path.join(hwpdir, "materials.txt"))
    stack       = loadStack(materials, os.path.join(hwpdir, "stack.txt"))      
    
    for theta in range(21):
        path = datadir + "%d_deg/"%(theta)

        if (os.path.exists(os.path.join(path, "Refl.npy"))):
            print("Skipping")
            continue
        os.makedirs(path, exist_ok = True)
            
        data = calcHWPSSCoeffs(theta = np.deg2rad(theta), reflected = False)
        np.save(path + "Trans", data)
        data = calcHWPSSCoeffs(theta = np.deg2rad(theta), reflected = True)
        np.save(path + "Refl", data)
##        
#    data = calcHWPSSCoeffs()
    
    
#    freqs= np.linspace(50 * GHz, 200 * GHz, 200)
#    A2upT, A4upT = fitAmplitudesBand(stack, freqs, np.deg2rad(0.), stokes = np.array([1,0,0,0]), reflected = False)
#    A2ppT, A4ppT = fitAmplitudesBand(stack, freqs, np.deg2rad(0.), stokes = np.array([1,1,0,0]), reflected = False)
#    
#    A2upR, A4upR = fitAmplitudesBand(stack, freqs, np.deg2rad(0.), stokes = np.array([1,0,0,0]), reflected = True)
#    A2ppR, A4ppR = fitAmplitudesBand(stack, freqs, np.deg2rad(0.), stokes = np.array([1,1,0,0]), reflected = True)
#    
#    labels = np.array(["freqs", "A2up", "A4up", "A2pp", "A4pp"])
#    
#    data1 = np.array([labels, freqs, A2upT, A4upT, A2ppT, A4ppT])
#    np.save("../HWP/3LayerSapphire/HWPSS_Trans_0deg",data1)
#    
#    data2 = np.array([labels, freqs, A2upR, A4upR, A2ppR, A4ppR])
#    np.save("../HWP/3LayerSapphire/HWPSS_Refl_0deg",data2)
    
    
#    plt.plot(freqs, A4ppT)
#    plt.show()
    
    
    
#    