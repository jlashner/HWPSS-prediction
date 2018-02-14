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
import os
import logging as log
#==============================================================================
# Constants
#==============================================================================
GHz = 1.0e9


class HWP:
    def __init__(self, hwpDir, temp, theta, det = None):
        matFile = os.path.join(hwpDir, "materials.txt")
        stackFile = os.path.join(hwpDir, "stack.txt")
        HWPSS_transFile = os.path.join(hwpDir, "HWPSS_Trans.npy")
        HWPSS_reflFile = os.path.join(hwpDir, "HWPSS_Refl.npy")
        self.theta = theta
        #labels = np.array(["Freqs", "A2up", "A4up", "A2pp", "A4pp"])
        self.materials = loadMaterials(matFile)
        self.stack = loadStack(self.materials, stackFile)
        
        _, self.freqs, self.A2upT, self.A4upT, self.A2ppT, self.A4ppT = np.load(HWPSS_transFile)
        _, _, self.A2upR, self.A4upR, self.A2ppR, self.A4ppR = np.load(HWPSS_reflFile)
        
        if det:
            self.MTave = self.MuellerAve(det.freqs, self.theta, reflected = False)
            self.MRave = self.MuellerAve(det.freqs, self.theta, reflected = True)
        

    def MuellerAve (self, freqs, theta, reflected = False):
        M = np.zeros((4,4))
        for f in freqs:
            M += self.Mueller(f, theta, reflected = reflected)
        return M / len(freqs)
        
    def Mueller(self, freq, theta, reflected = False):
        return tm.Mueller(self.stack, freq, theta, 0.0, reflected = reflected)            
        
    def getHWPSS(self, freq, stokes, reflected = False, fit = False):
        if not fit:
            I = stokes[0]
            Q = stokes[1]
            M = self.Mueller(freq, self.theta, reflected = reflected)
#            M = self.MRave if reflected else self.MTave

            A2 = .5 * ((I * M[1,0] + Q * M[0,1])**2 + (I * M[2,0] + Q * M[0,2])**2)**.5
            A4 = .25 * (Q * Q * ((M[1,2] + M[2,1])**2 + (M[1,1] - M[2,2])**2))**.5
            
            return A2, A4
        else:
            return 

    def toA4(self, freq, polarized = False, reflected = False, fit = False):
        
        if polarized and reflected:
            A4 = self.A4ppR
        elif polarized and (not reflected):
            A4 = self.A4ppT
        elif (not polarized) and reflected:
            A4 = self.A4upR
        else:
            A4 = self.A4upT
        
        return np.interp(freq, self.freqs, A4)
    
    def toA2(self, freq, polarized = False, reflected = False):
        if polarized and reflected:
            A2 = self.A2ppR
        elif polarized and (not reflected):
            A2 = self.A2ppT
        elif (not polarized) and reflected:
            A2 = self.A2upR
        else:
            A2 = self.A2upT
        
        return np.interp(freq, self.freqs, A2)
        
def demodFit(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, p1, p2, p3, p4, p5, p6, p7, p8):
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
    mats = {}
    name, no, ne, lto, lte, mtype = np.loadtxt(matFile, dtype=np.str, unpack=True)
    no = np.array(map(np.float, no))
    ne = np.array(map(np.float, ne))
    lto = 1.0e-4 * np.array(map(np.float, lto))
    lte = 1.0e-4 * np.array(map(np.float, lte))
    for (i,n) in enumerate(name):
        mats[n] = tm.material(no[i], ne[i], lto[i], lte[i], n, mtype[i])
    return mats

def loadStack(materials, stackFile):
    name, thick, angle = np.loadtxt(stackFile, dtype=np.str, unpack=True)
    mats = map(lambda n : materials[n] , name)
    thick = np.array(map(np.float, thick)) * 1.e-3
    angle = np.array(map(np.float, angle))
    angle = np.deg2rad(angle)
    return tm.Stack(thick, mats, angle)

def fitAmplitudes(stack, freq, theta, stokes = np.array([1, 0, 0, 0]), reflected = False,  p = None):
    det = .5 * np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])    
    chis = np.linspace(0, 2 * np.pi, 100)
    demod = []
    for chi in chis:
        M = tm.Mueller(stack, freq, theta, chi, reflected = False)
        demod.append(det.dot(M).dot(stokes)[0])
    if p != None:
        p = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        
    popt, pcov = optimize.curve_fit(demodFit, chis, demod, p)    
    return popt

def fitAmplitudesBand(stack, freqs, theta, stokes, reflected = False):
    popt = None
    A2 = []
    A4 = []
    for f in freqs:
        print f/ GHz
        popt = fitAmplitudes(stack, f, theta, stokes = stokes, reflected = reflected, p = popt)
        A2.append(popt[2])
        A4.append(popt[4])
    return np.array(A2), np.array(A4)
    

if __name__ == "__main__":
    hwp = HWP("../HWP/3LayerSapphire", 40, np.deg2rad(20.))
    print hwp.Mueller(150 * GHz, reflected = False)
    freqs = np.linspace(50 * GHz, 200 * GHz, 500)
    plt.plot(hwp.freqs, hwp.toA4(hwp.freqs, polarized = True))