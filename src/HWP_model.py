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
    
def fitAmplitudes(stack, freq, theta, stokes = np.array([1, 0, 0, 0]), makePlot = False):
    det = .5 * np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])    
    chis = np.linspace(0, 2 * np.pi, 100)
    demod = []
    for chi in chis:
        M = tm.Mueller(stack, freq, theta, chi, reflected = False)
        demod.append(det.dot(M).dot(stokes)[0])
    
    p = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    popt, pcov = optimize.curve_fit(demodFit, chis, demod, p)    
    if makePlot:
        testFit = demodFit(chis, *popt)
        plt.plot(chis, testFit)
        plt.plot(chis, demod, "o")
    
    
    
    return popt[2], popt[4]


mats = loadMaterials("../HWP_Mueller/materials.txt")
stack = loadStack(mats, "../HWP_Mueller/AHWP_stack.txt")
Sunpol = np.array([1,0,0,0])
Spol = np.array([1,1,0,0])

print fitAmplitudes(stack, 150*GHz, np.deg2rad(20.), stokes = Spol)
