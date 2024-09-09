#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:51:21 2017

@author: jlashner
"""

from pylab import *
import tmm
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg



#Constants
c =2.99792458 * 10**8 #speed of light [m/s]

#Units
GHz = 10 ** 9 # GHz -> Hz


def ARCoat(n, lam0, layers = 2):
    """Gets Index of refraction and thickness for AR coating centered around lam0"""
    ni= .00008
    if layers == 2:
        nAR = [real(n)**(1./3) + ni*1j, real(n)**(2./3) + ni * 1j]
    elif layers == 1:
        nAR = [real(n)**(1./2)]
        
    dAR = [lam0 / (4 * real(n)) for n in nAR]

    return nAR, dAR


def getCoeffs(n, d, freq, theta, pol):
    """Returns T, R, A coefficients for an optical stack of polarization pol ('s' or 'p')"""
    assert pol in ['s', 'p']
    lam_vac= c / freq * 1000 #vacuum wavelength in mm
    s = tmm.coh_tmm(pol, n, d, theta, lam_vac)
    return [s['T'], s['R'], 1 - s['T'] - s['R']]


def getDiffCoeffs(name, band_center, fbw, theta):
    """Returns the differential transmission and differential coefficient of an optical element"""
    
    #Determines band and calculates corresponding optimized wavelength    
    if band_center < 50 * GHz:   #LF Band
        nu0 = 33 * GHz
        lam0 = 3e3 / nu0 * 1000 #[mm]]
        lam0 = 9.09 #[mm]
        layers = 2
    elif band_center < 200 * GHz: #MF Band
        nu0 = 120 * GHz
        lam0 = 2.5 #[mm]
        layers = 2
    elif band_center < 300 * GHz: #UHF Band
        nu0 = 267 * GHz
        lam0 = 1.123 #[mm]
        layers = 1
    else:
        print("Frequency not in any band.")
        raise ValueError
            

    flo = band_center * (1 - fbw/ 2.)
    fhi = band_center * (1 + fbw/ 2.)

    if name == "Window":
        n0 =  1.5 + .0001j
        d0 = 10.0
    elif name == "AluminaF":
        n0 = 3.1 + .00008j
        d0 = 3.0
    else:
        return (0,0)
        
    nAR, dAR = ARCoat(n0, lam0, layers = layers)
    n_stack = [1.0] + nAR + [n0] + nAR[::-1] + [1.0]
    d_stack = [Inf] + dAR + [d0] + dAR[::-1] + [Inf]
    
    #Creates Frequency Array and gets T,R, and A coefficients accross bandwidth
    freqs = np.linspace(flo, fhi, 300)
    s_coeffs = [getCoeffs(n_stack, d_stack, f, theta, 's') for f in freqs]
    p_coeffs = [getCoeffs(n_stack, d_stack, f, theta, 'p') for f in freqs]
    
    Ts, Rs, As = np.transpose(s_coeffs)
    Tp, Rp, Ap = np.transpose(p_coeffs)
    

    
    
    #Band-averages differential transmission, reflection and absorption    
    diffTrans =  abs(intg.simpson((Ts - Tp)/2, x=freqs)/(band_center * fbw))
    diffRefl  =  abs(intg.simpson((Rs - Rp)/2, x=freqs)/(band_center * fbw))
    diffAbs   =  abs(intg.simpson((As - Ap)/2, x=freqs)/(band_center * fbw))
#    print("Absorption: ", abs(intg.simps((As + Ap)/2, freqs)/(band_center * fbw)))
    
    return (diffTrans, diffRefl, diffAbs)


if __name__ == "__main__":
    bc = np.array([27., 39., 93.0,145., 225., 278.]) * GHz # Band center [Hz]
    fbw = np.array([.222, .462, .376, .276, .267, .278]) #Fractional bandwidth
    theta = np.deg2rad(20.0)
    
    
    for index in [2,3]:
        T_filter, _, _ = getDiffCoeffs("AluminaF", bc[index], fbw[index], theta)
        T_window, _, _ =getDiffCoeffs("Window", bc[index], fbw[index], theta)
        print(f'Window ({bc[index]/GHz}): {T_window}')
        print(f'Filter ({bc[index]/GHz}): {T_filter}')




        
