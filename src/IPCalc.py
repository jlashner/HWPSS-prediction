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
        

    dAR = list(map(lambda x : lam0 / (4.0 * real(x)), nAR))
    return nAR, dAR


def getCoeffs(n, d, freq, theta, pol):
    """Returns T, R, A coefficients for an optical stack of polarization pol ('s' or 'p')"""
    assert pol in ['s', 'p']
    lam_vac= c / freq * 1000 #vacuum wavelength in mm
    s = tmm.coh_tmm(pol, n, d, theta, lam_vac)
    return [s['T'], s['R'], 1 - s['T'] - s['R']]


def getDiffCoeffs(name, band_center, fbw, theta):
    """Returns the differential transmission and differential coefficient of an optical element"""
    lam0 = 2.5 #[mm]
    lam0 = 1.1
    flo = band_center * (1 - fbw/ 2.)
    fhi = band_center * (1 + fbw/ 2.)
    flo = 200 * GHz
    fhi = 300 * GHz
    
    if name == "Window":
        n0 =  1.5 + .0001j
        d0 = 5.0
    elif name == "AluminaF":
        n0 = 3.1 + .00008j
        d0 = 2.0
    else:
        return (0,0)
        
    nAR, dAR = ARCoat(n0, lam0, layers = 2)
    n_stack = [1.0] + nAR + [n0] + nAR[::-1] + [1.0]
    d_stack = [Inf] + dAR + [d0] + dAR[::-1] + [Inf]
    
    #Creates Frequency Array and gets T,R, and A coefficients accross bandwidth
    freqs = np.linspace(flo, fhi, 300)
    s_coeffs = [getCoeffs(n_stack, d_stack, f, theta, 's') for f in freqs]
    p_coeffs = [getCoeffs(n_stack, d_stack, f, theta, 'p') for f in freqs]
    
    Ts, Rs, As = np.transpose(s_coeffs)
    Tp, Rp, Ap = np.transpose(p_coeffs)
    
    plt.plot(freqs, Ts)
    plt.plot(freqs, Rs)
    plt.plot(freqs, As)
    plt.ylim(0, 1)
    plt.show()
    
    #Band-averages differential transmission, reflection and absorption    
    diffTrans =  abs(intg.simps((Ts - Tp)/2, freqs)/(band_center * fbw))
    diffRefl  =  abs(intg.simps((Rs - Rp)/2, freqs)/(band_center * fbw))
    diffAbs   =  abs(intg.simps((As - Ap)/2, freqs)/(band_center * fbw))
    
    return (diffTrans, diffRefl, diffAbs)


if __name__ == "__main__":
    bc = np.array([93.0 * GHz,145. * GHz]) # Band center [Hz]
    fbw = np.array([.376, .276]) #Fractional bandwidth
    theta = np.deg2rad(15.0)
    
    print(getDiffCoeffs("AluminaF", bc[1], fbw[1], theta)    )


        