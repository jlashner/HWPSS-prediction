import scipy.integrate as intg
import numpy as np
import matplotlib.pyplot as plt


#Physical Constants
#Everything is in MKS units
h = 6.6261e-34  #Planck constant [J/s]
kB = 1.3806e-23 #Boltzmann constant [J/K]
c = 299792458.0 #Speed of light [m/s]
PI = np.pi      #Pi
eps0 = 8.85e-12 #Vacuum Permitivity
rho=2.417e-8    #Resistivity of the mirror
Tcmb = 2.725    #CMB Temperatrure [K]

#Conversions
GHz = 10 ** 9




#Calculates total black body power for a given temp and emis.
def bbSpec(freq,temp,emis):
    """Calculates the Black body spectrum for a given temp and emis"""    
    if temp==0:
        return 0
    occ = 1.0/(np.exp(h*freq/(temp*kB)) - 1)
    e = emis(freq) if  callable(emis) else emis
    return 2 * e * h * freq**3 /(c**2) * occ

def weightedSpec(freq,temp,emis):
    """Calculates the Black body spectrum for a given temp and emis weighted by AOmega"""    
    AOmega = (c/freq)**2
    return AOmega * bbSpec(freq, temp, emis)


def powFromSpec(freqs, spec):
    """Integrates spectrum"""
    return np.trapz(spec, freqs)


def spillEff(det):
    """" Calculates Efficiency of the Aperture"""
    D = det.pixSize
    F = det.f_num
    waistFact = det.waistFact
    freq = det.band_center
    return 1. - np.exp((-np.power(np.pi,2)/2.)*np.power((D/(waistFact*F*(c/freq))),2))


def getLambdaOpt(nu, chi):
    """Gets lambdal_opt for mirrors"""
    geom = (1 / np.cos(chi) - np.cos(chi))
    return - 2 * geom * np.sqrt(4 * PI * eps0 * rho * nu)

def aniPowSpec(emis, freq, temp=Tcmb):
    """Derivative of BB spectrum"""
    e = emis(freq) if callable(emis) else emis
    occ = 1.0/(np.exp(h*freq/(temp*kB)) - 1)
    return ((h**2)/kB)*e*(occ**2)*((freq**2)/(temp**2))*np.exp((h*freq)/(kB*temp))


if __name__=="__main__":
    print bbSpec(145 * GHz, 100, 1)
    