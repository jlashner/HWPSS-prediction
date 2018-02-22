import numpy as np
import thermo as th
from scipy import interpolate
from scipy import integrate as intg
import matplotlib.pyplot as plt
import transfer_matrix as tm
import HWP_model
import Detector as dt
import IPCalc


# Units and Constants
GHz = 1.e9 # GHz -> Hz
pW = 1.e12 # W -> pW


eps0 = 8.85e-12 #Vacuum Permitivity
rho=2.417e-8 #Resistivity of the mirror
c = 299792458.0 #Speed of light [m/s]


###############################################################################
# Class of objects used for optical chain
###############################################################################

class OpticalElement:
    """ Optical Element
        *****
        Optional Parmeters:
            :Thick: Thickness [m]
            :Index: Index of refraction
            :LossTan: Loss Tangent 
            :Absorb: Absorption Coefficient 
            :Refl: Reflection coefficient
            :Spill: Spillover Fraction
            :ScattFrac: Scattered Fraction 
            :ScattTemp: Scattered Temperature [K]
            :SpillTemp: Spillover Temperature [K]
            :IP: IP value 
            :PolRefl: Differential Reflection Coefficient
            :PolAbs: Polarized Absorption Coefficient
            :Chi: Incident angle of Mirror [rad]
            :Freqs: Frequency Array for 
            :EffCurve: Efficiency Function (Corresponding to frequency array)
            :IPCurve: IP Function
            :EmisCurve: Emission Function
    """
    def __init__(self, name, det, temp, params = {}):
        self.name = name
        self.det = det
        self.temp = temp
        
        #Sets Default Optical Element Parameters
        self.params = {"Thick": 0,      "Index": 1.0, "LossTan": 0, "Absorb": 0, \
                       "Absorb": 0,     "Spill": 0, "SpillTemp": 0, "Refl": 0, "ScattFrac": 0, \
                       "ScattTemp": 0,  "IP": 0, "PolRefl": 0, "PolAbs": 0,  "Chi": 0, \
                       "Freqs": None,   "EffCurve": None, "IPCurve": None, "EmisCurve": None, "PolEmisCurve": None};
        
        self.unpolIncident      = None
        self.unpolTransmitted   = None
        self.unpolEmitted       = None
        self.polTransmitted     = None
        self.polEmitted         = None
                       
        self.updateParams(params)

    def updateParams(self, modifiedParams):
        self.params.update(modifiedParams)
    
    
    #IP Coefficient 
    def Ip(self, freq):
        if self.params["IPCurve"] is not None:
            return np.interp(freq, self.params["Freqs"], self.params["IPCurve"])
        return self.params["IP"]
    
    #Absorption coefficient
    def Absorb(self, freq):
        if self.params["LossTan"] != 0:
            # Calculates absorption from Loss Tangent
            lamb = c / ( freq * self.params["Index"] )
            alpha = 2 * np.pi * self.params["Index"] * self.params["LossTan"] / lamb
            ab = 1.0 - np.exp(-alpha * self.params["Thick"])
            return ab
        else:
            return self.params["Absorb"]
    
    #Transmission Coefficient
    def Eff(self, freq):
        if self.params["EffCurve"] is not None:
            return np.interp(freq, self.params["Freqs"], self.params["EffCurve"])
        elif self.name == "Aperture":
            return th.spillEff(self.det)
        else:
            return  1 - self.Absorb(freq) - self.params["Spill"]- self.params["Refl"]
            
    #Polarized Efficiency
    def pEff(self, freq):
        return self.Eff(freq)

    def Emis(self, freq):
        if self.params["EmisCurve"] is not None:
            return np.interp(freq, self.params["Freqs"], self.params["EmisCurve"])
        
        if self.name == "Atm":
            # Returns 1 because we are using Rayleigh Jeans temperature
            return 1
        
        # Gets extra emissivity due to spillover 
        powFrac =  th.weightedSpec(freq, self.params["SpillTemp"], 1.)/th.weightedSpec(freq, self.temp, 1.)
        spillEmis = powFrac * self.params["Spill"]

        if self.name == "Aperture":
            return (1 - self.Eff(freq) + spillEmis)
        else:
            return self.Absorb(freq) + spillEmis

    #Polarized Emissivity
    def pEmis(self, freq):
        if self.params["PolEmisCurve"] is not None:
            return np.interp(freq, self.params["Freqs"], self.params["PolEmisCurve"])        
        
        if self.name == "Mirror":
            return - self.Ip(freq)
        
        if self.name == "HWP":
            ao = 8.7*10**(-5) * (freq/ GHz) + 3.1*10**(-7)*(freq/GHz)**2 + (3.0)*10**(-10) * (freq/GHz)**3 #1/cm
            ae = 1.47*10**(-7) * (freq/GHz)**(2.2) #1/cm
            Eotrans =  np.exp(-self.params["Thick"]*100.0 * ao * self.temp/300) 
            Eetrans =  np.exp(-self.params["Thick"]*100.0 * ae * self.temp/300)
            
            pemis = (abs(Eetrans)**2 - abs(Eotrans)**2) / 2            
            return pemis
        
        return self.params["PolAbs"]
    
    def Refl(self, freq):
        return self.params["Refl"]
    
    def pRefl(self, freq):
        return self.params["PolRefl"]


###############################################################################
#    Loads individual types of Optical Elements
###############################################################################
def loadAtm(atmFile, det):
    """Loads an optical element from specified atmosphere file"""
    freqs, temps, trans = np.loadtxt(atmFile, dtype=np.float, unpack=True, usecols=[0, 2, 3]) #frequency/tempRJ/efficiency arrays from input files
    freqs*=GHz # [Hz]
    
    atmTemp = 300. # [K]
    emis = temps / atmTemp
    e = OpticalElement("Atm", det, atmTemp, {"Freqs": freqs, "EffCurve": trans, "EmisCurve": emis})
    return e
    
    #Reads Rayleigh Jeans temperature from file and takes average
#    tempF = interpolate.interp1d(freqs, temps, kind = "linear")
#    x = np.linspace(det.flo, det.fhi, 400)
#    y = tempF(x)
#    aveTemp = intg.simps(y, x=x)/(det.fhi - det.flo)
#    e = OpticalElement("Atm", det, aveTemp, {"Freqs": freqs, "EffCurve": trans})
#    
#    return e


def loadHWP(hwp, theta, det):
    #(extra)ordinary index of reflection and LossTan for HWP
    n_e          = 3.38 
    n_o          = 3.05
    lossTan_e    = 1.25e-4
    lossTan_o    = 2.30e-4
#    lossTan_o = lossTan_e
    # n and lossTan for 1st and 2nd AR coatings
    nAR1 = 1.55 
    nAR2 = 2.52
    lossTanAR1 = 5.0e-5
    lossTanAR2 = 5.65e-3
    
    #Material definitions
    AR1mat=tm.material(nAR1,nAR1,lossTanAR1,lossTanAR1,'PTFE','isotropic')      #top AR-layer
    AR2mat=tm.material(nAR2,nAR2,lossTanAR2,lossTanAR2,'PTFE','isotropic')   #bottom AR-layer
    HWPmat=tm.material(n_o,n_e,lossTan_o,lossTan_e,'Sapphire','uniaxial')  
    mats = [AR1mat, AR2mat, HWPmat, HWPmat, HWPmat, AR2mat, AR1mat]
    #Material thicknesses
    HWPthick = 3.75e-3 
    AR1thick = 3.8e-4
    AR2thick = 2.7e-4
    thicks = [AR1thick, AR2thick, HWPthick, HWPthick, HWPthick, AR2thick, AR1thick]
    
    angles  = [0., 0.,0., np.deg2rad(54.), 0., 0., 0.] #Rotation of each layer in stack
    ff      = 120. * GHz           #Optimized frequency [Hz]
    
    stack = tm.Stack(thicks, mats, angles)           
        
    ##Calculates Band Averaged Mueller Matrix
    freqs = det.freqs
    
    mueller_t = np.zeros((4, 4))
    mueller_r = np.zeros((4, 4))
    
    for f in freqs:
        mueller_t += tm.Mueller(stack, f, theta, 0, reflected = False)
        mueller_r += tm.Mueller(stack, f, theta, 0, reflected = True)
    
    mueller_t /= len(freqs)
    mueller_r /= len(freqs)

    hwp.updateParams({"Mueller_T": mueller_t, "Mueller_R": mueller_r})
    return
    

def loadOpticalChain(opticsFile, det, hwpDir, theta = np.deg2rad(15./2)):
    """Returns list of optical elements from opticalChain.txt file. """
    
    elements = []
    
    data = np.loadtxt(opticsFile, dtype=np.str)
    keys = data[0]
    
    # Units of optical chain columns:
    # [sring, K, mm, NA, e-4, e6 S/m, NA, NA, K, um, NA, NA, K]
    conversions = [None, 1, 1e-3, 1, 1e-4, 1e6, 1, 1, 1, 1e-6, 1, 1, 1]
    
    for line in data[1:]:
        params = {}
        #Parses opticalChain.txt values and converts to correct units
        for (i, k) in enumerate(keys):
            value = line[i]
            if (value) == "NA":
                continue #Uses default optical element value
            try:
                v = eval(value)
                if type(v) == list:
                    params[k] = v[det.bid - 1] * conversions[i]
                else:
                    params[k] = v * conversions[i]
            except:
                params[k] = value       
                
        name = params["Element"]    
        
        if name == "HWP":
            elements.append(HWP_model.HWP(hwpDir, params["Temp"], theta,  det))
            continue
            
        ## Calculates differential transmission and absorption using tmm        
        if name == "AluminaF" or name == "Window":
            (ip,polRefl, polAbs) = IPCalc.getDiffCoeffs(name, det.band_center, det.fbw, theta)
            params.update({"IP": ip, "PolRefl": polRefl, "PolAbs": polAbs})

        e = OpticalElement(name, det, params["Temp"], params = params)        
        elements.append(e)
        
    return elements

