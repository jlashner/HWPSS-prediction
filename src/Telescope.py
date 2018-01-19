
import numpy as np
import scipy.integrate as intg

import thermo as th
import OpticalElement as opt
import Detector as dt
import matplotlib.pyplot as plt
import os
import json

Tcmb = 2.725 # CMB temp [K]

# Units and Constants
GHz = 1.e9 # GHz -> Hz
pW = 1.e12 # W -> pW

class Telescope:
    def __init__(self, config):
        """ Telescope object"""
        self.config = config
        
        expDir = config["ExperimentDirectory"]
        atmFile = config["AtmosphereFile"]
        hwpFile = config["HWPFile"]
        if not expDir:
            print "experiment directory not defined"
            raise AttributeError
        
        channelFile = os.path.join(expDir, "channels.txt")
        cameraFile =  os.path.join(expDir, "camera.txt")
        opticsFile =  os.path.join(expDir, "opticalChain.txt")
        
        
        #Imports detector data 
        self.det = dt.Detector(channelFile, cameraFile, config["bandID"], config)
        
        self.freqs = np.linspace(self.det.flo, self.det.fhi, 400) #Frequency array of the detector

        
        """Creating the Optical Chain"""
        self.elements = [] #List of optical elements
    
        self.elements.append(opt.OpticalElement("CMB", self.det, 2.725, {"Absorb": 1}))         #CMB 
        self.elements.append( opt.loadAtm(atmFile, self.det))     #Atmosphere
        self.elements += opt.loadOpticalChain(opticsFile, self.det, theta=config["theta"])       #Optical Chain
        self.elements.append(opt.OpticalElement("Detector", self.det, self.det.bath_temp, {"Absorb": 1 - self.det.det_eff})) #Detector
        
        #Finds HWP
        try:
            self.hwpIndex = [e.name for e in self.elements].index("HWP")
        except:
            print "No HWP in Optical Chain"
        
        #Adds HWP curves 
        fs, T, rho, _, _ = np.loadtxt(hwpFile, dtype=np.float, unpack=True)
        self.elements[self.hwpIndex].updateParams({"Freqs": fs, "EffCurve": T, "IPCurve": rho})
        
        #Calculates conversion from pW to Kcmb
        aniSpec  = lambda x: th.aniPowSpec(1, x, Tcmb)
        self.toKcmb = 1/intg.quad(aniSpec, self.det.flo, self.det.fhi)[0]
        #Conversion from pW to KRJ
        self.toKRJ = 1 /(th.kB *self.det.band_center * self.det.fbw)
        
        #Propagates Unpolarized Spectrum through each element
        self.propSpectrum()
        
        self.geta2()
        self.getA2()
        self.geta4()
        self.getA4()
        
        if config["WriteOutput"]:
            self.writeOutput(os.path.join(expDir, config["OutputDirectory"]))
                 
            
    def writeOutput(self, outputDir):
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)
            
        opticalTableFilename =  os.path.join(outputDir, "opticalTable.txt")
        configFilename =  os.path.join(outputDir, "config.json")
        hwpssFilename = os.path.join(outputDir, "hwpss.txt")
        
        with open(opticalTableFilename, 'w') as opticalTableFile:
            opticalTableFile.write(self.opticalTable())

        with open(configFilename, 'w') as configFile:
            json.dump(self.config, configFile, sort_keys=True, indent=4)
        
        with open(hwpssFilename, 'w') as hwpssFile:
            hwpssFile.write(self.hwpssTable())
            

    def cumEff(self, freq, start=1, end=-1):
        """Calculates the efficiency of all elements between the elements with indices "start" and "end" """
        cumEff = 1.
        
        if end == -1:
            elementsInBetween = self.elements[start+1:]
        else:
            elementsInBetween = self.elements[start+1:end]
                
        for e in elementsInBetween:
            cumEff *= e.Eff(freq)
            
        return cumEff
        
    
    def propSpectrum(self):
        
        self.elements[0].unpolIncident = np.array([0 for _ in self.freqs])
        
        for (i, el) in enumerate(self.elements):
            el.unpolTransmitted = el.unpolIncident * map(el.Eff, self.freqs)
            el.unpolEmitted = th.weightedSpec(self.freqs, el.temp, el.Emis)
            if i < len(self.elements) - 1:
                self.elements[i+1].unpolIncident = el.unpolTransmitted + el.unpolEmitted
            if i <= self.hwpIndex:
                el.polTransmitted = el.unpolIncident * map(el.Ip, self.freqs)
                el.polEmitted = th.weightedSpec(self.freqs, el.temp, el.pEmis)
            else:
                el.polTransmitted = [0 for _ in self.freqs]
                el.polEmitted = [0 for _ in self.freqs]                
            
                        
    def geta2(self):
        """Gets a2 by band-averaging the HWP IP coefficient"""
        hwp= self.elements[self.hwpIndex]
        self.a2 = abs(np.average(map (hwp.Ip, self.freqs) ))

    def getA2(self):
        """ Sets A2 power at the detector in pW"""        
        hwp = self.elements[self.hwpIndex]
        spectrumAtDetector = (hwp.polEmitted + hwp.polTransmitted) * self.cumEff(self.freqs, start = self.hwpIndex)
        self.A2 = .5 * abs(th.powFromSpec(self.freqs, spectrumAtDetector))
    
    def geta4(self):
        """Gets a4 by adding IP of all elements before the HWP (at band center)."""
        self.a4 = 0
        for e in self.elements[:self.hwpIndex]:
            self.a4 += e.Ip(self.det.band_center)
        
    def getA4(self):
        """Gets A4 power at the detector in pW"""        
        self.A4 = 0
        for (i,e) in enumerate(self.elements[:self.hwpIndex]):
            
            #polarized Spectrum outputted towards detector
            polSpecTotal = e.polEmitted #Output from polarized emission
            polSpecTotal += e.polTransmitted #Output from polarized transmission
        
            #Contribution from multiple reflections with skyside elements 
            #Loops over all pairs of elements
            for (j, e2) in enumerate(self.elements[i:]):
                # Light that is reflected by a later element and then polarized by e.
                polSpecTotal += (e2.unpolIncident * e2.Refl(self.freqs) + e2.unpolEmitted) * e.pRefl(self.freqs) * self.cumEff(self.freqs, start=i, end=j)
                
                if j < self.hwpIndex:
                    #Light that is polarized by a later element and then reflected by e.
                    # e2 must be before HWP or else it will be modulated on the way back.
                    polSpecTotal += (e2.unpolIncident * e2.pRefl(self.freqs) + e2.polEmitted) * e.Refl(self.freqs) * self.cumEff(self.freqs, start=i, end=j)
                
            
        
            specAtDetector = polSpecTotal * self.cumEff(self.freqs, start=i)
            ppTotal = .5 * abs(th.powFromSpec(self.freqs, specAtDetector))
            self.A4 += ppTotal    

    def _formatRow(self, row):
        return "\t".join(map( lambda x: "%-8s" % x, row)) + "\n"
        
    
    def hwpssTable(self):
        
        tableString = ""
        tableString += "Frequency: %i GHz\t fbw: %.3f\n"%(self.det.band_center/ GHz, self.det.fbw)
        headers = ["Location",  "A4", "A4", "A2", "A2"]
        units = ["", "[pW]", "[KRJ]", "[pW]", "[KRJ]"]
        atDet = np.array([self.A4 * pW, self.A4 * self.toKRJ, self.A2 * pW, self.A2 * self.toKRJ])
        atEntrance = atDet / self.cumEff(self.det.band_center)
        
        
        
        tableString += self._formatRow(headers)
        tableString += self._formatRow(units)
        tableString += "-"*70 + "\n"
        
        tableString += self._formatRow(["AtDetector"] + map(lambda x : "%.3e"%x, atDet ))
        tableString += self._formatRow(["AtEntrance"] + map(lambda x : "%.3e"%x, atEntrance ))
        
        return tableString
    
    def opticalTable(self):
        
        headers = ["Element", "unpolInc", "unpolEmitted", "IP", "polEmitted"]
        units = ["", "[pW]", "[pW]", "[pW]", "[pW]"]
        rows = []
        
        for e in self.elements:
            line = []
            line.append(e.name)
            spectrums= [e.unpolIncident, e.unpolEmitted, e.polTransmitted, e.polEmitted]
            line += map(lambda x : "%.3e"%(abs(th.powFromSpec(self.freqs, x)*pW)), spectrums)
            rows.append(line)
            
        tableString = ""
        tableString += "Frequency: %i GHz\t fbw: %.3f\n"%(self.det.band_center/ GHz, self.det.fbw)
        tableString += self._formatRow(headers)
        tableString += self._formatRow(units)
        tableString += "-" * 70 + "\n"
        
        for row in rows:
            tableString += self._formatRow(row)
        
        return tableString
        
        

if __name__=="__main__":
    expDir = "../Experiments/small_aperture/LargeTelescope/"    
    atmFile = "Atacama_1000um_60deg.txt"    
    theta = 20
    hwpFile = "../HWP_Mueller/Mueller_AR/Mueller_V2_nu150.0_no3p068_ne3p402_ARcoat_thetain%d.0.txt"%theta
    bid = 2
    
    opts = {'theta': np.deg2rad(theta)}
    
    tel = Telescope(expDir, atmFile, hwpFile, bid, **opts)
    
    print tel.A2 * pW
    print tel.A4 * pW
    tel.displayTable()