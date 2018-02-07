
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
        
        expDir      = config["ExperimentDirectory"]
        atmFile     = config["AtmosphereFile"]
        hwpFile     = config["HWPFile"]
        
        if not expDir:
            raise AttributeError("Experiment directory not defined.")
        if not atmFile:
            raise AttributeError("Atmosphere file not defined.")
        
        channelFile     = os.path.join(expDir, "channels.txt")
        cameraFile      =  os.path.join(expDir, "camera.txt")
        opticsFile      =  os.path.join(expDir, "opticalChain.txt")
        
        
        #Imports detector data 
        self.det = dt.Detector(channelFile, cameraFile, config["bandID"], config)
        
        self.freqs = np.linspace(self.det.flo, self.det.fhi, 400) #Frequency array of the detector

        
        """Creating the Optical Chain"""
        self.elements = [] #List of optical elements
    
        self.elements.append(opt.OpticalElement("CMB", self.det, 2.725, {"Absorb": 1}))         #CMB 
        self.elements.append(opt.loadAtm(atmFile, self.det))     #Atmosphere
        self.elements += opt.loadOpticalChain(opticsFile, self.det, theta=config["theta"])       #Optical Chain
        self.elements.append(opt.OpticalElement("Detector", self.det, self.det.bath_temp, {"Absorb": 1 - self.det.det_eff})) #Detector
        
        #Finds HWP
        try:
            self.hwpIndex = [e.name for e in self.elements].index("HWP")
            self.hwp = self.elements[self.hwpIndex]
        except:
            print "No HWP in Optical Chain"
       
        
        opt.loadHWP(self.hwp, config["theta"], self.det)

        #Adds HWP curves 
        fs, T, rho, _, _ = np.loadtxt(hwpFile, dtype=np.float, unpack=True)
        self.hwp.updateParams({"Freqs": fs, "EffCurve": T, "IPCurve": rho})
        
        #Calculates conversion from pW to Kcmb
        aniSpec  = lambda x: th.aniPowSpec(1, x, Tcmb)
        self.toKcmb = 1/intg.quad(aniSpec, self.det.flo, self.det.fhi)[0]
        #Conversion from pW to KRJ
        self.toKRJ = 1 /(th.kB *self.det.band_center * self.det.fbw)
        
        #Propagates Unpolarized Spectrum through each element
        self.propSpectrum()        
        self.getHWPSS()
        
            
        if config["WriteOutput"]:
            self.writeOutput(os.path.join(expDir, config["OutputDirectory"]))
                 
            

    def cumEff(self, freq, start=1, end=-1):
        """
        Calculates the efficiency of all elements between the elements with indices "start" and "end" 
        """
        cumEff = 1.
        
        if end == -1:
            elementsInBetween = self.elements[start+1:]
        else:
            elementsInBetween = self.elements[start+1:end]
                
        for e in elementsInBetween:
            cumEff *= e.Eff(freq)
            
        return cumEff
                
               
    def propSpectrum(self, ReflectionOrder = 2):
        """
        Propagates power through each element of the optical chain.
        For each element this function creates the spectra:
            :(un)polIncident: Spectrum incident on the sky-side
            :(un)polTransmitted: Spectrum transmitted through the element
            :(un)polEmitted: Spectrum emitted by element
            :(un)polReverse: Spectrum incident on the detector side
            :(un)polCreated: Spectrum added to the total signal by the element, through emission, Ip, or reflection
            :IpTransmitted: Polarized light created through Ip
        """
        
        for e in self.elements:
            e.unpolIncident     = np.zeros(len(self.freqs))   # Unpol incident on element
            e.unpolEmitted      = th.weightedSpec(self.freqs, e.temp, e.Emis)     # Unpol Emitted

            e.polIncident       = np.zeros(len(self.freqs))   # Pol incident on element
            e.polEmitted        = th.weightedSpec(self.freqs, e.temp, e.pEmis)     # Unpol Emitted
            
        for n in range(ReflectionOrder):
            for (i,e) in enumerate(self.elements):
                e.unpolCreated = e.unpolEmitted
                e.unpolTransmitted = e.unpolIncident * e.Eff(self.freqs)
                e.IpTransmitted = e.unpolIncident * e.Ip(self.freqs)
                e.polTransmitted = e.polIncident * e.pEff(self.freqs)
                

                
                e.unpolReverse = np.zeros(len(self.freqs))
                e.polReverse = np.zeros(len(self.freqs))
                if n != 0:
                    for (j,e2) in enumerate(self.elements[i+1:]):
                        eff = self.cumEff(self.freqs, start = i, end = i + j)
                        e.unpolReverse +=  e2.unpolIncident * e2.Refl(self.freqs) * eff
                        e.unpolReverse +=  e2.unpolEmitted * eff
                                        
                        if (j < self.hwpIndex):
                            e.polReverse += e2.polIncident * e2.Refl(self.freqs) * eff 
                            e.polReverse += e2.unpolIncident * e2.pRefl(self.freqs) * eff
                    
                            
                e.unpolCreated = e.unpolEmitted + e.unpolReverse * e.Refl(self.freqs)
                e.polCreated = e.polEmitted + e.IpTransmitted + e.polReverse * e.Refl(self.freqs) + e.unpolReverse * e.pRefl(self.freqs)
                
                #Sets incident power on next element
                if i + 1 < len(self.elements):
                    self.elements[i+1].unpolIncident    = e.unpolCreated + e.unpolTransmitted
                    self.elements[i+1].polIncident      = e.polCreated   + e.polTransmitted
                    
                
                
            
    def getHWPSS(self):
        """
        Calculates the expected HWPSS for the telescope. Computes a2, a4, A2, and A4.
        """
        hwp = self.hwp
        
        # Stokes params incident on HWP
        IT      = hwp.unpolIncident + hwp.polIncident
        QT      = hwp.polIncident
        # Stokes parameters reflected by HWP
        IR      = hwp.unpolReverse + hwp.polReverse
        QR      = hwp.polReverse
        #HWP Mueller matrices
        MuellerT = hwp.params["Mueller_T"]
        MuellerR = hwp.params["Mueller_R"]

        #######################################################################
        ####   A4 Calculation        
        A4specTransmitted = (MuellerT[0,0] - MuellerT[2,2])/2. * QT * self.cumEff(self.freqs, start = self.hwpIndex)
        A4specReflected   = (MuellerR[0,0] - MuellerR[2,2])/2. * QR * self.cumEff(self.freqs, start = self.hwpIndex)
        self.A4  = abs(.5 * th.powFromSpec(self.freqs, A4specTransmitted))
        self.A4 += abs(.5 * th.powFromSpec(self.freqs, A4specReflected))
         
        #######################################################################
        ####   A2 Calculation 
        A2specTransmitted  = (IT + QT) * MuellerT[0,1] * self.cumEff(self.freqs, start = self.hwpIndex)
        A2specReflected    = (IR + QR) * MuellerR[0,1] * self.cumEff(self.freqs, start = self.hwpIndex)
        A2specEmitted      = th.weightedSpec(self.freqs, self.hwp.temp, self.hwp.pEmis) * self.cumEff(self.freqs, start = self.hwpIndex) 
        
        self.A2  = abs(.5 * th.powFromSpec(self.freqs, A2specTransmitted))
        self.A2 += abs(.5 * th.powFromSpec(self.freqs, A2specReflected))
        self.A2 += abs(.5 * th.powFromSpec(self.freqs, A2specEmitted))
        
        #######################################################################
        ####   a4 Calculation
        self.a4  = 0
        for e in self.elements[:self.hwpIndex]:
            self.a4 += e.Ip(self.det.band_center)
        
        #######################################################################
        ####   a2 Calculation
        self.a2 = MuellerT[0,1]
        

    def _formatRow(self, row):
        return "\t".join(map( lambda x: "%-8s" % x, row)) + "\n"
        
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
            spectrums= [e.unpolIncident, e.unpolEmitted, e.IpTransmitted, e.polEmitted]
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

    config = json.load( open ("../run/config.json") )  
    tel = Telescope(config)
    print "Telescope A4: ", tel.A4 * pW
    print "Telescope A2: ", tel.A2 * pW
    
    print tel.opticalTable()