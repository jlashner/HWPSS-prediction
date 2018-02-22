
import numpy as np
import scipy.integrate as intg
import thermo as th
import OpticalElement as opt
import Detector as dt
import matplotlib.pyplot as plt
import HWP_model
import os
import json



#==============================================================================
#  Constants and Units
#==============================================================================
Tcmb = 2.725        # CMB temp [K]
GHz = 1.e9          # GHz -> Hz
pW = 1.e12          # W -> pW

class Telescope:
    def __init__(self, config):
        """ 
            Telescope object. Initiallizes and contains list of optical elements.
            Propagates light throughout the system and calculates A2 and A4.
        """
        
        #==============================================================================
        #   Config Initialization
        #==============================================================================
        self.config = config
        
        try:
            expDir      = config["ExperimentDirectory"]
            atmFile     = config["AtmosphereFile"]
            hwpDir     = config["HWPDirectory"]

            channelFile = os.path.join(expDir, "channels.txt")
            cameraFile  = os.path.join(expDir, "camera.txt")
            opticsFile  = os.path.join(expDir, "opticalChain.txt")
        except ValueError:
            print "Experiment directory, Atmosphere File, and HWP directory must all be defined in config file"
            raise 
            
        
        #==============================================================================
        #   Creates Detector and Optical Chain
        #==============================================================================
        #Imports detector data 
        self.det = dt.Detector(channelFile, cameraFile, config["bandID"], config)
        # Frequency array used throughout calculations
        self.freqs = self.det.freqs 
        
        self.elements = [] 
    
        self.elements.append(opt.OpticalElement("CMB", self.det, 2.725, {"Absorb": 1}))        
        self.elements.append(opt.loadAtm(atmFile, self.det))    
        self.elements += opt.loadOpticalChain(opticsFile, self.det, hwpDir , theta=config["theta"])       
        self.elements.append(opt.OpticalElement("Detector", self.det, self.det.bath_temp, {"Absorb": 1 - self.det.det_eff})) 
        
        # Finds index of HWP
        try:
            self.hwpIndex = [e.name for e in self.elements].index("HWP")
            self.hwp = self.elements[self.hwpIndex]
        except ValueError:
            print "No HWP in Optical Chain"
            raise
      
        
        #==============================================================================
        #   Calculates conversion to KRJ and KCMB for the telescope
        #==============================================================================
        # Watts to KCMB
        aniSpec  = lambda x: th.aniPowSpec(1, x, Tcmb)
        self.toKcmb = 1/intg.quad(aniSpec, self.det.flo, self.det.fhi)[0]
        # Watts to KRJ
        self.toKRJ = 1 /(th.kB *self.det.band_center * self.det.fbw)
        
        #Propagates Unpolarized Spectrum through each element
        self.propSpectrum()        
        # Gets A2, A4, a2 and a4
        self.getHWPSS(fit = True)

#        print self.hwpssTable()
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
                
                # Protects against multiple modulations by the HWP
                if e.name == "HWP":
                    e.polCreated = np.zeros(len(self.freqs))
                    
                #Sets incident power on next element
                if i + 1 < len(self.elements):
                    self.elements[i+1].unpolIncident    = e.unpolCreated + e.unpolTransmitted
                    self.elements[i+1].polIncident      = e.polCreated   + e.polTransmitted
                    
                
    def getHWPSS(self, fit = False):
        # Incoming and Reflected stokes parameters
        IT = self.hwp.unpolIncident + self.hwp.polIncident
        QT = self.hwp.polIncident
        
        IR = self.hwp.unpolReverse + self.hwp.polReverse
        QR = self.hwp.polReverse
    

        #==============================================================================
        #   Calculation of A2 and A4
        #==============================================================================
        A2TSpec,  A4TSpec     = [], []
        A2RSpec,  A4RSpec     = [], []
        # These are saved for final table
        A2spec = [[],[],[],[]]   # UnpolFW polFW unpolBW polBW
        A4spec = [[],[],[],[]]   # UnpolFW polFW unpolBW polBW
        
        # Gets A2 and A4 for transmission and reflection at each frequency
        for (i, f) in enumerate(self.freqs):
            
            
            
            A2T, A4T = self.hwp.getHWPSS(f, np.array([IT[i],QT[i], 0, 0]), reflected = False, fit = fit)                        
            A2TSpec.append(A2T)
            A4TSpec.append(A4T)
            
            A2R, A4R = self.hwp.getHWPSS(f, np.array([IR[i],QR[i], 0, 0]), reflected = True, fit = fit)
            A2RSpec.append(A2R)
            A4RSpec.append(A4R)
        
            A2emitted = .5 * self.hwp.polEmitted
        
        
        
        # Efficiency between HWP and detector
        eff = self.cumEff(self.freqs, start = self.hwpIndex)        
        # Modulated signal at the detector
        A2TSpec     = np.array(A2TSpec) * eff
        A2RSpec     = np.array(A2RSpec) * eff
        A2emitted   = np.array(A2emitted) * eff
        
        A4TSpec     = np.array(A4TSpec) * eff
        A4RSpec     = np.array(A4RSpec) * eff
        # Calculation of total A2 and A4 signals
        self.A4 =  th.powFromSpec(self.freqs, A4TSpec) + th.powFromSpec(self.freqs, A4RSpec)
        self.A2 =  th.powFromSpec(self.freqs, A2TSpec) + th.powFromSpec(self.freqs, A2RSpec) + th.powFromSpec(self.freqs, A2emitted)

        #==============================================================================
        #   a4 calculation    
        #==============================================================================
        self.a4  = 0
        for e in self.elements[:self.hwpIndex]:
            self.a4 += e.Ip(self.det.band_center)
        
        #==============================================================================
        #   a2 calculation    
        #==============================================================================
        self.a2 = self.hwp.MTave[0,1]




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
        
        #==============================================================================
        #   Prints HWP info  
        #==============================================================================
        
        tableString += "Frequency: %i GHz\t fbw: %.3f\n\n"%(self.det.band_center/ GHz, self.det.fbw)
        headers = ["Incident", "FW Unpol", "BW Unpol", "FW Pol", "BW Pol"]
        units = ["", "[pW]", "[pW]", "[pW]", "[pW]"]
        
        spectra = [self.hwp.unpolIncident, self.hwp.unpolReverse, self.hwp.polIncident, self.hwp.polReverse]
        A4conv = [self.hwp.A4upT, self.hwp.A4upR, self.hwp.A4ppT, self.hwp.A4ppR]
        A2conv = [self.hwp.A2upT, self.hwp.A2upR, self.hwp.A2ppT, self.hwp.A2ppR]
        A4row = []
        A2row = []
        for (i, s) in enumerate(spectra):
            A4row.append( th.powFromSpec(self.freqs, s * A4conv[i]) * pW) 
            A2row.append( th.powFromSpec(self.freqs, s * A2conv[i]) * pW)
        
#        A4spectra = np.array(spectra) * np.array([self.hwp.A4upT, self.hwp.A4upR, self.hwp.A4ppT, self.hwp.A4ppR])
#        A4row = np.array(map(lambda x : th.powFromSpec(self.freqs, x), A4spectra)) * pW
#        A2spectra = np.array(spectra) * np.array([self.hwp.A2upT, self.hwp.A2upR, self.hwp.A2ppT, self.hwp.A2ppR])
#        A2row = np.array(map(lambda x : th.powFromSpec(self.freqs, x), A2spectra)) * pW
        
        
        powers = map(lambda x : th.powFromSpec(self.freqs, x), spectra)
        powers = np.array(powers) * pW
        tableString += self._formatRow(headers)
        tableString += self._formatRow(units)
        tableString += "-" * 70 + "\n"
        tableString += self._formatRow(["Power"] +  map(lambda x : "%.3e"%x, powers))
        tableString += self._formatRow(["A4 (@HWP)"] +  map(lambda x : "%.3e"%x, A4row))
        tableString += self._formatRow(["A2 (@HWP)"] +  map(lambda x : "%.3e"%x, A2row))
        
        tableString += '\n'
        tableString += '-'*70+ '\n'
        tableString += '-'*70+ '\n'
        tableString += '\n'
        
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
    config["theta"] = np.deg2rad(20.)


    tel = Telescope(config)
    
    print tel.hwpssTable()
    
    print "Telescope A4: ", tel.A4 * pW / tel.cumEff(tel.det.band_center, start = tel.hwpIndex)
    print "Telescope A2: ", tel.A2 * pW
#    print tel.hwp.params["Mueller_T"]
    