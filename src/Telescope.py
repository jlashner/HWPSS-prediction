import numpy as np
import scipy.integrate as intg
import thermo as th
import OpticalElement as opt
import Detector as dt
import matplotlib.pyplot as plt
import HWP_model
import os
import json

try:
    from tqdm import *
except:
    tqdm = lambda x : x


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
        self.reflection_order = config["Reflection Order"]
        try:
            expDir      = config["ExperimentDirectory"]
            atmFile     = config["AtmosphereFile"]
            hwpDir     = config["HWPDirectory"]

            channelFile = os.path.join(expDir, "channels.txt")
            cameraFile  = os.path.join(expDir, "camera.txt")
            opticsFile  = os.path.join(expDir, "opticalChain.txt")

        except ValueError:
            print("Experiment directory, Atmosphere File, and HWP directory must all be defined in config file")
            raise 

        # ==============================================================================
        #   Creates Detector and Optical Chain
        # ==============================================================================
        #Imports detector data 
        self.det = dt.Detector(channelFile, cameraFile, config["bandID"], config)
        # Frequency array used throughout calculations
        self.freqs = self.det.freqs 
        
        #==============================================================================
        #   Calculates conversion to KRJ and KCMB for the telescope
        #==============================================================================
        # Watts to KCMB
        aniSpec = np.array([th.aniPowSpec(1, f) for f in self.freqs])
        self.toKcmb = 1/(np.trapz(aniSpec, self.freqs))

#        self.toKcmb /= self.cumEff(self.det.band_center)
        # Watts to KRJ
        self.toKRJ = 1 /(2 * th.kB *self.det.band_center * self.det.fbw)
        
        
        self.elements = [] 
    
        self.elements.append(opt.OpticalElement("CMB", self.det, 2.725, {"Absorb": 1}))        
        self.elements.append(opt.loadAtm(atmFile, self.det, self.toKcmb, self.toKRJ))    
        self.elements += opt.loadOpticalChain(opticsFile, self.det, hwpDir , theta=config["theta"])       
        self.elements.append(opt.OpticalElement("Detector", self.det, self.det.bath_temp, {"Absorb": 1 - self.det.det_eff})) 
        
        # Finds index of HWP
        try:
            self.hwpIndex = [e.name for e in self.elements].index("HWP")
            self.hwp = self.elements[self.hwpIndex]
        except ValueError:
            print("No HWP in Optical Chain")
            raise

#        self.toKRJ /= self.cumEff(self.det.band_center)
        
        # Propagates Unpolarized Spectrum through each element
        self.propSpectrum(ReflectionOrder=self.reflection_order)
        # Gets A2, A4, a2 and a4
        self.getHWPSS(fit=False)

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
                
               
    def propSpectrum(self, ReflectionOrder = 2, ignore_emis=False):
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
            
            if ignore_emis:
                e.unpolEmitted = np.zeros(len(self.freqs))
                e.polEmitted = np.zeros(len(self.freqs))
            
        
        for n in range(ReflectionOrder):
            for (i,e) in enumerate(self.elements):
                if i == 1 & ignore_emis:
                    e.unpolIncident = np.ones(len(self.freqs))
                
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
                    
                
    def getHWPSS(self, fit=False):
        # Incoming and Reflected stokes parameters
        IT = self.hwp.unpolIncident + self.hwp.polIncident
        QT = self.hwp.polIncident 
#        QT = np.zeros(np.shape(QT))
            
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
        
        a2spec = []
        test = []
        test2 = []
        # Gets A2 and A4 for transmission and reflection at each frequency
        for (i, f) in enumerate(self.freqs):
            
            A2T, A4T = self.hwp.getHWPSS(f, np.array([IT[i],QT[i], 0, 0]), reflected = False, fit = fit)            
            A2TSpec.append(A2T)
            A4TSpec.append(A4T)
            
            A2R, A4R = self.hwp.getHWPSS(f, np.array([IR[i],QR[i], 0, 0]), reflected = True, fit = fit)
            A2RSpec.append(A2R)
            A4RSpec.append(A4R)
            
            _, t = self.hwp.getHWPSS(f, np.array([IT[i] - QT[i],0, 0, 0]), reflected = False, fit = fit)
            test.append(t)
            
            _, t = self.hwp.getHWPSS(f, np.array([IR[i] - QR[i],0, 0, 0]), reflected = False, fit = fit)
            test2.append(t)
        print()

        # plt.plot(self.freqs, A2TSpec)
        # plt.plot(self.freqs, A4TSpec)
        # plt.show()

        A2emitted = .5 * self.hwp.polEmitted

        
        
        # Efficiency between HWP and detector
        eff = self.cumEff(self.freqs, start = self.hwpIndex)        
        # Modulated signal at the detector
        A2TSpec     = np.array(A2TSpec) * eff
        A2RSpec     = np.array(A2RSpec) * eff
        A2emitted   = np.array(A2emitted) * eff
        
        test = np.array(test) * eff
        
        A4TSpec     = np.array(A4TSpec) * eff
        A4RSpec     = np.array(A4RSpec) * eff
#       
#        plt.plot(self.freqs, A4TSpec / self.cumEff(self.freqs))
        
        print("A2 from Transmission: %f"%(th.powFromSpec(self.freqs, A2TSpec)* self.toKcmb * 2 / self.cumEff(self.det.band_center)))
        print("A2 from Reflection: %f"%(th.powFromSpec(self.freqs, A2RSpec)* self.toKcmb * 2 / self.cumEff(self.det.band_center)))
        print("A2 from Emission: %f"%(th.powFromSpec(self.freqs, A2emitted)* self.toKcmb * 2 / self.cumEff(self.det.band_center)))
        print("A4 from transmission: %f"%(th.powFromSpec(self.freqs, test)* self.toKcmb * 2 / self.cumEff(self.det.band_center)))
        print("A4 from Reflection: %f"%(th.powFromSpec(self.freqs, A4RSpec)* self.toKcmb * 2 / self.cumEff(self.det.band_center)))
        
#        A2_trans.append(th.powFromSpec(self.freqs, A2TSpec)* self.toKcmb * 2 / self.cumEff(self.det.band_center))
#        A2_refl.append(th.powFromSpec(self.freqs, A2RSpec)* self.toKcmb * 2 / self.cumEff(self.det.band_center))
#        A2_emit.append(th.powFromSpec(self.freqs, A2emitted)* self.toKcmb * 2 / self.cumEff(self.det.band_center))
#        hwp_A4[0].append(th.powFromSpec(self.freqs, test)* self.toKcmb * 2 / self.cumEff(self.det.band_center))
#        hwp_A4[1].append(th.powFromSpec(self.freqs, A4RSpec)* self.toKcmb * 2 / self.cumEff(self.det.band_center))
#   
#        print("A4 from Transmission: %f"%(th.powFromSpec(self.freqs, A4TSpec) * self.toKRJ))
#        print("A4 from Reflection: %f"%(th.powFromSpec(self.freqs, A4RSpec)*self.toKcmb))

        
        # A2 and A4 signals seen by detector
        self.A4 =  th.powFromSpec(self.freqs, A4TSpec) + th.powFromSpec(self.freqs, A4RSpec)
        self.A2 =  th.powFromSpec(self.freqs, A2TSpec) + th.powFromSpec(self.freqs, A2RSpec) + th.powFromSpec(self.freqs, A2emitted)
        
        
        
        # A2 and A4 in KRJ
        self.A2_KRJ = self.A2 * self.toKRJ * 2 / self.cumEff(self.det.band_center)
        self.A4_KRJ = self.A4 * self.toKRJ * 2 / self.cumEff(self.det.band_center)
        
        self.A2_Kcmb = self.A2 * self.toKcmb * 2 / self.cumEff(self.det.band_center)
        self.A4_Kcmb = self.A4 * self.toKcmb * 2 / self.cumEff(self.det.band_center)




        #==============================================================================
        #   a4 calculation    
        #==============================================================================
        self.propSpectrum(ReflectionOrder = self.reflection_order, ignore_emis = True)        
#        ip  = 0
#        for e in self.elements[:self.hwpIndex]:
#            ip += e.Ip(self.det.band_center)


        IT = self.hwp.unpolIncident + self.hwp.polIncident
        QT = self.hwp.polIncident 
            
        IR = self.hwp.unpolReverse + self.hwp.polReverse
        QR = self.hwp.polReverse


        a2spec = []
        a4spec = []

        for f in self.freqs:
            a2, a4 = self.hwp.getHWPSS(f, np.array([IT, QT, 0, 0]), reflected = False, fit = fit)
            a2spec.append(a2)
            a4spec.append(a4)
            
            
        eff = self.cumEff(self.det.band_center, end = self.hwpIndex)
        self.a2 = np.mean(a2spec) * 2 / eff
        self.a4 = np.mean(a4spec) * 2 / eff
        
        a2spec = []
        a4spec = []
        
        for f in self.freqs:
            a2, a4 = self.hwp.getHWPSS(f, np.array([IR, QR, 0, 0]), reflected = True, fit = fit)
            a2spec.append(a2)
            a4spec.append(a4)
            
            
        eff = self.cumEff(self.det.band_center, end = self.hwpIndex)
        self.a2 += np.mean(a2spec) * 2 / eff
        self.a4 += np.mean(a4spec) * 2 / eff
        
        self.propSpectrum(ReflectionOrder = self.reflection_order, ignore_emis = False)        

                            

if __name__=="__main__":

    config = json.load(open("../run/config.json"))
    config["theta"] = np.deg2rad(20.)
    config["WriteOutput"] = False


    config["bandID"] = 2
    config["ExperimentDirectory"] = "../Experiments/V3/MF_baseline/SmallTelescope"
    config["HWPDirectory"] = "../HWP/4LayerSapphire"    
    config["Reflection Order"] = 10
    tel = Telescope(config)

#    atm = tel.elements[1]
#    plt.plot(tel.freqs, atm.unpolEmitted * tel.toKRJ)
#    
#    win = tel.elements[2]

#    print("Atmosphere power: ",    th.powFromSpec(tel.freqs,atm.unpolEmitted) * tel.toKcmb)
#
#    print("Window power: ",    th.powFromSpec(tel.freqs,win.unpolEmitted) * tel.toKcmb)
#    
#    eff = tel.cumEff(tel.det.band_center)
#    
#    al_filter = tel.elements[7]
#    window = tel.elements[2]
#    print("Filter diff trans: ", al_filter.Ip(tel.det.band_center)*100)
#    print("Window diff trans: ", window.Ip(tel.det.band_center)*100)
#    
#    print("Filter IP: ", th.powFromSpec(tel.freqs, al_filter.IpTransmitted) * tel.toKcmb / tel.cumEff(tel.det.band_center, end=7))
#    print("Filter polReflection: ", th.powFromSpec(tel.freqs, al_filter.unpolReverse * al_filter.pRefl(tel.freqs)) * tel.toKcmb / tel.cumEff(tel.det.band_center, end=7))
#    print("Window IP: ", th.powFromSpec(tel.freqs, window.IpTransmitted) * tel.toKcmb / tel.cumEff(tel.det.band_center, end=3))
#    print("Window polReflection: ", th.powFromSpec(tel.freqs, window.unpolReverse * window.pRefl(tel.freqs)) * tel.toKcmb / tel.cumEff(tel.det.band_center, end=3))


    print("A2: %f"%(tel.A2_Kcmb))
    print("a2: %f"%(tel.a2 * 100))
    print("A4: %f"%(tel.A4_Kcmb))
    print("a4: %f"%(tel.a4 * 100))
    
#    angles = np.arange(0, 21)
#    #Trans, refl
#    win_A4 = [[],[]]
#    al_A4 = [[],[]]
#    hwp_A4 = [[],[]]
#    
#    A2_trans = []
#    A2_refl = []
#    A2_emit = []
    
#    for theta in tqdm(angles):
#        config["theta"] = np.deg2rad(theta)
#        tel = Telescope(config)
#    
#        al_filter = tel.elements[7]
#        window = tel.elements[2]
#        hwp = tel.hwp
#        
#        al_A4[0].append(th.powFromSpec(tel.freqs, al_filter.IpTransmitted) \
#             * tel.toKcmb / tel.cumEff(tel.det.band_center, end=7))
#        al_A4[1].append(th.powFromSpec(tel.freqs, al_filter.unpolReverse * al_filter.pRefl(tel.freqs)) * tel.toKcmb / tel.cumEff(tel.det.band_center, end=7))
#        
#        win_A4[0].append(th.powFromSpec(tel.freqs, window.IpTransmitted) * tel.toKcmb / tel.cumEff(tel.det.band_center, end=3))
#        win_A4[1].append(th.powFromSpec(tel.freqs, window.unpolReverse * window.pRefl(tel.freqs)) * tel.toKcmb / tel.cumEff(tel.det.band_center, end=3))

#    for band in ["LF", "MF", "UHF"]:
#        for bid in [1,2]:
#            config["bandID"] = bid
#            config["ExperimentDirectory"]="../Experiments/V3/%s_baseline/SmallTelescope"%band
#            tel = Telescope(config) 
#            print("Freq: %f GHz"%(tel.det.band_center / GHz))
#            print("A2: %.4f, A4: %.4f"%(tel.A2*tel.toKcmb, tel.A4*tel.toKcmb))
#            print("a2: %.4f, a4: %.4f"%(tel.a2*100, tel.a4*100))
    
#
#    
#    with open("../HWPSS_145GHz.txt", 'w') as file:
#         file.write("theta\tA2\tA4\ta2\ta4\n")
#         for theta in range(21):
#             config["theta"] = np.deg2rad(theta)
#             tel = Telescope(config)
#             file.write("%d\t%.5f\t%.5f\t%.5f\t%.5f\n"%(theta, tel.A2_Kcmb, tel.A4_Kcmb, tel.a2*100, tel.a4*100))
         

    