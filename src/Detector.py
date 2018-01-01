import numpy as np

# Units and Constants
GHz = 1.e9 # GHz -> Hz
pW = 1.e12 # W -> pW

class Detector:
    def __init__(self, channelFile, cameraFile, bandID, config):
        self.bid = bandID

        ch_str = np.loadtxt(channelFile, dtype=np.str)
        self.band_center = float(ch_str[bandID][2])*GHz #[Hz]
        self.fbw = float(ch_str[bandID][3]) #fractional bandwidth
        self.pixSize = float(ch_str[bandID][4])/1000.
        self.waistFact = float(ch_str[bandID][6])
        self.det_eff = float(ch_str[bandID][7])

        cam_str = np.loadtxt(cameraFile, dtype=np.str, usecols=[2])
        self.f_num = float(cam_str[2])
        self.bath_temp = float(cam_str[2])



        self.flo = self.band_center*(1 - .5 * self.fbw) #detector lower bound [Hz]
        self.fhi = self.band_center*(1 + .5 * self.fbw) #detector upper bound [Hz]

        if config["LowerFreq"] and config["UpperFreq"]:
            self.flo = config["LowerFreq"]
            self.fhi = config["UpperFreq"]
            self.band_center = (self.flo + self.fhi) / 2
            self.fbw =  (self.fhi - self.flo) / self.band_center