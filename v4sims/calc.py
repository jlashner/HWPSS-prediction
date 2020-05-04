import transfer_matrix as tm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.constants as const
GHz = 1e9


class Band:
    def __init__(self, input, shift=0.0):
        self.freqs, self.bandpass = np.loadtxt(input, unpack=True)
        self.freqs *= GHz

        self.freqs += shift

        self._spline = interp1d(self.freqs, self.bandpass, kind='quadratic')

        self.band_int = np.trapz(self.bandpass, x=self.freqs)
        self.band_center = np.trapz(self.freqs * self.bandpass, x=self.freqs) / self.band_int

    def bp(self, f):
        """Returns bandpass"""
        if isinstance(f, np.ndarray):
            out = np.zeros_like(f)
            mask = np.logical_and(f > self.freqs[0], f < self.freqs[-1])
            out[mask] = self._spline(f[mask])
            out[~mask] = 0
            return out

        else:
            if self.freqs[0] < f < self.freqs[-1]:
                return 0
            else:
                return self._spline(f)

    def band_average(self, freqs, signal):
        y = signal * self.bp(freqs)
        return np.trapz(y, x=freqs) / self.band_int

band_files = [f"inputs/bandpasses/MF{b}_w_OMT.txt" for b in [1,2]]
bands = [Band(f) for f in band_files]


def demodFit(x, *args):
    N = (len(args) - 1) // 2
    d = args[0]
    for n in range(1, N + 1):
        amp, phase = args[2 * n - 1], args[2 * n]
        d += amp * np.cos(n * x - phase)

    return d

def calc_params(chis, demod)
    p = [1, 1, 0, 1, 0, 1, 0, 1, np.deg2rad(70), 1, 0, 1, 0, 1, 0, 1, 0]
    popt, pcov = curve_fit(demodFit, chis, demod, p)

    A0, A4 = popt[0], popt[2 * 4 - 1]
    phi4 = popt[2 * 4]

    if A4 < 0:
        A4 = A4 * -1
        phi4 += np.pi

    return A4/A0,

def calc_modeffs(in_file):
    x = np.load(in_file)
    freqs, chis, muellers = x['freqs'], x['chis'], x['muellers']

    S_in = np.array([1, 1, 0, 0])
    M_det = .5 * np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])
    p = [1, 1, 0, 1, 0, 1, 0, 1, np.deg2rad(70), 1, 0, 1, 0, 1, 0, 1, 0]

    mod_effs = []
    pol_angles = []
    for i,f in enumerate(freqs):
        demod = [M_det.dot(m.dot(S_in))[0] for m in muellers[i]]
        popt, pcov = curve_fit(demodFit, chis, demod, p)
        A0, A4 = popt[0], popt[2*4 - 1]
        phi4 = popt[2*4]
        if A4 < 0:
            A4 = A4 * -1
            phi4 += np.pi
        pol_angles.append(phi4 / 4)
        mod_effs.append(np.abs(A4/A0))
    return freqs, mod_effs, pol_angles

def run_modeffs():
    for layer in [3,4,5]:
        print(f"Running layer {layer}")
        input = f'inputs/hwp_models/{layer}Layer/muellers_3LayerAR.npz'
        out = np.array(calc_modeffs(input)).T
        np.savetxt(f'outputs/out_{layer}Layer.txt', out, header='freq\tmod_eff\tpol_angle')


def calc_band_averaged(in_file, band, spec_in=None):
    x = np.load(in_file)
    freqs, chis, muellers = x['freqs'], x['chis'], x['muellers']

    demod = []
    M_det = .5 * np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])

    if spec_in is not None:
        spec_norm = np.trapz(band.bp(freqs) * spec_in(freqs), x=freqs)

    for i, chi in enumerate(chis):
        sig = []
        for j, freq in enumerate(freqs):
            m = muellers[j,i]

            if spec_in is None:
                S_in = np.array([1,1,0,0])
            else:
                S_in = spec_in(freq) * np.array([1,1,0,0]) / spec_norm
            # print(S_in)

            sig.append(M_det.dot(m.dot(S_in))[0])

        demod.append(band.band_average(freqs, sig))

    plt.plot(chis, demod)
    p = [1, 1, 0, 1, 0, 1, 0, 1, np.deg2rad(70), 1, 0, 1, 0, 1, 0, 1, 0]
    popt, pcov = curve_fit(demodFit, chis, demod, p)

    A0, A4 = popt[0], popt[2*4-1]
    phi4 = popt[2*4]

    if A4 < 0:
        A4 = A4 * -1
        phi4 += np.pi

    mod_eff = np.abs(A4 / A0)
    pol_angle = phi4 / 4
    return mod_eff, pol_angle


def run_band_average():
    layer = 3
    infile = f'inputs/hwp_models/{layer}Layer/muellers_3LayerAR.npz'

    def power_law(freq, index):
        return freq ** index

    def blackbody(freq, T, beta=0):
        return freq ** (3 + beta) / (np.exp(const.h * freq / (const.k * T)) - 1)

    specs = [
        lambda f: power_law(f, -3),
        lambda f: power_law(f, 2),
        lambda f: power_law(f, 0),
        lambda f: blackbody(f, 19.6, beta=1.59),
        lambda f: blackbody(f, 2.725)
    ]

    spec_labels = np.array(['pl_-3', 'pl_+2', 'pl_0', 'dust', 'bb_2.725'])
    shift_percs = np.array([0, 0.5, 1, 2, 3, 4, 5]) / 100

    band_labels = np.array([90,150])

    band_centers = [[],[]]

    polangles = np.zeros((2, len(spec_labels), len(shift_percs)))
    modeffs = np.zeros((2, len(spec_labels), len(shift_percs)))
    for i in [0, 1]:
        shifts = [bands[i].band_center * p for p in shift_percs]
        shiftbands = [Band(band_files[i], shift=s) for s in shifts]
        band_centers[i] = [b.band_center for b in shiftbands]
        for j,spec in enumerate(specs):
            for k,b in enumerate(shiftbands):

                print(i,j,k)
                m, p = calc_band_averaged(infile, b, spec_in=spec)
                modeffs[i,j,k] = m
                polangles[i,j,k] = p
    band_centers = np.array(band_centers)

    out_file = f'outputs/{layer}layer_spec_averages'
    np.savez(out_file, specs=spec_labels, shifts=shift_percs,
             band_centers=band_centers, band_labels=band_labels,
             polangles=polangles, modeffs=modeffs)


if __name__ == "__main__":
    run_modeffs()