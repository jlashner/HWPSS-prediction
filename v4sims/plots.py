import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GHz = 1e9

fig_dir = 'figs'

band_files = ["inputs/bandpasses/MF1_w_OMT.txt", "inputs/bandpasses/MF2_w_OMT.txt"]

bps = list(np.loadtxt(band_files[0], unpack=True))
bps.append(np.loadtxt(band_files[1], unpack=True)[1])

def plot_bps():

    for b in [1,2]:
        mask = bps[b] > 0.5

        xmin, xmax = bps[0][mask][0], bps[0][mask][-1]
        plt.axvspan(xmin, xmax, color='red', alpha=0.3)

def plot_bp_curve(ax):
    for b in [1,2]:
        ax.plot(bps[0], bps[1])
        ax.plot(bps[0], bps[2])


def mod_effs(layers):

    for l in layers:
        freqs, mod_effs, pol_angles = np.loadtxt(f'outputs/out_{l}Layer.txt', unpack=True)
        plt.plot(freqs/GHz, mod_effs * 100, label=f"{l} layer")

    plot_bps()

    plt.xlim(60, 180)
    plt.ylim(90, 101)
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Modulation Efficiency [%]")

    plt.legend(frameon=False, bbox_to_anchor=(.58, .2),)

    plt.savefig('outputs/plots/mod_effs_3_4_5.pdf')
    plt.show()


def pol_angles(show=True):
    layers = [3,  5]
    offsets = {3: 61, 4: 0, 5: 0}

    fig, ax = plt.subplots(figsize=(16, 12))
    for i, l in enumerate(layers):
        freqs, mod_effs, pol_angles = np.loadtxt(f'outputs/out_{l}Layer.txt', unpack=True)

        plt.plot(freqs / GHz, np.rad2deg(pol_angles) - offsets[l], label=f"{l} layer")

    plot_bps()
    if show:
        plt.xlim(60, 180)
        plt.ylim(-10, 10)
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Polarization Angle Variation [deg]")

        plt.legend(
            frameon=False, bbox_to_anchor=(.6, 1), bbox_transform=ax.transAxes
        )
        plt.tight_layout()

        plt.savefig(f'outputs/plots/pol_angles_3_5.pdf')
        plt.show()


def plot_band_averages():
    fig, ax = plt.subplots(1,1)

    x = np.load('outputs/3Layer_spec_averages.npz')
    bcs = x['band_centers'] / GHz
    specs = x['specs']
    shifts = x['shifts']
    angles = np.rad2deg(x['polangles'])

    for i in [0,1]:
        cmb = angles[i, 4, :]
        for j in [0,1,2,3]:
            rel_angles = angles[i,j,:] - cmb

            ax.plot(shifts, np.abs(rel_angles - rel_angles[0]), '-o', label=specs[j])

    plt.legend()
    plt.show()

def make_latex_table():
    x = np.load('outputs/3Layer_spec_averages.npz')
    bcs = x['band_centers'] / GHz
    specs = x['specs']
    shifts = x['shifts']
    angles = np.rad2deg(x['polangles'])

    print(angles.shape)
    cmb = angles[:, 4, :]
    sync = angles[:, 0, :]
    dust = angles[:, 3, :]

    def proc(ang, bid):
        d = (ang - cmb)[bid]
        return np.abs(d - d[0])

    df = pd.DataFrame({
        'shift': shifts,
        'sync (90 GHz)': proc(sync, 0),
        'sync (150 GHz)': proc(sync, 1),
        'dust (90 GHz)': proc(dust, 0),
        'dust (150 GHz)': proc(sync, 1),
    })
    print(df.to_latex(
        index=False, column_format='c',
        float_format="%.3f"
    ))




if __name__=='__main__':
    # mod_effs([3, 4, 5])
    pol_angles()
    # plot_band_averages()
    # make_latex_table()