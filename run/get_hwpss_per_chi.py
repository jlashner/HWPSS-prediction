# This script evaluates the full HWPSS as a function of HWP angle (chi)

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import transfer_matrix as tm
import Telescope as tp
import thermo as th
from Fit_HWPSS import calcHWPSSCoeffs

# theta is the incident angle
# chi is the HWP angle


fname_db = "hwpss_per_chi.pck"

bands = "LF", "MF", "UHF"
band_ids = 1, 2

n_chi = 100  # number of HWP angles to evaluate
chis = np.linspace(0, 2 * np.pi, n_chi)

n_theta = 21
thetas = np.arange(n_theta)

all_stokes = {}

hwpmodel = "3Layer"
index_in, index_out = 1, 1
n_stokes = 4

# Detector Mueller matrix

m_det = np.array(
    [
        # Q a
        0.5
        * np.vstack(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
        # Q b
        0.5
        * np.vstack(
            [
                [1, -1, 0, 0],
                [-1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
        # U a
        0.5
        * np.vstack(
            [
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 0],
            ]
        ),
        # U b
        0.5
        * np.vstack(
            [
                [1, 0, -1, 0],
                [0, 0, 0, 0],
                [-1, 0, 1, 0],
                [0, 0, 0, 0],
            ]
        ),
    ]
)
n_det = len(m_det)


def decompose(chis, sig):
    # Linear regression templates for measuring 2f, 4f and 6f

    templates = np.vstack(
        [
            np.ones(chis.size),
            np.cos(2 * chis),
            np.sin(2 * chis),
            np.cos(4 * chis),
            np.sin(4 * chis),
            np.cos(6 * chis),
            np.sin(6 * chis),
        ]
    )
    invcov = np.dot(templates, templates.T)
    cov = np.linalg.inv(invcov)
    proj = np.dot(templates, sig)
    coeff = np.dot(cov, proj)
    sig2f = np.dot(coeff[1:3], templates[1:3])
    sig4f = np.dot(coeff[3:5], templates[3:5])
    sig6f = np.dot(coeff[5:7], templates[5:7])
    return sig2f, sig4f, sig6f


def fit_hwpss(HWP_data_dir, theta_deg):
    path = os.path.join(HWP_data_dir, "{}_deg".format(theta_deg))
    os.makedirs(path, exist_ok=True)
    fname_trans = os.path.join(path, "Trans.npy")
    if not os.path.exists(fname_trans):
        print(
            f"{fname_trans} does not exist, Caching ...",
            flush=True,
        )
        data = calcHWPSSCoeffs(stack, theta=theta_rad, reflected=False, band=band)
        np.save(fname_trans[:-4], data)
    fname_refl = os.path.join(path, "Refl.npy")
    if not os.path.exists(fname_refl):
        print(
            f"{fname_refl} does not exist, Caching ...",
            flush=True,
        )
        data = calcHWPSSCoeffs(stack, theta=theta_rad, reflected=True, band=band)
        np.save(fname_refl[:-4], data)
    return


def get_telescope(theta_rad, band_id, HWP_dir):
    config = {
        "theta": theta_rad,
        "bandID": band_id,
        "AtmosphereFile": "../src/Atacama_1000um_50deg.txt",
        "ExperimentDirectory": f"../Experiments/V3/{band}_baseline/SmallTelescope",
        "HWPDirectory": HWP_dir,
        "Reflection Order": 10,
        "WriteOutput": False,
        # "OutputDirectory": "out",
        "LowerFreq": None,
        "UpperFreq": None,
        "BandCenter": None,
        "FBW": None,
        "getHWPSS": False,
    }
    tel = tp.Telescope(config)
    return tel


def get_per_freq(chis, n_chi, n_stokes, n_freq, tel):

    trans_per_freq = np.zeros([n_chi, n_stokes, n_freq])
    refl_per_freq = np.zeros([n_chi, n_stokes, n_freq])
    emit_per_freq = np.zeros([n_chi, n_stokes, n_freq])

    for i_freq, freq in enumerate(tqdm(tel.hwp.freqs)):
        # Evaluate the HWP Mueller matrices for
        # transmitted, reflected and emitted light
        m_trans = []
        m_refl = []
        m_emit = []

        # f = tel.det.band_center
        for chi in chis:
            transfer = tm.stackTransferMatrix(
                stack, freq, theta_rad, chi, index_in, index_out
            )
            jones = tm.TranToJones(transfer)
            m_trans.append(tm.JonesToMueller(jones[0]))
            m_refl.append(tm.JonesToMueller(jones[1]))
            # The emitted polarization is given for chi=0 so we must
            # rotate it ourselves
            m_emit.append(
                np.vstack(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(2 * chi), np.sin(2 * chi), 0],
                        [0, -np.sin(2 * chi), np.cos(2 * chi), 0],
                        [0, 0, 0, 1],
                    ]
                )
            )
        m_trans = np.array(m_trans)
        m_refl = np.array(m_refl)

        # Emitted, Incoming and Reflected Stokes parameters

        IT = (tel.hwp.unpolIncident + tel.hwp.polIncident)[i_freq]
        QT = tel.hwp.polIncident[i_freq]

        IR = (tel.hwp.unpolReverse + tel.hwp.polReverse)[i_freq]
        QR = tel.hwp.polReverse[i_freq]

        IE = (tel.hwp.unpolEmitted + tel.hwp.polEmitted)[i_freq]
        QE = tel.hwp.polEmitted[i_freq]

        incident = np.array([IT, QT, 0, 0])
        reflected = np.array([IR, QR, 0, 0])
        emitted = np.array([IE, QE, 0, 0])

        # Mueller matrices modify the Stokes vectors

        for i_chi, chi in enumerate(chis):
            trans_per_freq[i_chi, :, i_freq] = np.dot(m_trans[i_chi], incident)
            refl_per_freq[i_chi, :, i_freq] = np.dot(m_refl[i_chi], reflected)
            emit_per_freq[i_chi, :, i_freq] = np.dot(m_emit[i_chi], emitted)

    return trans_per_freq, refl_per_freq, emit_per_freq


def get_per_chi(n_chi, n_stokes, temperature):
    trans_per_chi = np.zeros([n_chi, n_stokes])
    refl_per_chi = np.zeros([n_chi, n_stokes])
    emit_per_chi = np.zeros([n_chi, n_stokes])
    for i_chi in range(n_chi):
        for i_stokes in range(n_stokes):
            trans_per_chi[i_chi, i_stokes] = temperature(
                trans_per_freq[i_chi, i_stokes]
            )
            refl_per_chi[i_chi, i_stokes] = temperature(refl_per_freq[i_chi, i_stokes])
            emit_per_chi[i_chi, i_stokes] = temperature(emit_per_freq[i_chi, i_stokes])
    return trans_per_chi, refl_per_chi, emit_per_chi


def cache_results(
    all_stokes, i_theta, trans_per_chi, refl_per_chi, emit_per_chi, fname_db
):
    all_stokes[band_name]["transmission"][i_theta] = trans_per_chi
    all_stokes[band_name]["reflection"][i_theta] = refl_per_chi
    all_stokes[band_name]["emission"][i_theta] = emit_per_chi

    with open(fname_db, "wb") as fout:
        pickle.dump([thetas, chis, all_stokes], fout)
        print(f"Saved results to {fname_db}")

    return


def plot_hwpss(full, trans, refl, emit, hwpmodel, band_name, theta_deg):

    fig = plt.figure(figsize=[18, 12])
    fig.suptitle(
        f"{hwpmodel} HWP, {band_name}GHz, " f"incident angle = {theta_deg} deg",
    )
    nrow, ncol = 2, 2
    scale = 1e3
    units = "mK_CMB"

    for i, (sig, name) in enumerate(
        [
            [full, "Total HWPSS"],
            [trans, "Transmitted HWPSS"],
            [refl, "Reflected HWPSS"],
            [emit, "Emitted HWPSS"],
        ]
    ):
        ax = fig.add_subplot(nrow, ncol, 1 + i)
        ax.set_title(name)
        offset = np.round(np.mean(sig), 3)
        ax.plot(
            chis, (sig[0] - offset) * scale, lw=4, label=f"Full A - {offset} {units}"
        )
        ax.plot(
            chis, (sig[1] - offset) * scale, "--", label=f"Full B - {offset} {units}"
        )
        sig2f, sig4f, sig6f = decompose(chis, sig[0])
        ax.plot(chis, sig2f * scale, label="2f")
        ax.plot(chis, sig4f * scale, label="4f")
        ax.plot(chis, sig6f * scale, label="6f")
        ax.legend(loc="upper right")
        ax.set_xlabel("HWP angle")
        ax.set_ylabel(f"[{units}]")

    plt.savefig(
        f"hwpss_per_chi.{hwpmodel}.{band_name}GHz." f"angle{theta_deg:03}.png",
    )
    plt.close()

    return


def load_results(fname_db):
    if os.path.isfile(fname_db):
        with open(fname_db, "rb") as fin:
            print(f"Loading an existing databse from {fname_db}")
            thetas_test, chis_test, all_stokes = pickle.load(fin)
        if not np.array_equal(thetas, thetas_test) or not np.array_equal(
            chis, chis_test
        ):
            print(
                "WARNING: existing database does not match specified grid, "
                "will erase and start anew."
            )
            all_stokes = {}
    else:
        all_stokes = {}
    return all_stokes


all_stokes = load_results(fname_db)

for band in bands:
    HWP_dir = f"../HWP/{hwpmodel}Sapphire"
    HWP_band_dir = f"../HWP/{hwpmodel}Sapphire/{band}"
    HWP_data_dir = f"../HWP/{hwpmodel}Sapphire/{band}/HWPSS_data"
    # Load HWP Models
    mats = tm.loadMaterials(os.path.join(HWP_band_dir, "materials.txt"))
    stack = tm.loadStack(mats, os.path.join(HWP_band_dir, "stack.txt"))

    for band_id in band_ids:
        for i_theta, theta_deg in enumerate(thetas):
            theta_rad = np.radians(theta_deg)

            # Ensure the HWPSS is evaluated
            fit_hwpss(HWP_data_dir, theta_deg)

            tel = get_telescope(theta_rad, band_id, HWP_dir)
            band_center_ghz = int(tel.det.band_center * 1e-9)
            band_name = f"{band_center_ghz:03}"
            if band_name not in all_stokes:
                all_stokes[band_name] = {
                    "transmission": np.zeros([n_theta, n_chi, n_stokes]),
                    "reflection": np.zeros([n_theta, n_chi, n_stokes]),
                    "emission": np.zeros([n_theta, n_chi, n_stokes]),
                }
            else:
                if np.any(all_stokes[band_name]["transmission"][i_theta] != 0):
                    print(f"{band_name} {theta_deg} already evaluated")
                    continue
            print(
                f"\nband = {band}, ID = {band_id}, freq = {band_name}, "
                f"ang = {theta_deg}"
            )
            eff = tel.cumEff(tel.freqs, start=tel.hwpIndex)
            eff_center = tel.cumEff(tel.det.band_center)

            def temperature(ps):
                return th.powFromSpec(tel.freqs, ps * eff) * tel.toKcmb * 2 / eff_center

            print("Incoming unpol:", temperature(tel.hwp.unpolIncident))
            print("Incoming pol:", temperature(tel.hwp.polIncident))
            print("Reflected unpol", temperature(tel.hwp.unpolReverse))
            print("Reflected pol:", temperature(tel.hwp.polReverse))
            print("Emitted unpol:", temperature(tel.hwp.unpolEmitted))
            print("Emitted pol", temperature(tel.hwp.polEmitted))

            # Calculate a Mueller matrix for each frequency and rotation angle

            n_freq = len(tel.hwp.freqs)

            trans_per_freq, refl_per_freq, emit_per_freq = get_per_freq(
                chis,
                n_chi,
                n_stokes,
                n_freq,
                tel,
            )

            trans_per_chi, refl_per_chi, emit_per_chi = get_per_chi(
                n_chi, n_stokes, temperature
            )

            cache_results(
                all_stokes, i_theta, trans_per_chi, refl_per_chi, emit_per_chi, fname_db
            )

            trans = np.zeros([n_det, n_chi])
            refl = np.zeros([n_det, n_chi])
            emit = np.zeros([n_det, n_chi])
            for i_det in range(n_det):
                for i_chi in range(n_chi):
                    trans[i_det, i_chi] = np.dot(m_det[i_det], trans_per_chi[i_chi])[0]
                    refl[i_det, i_chi] = np.dot(m_det[i_det], refl_per_chi[i_chi])[0]
                    emit[i_det, i_chi] = np.dot(m_det[i_det], emit_per_chi[i_chi])[0]
            full = trans + refl + emit

            plot_hwpss(full, trans, refl, emit, hwpmodel, band_name, theta_deg)
