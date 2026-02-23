from pyPPG import PPG, Fiducials, Biomarkers
from pyPPG.datahandling import plot_fiducials
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI

import numpy as np
import pandas as pd
import scipy
from dotmap import DotMap
import matplotlib.pyplot as plt

def extract_ppg_features(signal, fs):
    # Initialize signal object
    signal_obj = DotMap()
    signal_obj.v = signal
    signal_obj.fs = fs
    signal_obj.name = "example"

    # Plot raw PPG signal
    # t = np.arange(0, len(signal)) / fs
    # plt.figure()
    # plt.plot(t, signal, color='blue')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Raw PPG')
    # plt.show()

    # Configure preprocessing parameters
    signal_obj.filtering = True
    signal_obj.fL = 0.5000001
    signal_obj.fH = 12
    signal_obj.order = 4
    signal_obj.sm_wins = {'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}

    # Preprocess signals
    prep = PP.Preprocess(fL=signal_obj.fL, fH=signal_obj.fH, order=signal_obj.order, sm_wins=signal_obj.sm_wins)
    signal_obj.ppg, signal_obj.vpg, signal_obj.apg, signal_obj.jpg = prep.get_signals(s=signal_obj)

    # # Plot preprocessed signals
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    # t = np.arange(0, len(signal_obj.ppg)) / fs
    # ax1.plot(t, signal_obj.ppg)
    # ax1.set_ylabel('PPG')
    # ax2.plot(t, signal_obj.vpg)
    # ax2.set_ylabel("PPG'")
    # ax3.plot(t, signal_obj.apg)
    # ax3.set_ylabel("PPG''")
    # ax4.plot(t, signal_obj.jpg)
    # ax4.set_ylabel("PPG'''")
    # ax4.set_xlabel('Time (s)')
    # plt.show()

    # Fiducial points correction setup
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction = pd.DataFrame({key: [True] for key in corr_on})
    signal_obj.correction = correction

    # Create PPG object
    s = PPG(signal_obj)

    # Extract fiducials
    fpex = FP.FpCollection(s=s)
    fiducials = fpex.get_fiducials(s=s)
    fp = Fiducials(fp=fiducials)

    # Calculate PPG SQI
    ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
    # print('Mean PPG SQI: ', ppgSQI, '%')

    # Extract biomarkers
    bmex = BM.BmCollection(s=s, fp=fp)
    bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
    bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)

    # Compute feature vector
    feature = np.concatenate((
        bm_stats['ppg_sig'].iloc[[0, 2, 5]].values.flatten(),
        bm_stats['sig_ratios'].iloc[[0, 2, 5]].values.flatten(),
        bm_stats['ppg_derivs'].iloc[[0, 2, 5]].values.flatten(),
        bm_stats['derivs_ratios'].iloc[[0, 2, 5]].values.flatten()
    ))

    return feature




