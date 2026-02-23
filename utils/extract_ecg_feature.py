import numpy as np
import neurokit2 as nk

def extract_ecg_feature(ecg_signal, fs):
    """
    Extracts morphological, HRV, and SQI features from an ECG signal.

    Args:
        ecg_signal (ArrayLike): The ECG signal data.
        fs (float): The sampling rate of the ECG signal.

    Returns:
        dict: Dictionary of ECG features including morphological, HRV, and SQI features.
    """
    # Clean the ECG signal
    ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=fs)

    # Detect R-peaks
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)

    if len(rpeaks['ECG_R_Peaks']) <= 3:
        return np.zeros(54)  # Return a zero array if there are insufficient R-peaks

    # Delineate waves
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="peak")

    # Extract morphological ECG features
    features_rpeaks = from_Rpeaks(ecg_signal, rpeaks['ECG_R_Peaks'], sampling_rate=fs, average=True)

    features_waveforms = from_waves(ecg_signal, rpeaks['ECG_R_Peaks'], waves_peak, sampling_rate=fs, average=True)

    # Extract HRV features
    features_hrv = np.nan_to_num(nk.hrv_time(rpeaks['ECG_R_Peaks'], sampling_rate=fs)).squeeze() #.values().fillna(0)# .fillna(0).to_dict(orient='records')[0]

    try:
        # Extract signal quality index (SQI)
        sqi = nk.ecg_quality(ecg_signal, rpeaks=rpeaks['ECG_R_Peaks'], sampling_rate=fs, method='zhao2018', approach='fuzzy')
        sqi_value = {'Excellent': [1], 'Unacceptable': [0], 'Barely acceptable': [0.5]}
        sqi = sqi_value[sqi]
    except:
        sqi = 0

    # Combine all features into a single vector
    features = np.concatenate([
        np.array(list(features_rpeaks.values())),
        np.array(list(features_waveforms.values())),
        np.array(features_hrv),
        np.array(sqi)
    ])

    return features

def from_Rpeaks(sig, peaks_locs, sampling_rate, prefix="ecg", average=False):
    """Calculates R-peak-based ECG features and returns a dictionary of features for each heart beat.

    Args:
        sig (ArrayLike): ECG signal segment.
        peaks_locs (ArrayLike): ECG R-peak locations.
        sampling_rate (float): Sampling rate of the ECG signal (Hz).
        prefix (str, optional): Prefix for the feature. Defaults to 'ecg'.
        average (bool, optional): If True, averaged features are returned. Defaults to False.

    Returns:
        dict: Dictionary of ECG features.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    features_rpeaks = {}
    for m in range(1, len(peaks_locs) - 2):
        features = {}
        for key, func in FEATURES_RPEAKS.items():
            try:
                features["_".join([prefix, key])] = func(sig, sampling_rate, peaks_locs=peaks_locs, beatno=m)
            except:
                features["_".join([prefix, key])] = np.nan
        features_rpeaks[m] = features

    if average:
        features_avr = {}

        features_ = {}
        for subdict in features_rpeaks.values():
            for key, value in subdict.items():
                if key not in features_:
                    features_[key] = [value]
                else:
                    features_[key].append(value)

        for k in features_.keys():
            features_avr[k] = np.mean(features_[k])

        return features_avr

    else:
        return features_rpeaks


# Define necessary functions for R-peak features
def amplitude_R(sig, sampling_rate, peaks_locs, beatno):
    return sig[peaks_locs[beatno]]


def RR0(sig, sampling_rate, peaks_locs, beatno):
    return (peaks_locs[beatno] - peaks_locs[beatno - 1]) / sampling_rate


def RR1(sig, sampling_rate, peaks_locs, beatno):
    return (peaks_locs[beatno + 1] - peaks_locs[beatno]) / sampling_rate


def RR2(sig, sampling_rate, peaks_locs, beatno):
    return (peaks_locs[beatno + 2] - peaks_locs[beatno + 1]) / sampling_rate


def RRm(sig, sampling_rate, peaks_locs, beatno):
    return np.mean([RR0(sig, sampling_rate, peaks_locs, beatno), RR1(sig, sampling_rate, peaks_locs, beatno),
                    RR2(sig, sampling_rate, peaks_locs, beatno)])


def RR_0_1(sig, sampling_rate, peaks_locs, beatno):
    return RR0(sig, sampling_rate, peaks_locs, beatno) / RR1(sig, sampling_rate, peaks_locs, beatno)


def RR_2_1(sig, sampling_rate, peaks_locs, beatno):
    return RR2(sig, sampling_rate, peaks_locs, beatno) / RR1(sig, sampling_rate, peaks_locs, beatno)


def RR_m_1(sig, sampling_rate, peaks_locs, beatno):
    return RRm(sig, sampling_rate, peaks_locs, beatno) / RR1(sig, sampling_rate, peaks_locs, beatno)


# Define the feature functions in a dictionary
FEATURES_RPEAKS = {
    'a_R': amplitude_R,
    'RR0': RR0,
    'RR1': RR1,
    'RR2': RR2,
    'RRm': RRm,
    'RR_0_1': RR_0_1,
    'RR_2_1': RR_2_1,
    'RR_m_1': RR_m_1
}


# Define the feature functions for wave features
def t_PR(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(P_peaks) and P_peaks[beatno] is not None:
        return (R_peaks[beatno] - P_peaks[beatno]) / sampling_rate
    return np.nan

def t_QR(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(Q_peaks) and Q_peaks[beatno] is not None:
        return (R_peaks[beatno] - Q_peaks[beatno]) / sampling_rate
    return np.nan

def t_RS(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(S_peaks) and S_peaks[beatno] is not None:
        return (S_peaks[beatno] - R_peaks[beatno]) / sampling_rate
    return np.nan

def t_RT(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(T_peaks) and T_peaks[beatno] is not None:
        return (T_peaks[beatno] - R_peaks[beatno]) / sampling_rate
    return np.nan

def t_PQ(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(P_peaks) and beatno < len(Q_peaks) and P_peaks[beatno] is not None and Q_peaks[beatno] is not None:
        return (Q_peaks[beatno] - P_peaks[beatno]) / sampling_rate
    return np.nan

def t_PS(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(P_peaks) and beatno < len(S_peaks) and P_peaks[beatno] is not None and S_peaks[beatno] is not None:
        return (S_peaks[beatno] - P_peaks[beatno]) / sampling_rate
    return np.nan

def t_PT(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(P_peaks) and beatno < len(T_peaks) and P_peaks[beatno] is not None and T_peaks[beatno] is not None:
        return (T_peaks[beatno] - P_peaks[beatno]) / sampling_rate
    return np.nan

def t_QS(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(Q_peaks) and beatno < len(S_peaks) and Q_peaks[beatno] is not None and S_peaks[beatno] is not None:
        return (S_peaks[beatno] - Q_peaks[beatno]) / sampling_rate
    return np.nan

def t_QT(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(Q_peaks) and beatno < len(T_peaks) and Q_peaks[beatno] is not None and T_peaks[beatno] is not None:
        return (T_peaks[beatno] - Q_peaks[beatno]) / sampling_rate
    return np.nan

def t_ST(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(S_peaks) and beatno < len(T_peaks) and S_peaks[beatno] is not None and T_peaks[beatno] is not None:
        return (T_peaks[beatno] - S_peaks[beatno]) / sampling_rate
    return np.nan

def a_PQ(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(P_peaks) and beatno < len(Q_peaks) and P_peaks[beatno] is not None and Q_peaks[beatno] is not None:
        return sig[Q_peaks[beatno]] - sig[P_peaks[beatno]]
    return np.nan

def a_QR(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(Q_peaks) and beatno < len(R_peaks) and Q_peaks[beatno] is not None and R_peaks[beatno] is not None:
        return sig[R_peaks[beatno]] - sig[Q_peaks[beatno]]
    return np.nan

def a_RS(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(R_peaks) and beatno < len(S_peaks) and R_peaks[beatno] is not None and S_peaks[beatno] is not None:
        return sig[S_peaks[beatno]] - sig[R_peaks[beatno]]
    return np.nan

def a_ST(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(S_peaks) and beatno < len(T_peaks) and S_peaks[beatno] is not None and T_peaks[beatno] is not None:
        return sig[T_peaks[beatno]] - sig[S_peaks[beatno]]
    return np.nan

def a_PS(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(P_peaks) and beatno < len(S_peaks) and P_peaks[beatno] is not None and S_peaks[beatno] is not None:
        return sig[S_peaks[beatno]] - sig[P_peaks[beatno]]
    return np.nan

def a_PT(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(P_peaks) and beatno < len(T_peaks) and P_peaks[beatno] is not None and T_peaks[beatno] is not None:
        return sig[T_peaks[beatno]] - sig[P_peaks[beatno]]
    return np.nan

def a_QS(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(Q_peaks) and beatno < len(S_peaks) and Q_peaks[beatno] is not None and S_peaks[beatno] is not None:
        return sig[S_peaks[beatno]] - sig[Q_peaks[beatno]]
    return np.nan

def a_QT(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(Q_peaks) and beatno < len(T_peaks) and Q_peaks[beatno] is not None and T_peaks[beatno] is not None:
        return sig[T_peaks[beatno]] - sig[Q_peaks[beatno]]
    return np.nan

def a_ST_QS(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(S_peaks) and beatno < len(T_peaks) and beatno < len(Q_peaks) and Q_peaks[beatno] is not None and S_peaks[beatno] is not None and T_peaks[beatno] is not None:
        return a_ST(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno) / a_QS(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno)
    return np.nan

def a_RS_QR(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(R_peaks) and beatno < len(S_peaks) and beatno < len(Q_peaks) and Q_peaks[beatno] is not None and R_peaks[beatno] is not None and S_peaks[beatno] is not None:
        return a_RS

def a_RS_QR(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno):
    if beatno < len(R_peaks) and beatno < len(S_peaks) and beatno < len(Q_peaks) and Q_peaks[beatno] is not None and R_peaks[beatno] is not None and S_peaks[beatno] is not None:
        return a_RS(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno) / a_QR(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno)
    return np.nan

# Define the feature functions in a dictionary
FEATURES_WAVES = {
    't_PR': t_PR,
    't_QR': t_QR,
    't_RS': t_RS,
    't_RT': t_RT,
    't_PQ': t_PQ,
    't_PS': t_PS,
    't_PT': t_PT,
    't_QS': t_QS,
    't_QT': t_QT,
    't_ST': t_ST,
    'a_PQ': a_PQ,
    'a_QR': a_QR,
    'a_RS': a_RS,
    'a_ST': a_ST,
    'a_PS': a_PS,
    'a_PT': a_PT,
    'a_QS': a_QS,
    'a_QT': a_QT,
    'a_ST_QS': a_ST_QS,
    'a_RS_QR': a_RS_QR,
    # You can add other feature functions here
}

# Now, use these feature functions in the from_waves function
def from_waves(sig, R_peaks, fiducials, sampling_rate, prefix="ecg", average=False):
    """Calculates ECG features from the given fiducials and returns a dictionary of features.

    Args:
        sig (ArrayLike): ECG signal segment.
        R_peaks (ArrayLike): ECG R-peak locations.
        fiducials (dict): Dictionary of fiducial locations (keys: "ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks").
        sampling_rate (float): Sampling rate of the ECG signal (Hz).
        prefix (str, optional): Prefix for the feature. Defaults to 'ecg'.
        average (bool, optional): If True, averaged features are returned. Defaults to False.

    Returns:
        dict: Dictionary of ECG features.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    feature_list = FEATURES_WAVES.copy()

    fiducial_names = ["ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks"]
    fiducials = {key: fiducials.get(key, []) for key in fiducial_names}

    P_peaks = fiducials["ECG_P_Peaks"]
    Q_peaks = fiducials["ECG_Q_Peaks"]
    S_peaks = fiducials["ECG_S_Peaks"]
    T_peaks = fiducials["ECG_T_Peaks"]

    if len(P_peaks) == 0:
        P_features = [
            "t_PR", "t_PQ", "t_PS", "t_PT", "a_PQ", "a_PS", "a_PT"
        ]
        for key in P_features:
            feature_list[key] = lambda *args, **kwargs: 0

    if len(Q_peaks) == 0:
        Q_features = [
            "t_QR", "t_PQ", "t_QS", "t_QT", "a_PQ", "a_QR", "a_QS", "a_QT", "a_ST_QS", "a_RS_QR"
        ]
        for key in Q_features:
            feature_list[key] = lambda *args, **kwargs: 0

    if len(S_peaks) == 0:
        S_features = [
            "t_RS", "t_PS", "t_QS", "t_ST", "a_RS", "a_ST", "a_PS", "a_QS", "a_ST_QS", "a_RS_QR"
        ]
        for key in S_features:
            feature_list[key] = lambda *args, **kwargs: 0

    if len(T_peaks) == 0:
        T_features = [
            "t_RT", "t_PT", "t_QT", "t_ST", "a_ST", "a_PT", "a_QT", "a_ST_QS", "a_RS_QR"
        ]
        for key in T_features:
            feature_list[key] = lambda *args, **kwargs: 0

    features_waves = {}
    for m in range(len(R_peaks)):
        features = {}
        for key, func in feature_list.items():
            try:
                features["_".join([prefix, key])] = func(
                    sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno=m
                )
            except:
                features["_".join([prefix, key])] = np.nan
        features_waves[m] = features

    if average:
        features_avr = {}

        features_ = {}
        for subdict in features_waves.values():
            for key, value in subdict.items():
                if key not in features_:
                    features_[key] = [value]
                else:
                    features_[key].append(value)

        for k in features_.keys():
            features_avr[k] = np.nanmean(features_[k])

        return features_avr

    else:
        return features_waves
