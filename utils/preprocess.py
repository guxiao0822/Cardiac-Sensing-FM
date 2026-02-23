import neurokit2 as nk
import numpy as np
import scipy.signal

fs_target = 250  # Target resampling frequency
signal_length = 2500  # Final signal length after truncation (samples)

def truncate_signal(signal, length=signal_length):
    if signal.shape[1] < length:
        # Pad with zeros if too short
        pad_width = length - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
    elif signal.shape[1] > length:
        # Random crop if too long
        start = np.random.randint(0, signal.shape[1] - length)
        signal = signal[:, start:start + length]
    return signal

def preprocess_ecg(ecg, fs=250):
    '''
    Standardize the format of ECG input and apply preprocessing:
    - Resample to fs_target
    - Clean using neurokit2
    - Z-normalize

    Parameters:
        ecg (np.ndarray): 1D or 2D array (channels x time or time,)
        fs (int): Original sampling rate

    Returns:
        np.ndarray: Preprocessed ECG (channels x time)
    '''
    # Ensure 2D shape (channels x time)
    if ecg.ndim == 1:
        ecg = ecg[np.newaxis, :]  # shape: (1, time)
    elif ecg.shape[0] > ecg.shape[1]:
        ecg = ecg.T  # shape: (channels, time)

    # Resample to target frequency
    if fs != fs_target:
        ecg = scipy.signal.resample(ecg, int(ecg.shape[1] * fs_target / fs), axis=1)

    # Clean and z-normalize each channel
    ecg_clean = []
    for i in range(ecg.shape[0]):
        clean = nk.ecg_clean(ecg[i], sampling_rate=fs_target)
        ecg_clean.append(clean)

    ecg_clean = np.stack(ecg_clean, axis=0)
    ecg_clean = truncate_signal(ecg_clean, length=signal_length)

    mean = np.mean(ecg_clean, axis=1, keepdims=True)
    std = np.std(ecg_clean, axis=1, keepdims=True)
    ecg_clean = (ecg_clean - mean) / (std + 1e-8)

    return ecg_clean


def preprocess_ppg(ppg, fs=250):
    '''
    Standardize the format of PPG input and apply preprocessing:
    - Resample to fs_target
    - Clean using neurokit2
    - Z-normalize

    Parameters:
        ppg (np.ndarray): 1D or 2D array (channels x time or time,)
        fs (int): Original sampling rate

    Returns:
        np.ndarray: Preprocessed PPG (channels x time)
    '''
    # Ensure 2D shape (channels x time)
    if ppg.ndim == 1:
        ppg = ppg[np.newaxis, :]  # shape: (1, time)
    elif ppg.shape[0] > ppg.shape[1]:
        ppg = ppg.T  # shape: (channels, time)

    # Resample to target frequency
    if fs != fs_target:
        ppg = scipy.signal.resample(ppg, int(ppg.shape[1] * fs_target / fs), axis=1)

    # Clean and z-normalize each channel
    ppg_clean = nk.ppg_clean(ppg[0], sampling_rate=fs_target)
    ppg_clean = ppg_clean[np.newaxis, :]
    ppg_clean = truncate_signal(ppg_clean)

    mean = np.mean(ppg_clean, axis=1, keepdims=True)
    std = np.std(ppg_clean, axis=1, keepdims=True)
    ppg_clean = (ppg_clean - mean) / (std + 1e-8)

    return ppg_clean

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs_original = 100  # Original sampling rate
    duration = 12  # seconds
    t = np.linspace(0, duration, duration * fs_original)

    ecg_raw = nk.ecg_simulate(duration=duration, sampling_rate=fs_original)
    ppg_raw = nk.ppg_simulate(duration=duration, sampling_rate=fs_original)

    print("Raw ECG shape:", ecg_raw.shape)
    print("Raw PPG shape:", ppg_raw.shape)

    ecg_clean = preprocess_ecg(ecg_raw, fs=fs_original)
    ppg_clean = preprocess_ppg(ppg_raw, fs=fs_original)

    print("Processed ECG shape:", ecg_clean.shape)  # Should be (1, 2500)
    print("Processed PPG shape:", ppg_clean.shape)  # Should be (1, 2500)

    t_ecg = np.linspace(0, signal_length / fs_target, signal_length)
    t_ppg = np.linspace(0, signal_length / fs_target, signal_length)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_ecg, ecg_clean[0], label="Preprocessed ECG")
    plt.title("ECG", fontsize=16)
    plt.xlabel("Time (s)", fontsize=16)

    plt.subplot(1, 2, 2)
    plt.plot(t_ppg, ppg_clean[0], label="Preprocessed PPG", color='orange')
    plt.title("PPG", fontsize=16)
    plt.xlabel("Time (s)", fontsize=16)

    plt.tight_layout()
    plt.show()

