from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

TARGET_SAMPLE_RATE = 8_000
BANDPASS_LOW_HZ = 40.0
BANDPASS_HIGH_HZ = 2_000.0


def load_and_preprocess(path: Path) -> tuple[np.ndarray, int]:
    signal, sample_rate = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True)
    signal = signal.astype(np.float64, copy=False)
    signal -= np.mean(signal)

    nyquist = sample_rate / 2
    sos = butter(
        6,
        [BANDPASS_LOW_HZ / nyquist, BANDPASS_HIGH_HZ / nyquist],
        btype="bandpass",
        output="sos",
    )
    signal = sosfiltfilt(sos, signal)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return signal.astype(np.float32), sample_rate


def _summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(flattened)),
        f"{prefix}_std": float(np.std(flattened)),
        f"{prefix}_p10": float(np.quantile(flattened, 0.10)),
        f"{prefix}_p90": float(np.quantile(flattened, 0.90)),
    }


def _band_energy(power: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def extract_features(path: Path) -> dict[str, float]:
    signal, sample_rate = load_and_preprocess(path)
    n_fft = 512
    hop_length = 128

    spectrum = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    power = np.square(spectrum)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    total_power = float(np.sum(power)) + 1e-12

    features: dict[str, float] = {
        "duration_seconds": float(len(signal) / sample_rate),
        "signal_rms": float(np.sqrt(np.mean(np.square(signal)))),
        "signal_abs_p95": float(np.quantile(np.abs(signal), 0.95)),
        "low_band_fraction": _band_energy(power, frequencies, 80, 400) / total_power,
        "high_band_fraction": _band_energy(power, frequencies, 500, 1_700) / total_power,
    }

    features.update(_summary("zcr", librosa.feature.zero_crossing_rate(signal)))
    features.update(
        _summary("centroid", librosa.feature.spectral_centroid(S=spectrum, sr=sample_rate))
    )
    features.update(
        _summary("bandwidth", librosa.feature.spectral_bandwidth(S=spectrum, sr=sample_rate))
    )
    features.update(
        _summary("rolloff", librosa.feature.spectral_rolloff(S=spectrum, sr=sample_rate))
    )
    features.update(_summary("flatness", librosa.feature.spectral_flatness(S=spectrum)))

    mfcc = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
    )
    for index, coefficient in enumerate(mfcc, start=1):
        features[f"mfcc_{index:02d}_mean"] = float(np.mean(coefficient))
        features[f"mfcc_{index:02d}_std"] = float(np.std(coefficient))

    return features
