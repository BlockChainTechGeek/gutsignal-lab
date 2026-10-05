"""Deterministic perturbations of generated demonstration audio, not clinical evidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from gutsignal.audio import AudioClip
from gutsignal.inference import predict_clip

STRESS_CASES = {
    "Unchanged": ("Unchanged", 0.0),
    "Gain -12 dB": ("Gain", -12.0),
    "Gain -30 dB": ("Gain", -30.0),
    "Noise 20 dB SNR": ("Noise", 20.0),
    "Noise 10 dB SNR": ("Noise", 10.0),
    "Noise 0 dB SNR": ("Noise", 0.0),
    "Clipping x8": ("Clipping", 8.0),
    "Dropout 25%": ("Dropout", 25.0),
    "Dropout 50%": ("Dropout", 50.0),
    "Silence": ("Silence", 0.0),
    "Constant DC": ("Constant DC", 0.0),
}


def perturb(clip: AudioClip, kind: str, level: float, seed: int = 42) -> AudioClip:
    signal = clip.signal.copy()
    if not np.isfinite(level):
        raise ValueError("Expected a finite perturbation level")
    if kind == "Noise":
        if not 0 <= level <= 40:
            raise ValueError("Noise SNR must be between 0 and 40 dB")
        noise = np.random.default_rng(seed).normal(size=signal.size)
        noise /= np.sqrt(np.mean(noise**2))
        rms = np.sqrt(np.mean(signal**2))
        signal += noise * rms / (10 ** (level / 20))
    elif kind == "Gain":
        if not -60 <= level <= 0:
            raise ValueError("Gain must be between -60 and 0 dB")
        signal *= 10 ** (level / 20)
    elif kind == "Clipping":
        if not 1 <= level <= 20:
            raise ValueError("Clipping multiplier must be between 1 and 20")
        signal = np.clip(signal * level, -1, 1)
    elif kind == "Dropout":
        if not 0 <= level <= 100:
            raise ValueError("Dropout must be between 0 and 100 percent")
        lost = round(signal.size * level / 100)
        start = (signal.size - lost) // 2
        signal[start : start + lost] = 0
    elif kind == "Silence":
        signal[:] = 0
    elif kind == "Constant DC":
        signal[:] = 0.25
    elif kind != "Unchanged":
        raise ValueError("Unknown perturbation")
    # Keep the noise experiment within the digital range without changing its SNR.
    if kind == "Noise":
        peak = np.max(np.abs(signal))
        if peak > 0.98:
            signal *= 0.98 / peak
    return AudioClip(signal, clip.sample_rate)


def stress_matrix(clips: dict[str, AudioClip], bundle: dict[str, Any]) -> list[dict]:
    rows = []
    for name, clip in clips.items():
        clean = predict_clip(clip, bundle)
        for case, (kind, level) in STRESS_CASES.items():
            result = predict_clip(perturb(clip, kind, level), bundle)
            delta = (
                100 * (result.probability - clean.probability)
                if result.probability is not None and clean.probability is not None
                else None
            )
            rows.append(
                {
                    "example": name,
                    "condition": case,
                    "clean_score": clean.probability,
                    "score": result.probability,
                    "change_percentage_points": delta,
                    "state": result.state,
                    "state_changed": result.state != clean.state,
                    "quality_accepted": result.quality.accepted,
                    "reason": " ".join(result.quality.reasons),
                }
            )
    return rows
