"""Bounded, in-memory decoding and explicit engineering input checks.

These rules are prototype heuristics, not validated acoustic-quality detectors.
Passing a check does not establish that audio is abdominal or in distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import soundfile as sf

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MIN_DURATION = 1.9
MAX_DURATION = 2.1


class AudioInputError(ValueError):
    """Safe error text suitable for showing to a reviewer."""


@dataclass(frozen=True)
class AudioClip:
    signal: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class QualityCheck:
    accepted: bool
    reasons: tuple[str, ...]
    duration_seconds: float
    rms: float
    clipped_fraction: float
    flat_fraction: float


def decode_wav(payload: bytes) -> AudioClip:
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise AudioInputError("Choose a non-empty WAV file smaller than 8 MiB.")
    try:
        with sf.SoundFile(BytesIO(payload)) as source:
            if source.format not in {"WAV", "WAVEX"}:
                raise AudioInputError("The file content must be WAV, not just the filename.")
            if source.channels != 1:
                raise AudioInputError("Choose mono audio. Stereo channel mixing is not validated.")
            if not 8_000 <= source.samplerate <= 192_000:
                raise AudioInputError("Choose audio sampled between 8 kHz and 192 kHz.")
            duration = source.frames / source.samplerate
            if not MIN_DURATION <= duration <= MAX_DURATION:
                raise AudioInputError("Choose a two-second clip (1.9 to 2.1 seconds).")
            signal = source.read(dtype="float64")
            sample_rate = source.samplerate
    except (sf.SoundFileError, ValueError, OSError) as error:
        if isinstance(error, AudioInputError):
            raise
        raise AudioInputError("This WAV could not be decoded. Choose another file.") from error
    if not np.isfinite(signal).all():
        raise AudioInputError("The audio contains invalid numeric samples.")
    return AudioClip(signal, sample_rate)


def check_quality(clip: AudioClip) -> QualityCheck:
    signal = np.asarray(clip.signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
        raise AudioInputError("Expected non-empty, finite mono audio.")
    if not 8_000 <= clip.sample_rate <= 192_000:
        raise AudioInputError("Unsupported audio sample rate.")
    duration = signal.size / clip.sample_rate
    # Centre before measuring level so constant DC is not mistaken for useful audio.
    rms = float(np.sqrt(np.mean((signal - signal.mean()) ** 2)))
    clipped = float(np.mean(np.abs(signal) >= 0.999))
    # Flat 20 ms windows are a coarse dropout/contact-loss proxy, not an SNR measure.
    window_size = max(1, round(clip.sample_rate * 0.02))
    count = signal.size // window_size
    flat = (
        float(
            np.mean(
                np.ptp(signal[: count * window_size].reshape(count, window_size), axis=1) < 1e-8
            )
        )
        if count
        else 1.0
    )
    reasons = []
    if not MIN_DURATION <= duration <= MAX_DURATION:
        reasons.append("Outside the two-second input window.")
    if rms < 1e-5:
        reasons.append("No usable variation: silence, a constant signal or very low level.")
    if clipped > 0.01:
        reasons.append("More than 1% of samples are near full scale: possible clipping.")
    if flat >= 0.20:
        reasons.append("At least 20% of short windows are flat: possible signal loss.")
    return QualityCheck(not reasons, tuple(reasons), duration, rms, clipped, flat)


def encode_wav(clip: AudioClip) -> bytes:
    """Generated workbench preview only; uploads are never written to disk."""
    buffer = BytesIO()
    sf.write(buffer, clip.signal, clip.sample_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()
