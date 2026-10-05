from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from gutsignal.audio import (
    MAX_UPLOAD_BYTES,
    AudioClip,
    AudioInputError,
    check_quality,
    decode_wav,
    encode_wav,
)


def tone(duration: float = 2.0, rate: int = 8000) -> AudioClip:
    time = np.arange(round(duration * rate)) / rate
    return AudioClip(0.2 * np.sin(2 * np.pi * 180 * time), rate)


def test_valid_audio_roundtrip() -> None:
    clip = tone()
    decoded = decode_wav(encode_wav(clip))
    np.testing.assert_allclose(decoded.signal, clip.signal, atol=1e-7)
    assert check_quality(decoded).accepted


@pytest.mark.parametrize("payload", [b"", b"not a wav", b"x" * (MAX_UPLOAD_BYTES + 1)])
def test_invalid_payload_is_rejected(payload: bytes) -> None:
    with pytest.raises(AudioInputError):
        decode_wav(payload)


@pytest.mark.parametrize("duration", [0.01, 1.0, 3.0, 60.0])
def test_wrong_duration_is_rejected_before_feature_extraction(duration: float) -> None:
    with pytest.raises(AudioInputError, match="two-second"):
        decode_wav(encode_wav(tone(duration)))


def test_stereo_is_not_silently_mixed() -> None:
    signal = np.column_stack([tone().signal, -tone().signal])
    buffer = BytesIO()
    sf.write(buffer, signal, 8000, format="WAV")
    with pytest.raises(AudioInputError, match="mono"):
        decode_wav(buffer.getvalue())


def test_non_wav_content_is_rejected() -> None:
    buffer = BytesIO()
    sf.write(buffer, tone().signal, 8000, format="FLAC")
    with pytest.raises(AudioInputError, match="content must be WAV"):
        decode_wav(buffer.getvalue())


@pytest.mark.parametrize("rate", [4000, 384000])
def test_unsupported_sample_rate_is_rejected(rate: int) -> None:
    with pytest.raises(AudioInputError, match="sampled between"):
        decode_wav(encode_wav(tone(rate=rate)))


@pytest.mark.parametrize("value", [0.0, 0.25])
def test_silence_and_dc_are_withheld(value: float) -> None:
    quality = check_quality(AudioClip(np.full(16000, value), 8000))
    assert not quality.accepted
    assert quality.rms == 0


def test_near_silence_is_withheld_before_normalisation() -> None:
    clip = tone()
    assert not check_quality(AudioClip(clip.signal * 1e-6, 8000)).accepted


def test_clipping_is_withheld() -> None:
    clip = tone()
    quality = check_quality(AudioClip(np.clip(clip.signal * 20, -1, 1), 8000))
    assert not quality.accepted
    assert quality.clipped_fraction > 0.01


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_audio_is_rejected(bad: float) -> None:
    clip = tone()
    clip.signal[30] = bad
    with pytest.raises(AudioInputError, match="invalid numeric"):
        decode_wav(encode_wav(clip))


def test_empty_array_is_rejected() -> None:
    with pytest.raises(AudioInputError):
        check_quality(AudioClip(np.array([]), 8000))
