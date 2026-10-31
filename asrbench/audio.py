from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def to_mono_16k(audio: tuple[int, np.ndarray] | np.ndarray, source_sr: int | None = None, normalize: bool = True) -> tuple[np.ndarray, int]:
    if isinstance(audio, tuple):
        sr, data = audio
    else:
        sr = source_sr
        data = audio
    if sr is None:
        raise ValueError("missing sample rate")
    data = np.asarray(data)
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    data = data.astype(np.float32)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    if normalize:
        peak = np.max(np.abs(data)) if data.size else 0.0
        if peak > 0:
            data = data / peak * 0.95
    return data, 16000


def save_wav(audio: np.ndarray, sr: int = 16000) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    sf.write(path, audio, sr)
    return path


def duration(audio: np.ndarray, sr: int = 16000) -> float:
    return float(len(audio) / sr)