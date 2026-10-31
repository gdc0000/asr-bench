from __future__ import annotations

import sounddevice as sd
import numpy as np

from asrbench.audio import to_mono_16k


def record(seconds: float = 5.0) -> tuple[np.ndarray, int]:
    sr = 16000
    frames = int(seconds * sr)
    data = sd.rec(frames, samplerate=sr, channels=1, dtype="float32", device=0)
    sd.wait()
    audio = data.reshape(-1)
    audio, sr = to_mono_16k((sr, audio))
    return audio, sr