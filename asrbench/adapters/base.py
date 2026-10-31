from __future__ import annotations

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ModelInfo:
    id: str
    name: str
    license: str
    languages: str
    params: str = ""
    size: str = ""
    streaming_type: str = ""
    streaming_detail: str = ""
    features: str = ""
    notes: str = ""


class Adapter(Protocol):
    info: ModelInfo

    def load(self) -> None: ...

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str: ...  # noqa: F821

    def unload(self) -> None: ...