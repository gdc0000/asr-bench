from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ModelInfo:
    """Scheda modello: registry per UI e selezione adapter."""
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
    """Protocollo adapter: load, transcribe, unload. Un modello in memoria alla volta."""

    info: ModelInfo

    def load(self) -> None:
        """Scarica pesi da HF, sposta su device, mette in eval."""
        ...

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:  # noqa: F821
        """Trascrive audio mono 16k. Ritorna testo pulito."""
        ...

    def unload(self) -> None:
        """Rilascia pesi e libera VRAM."""
        ...