from __future__ import annotations

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asrbench.adapters.base import ModelInfo
from asrbench.core.registry import MODELS


class WhisperAdapter:
    """Encoder-decoder autoregressivo, forzato su italiano."""

    def __init__(self, model_id: str = "openai/whisper-small"):
        """Lega la scheda registry, pesi ancora vuoti."""
        info = next(m for m in MODELS if m.id == model_id)
        self.info = info
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load(self) -> None:
        """Carica WhisperProcessor + modello in fp16, eval."""
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()

    def transcribe(self, audio, sr: int = 16000) -> str:
        """Log-mel -> generate(language=it, task=transcribe) -> decode."""
        if self.model is None:
            self.load()
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs["input_features"].to(self.model.device, dtype=torch.float16)
        with torch.no_grad():
            generated = self.model.generate(
                input_features,
                language="it",
                task="transcribe",
                max_new_tokens=256,
            )
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()

    def unload(self) -> None:
        """Rilascia pesi e libera VRAM."""
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()