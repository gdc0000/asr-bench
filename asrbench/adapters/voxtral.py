from __future__ import annotations

import torch
from transformers import VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor

from asrbench.adapters.base import ModelInfo
from asrbench.registry import MODELS


class VoxtralAdapter:
    def __init__(self, model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602"):
        info = next(m for m in MODELS if m.id == model_id)
        self.info = info
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load(self) -> None:
        self.processor = VoxtralRealtimeProcessor.from_pretrained(self.model_id)
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

    def transcribe(self, audio, sr: int = 16000) -> str:
        if self.model is None:
            self.load()
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {
            k: (v.to(self.model.device, dtype=torch.bfloat16) if torch.is_tensor(v) and torch.is_floating_point(v) else (v.to(self.model.device) if torch.is_tensor(v) else v))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()

    def unload(self) -> None:
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()
