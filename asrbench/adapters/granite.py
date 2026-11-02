from __future__ import annotations

import torch
from transformers import AutoModelForCTC, AutoProcessor

from asrbench.adapters.base import ModelInfo
from asrbench.core.registry import MODELS


class GraniteAdapter:
    """CTC greedy frame-wise: forward + argmax, nessun generate."""

    def __init__(self, model_id: str = "ibm-granite/granite-speech-5.0-470m-turboctc"):
        """Lega la scheda registry, pesi ancora vuoti."""
        info = next(m for m in MODELS if m.id == model_id)
        self.info = info
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load(self) -> None:
        """Carica AutoProcessor + AutoModelForCTC in fp16, eval."""
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCTC.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        ).to("cuda")  # ZeroGPU: niente device_map (serve accelerate e rompe l'hijack)
        self.model.eval()

    def transcribe(self, audio, sr: int = 16000) -> str:
        """Logit CTC -> argmax per frame -> batch_decode."""
        if self.model is None:
            self.load()
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(ids)[0]
        return text.strip()

    def unload(self) -> None:
        """Rilascia riferimenti. I pesi scope restano (condivisi tra main e worker)."""
        self.model = None
        self.processor = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # main in emulation o contesto senza CUDA: niente da liberare