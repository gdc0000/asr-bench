from __future__ import annotations

import re
import torch
from transformers import Nemotron3_5AsrForRNNT, Nemotron3_5AsrProcessor

from asrbench.adapters.base import ModelInfo
from asrbench.core.registry import MODELS

LANG_TAG = re.compile(r"\s*<[A-Za-z]{2,3}(-[A-Za-z]{2})?>\s*")


def clean_text(t: str) -> str:
    """Via eco del prompt lingua (es. <en-US>): artefatto, non parlato."""
    return re.sub(r"\s+", " ", LANG_TAG.sub(" ", t or "")).strip()


def _to_float(a) -> "np.ndarray":
    """Mic Gradio = float in scala int16: riporta in [-1, 1]. Idempotente."""
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 1.5:
        a = a / 32768.0
    return a


class NemotronStreamSession:
    """Streaming RNNT nativo: chunk mel a misura fissa + cache encoder/decoder.

    Pezzi: primo 1+8*R frame mel, successivi 8*(R+1). Cache encoder
    (past_key_values + padding_cache) e decoder LSTM portate avanti.
    Replicazione fedele del loop greedy HF (blank inclusi come no-op).
    """

    FIRST_N = 25
    NEXT_N = 32
    R = 3

    def __init__(self, adapter: "NemotronAdapter"):
        """Stato vergine: cache vuote, ultimo token = blank, zero frame consumati."""
        from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import (
            Nemotron3_5AsrRNNTDecoderCache,
        )
        self.adapter = adapter
        model = adapter.model
        self.dev = model.device
        self.dtype = model.dtype
        self.blank = model.config.blank_token_id
        self.max_sym = model.max_symbols_per_step
        self.raw: list = []
        self.fed_mel = 0
        self.first_done = False
        self.kv = None
        self.pad = None
        self.prompt = None
        self.last = torch.full((1, 1), self.blank, dtype=torch.long, device=self.dev)
        self.dec_cache = Nemotron3_5AsrRNNTDecoderCache(model.config)
        self.emitted: list[int] = []

    def _mel(self) -> "torch.Tensor":
        """Mel time-major [B, T, 128] dal buffer intero. Prompt lingua fissato al primo uso."""
        audio = _to_float(self.raw)
        out = self.adapter.processor(audio, sampling_rate=16000, return_tensors="pt")
        if self.prompt is None:
            self.prompt = out["prompt_ids"].to(self.dev)
        return out["input_features"]

    def _decode_frames(self, frames) -> None:
        """Loop greedy RNNT per frame: joint decoder+encoder, argmax, emit se non blank.

        Il decoder riceve anche i blank (no-op che preserva lo stato LSTM).
        Max 10 simboli per frame, come il generate ufficiale.
        """
        model = self.adapter.model
        for f in range(frames.shape[0]):
            enc_f = frames[f].view(1, 1, 1, -1)
            syms = 0
            while True:
                dec_out = model.decoder(self.last, cache=self.dec_cache).view(1, 1, 1, -1)
                tok = int(model.joint(dec_out, enc_f).argmax(-1).item())
                self.last = torch.full((1, 1), tok, dtype=torch.long, device=self.dev)
                if tok == self.blank:
                    break
                self.emitted.append(tok)
                syms += 1
                if syms >= self.max_sym:
                    break

    def _feed_pieces(self, mel) -> None:
        """Consuma pezzi completi (25 primo, 32 poi) con cache encoder portate avanti."""
        import numpy as np
        model = self.adapter.model
        T = mel.shape[1]
        while True:
            need = self.NEXT_N if self.first_done else self.FIRST_N
            if T - self.fed_mel < need:
                return
            piece = mel[:, self.fed_mel:self.fed_mel + need, :].to(self.dev, dtype=self.dtype)
            kw = dict(input_features=piece, num_lookahead_tokens=self.R, use_cache=True)
            if self.kv is not None:
                kw["past_key_values"] = self.kv
                kw["padding_cache"] = self.pad
            elif self.prompt is not None:
                kw["prompt_ids"] = self.prompt
            enc = model.get_audio_features(**kw)
            self.kv, self.pad = enc.past_key_values, enc.padding_cache
            self._decode_frames(enc.pooler_output[0])
            self.fed_mel += need
            self.first_done = True

    def feed(self, raw_chunk) -> str:
        """Accoda audio grezzo, avanza i pezzi completi, ritorna il parziale corrente."""
        self.raw.extend(_to_float(raw_chunk).tolist())
        with torch.no_grad():
            self._feed_pieces(self._mel())
        return self.text()

    def finish(self) -> str:
        """Svuota la coda con zero-padding e ritorna il testo finale."""
        import numpy as np
        with torch.no_grad():
            mel = self._mel()
            T = mel.shape[1]
            need = self.NEXT_N if self.first_done else self.FIRST_N
            rem = T - self.fed_mel
            if rem > 0:
                piece = mel[:, self.fed_mel:T, :]
                if rem < need:
                    pad = torch.zeros(1, need - rem, piece.shape[2])
                    piece = torch.cat([piece, pad], dim=1)
                kw = dict(input_features=piece.to(self.dev, dtype=self.dtype),
                          num_lookahead_tokens=self.R, use_cache=True)
                if self.kv is not None:
                    kw["past_key_values"] = self.kv
                    kw["padding_cache"] = self.pad
                elif self.prompt is not None:
                    kw["prompt_ids"] = self.prompt
                enc = self.adapter.model.get_audio_features(**kw)
                self._decode_frames(enc.pooler_output[0])
                self.fed_mel = T
        return self.text()

    def text(self) -> str:
        """Decode degli id emessi fin qui. Vuoto se nessun token."""
        if not self.emitted:
            return ""
        return clean_text(self.adapter.processor.batch_decode([self.emitted])[0])


class NemotronAdapter:
    """Batch via generate + streaming nativo via NemotronStreamSession."""

    def __init__(self, model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"):
        """Lega la scheda registry, pesi e sessione ancora vuoti."""
        info = next(m for m in MODELS if m.id == model_id)
        self.info = info
        self.model_id = model_id
        self.processor = None
        self.model = None
        self._sess = None

    def load(self) -> None:
        """Carica processor + modello RNNT in fp16, eval."""
        self.processor = Nemotron3_5AsrProcessor.from_pretrained(self.model_id)
        self.model = Nemotron3_5AsrForRNNT.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        ).to("cuda")  # ZeroGPU: niente device_map (serve accelerate e rompe l'hijack)
        self.model.eval()

    def transcribe(self, audio, sr: int = 16000) -> str:
        """Batch offline: generate sul buffer intero, eco prompt rimossa."""
        if self.model is None:
            self.load()
        audio = _to_float(audio)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {
            k: (v.to(self.model.device, dtype=torch.float16) if torch.is_tensor(v) and torch.is_floating_point(v) else (v.to(self.model.device) if torch.is_tensor(v) else v))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256)
        sequences = out.sequences if hasattr(out, "sequences") else out
        text = self.processor.batch_decode(sequences, skip_special_tokens=True)[0]
        return clean_text(text)

    def unload(self) -> None:
        """Rilascia riferimenti e sessione. I pesi scope restano (condivisi)."""
        self.model = None
        self.processor = None
        self._sess = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # main in emulation o contesto senza CUDA: niente da liberare

    def stream_start(self) -> None:
        """Apre una sessione streaming nativa."""
        if self.model is None:
            self.load()
        self._sess = NemotronStreamSession(self)

    def stream_feed(self, raw_chunk) -> str:
        """Avanza la sessione con audio nuovo, ritorna il parziale."""
        return self._sess.feed(raw_chunk)

    def stream_finish(self) -> str:
        """Chiude la sessione e ritorna il testo finale."""
        text, self._sess = self._sess.finish(), None
        return text
