from __future__ import annotations

import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor, Qwen3ASRConfig

from asrbench.adapters.base import ModelInfo
from asrbench.core.registry import MODELS


class QwenAdapter:
    """LLM audio: checkpoint thinker.* rimappato, prompt con lingua fissata a italiano."""

    def __init__(self, model_id: str = "Qwen/Qwen3-ASR-0.6B"):
        """Lega la scheda registry, pesi ancora vuoti."""
        info = next(m for m in MODELS if m.id == model_id)
        self.info = info
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load(self) -> None:
        """Ricostruisce config da thinker_config, rimappa pesi, sposta su CUDA fp16."""
        self.processor = Qwen3ASRProcessor.from_pretrained(self.model_id)
        import json
        from huggingface_hub import hf_hub_download
        from transformers import Qwen3ASREncoderConfig, Qwen3Config
        path = hf_hub_download(self.model_id, "config.json")
        raw = json.load(open(path))
        tc = raw["thinker_config"]
        audio_cfg = {k: v for k, v in tc["audio_config"].items() if k != "model_type"}
        text_cfg = {k: v for k, v in tc["text_config"].items() if k != "model_type"}
        config = Qwen3ASRConfig(
            audio_config=Qwen3ASREncoderConfig(**audio_cfg),
            text_config=Qwen3Config(**text_cfg),
            audio_token_id=tc["audio_token_id"],
        )
        self.model = Qwen3ASRForConditionalGeneration(config)
        weight_path = hf_hub_download(self.model_id, "model.safetensors")
        raw_state = load_file(weight_path)
        state = {}
        for key, value in raw_state.items():
            if key == "thinker.audio_tower.proj1.weight":
                state["model.multi_modal_projector.linear_1.weight"] = value
            elif key == "thinker.audio_tower.proj1.bias":
                state["model.multi_modal_projector.linear_1.bias"] = value
            elif key == "thinker.audio_tower.proj2.weight":
                state["model.multi_modal_projector.linear_2.weight"] = value
            elif key == "thinker.audio_tower.proj2.bias":
                state["model.multi_modal_projector.linear_2.bias"] = value
            elif key.startswith("thinker.audio_tower."):
                state["model.audio_tower." + key[len("thinker.audio_tower."):]] = value
            elif key.startswith("thinker.model."):
                state["model.language_model." + key[len("thinker.model."):]] = value
            elif key == "thinker.lm_head.weight":
                state["lm_head.weight"] = value
        self.model.load_state_dict(state, strict=True)
        self.model.to("cuda", dtype=torch.float16)
        self.model.eval()

    def transcribe(self, audio, sr: int = 16000) -> str:
        """Prompt chat + audio -> generate -> taglia il prompt -> decode."""
        if self.model is None:
            self.load()
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        text_prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        text_prompt = text_prompt + "language Italian<asr_text>"
        inputs = self.processor(text=text_prompt, audio=audio, sampling_rate=sr, return_tensors="pt")
        inputs = {
            k: (v.to(self.model.device, dtype=torch.float16) if torch.is_floating_point(v) else v.to(self.model.device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=256)
        if "input_ids" in inputs:
            generated = generated[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()

    def unload(self) -> None:
        """Rilascia pesi e libera VRAM."""
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()