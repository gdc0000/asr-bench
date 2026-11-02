from __future__ import annotations

from asrbench.adapters.base import ModelInfo

MODELS = [
    ModelInfo(
        id="ibm-granite/granite-speech-5.0-470m-turboctc",
        name="Granite Speech 5.0 470M TurboCTC",
        license="Apache 2.0",
        languages="en fr de es pt it ja",
        params="470M",
        size="~1GB",
        streaming_type="Pseudo-streaming",
        streaming_detail="Batch CTC. Live: accumulo chunk e trascrizione ogni 2s.",
        features="CTC, punteggiatura nativa, veloce",
    ),
    ModelInfo(
        id="Qwen/Qwen3-ASR-0.6B",
        name="Qwen3 ASR 0.6B",
        license="Apache 2.0",
        languages="zh en yue ar de fr es pt id it ja ko ru th vi tr hi ... (52)",
        params="0.6B",
        size="~1.2GB",
        streaming_type="Nativo (solo vLLM/qwen-asr)",
        streaming_detail="Finestra attenzione dinamica. Locale: pseudo-streaming.",
        features="LID, allineatore opzionale, torch.compile",
    ),
    ModelInfo(
        id="openai/whisper-small",
        name="Whisper Small",
        license="MIT",
        languages="99 lingue incl. it",
        params="244M",
        size="~460MB",
        streaming_type="Pseudo-streaming",
        streaming_detail="Batch encoder-decoder. Live: accumulo chunk e trascrizione ogni 2s.",
        features="Baseline robusto",
    ),
    ModelInfo(
        id="nvidia/nemotron-3.5-asr-streaming-0.6b",
        name="Nemotron 3.5 ASR Streaming",
        license="Apache 2.0",
        languages="en (multi?)",
        params="600M",
        size="~1.2GB",
        streaming_type="Nativo (cache-aware)",
        streaming_detail="Chunk 80/160/560/1120ms, cache precedenti.",
        features="Punteggiatura e maiuscole native",
    ),
]
