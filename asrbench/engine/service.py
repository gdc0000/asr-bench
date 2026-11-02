"""Gestione modelli: load a module-scope, inferenza GPU, Attiva/Scarica.

ZeroGPU: i pesi NON possono attraversare il fork (niente tensori CUDA in
argomenti/return dei @spaces.GPU). Perciò tutti i modelli si caricano a
module-scope con .to("cuda") (pack su disco, stream in VRAM per worker) e gli
handler GPU prendono solo (model_id, array numpy) e ritornano str.
"""
from __future__ import annotations

import os
import time

import gradio as gr
import numpy as np

from asrbench.core import state
from asrbench.core.logbuffer import event, tail as log_tail
from asrbench.engine.helpers import short, sname
from asrbench.ui.i18n import T
from asrbench.ui.status import live_btn_state, status_html

try:
    import spaces
except ImportError:  # dev locale senza pacchetto spaces
    class _SpacesFallback:
        """No-op locale quando il pacchetto spaces manca."""

        @staticmethod
        def GPU(task=None, **kwargs):
            """Mima il decoratore senza accodare GPU."""
            if task is None:
                return lambda fn: fn
            return task
    spaces = _SpacesFallback()

#: Adapter pronti (scope-load a boot). Nel worker sono tensori VRAM reali,
#: nel main sono alias su disco (emulation). Mai passarli come arg/return GPU.
MODELS_LIVE: dict = {}
#: model_id -> hint leggibile se lo scope-load è fallito (es. repo gated).
LOAD_ERROR: dict[str, str] = {}


def _preload_all(lang: str = "it"):
    """Carica tutti gli adapter a module-scope. Errori isolati per modello.

    Chiamata una volta da app.py a boot. Su ZeroGPU questo .to("cuda") fa il
    pack su disco: zero quota, VRAM allocata solo per worker a richiesta.
    """
    for mid, cls in state.ADAPTERS.items():
        start = time.perf_counter()
        try:
            adapter = cls(mid)
            adapter.load()
            MODELS_LIVE[mid] = adapter
            secs = round(time.perf_counter() - start, 1)
            state.MODEL_STATUS[mid] = {"state": "ready", "secs": secs}
            event("load", f"pronto {short(mid)} scope in {secs}s")
        except Exception as exc:
            LOAD_ERROR[mid] = gated_hint(mid, exc, lang)
            state.MODEL_STATUS[mid] = {"state": "error", "detail": LOAD_ERROR[mid]}
            event("load", f"ERRORE scope {short(mid)}: {exc}")


def get_adapter(model_id: str, lang: str = "it"):
    """Lookup istantanea, zero quota. Alza RuntimeError se lo scope-load fallì."""
    if model_id in MODELS_LIVE:
        return MODELS_LIVE[model_id]
    raise RuntimeError(LOAD_ERROR.get(model_id, "?"))


def gated_hint(model_id: str, exc: Exception, lang: str = "it") -> str:
    """Messaggio corto: HF_TOKEN se errore 401/403/gated, altrimenti prime 120 lettere."""
    msg = str(exc).lower()
    if any(k in msg for k in ("401", "403", "gated", "access", "token", "unauthorized")):
        return T(lang, "gated")
    return str(exc)[:120]


@spaces.GPU(duration=45)
def run_transcribe(model_id: str, audio, sr: int = 16000):
    """Trascrizione batch su modello scope-loaded. Solo tipi picklabili in/out."""
    adapter = MODELS_LIVE[model_id]
    return adapter.transcribe(np.asarray(audio, dtype=np.float32), sr)


@spaces.GPU(duration=45)
def nemotron_feed(adapter, raw_chunk):
    """Un feed streaming nativo Nemotron. Solo locale: su Space gated in trials."""
    if os.environ.get("SPACES_ZERO_GPU") == "1":
        raise RuntimeError("streaming nativo disabilitato su Space, uso re-tx")
    return adapter.stream_feed(raw_chunk)


@spaces.GPU(duration=45)
def nemotron_finish(adapter):
    """Chiusura sessione streaming nativa. Solo locale: su Space gated in trials."""
    if os.environ.get("SPACES_ZERO_GPU") == "1":
        raise RuntimeError("streaming nativo disabilitato su Space, uso re-tx")
    return adapter.stream_finish()


def preload(model_id: str, lang: str = "it"):
    """Attiva: seleziona il modello (già scope-loaded a boot, zero quota).

    Rifiuta se una prova è aperta.
    """
    event("ui", f"click Attiva model={short(model_id)}")
    if state.TRIAL_ACTIVE:
        gr.Warning(T(lang, "close_warn", op=T(lang, "op_switch")))
        event("ui", "click Attiva rifiutato: prova in corso")
        return (T(lang, "trial_busy", op=T(lang, "op_switch")),
                status_html(lang), log_tail(30), live_btn_state(lang))
    if state.CURRENT.get("id") == model_id and state.CURRENT.get("adapter") is not None:
        event("load", f"{short(model_id)} gia' attivo")
        return T(lang, "model_already", m=short(model_id)), status_html(lang), log_tail(60), live_btn_state(lang)
    try:
        adapter = get_adapter(model_id, lang)
    except Exception as exc:
        event("load", f"ERRORE scope {short(model_id)}: {exc}")
        return T(lang, "load_error", e=exc), status_html(lang), log_tail(60), live_btn_state(lang)
    state.CURRENT["adapter"] = adapter
    state.CURRENT["id"] = model_id
    event("load", f"attivo {short(model_id)} (scope, zero quota)")
    return T(lang, "model_ready", m=short(model_id)), status_html(lang), log_tail(60), live_btn_state(lang)


def unload_model(lang: str = "it"):
    """Scarica = deseleziona. I pesi restano scope-loaded (condivisi, no VRAM da liberare)."""
    if state.TRIAL_ACTIVE:
        gr.Warning(T(lang, "close_warn", op=T(lang, "op_unload")))
        event("ui", "click Scarica rifiutato: prova in corso")
        return (T(lang, "trial_busy", op=T(lang, "op_unload")),
                status_html(lang), log_tail(30), live_btn_state(lang))
    mid = state.CURRENT.get("id")
    if state.CURRENT.get("adapter") is None or mid is None:
        event("ui", "click Scarica ma nessun modello attivo")
        return T(lang, "no_active"), status_html(lang), log_tail(60), live_btn_state(lang)
    event("ui", f"click Scarica: deseleziono {short(mid)} (pesi scope restano)")
    state.CURRENT["adapter"] = None
    state.CURRENT["id"] = None
    return T(lang, "unloaded", m=short(mid)), status_html(lang), log_tail(60), live_btn_state(lang)
