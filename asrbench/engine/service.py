"""Gestione modelli: load/unload, inferenza GPU, Attiva/Scarica."""
from __future__ import annotations

import time

import gradio as gr

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


def get_adapter(model_id: str, lang: str = "it"):
    """Cache-hit diretto, altrimenti unload + load. Un solo modello in memoria."""
    if state.CURRENT["id"] != model_id:
        if state.CURRENT["adapter"] is not None:
            event("load", f"unload {short(state.CURRENT['id'])}")
            state.CURRENT["adapter"].unload()
        state.MODEL_STATUS[model_id] = {"state": "loading", "detail": ""}
        event("load", f"carico {short(model_id)} ...")
        start = time.perf_counter()
        try:
            adapter = _load_new_adapter(model_id)
        except Exception as exc:
            state.MODEL_STATUS[model_id] = {"state": "error", "detail": gated_hint(model_id, exc, lang)}
            event("load", f"ERRORE {short(model_id)}: {exc}")
            raise
        state.CURRENT["adapter"] = adapter
        state.CURRENT["id"] = model_id
        dt = time.perf_counter() - start
        state.MODEL_STATUS[model_id] = {"state": "ready", "secs": round(dt, 1)}
        event("load", f"pronto {short(model_id)} in {dt:.1f}s")
    return state.CURRENT["adapter"]


def gated_hint(model_id: str, exc: Exception, lang: str = "it") -> str:
    """Messaggio corto: HF_TOKEN se errore 401/403/gated, altrimenti prime 120 lettere."""
    msg = str(exc).lower()
    if any(k in msg for k in ("401", "403", "gated", "access", "token", "unauthorized")):
        return T(lang, "gated")
    return str(exc)[:120]


@spaces.GPU(duration=180)
def _load_new_adapter(model_id: str):
    """Istanzia e carica l'adapter. 180s di GPU su ZeroGPU."""
    adapter = state.ADAPTERS[model_id](model_id)
    adapter.load()
    return adapter


@spaces.GPU(duration=60)
def run_transcribe(adapter, audio, sr: int = 16000):
    """Trascrizione batch con 60s di GPU."""
    return adapter.transcribe(audio, sr)


@spaces.GPU(duration=60)
def nemotron_feed(adapter, raw_chunk):
    """Un feed streaming nativo Nemotron con 60s di GPU."""
    return adapter.stream_feed(raw_chunk)


@spaces.GPU(duration=60)
def nemotron_finish(adapter):
    """Chiusura sessione streaming nativa con 60s di GPU."""
    return adapter.stream_finish()


def preload(model_id: str, lang: str = "it"):
    """Attiva: carica il modello scelto. Rifiuta se una prova è aperta."""
    event("ui", f"click Attiva model={short(model_id)}")
    if state.TRIAL_ACTIVE:
        gr.Warning(T(lang, "close_warn", op=T(lang, "op_switch")))
        event("ui", "click Attiva rifiutato: prova in corso")
        yield (T(lang, "trial_busy", op=T(lang, "op_switch")),
               status_html(lang), log_tail(30), live_btn_state(lang))
        return
    if state.CURRENT.get("id") == model_id and state.CURRENT.get("adapter") is not None:
        event("load", f"{short(model_id)} gia' attivo")
        yield T(lang, "model_already", m=short(model_id)), status_html(lang), log_tail(60), live_btn_state(lang)
        return
    state.MODEL_STATUS[model_id] = {"state": "loading", "detail": ""}
    yield T(lang, "loading", m=short(model_id)), status_html(lang), log_tail(60), live_btn_state(lang)
    try:
        get_adapter(model_id, lang)
    except Exception as exc:
        yield T(lang, "load_error", e=exc), status_html(lang), log_tail(60), live_btn_state(lang)
        return
    yield T(lang, "model_ready", m=short(model_id)), status_html(lang), log_tail(60), live_btn_state(lang)


def unload_model(lang: str = "it"):
    """Scarica il modello attivo, libera VRAM. Rifiuta a prova aperta."""
    if state.TRIAL_ACTIVE:
        gr.Warning(T(lang, "close_warn", op=T(lang, "op_unload")))
        event("ui", "click Scarica rifiutato: prova in corso")
        return (T(lang, "trial_busy", op=T(lang, "op_unload")),
                status_html(lang), log_tail(30), live_btn_state(lang))
    mid = state.CURRENT.get("id")
    if state.CURRENT.get("adapter") is None or mid is None:
        event("ui", "click Scarica ma nessun modello attivo")
        return T(lang, "no_active"), status_html(lang), log_tail(60), live_btn_state(lang)
    event("ui", f"click Scarica modello attivo {short(mid)}")
    try:
        state.CURRENT["adapter"].unload()
    except Exception as exc:
        event("load", f"ERRORE unload {short(mid)}: {exc}")
    event("load", f"unload {short(mid)} completato")
    state.MODEL_STATUS[mid] = {"state": "idle", "detail": ""}
    state.CURRENT["adapter"] = None
    state.CURRENT["id"] = None
    return T(lang, "unloaded", m=short(mid)), status_html(lang), log_tail(60), live_btn_state(lang)
