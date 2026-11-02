"""Stato modelli per sidebar e bottone Live."""
from __future__ import annotations

import gradio as gr

from asrbench.core import state
from asrbench.core.logbuffer import tail as log_tail
from asrbench.core.registry import MODELS
from asrbench.engine.helpers import default_detail, model_choices, short, sname
from asrbench.ui.i18n import T

STATE_LABEL = {"ready": "st_ready", "loading": "st_loading",
               "idle": "st_idle", "error": "st_error"}


def status_badge(state_key: str, lang: str = "it") -> str:
    """Pallino colorato per stato modello."""
    color = {"ready": "green", "loading": "orange", "idle": "gray", "error": "red"}.get(state_key, "gray")
    return f"<b style='color:{color}'>● {T(lang, STATE_LABEL.get(state_key, 'st_idle'))}</b>"


def usage_of(mid: str, lang: str = "it") -> str:
    """Attivo / selezionato / entrambi, per la tabella stato."""
    tags = []
    if state.CURRENT.get("id") == mid:
        tags.append(T(lang, "use_active"))
    if state.SELECTED.get("model") == mid:
        tags.append(T(lang, "use_selected"))
    return ", ".join(tags) if tags else "—"


def status_html(lang: str = "it") -> str:
    """Tabella stato modelli per la sidebar."""
    rows = []
    for m in MODELS:
        if m.id not in state.ADAPTERS:
            continue
        s = state.MODEL_STATUS.get(m.id, {"state": "?", "detail": ""})
        key = s["state"]
        if key == "ready":
            sub = f"<br><small>{T(lang, 'detail_ready', s=s.get('secs', 0.0))}</small>"
        elif key == "loading":
            sub = f"<br><small>{T(lang, 'detail_loading')}</small>"
        elif key == "error":
            sub = f"<br><small>{s.get('detail', '')}</small>"
        else:
            sub = f"<br><small>{default_detail(m.id)}</small>"
        rows.append(
            f"<tr><td>{sname(m.id)}</td><td>{status_badge(key, lang)}{sub}</td>"
            f"<td>{usage_of(m.id, lang)}</td></tr>"
        )
    return (
        f"<table><tr><th>{T(lang, 'th_model')}</th><th>{T(lang, 'th_state')}</th><th>{T(lang, 'th_use')}</th></tr>"
        + "".join(rows) + "</table>"
    )


def side_update(lang: str = "it"):
    """Refresh sidebar: tabella stati + coda log."""
    return status_html(lang), log_tail(60)


def live_btn_state(lang: str = "it"):
    """Live cliccabile solo se il modello selezionato è attivo."""
    ready = state.CURRENT.get("adapter") is not None and state.CURRENT.get("id") == state.SELECTED.get("model")
    if ready:
        return gr.update(value=T(lang, "live_on"), variant="primary", interactive=True)
    return gr.update(value=T(lang, "live_need", m=short(state.SELECTED.get("model"))),
                     variant="secondary", interactive=False)


def model_status_init():
    """Stati iniziali + seleziona il primo modello. Non tocca stati già scope-loaded."""
    for m in MODELS:
        if m.id in state.ADAPTERS and m.id not in state.MODEL_STATUS:
            state.MODEL_STATUS[m.id] = {"state": "idle", "detail": ""}
    state.SELECTED["model"] = model_choices()[0][1]
