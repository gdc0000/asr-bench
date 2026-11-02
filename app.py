from __future__ import annotations

import csv
import json
import time

import gradio as gr
import numpy as np

from asrbench.adapters.granite import GraniteAdapter
from asrbench.adapters.nemotron import NemotronAdapter
from asrbench.adapters.qwen import QwenAdapter
from asrbench.adapters.whisper import WhisperAdapter
from asrbench.audio import to_mono_16k
from asrbench.logbuffer import RUN_ID, tail as log_tail, clear as log_clear, event, path as log_path
from asrbench.registry import MODELS
from asrbench.scores import cer, fmt, wer

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

ADAPTERS = {
    "ibm-granite/granite-speech-5.0-470m-turboctc": GraniteAdapter,
    "Qwen/Qwen3-ASR-0.6B": QwenAdapter,
    "openai/whisper-small": WhisperAdapter,
    "nvidia/nemotron-3.5-asr-streaming-0.6b": NemotronAdapter,
}

CURRENT = {"adapter": None, "id": None}
SELECTED = {"model": None}
MODEL_STATUS = {}
TRIAL_SEQ = 0
VERBOSE = {"live_chunks": False}
SESSION: list = []

CSV_PATH = f"logs/session-{RUN_ID}.csv"
JSONL_PATH = f"logs/session-{RUN_ID}.jsonl"
CSV_COLS = ["prova", "ora", "modello", "audio_s", "gt", "testo",
            "wer", "cer", "latenza_primo_testo_s", "inferenza_s", "rtf", "revisioni"]

CSS = """
#asr-sidebar { min-width: 300px; }
#asr-sidebar .prose table { font-size: 12px; }
#asr-drag { position: absolute; top: 64px; bottom: 0; left: 0; width: 10px;
  cursor: ew-resize; z-index: 1002; display: flex; align-items: center; justify-content: center; }
#asr-drag::after { content: ""; width: 3px; height: 48px; border-radius: 2px;
  background: var(--border-color-primary); }
#asr-drag:hover::after { background: var(--body-text-color); }
h2, h3 { margin-top: 6px !important; margin-bottom: 4px !important; }
"""

SIDEBAR_JS = """
<script>
(function () {
  window.asrSideW = window.asrSideW || 380;
  window.asrToggleSidebar = function () {
    var bar = document.getElementById('asr-sidebar');
    if (!bar) return false;
    var btn = bar.querySelector('.toggle-button');
    if (btn) { btn.click(); return true; }
    bar.classList.toggle('open');
    return true;
  };
  function applyW(bar, w) {
    w = Math.max(300, Math.min(Math.floor(window.innerWidth * 0.65), w));
    window.asrSideW = w;
    bar.style.width = w + 'px';
    bar.style.right = 'calc(' + w + 'px * -1)';
  }
  function install(bar) {
    if (!bar || bar.dataset.asrDrag) return;
    bar.dataset.asrDrag = '1';
    applyW(bar, window.asrSideW);
    var h = document.createElement('div');
    h.id = 'asr-drag';
    h.title = 'Trascina per ridimensionare il pannello';
    h.addEventListener('mousedown', function (e) {
      e.preventDefault();
      var sx = e.clientX, sw = bar.offsetWidth;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      function mv(ev) { applyW(bar, sw + (sx - ev.clientX)); }
      function up() {
        document.removeEventListener('mousemove', mv);
        document.removeEventListener('mouseup', up);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
      document.addEventListener('mousemove', mv);
      document.addEventListener('mouseup', up);
    });
    bar.appendChild(h);
  }
  function scan() { install(document.getElementById('asr-sidebar')); }
  new MutationObserver(scan).observe(document.documentElement, {childList: true, subtree: true});
  window.addEventListener('resize', function () {
    var bar = document.getElementById('asr-sidebar');
    if (bar) applyW(bar, window.asrSideW);
  });
  scan();
})();
</script>
"""


def short(mid: str) -> str:
    """Ultimo segmento dell'id HF, per log compatti."""
    return mid.split("/")[-1] if mid else "?"


VAD_RMS = 0.02  # soglia voce su scala -1..1


def chunk_is_speech(wav: np.ndarray) -> bool:
    """RMS sopra soglia. Scala auto: peak > 1.5 = int16."""
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= 0:
        return False
    scale = 32768.0 if peak > 1.5 else 1.0
    rms = float(np.sqrt(np.mean((wav / scale) ** 2)))
    return rms > VAD_RMS


SHORT = {
    "ibm-granite/granite-speech-5.0-470m-turboctc": "Granite",
    "Qwen/Qwen3-ASR-0.6B": "Qwen3",
    "openai/whisper-small": "Whisper",
    "nvidia/nemotron-3.5-asr-streaming-0.6b": "Nemotron",
}
DASH = "—"


def sname(mid: str) -> str:
    """Nome corto per UI (Granite, Qwen3, Whisper, Nemotron)."""
    return SHORT.get(mid, short(mid))


def default_detail(mid: str) -> str:
    """Riga params | size dalla scheda registry."""
    m = next((x for x in MODELS if x.id == mid), None)
    return f"{m.params} | {m.size}" if m else ""


def model_status_init():
    """Stati iniziali + seleziona il primo modello."""
    for m in MODELS:
        if m.id in ADAPTERS:
            MODEL_STATUS[m.id] = {"state": "non caricato", "detail": default_detail(m.id)}
    SELECTED["model"] = model_choices()[0][1]


def model_choices():
    """Coppie (etichetta, id) per il dropdown. Solo modelli wired."""
    return [(f"{m.name} [{m.streaming_type}]", m.id) for m in MODELS if m.id in ADAPTERS]


def status_badge(state: str) -> str:
    """Pallino colorato per stato modello."""
    color = {"pronto": "green", "caricamento": "orange", "non caricato": "gray", "errore": "red"}.get(state, "gray")
    return f"<b style='color:{color}'>● {state}</b>"


def usage_of(mid: str) -> str:
    """Attivo / selezionato / entrambi, per la tabella stato."""
    tags = []
    if CURRENT.get("id") == mid:
        tags.append("attivo")
    if SELECTED.get("model") == mid:
        tags.append("selezionato")
    return ", ".join(tags) if tags else "—"


def status_html() -> str:
    """Tabella stato modelli per la sidebar."""
    rows = []
    for m in MODELS:
        if m.id not in ADAPTERS:
            continue
        s = MODEL_STATUS.get(m.id, {"state": "?", "detail": ""})
        sub = f"<br><small>{s['detail']}</small>" if s["state"] in ("pronto", "errore") else ""
        rows.append(
            f"<tr><td>{sname(m.id)}</td><td>{status_badge(s['state'])}{sub}</td>"
            f"<td>{usage_of(m.id)}</td></tr>"
        )
    return (
        "<table><tr><th>Modello</th><th>Stato</th><th>Uso</th></tr>"
        + "".join(rows) + "</table>"
    )


def side_update():
    """Refresh sidebar: tabella stati + coda log."""
    return status_html(), log_tail(60)


FLOW_CSS = """
<style>
.asrflow { display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; margin: 4px 0; }
.asrflow-node { border: 1px solid var(--border-color-primary); border-radius: 8px;
  padding: 6px 10px; min-width: 96px; max-width: 150px; background: var(--background-fill-primary); }
.asrflow-node b { display: block; font-size: 13px; }
.asrflow-node span { font-size: 11px; opacity: 0.85; }
.asrflow-node.first { border-color: var(--color-accent); }
.asrflow-node.last { border-color: green; box-shadow: 0 0 6px rgba(0,128,0,.35);
  animation: asrflow-glow 2s ease-in-out infinite; }
@keyframes asrflow-glow { 50% { box-shadow: 0 0 12px rgba(0,128,0,.6); } }
.asrflow-link { position: relative; flex: 1 1 28px; min-width: 28px; align-self: center;
  height: 2px; background: var(--border-color-primary); overflow: visible; }
.asrflow-link.loop { background: repeating-linear-gradient(90deg, orange 0 6px, transparent 6px 12px);
  animation: asrflow-dash 1s linear infinite; }
@keyframes asrflow-dash { to { background-position: 12px 0; } }
.asrflow-dot { position: absolute; top: -3px; left: 0; width: 8px; height: 8px; border-radius: 50%;
  background: var(--color-accent); animation: asrflow-move 1.2s linear infinite; }
@keyframes asrflow-move { to { left: calc(100% - 8px); } }
.asrflow-note { font-size: 12px; opacity: 0.9; margin-top: 2px; }
.asrflow-badge { display: inline-block; font-size: 11px; font-weight: bold; border-radius: 4px;
  padding: 1px 8px; margin-bottom: 4px; }
.asrflow-badge.nat { background: rgba(0,128,0,.15); color: green; }
.asrflow-badge.retx { background: rgba(255,165,0,.18); color: #b26a00; }
</style>
"""

MODEL_FLOW = {
    "ibm-granite/granite-speech-5.0-470m-turboctc": {
        "mode": "re-tx",
        "steps": [("Mic", "chunk 1s"), ("Buffer", "accumulo, max 30s"),
                  ("Ogni 2s", "ritrascrive tutto"), ("CTC greedy", "frame -> argmax"),
                  ("Testo", "parziale riscritto")],
    },
    "Qwen/Qwen3-ASR-0.6B": {
        "mode": "re-tx",
        "steps": [("Mic", "chunk 1s"), ("Buffer", "accumulo, max 30s"),
                  ("Ogni 2s", "ritrascrive tutto"), ("Prompt chat", "lingua Italian"),
                  ("Generate LLM", "0.6B autoregressivo"), ("Testo", "prompt tagliato")],
    },
    "openai/whisper-small": {
        "mode": "re-tx",
        "steps": [("Mic", "chunk 1s"), ("Buffer", "accumulo, max 30s"),
                  ("Ogni 2s", "ritrascrive tutto"), ("Log-mel", "80 bin"),
                  ("Generate it", "autoregressivo"), ("Testo", "parziale riscritto")],
    },
    "nvidia/nemotron-3.5-asr-streaming-0.6b": {
        "mode": "nativo",
        "steps": [("Mic", "chunk 1s"), ("Feed 0.3s", "mel a pezzi 25/32"),
                  ("Encoder cache", "KV + conv portate avanti"), ("RNNT greedy", "frame -> token"),
                  ("Testo", "committed, niente rewrite")],
    },
}

FLOW_NOTE = {
    "re-tx": "↻ Il buffer intero viene ritrascritto a ogni giro: costo O(n²), parziali riscritti.",
    "nativo": "→ Ogni frame è codificato e decodificato una volta sola: costo O(n), parziali stabili.",
}


def flow_html(model_id: str) -> str:
    """Diagramma di flusso animato (CSS) dello streaming del modello."""
    spec = MODEL_FLOW.get(model_id, {"mode": "re-tx", "steps": []})
    mode = spec["mode"]
    badge = ("<span class='asrflow-badge nat'>NATIVO</span>"
             if mode == "nativo" else "<span class='asrflow-badge retx'>RE-TX OGNI 2s</span>")
    parts = [FLOW_CSS, badge, "<div class='asrflow'>"]
    steps = spec["steps"]
    for i, (title, sub) in enumerate(steps):
        cls = "first" if i == 0 else ("last" if i == len(steps) - 1 else "")
        parts.append(f"<div class='asrflow-node {cls}'><b>{i + 1}. {title}</b><span>{sub}</span></div>")
        if i < len(steps) - 1:
            link_cls = "loop" if (mode == "re-tx" and i == 1) else ""
            parts.append(f"<div class='asrflow-link {link_cls}'><span class='asrflow-dot'></span></div>")
    parts.append("</div>")
    parts.append(f"<div class='asrflow-note'>{FLOW_NOTE[mode]}</div>")
    return "".join(parts)


def model_info(model_id: str) -> str:
    """Scheda markdown del modello per l'accordion dettagli, con link alla fonte HF."""
    model = next(m for m in MODELS if m.id == model_id)
    lines = [
        f"**{model.name}** · [scheda Hugging Face](https://huggingface.co/{model.id})",
        "",
        f"Licenza: {model.license} | Parametri: {model.params} | Dimensione: {model.size}",
        f"Lingue: {model.languages}",
        f"Streaming: {model.streaming_type} — {model.streaming_detail}",
        f"Caratteristiche: {model.features}",
    ]
    if model.notes:
        lines.append(f"Note: {model.notes}")
    return "\n".join(lines)


def live_btn_state():
    """Live cliccabile solo se il modello selezionato è attivo."""
    ready = CURRENT.get("adapter") is not None and CURRENT.get("id") == SELECTED.get("model")
    if ready:
        return gr.update(value="● Live ON", variant="primary", interactive=True)
    return gr.update(value=f"Attiva {short(SELECTED.get('model'))} per la prova",
                     variant="secondary", interactive=False)


def on_select(model_id: str):
    """Cambio dropdown: aggiorna scheda, diagramma flusso, stato, log, bottone live."""
    prev = SELECTED.get("model")
    SELECTED["model"] = model_id
    event("ui", f"select {short(prev)} -> {short(model_id)} (attivo={short(CURRENT.get('id'))})")
    return model_info(model_id), flow_html(model_id), status_html(), log_tail(60), live_btn_state()


def get_adapter(model_id: str):
    """Cache-hit diretto, altrimenti unload + load. Un solo modello in memoria."""
    if CURRENT["id"] != model_id:
        if CURRENT["adapter"] is not None:
            event("load", f"unload {short(CURRENT['id'])}")
            CURRENT["adapter"].unload()
        MODEL_STATUS[model_id] = {"state": "caricamento", "detail": "download pesi + init..."}
        event("load", f"carico {short(model_id)} ...")
        start = time.perf_counter()
        try:
            adapter = _load_new_adapter(model_id)
        except Exception as exc:
            MODEL_STATUS[model_id] = {"state": "errore", "detail": gated_hint(model_id, exc)}
            event("load", f"ERRORE {short(model_id)}: {exc}")
            raise
        CURRENT["adapter"] = adapter
        CURRENT["id"] = model_id
        dt = time.perf_counter() - start
        MODEL_STATUS[model_id] = {"state": "pronto", "detail": f"caricato in {dt:.1f}s"}
        event("load", f"pronto {short(model_id)} in {dt:.1f}s")
    return CURRENT["adapter"]


def gated_hint(model_id: str, exc: Exception) -> str:
    """Messaggio corto: HF_TOKEN se errore 401/403/gated, altrimenti prime 120 lettere."""
    msg = str(exc).lower()
    if any(k in msg for k in ("401", "403", "gated", "access", "token", "unauthorized")):
        return "serve HF_TOKEN con accesso approvato al repo"
    return str(exc)[:120]


@spaces.GPU(duration=180)
def _load_new_adapter(model_id: str):
    """Istanzia e carica l'adapter. 180s di GPU su ZeroGPU."""
    adapter = ADAPTERS[model_id](model_id)
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


NEMOTRON_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
TRIAL_ACTIVE = False


def preload(model_id: str):
    """Attiva: carica il modello scelto. Rifiuta se una prova è aperta."""
    event("ui", f"click Attiva model={short(model_id)}")
    if TRIAL_ACTIVE:
        gr.Warning("Chiudi la prova prima di cambiare modello.")
        event("ui", "click Attiva rifiutato: prova in corso")
        yield ("Prova in corso: premi Live OFF prima di cambiare modello.",
               status_html(), log_tail(30), live_btn_state())
        return
    if CURRENT.get("id") == model_id and CURRENT.get("adapter") is not None:
        event("load", f"{short(model_id)} gia' attivo")
        yield f"Modello {short(model_id)} gia' attivo.", status_html(), log_tail(60), live_btn_state()
        return
    MODEL_STATUS[model_id] = {"state": "caricamento", "detail": "download pesi + init..."}
    yield f"Caricamento {short(model_id)}... (puo' durare minuti al primo uso)", status_html(), log_tail(60), live_btn_state()
    try:
        get_adapter(model_id)
    except Exception as exc:
        yield f"Errore caricamento: {exc}", status_html(), log_tail(60), live_btn_state()
        return
    yield f"Modello {short(model_id)} pronto e attivo.", status_html(), log_tail(60), live_btn_state()


def unload_model():
    """Scarica il modello attivo, libera VRAM. Rifiuta a prova aperta."""
    if TRIAL_ACTIVE:
        gr.Warning("Chiudi la prova prima di scaricare il modello.")
        event("ui", "click Scarica rifiutato: prova in corso")
        return ("Prova in corso: premi Live OFF prima di scaricare.",
                status_html(), log_tail(30), live_btn_state())
    mid = CURRENT.get("id")
    if CURRENT.get("adapter") is None or mid is None:
        event("ui", "click Scarica ma nessun modello attivo")
        return "Nessun modello attivo.", status_html(), log_tail(60), live_btn_state()
    event("ui", f"click Scarica modello attivo {short(mid)}")
    try:
        CURRENT["adapter"].unload()
    except Exception as exc:
        event("load", f"ERRORE unload {short(mid)}: {exc}")
    event("load", f"unload {short(mid)} completato")
    MODEL_STATUS[mid] = {"state": "non caricato", "detail": default_detail(mid)}
    CURRENT["adapter"] = None
    CURRENT["id"] = None
    return f"Modello {short(mid)} scaricato. VRAM liberata.", status_html(), log_tail(60), live_btn_state()


def new_trial():
    """Stato prova vergine: buffer, tempi, metriche."""
    return {"active": False, "n": 0, "mid": None, "gt": "", "samples": [],
            "last_len": 0, "last_text": "", "last_t": 0.0, "t_on": 0.0,
            "t_first": None, "t_speech": None, "infer_s": 0.0, "revs": 0, "k": 0,
            "last_log": 0.0, "last_sr": None, "last_feed_len": 0, "native": False}


def table_view():
    """Righe sessione per il dataframe: # modello WER CER lat RTF audio."""
    rows = []
    for r in SESSION:
        rows.append([r["prova"], sname(r["modello"]),
                     fmt(r["wer"]), fmt(r["cer"]),
                     f"{r['lat_s']:.1f}" if r["lat_s"] is not None else DASH,
                     f"{r['rtf']:.2f}", f"{r['audio_s']:.1f}"])
    return rows


TABLE_HEADERS = ["#", "Modello", "WER", "CER", "Lat", "RTF", "Audio"]


def last_card_view():
    """Card markdown dell'ultima prova chiusa, o placeholder."""
    if not SESSION:
        return "_Nessuna prova in sessione._"
    r = SESSION[-1]
    gt = r["gt"] or "_non fornita_"
    lat = f"{r['lat_s']}s" if r["lat_s"] is not None else DASH
    return (f"**Prova #{r['prova']} — {sname(r['modello'])}**\n\n"
            f"GT: {gt}\n\nTesto: {r['testo']}\n\n"
            f"WER {fmt(r['wer'])} | CER {fmt(r['cer'])} | lat {lat} | RTF {r['rtf']}")


def write_session_files():
    """Riscrive CSV + JSONL della sessione dopo ogni prova."""
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in SESSION:
            w.writerow([r["prova"], r["ora"], r["modello"], f"{r['audio_s']:.2f}",
                        r["gt"], r["testo"],
                        "" if r["wer"] is None else round(r["wer"] * 100, 1),
                        "" if r["cer"] is None else round(r["cer"] * 100, 1),
                        "" if r["lat_s"] is None else round(r["lat_s"], 2),
                        round(r["infer_s"], 2), round(r["rtf"], 2), r["revs"]])
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in SESSION:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def live_toggle(model_id: str, gt: str, st):
    """Un bottone: ON apre la prova (GT congelato), OFF la chiude e la archivia.

    Audio sotto 1s scartato. Ramo nativo per Nemotron, batch per gli altri.
    """
    global TRIAL_SEQ, TRIAL_ACTIVE
    st = st or new_trial()
    if not st.get("active"):
        if CURRENT.get("id") != model_id or CURRENT.get("adapter") is None:
            event("ui", "click Live ON rifiutato: modello non pronto, premi Attiva")
            gr.Warning(f"Modello {sname(model_id)} non attivo: premi Attiva prima.")
            return (gr.update(), gr.update(), gr.update(),
                    st["last_text"] or DASH,
                    f"Modello {sname(model_id)} non attivo: premi Attiva prima.",
                    table_view(), last_card_view(), DASH, DASH, DASH)
        TRIAL_SEQ += 1
        st.update(new_trial())
        st.update({"active": True, "n": TRIAL_SEQ, "mid": model_id,
                   "gt": (gt or "").strip(), "t_on": time.perf_counter(),
                   "native": False})
        native = (model_id == NEMOTRON_ID)
        if native:
            try:
                CURRENT["adapter"].stream_start()
                st["native"] = True
            except Exception as exc:
                event("live", f"prova#{TRIAL_SEQ} stream nativo non avviato ({exc}): fallback re-tx")
                st["native"] = False
        TRIAL_ACTIVE = True
        event("ui", f"prova#{TRIAL_SEQ} Live ON model={short(model_id)} "
                    f"gt={'si' if st['gt'] else 'no'} {'nativo' if st['native'] else 're-tx'}")
        tag = "nativo" if st["native"] else "re-tx"
        return (gr.update(value="■ Live OFF", variant="stop", interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "In ascolto... parla pure.",
                f"Prova #{TRIAL_SEQ} | {sname(model_id)} [{tag}] | premi il microfono e parla.",
                table_view(), last_card_view(), DASH, DASH, DASH)
    # chiusura prova
    audio_s = len(st["samples"]) / 16000
    st["active"] = False
    TRIAL_ACTIVE = False
    if audio_s < 1.0:
        event("ui", f"prova#{st['n']} Live OFF scartata: audio {audio_s:.1f}s < 1s")
        gr.Warning(f"Prova #{st['n']} scartata: audio troppo corto.")
        out = (live_btn_state(), gr.update(interactive=True), gr.update(interactive=True),
               st["last_text"] or DASH,
               f"Prova #{st['n']} scartata: audio troppo corto ({audio_s:.1f}s).",
               table_view(), last_card_view(), DASH, DASH, DASH)
        st.update(new_trial())
        return out
    if st.get("native"):
        try:
            adapter = get_adapter(st["mid"])
        except Exception as exc:
            event("ui", f"prova#{st['n']} ERRORE reload modello: {exc}")
            gr.Warning(f"Errore modello: {exc}")
            return (live_btn_state(), gr.update(interactive=True), gr.update(interactive=True),
                    st["last_text"] or DASH, f"Errore modello: {exc}",
                    table_view(), last_card_view(), DASH, DASH, DASH)
        t0 = time.perf_counter()
        try:
            text = nemotron_finish(adapter)
        except Exception as exc:
            event("ui", f"prova#{st['n']} ERRORE finish nativo: {exc}")
            gr.Warning(f"Errore modello: {exc}")
            return (live_btn_state(), gr.update(interactive=True), gr.update(interactive=True),
                    st["last_text"] or DASH, f"Errore modello: {exc}",
                    table_view(), last_card_view(), DASH, DASH, DASH)
        st["infer_s"] += time.perf_counter() - t0
        st["last_text"] = text
    else:
        try:
            adapter = get_adapter(st["mid"])
        except Exception as exc:
            event("ui", f"prova#{st['n']} ERRORE reload modello: {exc}")
            gr.Warning(f"Errore modello: {exc}")
            return (live_btn_state(), gr.update(interactive=True), gr.update(interactive=True),
                    st["last_text"] or DASH, f"Errore modello: {exc}",
                    table_view(), last_card_view(), DASH, DASH, DASH)
        audio = np.asarray(st["samples"], dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0:
            audio = audio / peak * 0.95
        t0 = time.perf_counter()
        try:
            text = run_transcribe(adapter, audio, 16000)
        except Exception as exc:
            event("ui", f"prova#{st['n']} ERRORE trascrizione finale: {exc}")
            gr.Warning(f"Errore modello: {exc}")
            return (live_btn_state(), gr.update(interactive=True), gr.update(interactive=True),
                    st["last_text"] or DASH, f"Errore modello: {exc}",
                    table_view(), last_card_view(), DASH, DASH, DASH)
        st["infer_s"] += time.perf_counter() - t0
        st["last_text"] = text
    rtf = st["infer_s"] / audio_s if audio_s else 0.0
    base = st["t_speech"] if st["t_speech"] else st["t_on"]
    lat = max(0.0, st["t_first"] - base) if st["t_first"] else None
    if st["t_speech"] is None:
        event("ui", f"prova#{st['n']} nessuna voce rilevata: Lat dal click ON")
    w, c = (wer(st["gt"], text), cer(st["gt"], text)) if st["gt"] else (None, None)
    row = {"prova": st["n"], "ora": time.strftime("%H:%M:%S"),
           "modello": st["mid"], "audio_s": round(audio_s, 2),
           "gt": st["gt"], "testo": text, "wer": w, "cer": c,
           "lat_s": round(lat, 2) if lat is not None else None,
           "infer_s": round(st["infer_s"], 2), "rtf": round(rtf, 2), "revs": st["revs"]}
    SESSION.append(row)
    write_session_files()
    event("ui", f"prova#{st['n']} done model={short(st['mid'])} audio={audio_s:.1f}s "
                f"wer={fmt(w)} cer={fmt(c)} lat={row['lat_s']} rtf={row['rtf']:.2f}")
    gr.Info(f"Prova #{st['n']} chiusa: WER {fmt(w)} | CER {fmt(c)} | RTF {rtf:.2f}")
    n = st["n"]
    st.update(new_trial())
    lat_txt = f"{row['lat_s']}" if row["lat_s"] is not None else DASH
    return (live_btn_state(), gr.update(interactive=True), gr.update(interactive=True),
            text,
            f"Prova #{n} chiusa: WER {fmt(w)} | CER {fmt(c)} | RTF {rtf:.2f}.",
            table_view(), last_card_view(), lat_txt, f"{rtf:.2f}", fmt(w))


def live_transcribe(model_id: str, chunk, st):
    """Generatore sui chunk mic: accumula buffer, VAD, parziali live.

    Nativo Nemotron: feed ogni 0.3s di audio nuovo, parziali committed.
    Altri: ritrascrizione batch ogni 2s di audio nuovo. Nessun parziale se inattivo.
    """
    now = time.perf_counter()
    st = st or new_trial()
    if not st.get("active"):
        yield st, "Premi Live ON e parla.", "In attesa. Attiva il modello, poi Live ON."
        return
    if st.get("mid") != model_id:
        event("live", f"prova#{st['n']} dropdown cambiato durante la prova: ignoro, uso {short(st['mid'])}")
    if chunk is None:
        audio_s = len(st["samples"]) / 16000
        if VERBOSE["live_chunks"]:
            event("live", f"prova#{st['n']} chunk=None buf={audio_s:.1f}s")
        yield st, st["last_text"] or "In ascolto... parla pure.", f"Prova #{st['n']} | audio {audio_s:.1f}s | in attesa chunk."
        return
    try:
        wav, sr = to_mono_16k(chunk, normalize=False)
    except Exception as exc:
        event("live", f"prova#{st['n']} chunk ignorato: {exc}")
        yield st, st["last_text"] or "In ascolto... parla pure.", f"Prova #{st['n']} | chunk ignorato."
        return
    if st.get("last_sr") != sr:
        event("live", f"prova#{st['n']} sr chunk {st.get('last_sr')} -> {sr} len={len(wav)}")
        st["last_sr"] = sr
    elif VERBOSE["live_chunks"]:
        event("live", f"prova#{st['n']} chunk sr={sr} len={len(wav)}")
    if st["t_speech"] is None and chunk_is_speech(wav):
        st["t_speech"] = now
        event("live", f"prova#{st['n']} voce dopo {now - st['t_on']:.1f}s da ON")
    st["samples"].extend(wav.tolist())
    max_keep = 16000 * 30
    if len(st["samples"]) > max_keep:
        drop = len(st["samples"]) - max_keep
        st["samples"] = st["samples"][drop:]
        st["last_len"] = max(0, st["last_len"] - drop)
    audio_s = len(st["samples"]) / 16000
    new_s = (len(st["samples"]) - st["last_len"]) / 16000
    if st.get("native"):
        if len(st["samples"]) - st.get("last_feed_len", 0) >= 4800:
            try:
                adapter = get_adapter(st["mid"])
            except Exception as exc:
                event("live", f"prova#{st['n']} ERRORE caricamento: {exc}")
                st["native"] = False
            else:
                try:
                    t0 = time.perf_counter()
                    text = nemotron_feed(adapter, np.asarray(
                        st["samples"][st.get("last_feed_len", 0):], dtype=np.float32))
                    st["infer_s"] += time.perf_counter() - t0
                    st["last_feed_len"] = len(st["samples"])
                    if text.strip() and st["t_first"] is None:
                        st["t_first"] = time.perf_counter()
                        event("live", f"prova#{st['n']} primo testo dopo {st['t_first']-st['t_on']:.1f}s (nativo)")
                    if text != st["last_text"]:
                        st["revs"] += 1
                    st["last_text"] = text
                except Exception as exc:
                    event("live", f"prova#{st['n']} ERRORE nativo ({exc}): fallback re-tx")
                    st["native"] = False
        lat_txt = f"lat {st['t_first']-st['t_on']:.1f}s" if st["t_first"] else "lat --"
        voice_txt = "voce ok" if st["t_speech"] else "in attesa voce..."
        tag = "nativo" if st.get("native") else "re-tx"
        yield st, st["last_text"] or "In ascolto... parla pure.", \
            f"Prova #{st['n']} | audio {audio_s:.1f}s | {voice_txt} | {lat_txt} | {sname(st['mid'])} [{tag}]"
        return
    if new_s >= 2.0 and now - st["last_t"] >= 1.5:
        try:
            adapter = get_adapter(st["mid"])
        except Exception as exc:
            event("live", f"prova#{st['n']} ERRORE caricamento: {exc}")
            yield st, f"Errore caricamento: {exc}", f"Prova #{st['n']} | errore."
            return
        audio = np.asarray(st["samples"], dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0:
            audio = audio / peak * 0.95
        st["last_t"] = time.perf_counter()
        try:
            t0 = time.perf_counter()
            text = run_transcribe(adapter, audio, 16000)
            st["infer_s"] += time.perf_counter() - t0
            if text.strip() and st["t_first"] is None:
                st["t_first"] = time.perf_counter()
                event("live", f"prova#{st['n']} primo testo dopo {st['t_first']-st['t_on']:.1f}s")
            if text != st["last_text"]:
                st["revs"] += 1
            st["last_text"] = text
            st["last_len"] = len(st["samples"])
        except Exception as exc:
            event("live", f"prova#{st['n']} ERRORE modello: {exc}")
            st["last_text"] = f"Errore modello: {exc}"
    lat_txt = f"lat {st['t_first']-st['t_on']:.1f}s" if st["t_first"] else "lat --"
    voice_txt = "voce ok" if st["t_speech"] else "in attesa voce..."
    yield st, st["last_text"] or "In ascolto... parla pure.", \
        f"Prova #{st['n']} | audio {audio_s:.1f}s | {voice_txt} | {lat_txt} | {sname(st['mid'])}"


def set_verbose(on: bool):
    """Accende/spegne il log dei chunk live dalla sidebar."""
    VERBOSE["live_chunks"] = bool(on)
    event("ui", f"sidebar: log chunk live {'ON' if on else 'OFF'}")
    return status_html(), log_tail(60)


def ui_clear():
    """Pulisce il buffer log mostrato in sidebar. Il file resta."""
    log_clear()
    event("ui", "sidebar: log pulito da utente")
    return status_html(), log_tail(60)


def reset_session():
    """Azzera righe e file sessione. Il contatore prove continua."""
    SESSION.clear()
    write_session_files()
    event("ui", "sessione azzerata da utente (contatore prove continua)")
    gr.Info("Sessione azzerata.")
    return table_view(), last_card_view(), DASH, DASH, DASH, status_html(), log_tail(60)


model_status_init()
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(CSV_COLS)
open(JSONL_PATH, "w", encoding="utf-8").close()
event("sys", f"app init modelli={len([m for m in MODELS if m.id in ADAPTERS])} log={log_path()}")

with gr.Blocks(title="asr-bench", css=CSS) as demo:
    gr.HTML(SIDEBAR_JS)
    with gr.Row():
        gr.Markdown("# asr-bench — una prova alla volta", elem_id="asr-title")
        panel_btn = gr.Button("Pannello", size="sm", scale=0, min_width=120)
    panel_btn.click(None, js="() => { window.asrToggleSidebar(); }")
    with gr.Sidebar(label="Stato e log", open=True, position="right", elem_id="asr-sidebar", width=380):
        gr.Markdown("### Stato modelli")
        side_models = gr.HTML(status_html())
        side_log = gr.Textbox(label="Log · auto 2s", lines=10, max_lines=20,
                              value=log_tail(30), info=f"{log_path()}")
        verbose_box = gr.Checkbox(label="chunk live (verbose)", value=False)
        with gr.Row():
            side_refresh = gr.Button("Aggiorna", size="sm")
            side_clear = gr.Button("Pulisci", size="sm")
            side_dl = gr.DownloadButton(label="Log", value=log_path(), size="sm")
        side_refresh.click(lambda: (status_html(), log_tail(60)), inputs=None, outputs=[side_models, side_log])
        side_clear.click(ui_clear, inputs=None, outputs=[side_models, side_log])
        verbose_box.change(set_verbose, inputs=[verbose_box], outputs=[side_models, side_log])
        side_timer = gr.Timer(2.0)
        side_timer.tick(lambda: (status_html(), log_tail(60)), inputs=None, outputs=[side_models, side_log])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1) Frase e modello")
            gt_box = gr.Textbox(label="Frase da pronunciare (opzionale)",
                                placeholder="es. il gatto dorme sul divano",
                                lines=1,
                                info="Senza frase, niente WER. Si blocca durante la prova.")
            model_id = gr.Dropdown(choices=model_choices(), value=model_choices()[0][1], label="Modello")
            with gr.Row():
                attiva_btn = gr.Button("Attiva", variant="primary", size="sm")
                scarica_btn = gr.Button("Scarica attivo", size="sm")
            attiva_status = gr.Markdown("_Nessun modello attivo._")
        with gr.Column():
            gr.Markdown("### Dettagli e KPI")
            with gr.Accordion("Dettagli modello", open=False):
                info = gr.Markdown(model_info(model_choices()[0][1]))
            with gr.Row():
                lat_kpi = gr.Textbox(label="Lat 1° (s)", value=DASH, lines=1, interactive=False)
                rtf_kpi = gr.Textbox(label="RTF", value=DASH, lines=1, interactive=False)
                wer_kpi = gr.Textbox(label="WER", value=DASH, lines=1, interactive=False)

    with gr.Row():
        gr.Markdown("### Come streamma")
    with gr.Row():
        flow = gr.HTML(flow_html(model_choices()[0][1]))

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 2) Prova")
            live_audio = gr.Audio(sources=["microphone"], type="numpy", streaming=True,
                                  label="1) Microfono: premi Record qui")
            live_btn = gr.Button("2) Attiva un modello per la prova", variant="secondary", interactive=False)
        with gr.Column():
            gr.Markdown("### Live")
            live_text = gr.Textbox(label="Trascrizione live", lines=4)
            live_status = gr.Textbox(label="Stato prova", lines=1)
    trial_state = gr.State(value=None)

    gr.Markdown("### 3) Sessione")
    with gr.Row():
        session_table = gr.Dataframe(headers=TABLE_HEADERS, value=[], interactive=False,
                                     label="Prove · Lat dal primo suono", max_height=220)
        last_card = gr.Markdown("_Nessuna prova in sessione._")
    with gr.Row():
        dl_csv = gr.DownloadButton(label="CSV", value=CSV_PATH, size="sm")
        dl_jsonl = gr.DownloadButton(label="JSONL", value=JSONL_PATH, size="sm")
        reset_btn = gr.Button("Azzera sessione", size="sm")

    model_id.change(on_select, inputs=model_id, outputs=[info, flow, side_models, side_log, live_btn])
    attiva_btn.click(preload, inputs=[model_id], outputs=[attiva_status, side_models, side_log, live_btn])
    scarica_btn.click(unload_model, inputs=None, outputs=[attiva_status, side_models, side_log, live_btn])
    live_btn.click(live_toggle, inputs=[model_id, gt_box, trial_state],
                   outputs=[live_btn, gt_box, model_id, live_text, live_status,
                            session_table, last_card, lat_kpi, rtf_kpi, wer_kpi])
    live_audio.stream(live_transcribe, inputs=[model_id, live_audio, trial_state],
                      outputs=[trial_state, live_text, live_status],
                      stream_every=1.0, time_limit=300)
    reset_btn.click(reset_session, inputs=None,
                    outputs=[session_table, last_card, lat_kpi, rtf_kpi, wer_kpi,
                             side_models, side_log])

if __name__ == "__main__":
    demo.launch()  # su HF Space: bind 0.0.0.0:7860 automatico
