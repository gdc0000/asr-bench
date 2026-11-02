"""Prove live: stato trial, loop mic, chiusura e archiviazione."""
from __future__ import annotations

import csv
import json
import os
import time

import gradio as gr
import numpy as np

from asrbench.core import state
from asrbench.core.logbuffer import event
from asrbench.core.logbuffer import tail as log_tail
from asrbench.engine.audio import to_mono_16k
from asrbench.engine.helpers import chunk_is_speech, short, sname
from asrbench.engine.scores import cer, fmt, wer
from asrbench.engine.service import get_adapter, nemotron_feed, nemotron_finish, run_transcribe
from asrbench.ui.i18n import T
from asrbench.ui.status import live_btn_state, status_html


def new_trial():
    """Stato prova vergine: buffer, tempi, metriche."""
    return {"active": False, "n": 0, "mid": None, "gt": "", "samples": [],
            "last_len": 0, "last_text": "", "last_t": 0.0, "t_on": 0.0,
            "t_first": None, "t_speech": None, "infer_s": 0.0, "revs": 0, "k": 0,
            "last_log": 0.0, "last_sr": None, "last_feed_len": 0, "native": False}


def table_view():
    """Righe sessione per il dataframe: # modello WER CER lat RTF audio."""
    rows = []
    for r in state.SESSION:
        rows.append([r["prova"], sname(r["modello"]),
                     fmt(r["wer"]), fmt(r["cer"]),
                     f"{r['lat_s']:.1f}" if r["lat_s"] is not None else state.DASH,
                     f"{r['rtf']:.2f}", f"{r['audio_s']:.1f}"])
    return rows


def last_card_view(lang: str = "it"):
    """Card markdown dell'ultima prova chiusa, o placeholder."""
    if not state.SESSION:
        return T(lang, "no_session")
    r = state.SESSION[-1]
    gt = r["gt"] or T(lang, "mi_not_given")
    lat = f"{r['lat_s']}s" if r["lat_s"] is not None else state.DASH
    return (f"**Prova #{r['prova']} — {sname(r['modello'])}**\n\n"
            f"{T(lang, 'gt_word')}: {gt}\n\n{T(lang, 'text_word')}: {r['testo']}\n\n"
            f"WER {fmt(r['wer'])} | CER {fmt(r['cer'])} | lat {lat} | RTF {r['rtf']}")


def write_session_files():
    """Riscrive CSV + JSONL della sessione dopo ogni prova."""
    with open(state.CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(state.CSV_COLS)
        for r in state.SESSION:
            w.writerow([r["prova"], r["ora"], r["modello"], f"{r['audio_s']:.2f}",
                        r["gt"], r["testo"],
                        "" if r["wer"] is None else round(r["wer"] * 100, 1),
                        "" if r["cer"] is None else round(r["cer"] * 100, 1),
                        "" if r["lat_s"] is None else round(r["lat_s"], 2),
                        round(r["infer_s"], 2), round(r["rtf"], 2), r["revs"]])
    with open(state.JSONL_PATH, "w", encoding="utf-8") as f:
        for r in state.SESSION:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def reset_session(lang: str = "it"):
    """Azzera righe e file sessione. Il contatore prove continua."""
    state.SESSION.clear()
    write_session_files()
    event("ui", "sessione azzerata da utente (contatore prove continua)")
    gr.Info(T(lang, "session_cleared"))
    return table_view(), last_card_view(lang), state.DASH, state.DASH, state.DASH, status_html(lang), log_tail(60)


def live_toggle(model_id: str, gt: str, st, lang: str = "it"):
    """Un bottone: ON apre la prova (GT congelato), OFF la chiude e la archivia.

    Audio sotto 1s scartato. Ramo nativo per Nemotron, batch per gli altri.
    """
    st = st or new_trial()
    if not st.get("active"):
        if state.CURRENT.get("id") != model_id or state.CURRENT.get("adapter") is None:
            event("ui", "click Live ON rifiutato: modello non pronto, premi Attiva")
            gr.Warning(T(lang, "activate_first", m=sname(model_id)))
            return (gr.update(), gr.update(), gr.update(),
                    st["last_text"] or state.DASH,
                    T(lang, "activate_first", m=sname(model_id)),
                    table_view(), last_card_view(lang), state.DASH, state.DASH, state.DASH, st)
        state.TRIAL_SEQ += 1
        st.update(new_trial())
        st.update({"active": True, "n": state.TRIAL_SEQ, "mid": model_id,
                   "gt": (gt or "").strip(), "t_on": time.perf_counter(),
                   "native": False})
        native = (model_id == state.NEMOTRON_ID
                  and os.environ.get("SPACES_ZERO_GPU") != "1")
        if native:  # nativo solo locale: su Space la sessione non attraversa i worker
            try:
                state.CURRENT["adapter"].stream_start()
                st["native"] = True
            except Exception as exc:
                event("live", f"prova#{state.TRIAL_SEQ} stream nativo non avviato ({exc}): fallback re-tx")
                st["native"] = False
        state.TRIAL_ACTIVE = True
        event("ui", f"prova#{state.TRIAL_SEQ} Live ON model={short(model_id)} "
                    f"gt={'si' if st['gt'] else 'no'} {'nativo' if st['native'] else 're-tx'}")
        tag = "nativo" if st["native"] else "re-tx"
        return (gr.update(value=T(lang, "live_off"), variant="stop", interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                T(lang, "listening"),
                T(lang, "trial_on", n=state.TRIAL_SEQ, m=sname(model_id), tag=tag),
                table_view(), last_card_view(lang), state.DASH, state.DASH, state.DASH, st)
    # chiusura prova
    audio_s = len(st["samples"]) / 16000
    st["active"] = False
    state.TRIAL_ACTIVE = False
    if audio_s < 1.0:
        event("ui", f"prova#{st['n']} Live OFF scartata: audio {audio_s:.1f}s < 1s")
        gr.Warning(T(lang, "too_short", n=st["n"], a=f"{audio_s:.1f}"))
        out = (live_btn_state(lang), gr.update(interactive=True), gr.update(interactive=True),
               st["last_text"] or state.DASH,
               T(lang, "too_short", n=st["n"], a=f"{audio_s:.1f}"),
               table_view(), last_card_view(lang), state.DASH, state.DASH, state.DASH)
        st.update(new_trial())
        return out + (st,)
    if st.get("native"):
        try:
            adapter = get_adapter(st["mid"], lang)
        except Exception as exc:
            event("ui", f"prova#{st['n']} ERRORE reload modello: {exc}")
            gr.Warning(T(lang, "model_error", e=exc))
            return (live_btn_state(lang), gr.update(interactive=True), gr.update(interactive=True),
                    st["last_text"] or state.DASH, T(lang, "model_error", e=exc),
                    table_view(), last_card_view(lang), state.DASH, state.DASH, state.DASH, st)
        t0 = time.perf_counter()
        try:
            text = nemotron_finish(adapter)
        except Exception as exc:
            event("ui", f"prova#{st['n']} ERRORE finish nativo: {exc}")
            gr.Warning(T(lang, "model_error", e=exc))
            return (live_btn_state(lang), gr.update(interactive=True), gr.update(interactive=True),
                    st["last_text"] or state.DASH, T(lang, "model_error", e=exc),
                    table_view(), last_card_view(lang), state.DASH, state.DASH, state.DASH, st)
        st["infer_s"] += time.perf_counter() - t0
        st["last_text"] = text
    else:
        try:
            adapter = get_adapter(st["mid"], lang)
        except Exception as exc:
            event("ui", f"prova#{st['n']} ERRORE reload modello: {exc}")
            gr.Warning(T(lang, "model_error", e=exc))
            return (live_btn_state(lang), gr.update(interactive=True), gr.update(interactive=True),
                    st["last_text"] or state.DASH, T(lang, "model_error", e=exc),
                    table_view(), last_card_view(lang), state.DASH, state.DASH, state.DASH, st)
        audio = np.asarray(st["samples"], dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0:
            audio = audio / peak * 0.95
        t0 = time.perf_counter()
        try:
            text = run_transcribe(st["mid"], audio, 16000)
        except Exception as exc:
            event("ui", f"prova#{st['n']} ERRORE trascrizione finale: {exc}")
            gr.Warning(T(lang, "model_error", e=exc))
            return (live_btn_state(lang), gr.update(interactive=True), gr.update(interactive=True),
                    st["last_text"] or state.DASH, T(lang, "model_error", e=exc),
                    table_view(), last_card_view(lang), state.DASH, state.DASH, state.DASH, st)
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
    state.SESSION.append(row)
    write_session_files()
    event("ui", f"prova#{st['n']} done model={short(st['mid'])} audio={audio_s:.1f}s "
                f"wer={fmt(w)} cer={fmt(c)} lat={row['lat_s']} rtf={row['rtf']:.2f}")
    gr.Info(T(lang, "closed", n=st["n"], w=fmt(w), c=fmt(c), r=f"{rtf:.2f}"))
    n = st["n"]
    st.update(new_trial())
    lat_txt = f"{row['lat_s']}" if row["lat_s"] is not None else state.DASH
    return (live_btn_state(lang), gr.update(interactive=True), gr.update(interactive=True),
            text,
            T(lang, "closed", n=n, w=fmt(w), c=fmt(c), r=f"{rtf:.2f}"),
            table_view(), last_card_view(lang), lat_txt, f"{rtf:.2f}", fmt(w), st)


def live_transcribe(model_id: str, chunk, st, lang: str = "it"):
    """Generatore sui chunk mic: accumula buffer, VAD, parziali live.

    Nativo Nemotron: feed ogni 0.3s di audio nuovo, parziali committed.
    Altri: ritrascrizione batch ogni 2s di audio nuovo. Nessun parziale se inattivo.
    """
    now = time.perf_counter()
    st = st or new_trial()
    if not st.get("active"):
        yield st, T(lang, "idle_press"), T(lang, "idle_wait")
        return
    if st.get("mid") != model_id:
        event("live", f"prova#{st['n']} dropdown cambiato durante la prova: ignoro, uso {short(st['mid'])}")
    if chunk is None:
        audio_s = len(st["samples"]) / 16000
        if state.VERBOSE["live_chunks"]:
            event("live", f"prova#{st['n']} chunk=None buf={audio_s:.1f}s")
        pre = f"Prova #{st['n']}" if lang == "it" else f"Trial #{st['n']}"
        yield st, st["last_text"] or T(lang, "listening"), \
            f"{pre} | audio {audio_s:.1f}s | {T(lang, 'waiting_chunks')}"
        return
    try:
        wav, sr = to_mono_16k(chunk, normalize=False)
    except Exception as exc:
        event("live", f"prova#{st['n']} chunk ignorato: {exc}")
        pre = f"Prova #{st['n']}" if lang == "it" else f"Trial #{st['n']}"
        yield st, st["last_text"] or T(lang, "listening"), \
            f"{pre} | {T(lang, 'chunk_ignored')}"
        return
    if st.get("last_sr") != sr:
        event("live", f"prova#{st['n']} sr chunk {st.get('last_sr')} -> {sr} len={len(wav)}")
        st["last_sr"] = sr
    elif state.VERBOSE["live_chunks"]:
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
                adapter = get_adapter(st["mid"], lang)
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
        lat_txt = (f"lat {st['t_first']-st['t_on']:.1f}s" if st["t_first"]
                   else T(lang, "lat_none"))
        voice_txt = T(lang, "voice_ok") if st["t_speech"] else T(lang, "waiting_voice")
        tag = "nativo" if st.get("native") else "re-tx"
        yield st, st["last_text"] or T(lang, "listening"), \
            T(lang, "trial_status", n=st["n"], a=f"{audio_s:.1f}", v=voice_txt,
              l=lat_txt, m=sname(st["mid"]), tag=f" [{tag}]")
        return
    if new_s >= 2.0 and now - st["last_t"] >= 1.5:
        try:
            adapter = get_adapter(st["mid"], lang)
        except Exception as exc:
            event("live", f"prova#{st['n']} ERRORE caricamento: {exc}")
            pre = f"Prova #{st['n']}" if lang == "it" else f"Trial #{st['n']}"
            err = "errore" if lang == "it" else "error"
            yield st, T(lang, "load_error_short", e=exc), f"{pre} | {err}."
            return
        audio = np.asarray(st["samples"], dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0:
            audio = audio / peak * 0.95
        st["last_t"] = time.perf_counter()
        try:
            t0 = time.perf_counter()
            text = run_transcribe(st["mid"], audio, 16000)
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
            st["last_text"] = T(lang, "model_error", e=exc)
    lat_txt = (f"lat {st['t_first']-st['t_on']:.1f}s" if st["t_first"]
               else T(lang, "lat_none"))
    voice_txt = T(lang, "voice_ok") if st["t_speech"] else T(lang, "waiting_voice")
    yield st, st["last_text"] or T(lang, "listening"), \
        T(lang, "trial_status", n=st["n"], a=f"{audio_s:.1f}", v=voice_txt,
          l=lat_txt, m=sname(st["mid"]), tag="")
