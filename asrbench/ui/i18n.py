"""Copy IT/EN, schemi di flusso e deep-dive ingegneristici per la UI."""
from __future__ import annotations

from asrbench.core.registry import MODELS


def T(lang: str, key: str, **kw) -> str:
    """Stringa UI in lingua, con formattazione."""
    return STRINGS.get(lang, STRINGS["it"])[key].format(**kw)


STRINGS = {
    "it": {
        "title": "# asr-bench — una prova alla volta",
        "guide": ("**Cosa stai facendo:** 1. scrivi la frase e scegli il modello · "
                  "2. premi Attiva, poi Live ON e parla · "
                  "3. leggi WER/CER in sessione e confronta i modelli."),
        "panel": "Pannello",
        "side_title": "### Stato modelli",
        "log_label": "Log · auto 2s",
        "verbose": "chunk live (verbose)",
        "refresh": "Aggiorna",
        "clear": "Pulisci",
        "s1": "### 1) Frase e modello",
        "gt_label": "Frase da pronunciare (opzionale)",
        "gt_placeholder": "es. il gatto dorme sul divano",
        "gt_info": "Senza frase, niente WER. Si blocca durante la prova.",
        "model_label": "Modello",
        "attiva": "Attiva",
        "scarica": "Scarica attivo",
        "no_model": "_Nessun modello attivo._",
        "s2": "### Dettagli e KPI",
        "acc_details": "Dettagli modello",
        "lat_label": "Lat 1° (s)",
        "flow_title": "### Come streamma",
        "deep_hint": "Clicca uno step per il dettaglio ingegneristico.",
        "deep_close": "Chiudi",
        "s_trial": "### 2) Prova",
        "mic_label": "1) Microfono: premi Record qui",
        "live_need": "2) Attiva {m} per la prova",
        "live_on": "● Live ON",
        "live_off": "■ Live OFF",
        "s_live": "### Live",
        "live_text": "Trascrizione live",
        "live_status": "Stato prova",
        "s_session": "### 3) Sessione",
        "table_label": "Prove · Lat dal primo suono",
        "no_session": "_Nessuna prova in sessione._",
        "reset": "Azzera sessione",
        "idle_press": "Premi Live ON e parla.",
        "idle_wait": "In attesa. Attiva il modello, poi Live ON.",
        "listening": "In ascolto... parla pure.",
        "waiting_chunks": "in attesa chunk.",
        "chunk_ignored": "chunk ignorato.",
        "waiting_voice": "in attesa voce...",
        "voice_ok": "voce ok",
        "lat_none": "lat --",
        "activate_first": "Modello {m} non attivo: premi Attiva prima.",
        "trial_busy": "Prova in corso: premi Live OFF prima di {op}.",
        "op_switch": "cambiare modello",
        "op_unload": "scaricare",
        "too_short": "Prova #{n} scartata: audio troppo corto ({a}s).",
        "closed": "Prova #{n} chiusa: WER {w} | CER {c} | RTF {r}.",
        "session_cleared": "Sessione azzerata.",
        "model_ready": "Modello {m} pronto e attivo.",
        "model_already": "Modello {m} gia' attivo.",
        "loading": "Caricamento {m}... (puo' durare minuti al primo uso)",
        "load_error": "Errore caricamento: {e}",
        "model_error": "Errore modello: {e}",
        "load_error_short": "Errore caricamento: {e}",
        "no_active": "Nessun modello attivo.",
        "unloaded": "Modello {m} scaricato. VRAM liberata.",
        "close_warn": "Chiudi la prova prima di {op}.",
        "press_mic": "premi il microfono e parla.",
        "trial_on": "Prova #{n} | {m} [{tag}] | premi il microfono e parla.",
        "trial_status": "Prova #{n} | audio {a}s | {v} | {l} | {m}{tag}",
        "gt_word": "GT",
        "text_word": "Testo",
        "st_ready": "pronto",
        "st_loading": "caricamento",
        "st_idle": "non caricato",
        "st_error": "errore",
        "use_active": "attivo",
        "use_selected": "selezionato",
        "detail_loading": "download pesi + init...",
        "detail_ready": "caricato in {s}s",
        "gated": "serve HF_TOKEN con accesso approvato al repo",
        "th_model": "Modello",
        "th_state": "Stato",
        "th_use": "Uso",
        "hf_page": "scheda Hugging Face",
        "mi_license": "Licenza",
        "mi_params": "Parametri",
        "mi_size": "Dimensione",
        "mi_lang": "Lingue",
        "mi_stream": "Streaming",
        "mi_feat": "Caratteristiche",
        "mi_notes": "Note",
        "mi_not_given": "_non fornita_",
    },
    "en": {
        "title": "# asr-bench — one trial at a time",
        "guide": ("**What you are doing:** 1. type the sentence and pick the model · "
                  "2. press Activate, then Live ON and speak · "
                  "3. read WER/CER in the session and compare models."),
        "panel": "Panel",
        "side_title": "### Model status",
        "log_label": "Log · auto 2s",
        "verbose": "live chunks (verbose)",
        "refresh": "Refresh",
        "clear": "Clear",
        "s1": "### 1) Sentence & model",
        "gt_label": "Sentence to speak (optional)",
        "gt_placeholder": "e.g. the cat sleeps on the couch",
        "gt_info": "No sentence, no WER. Frozen during the trial.",
        "model_label": "Model",
        "attiva": "Activate",
        "scarica": "Unload active",
        "no_model": "_No active model._",
        "s2": "### Details & KPIs",
        "acc_details": "Model details",
        "lat_label": "1st lat (s)",
        "flow_title": "### How it streams",
        "deep_hint": "Click a step for engineering detail.",
        "deep_close": "Close",
        "s_trial": "### 2) Trial",
        "mic_label": "1) Microphone: press Record here",
        "live_need": "2) Activate {m} for the trial",
        "live_on": "● Live ON",
        "live_off": "■ Live OFF",
        "s_live": "### Live",
        "live_text": "Live transcription",
        "live_status": "Trial status",
        "s_session": "### 3) Session",
        "table_label": "Trials · Lat from first sound",
        "no_session": "_No trials in session._",
        "reset": "Reset session",
        "idle_press": "Press Live ON and speak.",
        "idle_wait": "Waiting. Activate the model, then Live ON.",
        "listening": "Listening... speak.",
        "waiting_chunks": "waiting for chunks.",
        "chunk_ignored": "chunk ignored.",
        "waiting_voice": "waiting for voice...",
        "voice_ok": "voice ok",
        "lat_none": "lat --",
        "activate_first": "Model {m} not active: press Activate first.",
        "trial_busy": "Trial running: press Live OFF before {op}.",
        "op_switch": "switching models",
        "op_unload": "unloading",
        "too_short": "Trial #{n} discarded: audio too short ({a}s).",
        "closed": "Trial #{n} closed: WER {w} | CER {c} | RTF {r}.",
        "session_cleared": "Session cleared.",
        "model_ready": "Model {m} ready and active.",
        "model_already": "Model {m} already active.",
        "loading": "Loading {m}... (may take minutes on first use)",
        "load_error": "Loading error: {e}",
        "model_error": "Model error: {e}",
        "load_error_short": "Loading error: {e}",
        "no_active": "No active model.",
        "unloaded": "Model {m} unloaded. VRAM freed.",
        "close_warn": "Close the trial before {op}.",
        "press_mic": "press the microphone and speak.",
        "trial_on": "Trial #{n} | {m} [{tag}] | press the microphone and speak.",
        "trial_status": "Trial #{n} | audio {a}s | {v} | {l} | {m}{tag}",
        "gt_word": "GT",
        "text_word": "Text",
        "st_ready": "ready",
        "st_loading": "loading",
        "st_idle": "not loaded",
        "st_error": "error",
        "use_active": "active",
        "use_selected": "selected",
        "detail_loading": "downloading weights + init...",
        "detail_ready": "loaded in {s}s",
        "gated": "needs HF_TOKEN with approved repo access",
        "th_model": "Model",
        "th_state": "State",
        "th_use": "Use",
        "hf_page": "Hugging Face page",
        "mi_license": "License",
        "mi_params": "Parameters",
        "mi_size": "Size",
        "mi_lang": "Languages",
        "mi_stream": "Streaming",
        "mi_feat": "Features",
        "mi_notes": "Notes",
        "mi_not_given": "_not given_",
    },
}

TABLE_HEADERS = {
    "it": ["#", "Modello", "WER", "CER", "Lat", "RTF", "Audio"],
    "en": ["#", "Model", "WER", "CER", "Lat", "RTF", "Audio"],
}

FLOW = {
    "ibm-granite/granite-speech-5.0-470m-turboctc": {
        "mode": "re-tx",
        "it": [("Mic", "chunk 1s"), ("Buffer", "accumulo, max 30s"),
               ("Ogni 2s", "ritrascrive tutto"), ("CTC greedy", "frame → argmax"),
               ("Testo", "parziale riscritto")],
        "en": [("Mic", "1s chunks"), ("Buffer", "accumulates, max 30s"),
               ("Every 2s", "re-decodes all"), ("CTC greedy", "frame → argmax"),
               ("Text", "rewritten partial")],
    },
    "Qwen/Qwen3-ASR-0.6B": {
        "mode": "re-tx",
        "it": [("Mic", "chunk 1s"), ("Buffer", "accumulo, max 30s"),
               ("Ogni 2s", "ritrascrive tutto"), ("Prompt chat", "lingua Italian"),
               ("Generate LLM", "0.6B autoregressivo"), ("Testo", "prompt tagliato")],
        "en": [("Mic", "1s chunks"), ("Buffer", "accumulates, max 30s"),
               ("Every 2s", "re-decodes all"), ("Chat prompt", "language Italian"),
               ("LLM generate", "0.6B autoregressive"), ("Text", "prompt trimmed")],
    },
    "openai/whisper-small": {
        "mode": "re-tx",
        "it": [("Mic", "chunk 1s"), ("Buffer", "accumulo, max 30s"),
               ("Ogni 2s", "ritrascrive tutto"), ("Log-mel", "80 bin"),
               ("Generate it", "autoregressivo"), ("Testo", "parziale riscritto")],
        "en": [("Mic", "1s chunks"), ("Buffer", "accumulates, max 30s"),
               ("Every 2s", "re-decodes all"), ("Log-mel", "80 bins"),
               ("Generate it", "autoregressive"), ("Text", "rewritten partial")],
    },
    "nvidia/nemotron-3.5-asr-streaming-0.6b": {
        "mode": "nativo",
        "it": [("Mic", "chunk 1s"), ("Feed 0.3s", "mel a pezzi 25/32"),
               ("Encoder cache", "KV + conv portate avanti"), ("RNNT greedy", "frame → token"),
               ("Testo", "committed, niente rewrite")],
        "en": [("Mic", "1s chunks"), ("Feed 0.3s", "mel in 25/32 pieces"),
               ("Cached encoder", "KV + conv carried over"), ("RNNT greedy", "frame → token"),
               ("Text", "committed, no rewrites")],
    },
}

FLOW_NOTE = {
    "re-tx": {"it": "↻ Il buffer intero viene ritrascritto a ogni giro: costo O(n²), parziali riscritti.",
              "en": "↻ The whole buffer is re-decoded every cycle: O(n²) cost, rewritten partials."},
    "nativo": {"it": "→ Ogni frame è codificato e decodificato una volta sola: costo O(n), parziali stabili.",
               "en": "→ Each frame is encoded and decoded once: O(n) cost, stable partials."},
}

# Dettagli per step, allineati a FLOW[model][lang]. HTML con <code>, niente < > grezzi.
DEEP = {
    "ibm-granite/granite-speech-5.0-470m-turboctc": {
        "it": [
            ("1. Mic", ["<code>gr.Audio(streaming, stream_every=1.0)</code> → tupla <code>(sr, np.ndarray)</code>",
                        "<code>to_mono_16k()</code>: mono float32 a 16 kHz",
                        "VAD <code>chunk_is_speech()</code>: RMS &gt; 0.02, scala auto int16"]),
            ("2. Buffer", ["lista <code>st[\"samples\"]</code>, accoda ogni chunk",
                           "tetto 30 s (<code>max_keep</code>), poi finestra scorrevole",
                           "cursore <code>last_len</code>: cosa è già stato trascritto"]),
            ("3. Ogni 2s", ["scatta con 2 s di audio nuovo e 1.5 s dall'ultimo giro",
                            "<code>run_transcribe()</code> su tutto il buffer, normalizzato a picco 0.95",
                            "costo O(n²): più parli, più ogni giro costa"]),
            ("4. CTC greedy", ["<code>AutoProcessor</code> → <code>input_features</code>",
                               "<code>AutoModelForCTC(**inputs).logits</code> forma <code>[1,T,vocab]</code>",
                               "<code>argmax(dim=-1)</code> frame per frame, poi <code>batch_decode()</code>. Niente <code>generate</code>"]),
            ("5. Testo", ["<code>last_text</code> mostrato live, riscritto a ogni giro",
                          "<code>revs</code> conta le riscritture",
                          "<code>t_first</code>: prima emissione; latenza dal primo suono"]),
        ],
        "en": [
            ("1. Mic", ["<code>gr.Audio(streaming, stream_every=1.0)</code> → tuple <code>(sr, np.ndarray)</code>",
                        "<code>to_mono_16k()</code>: mono float32 at 16 kHz",
                        "VAD <code>chunk_is_speech()</code>: RMS &gt; 0.02, auto int16 scale"]),
            ("2. Buffer", ["list <code>st[\"samples\"]</code>, appends every chunk",
                           "30 s cap (<code>max_keep</code>), then sliding window",
                           "cursor <code>last_len</code>: what is already transcribed"]),
            ("3. Every 2s", ["fires with 2 s of new audio and 1.5 s since last pass",
                             "<code>run_transcribe()</code> on the whole buffer, peak-normalized to 0.95",
                             "O(n²) cost: the longer you speak, the pricier each pass"]),
            ("4. CTC greedy", ["<code>AutoProcessor</code> → <code>input_features</code>",
                               "<code>AutoModelForCTC(**inputs).logits</code> shape <code>[1,T,vocab]</code>",
                               "<code>argmax(dim=-1)</code> frame by frame, then <code>batch_decode()</code>. No <code>generate</code>"]),
            ("5. Text", ["<code>last_text</code> shown live, rewritten each pass",
                         "<code>revs</code> counts rewrites",
                         "<code>t_first</code>: first emission; latency from first sound"]),
        ],
    },
    "Qwen/Qwen3-ASR-0.6B": {
        "it": [
            ("1. Mic", ["<code>gr.Audio(streaming, stream_every=1.0)</code> → tupla <code>(sr, np.ndarray)</code>",
                        "<code>to_mono_16k()</code>: mono float32 a 16 kHz",
                        "VAD <code>chunk_is_speech()</code>: RMS &gt; 0.02, scala auto int16"]),
            ("2. Buffer", ["lista <code>st[\"samples\"]</code>, accoda ogni chunk",
                           "tetto 30 s (<code>max_keep</code>), poi finestra scorrevole",
                           "cursore <code>last_len</code>: cosa è già stato trascritto"]),
            ("3. Ogni 2s", ["scatta con 2 s di audio nuovo e 1.5 s dall'ultimo giro",
                            "<code>run_transcribe()</code> su tutto il buffer, normalizzato a picco 0.95",
                            "costo O(n²) più autoregressione LLM a ogni giro"]),
            ("4. Prompt chat", ["<code>apply_chat_template(..., add_generation_prompt=True)</code>",
                                "suffisso <code>language Italian&lt;asr_text&gt;</code>: lingua fissata",
                                "<code>Qwen3ASRProcessor(text, audio)</code> → <code>input_ids, input_features, attention_mask</code>"]),
            ("5. Generate LLM", ["config ricostruita da <code>thinker_config</code>",
                                 "remap pesi <code>thinker.audio_tower.* → model.audio_tower.*</code>, <code>thinker.model.* → model.language_model.*</code>, <code>proj1/2 → multi_modal_projector.linear_1/2</code>",
                                 "<code>.to(\"cuda\", fp16)</code>, <code>generate(max_new_tokens=256)</code>, taglio <code>generated[:, input_ids.shape[1]:]</code>"]),
            ("6. Testo", ["<code>batch_decode(skip_special_tokens=True)</code>",
                          "<code>last_text</code> riscritto a ogni giro, <code>revs</code> li conta",
                          "<code>t_first</code>: prima emissione; latenza dal primo suono"]),
        ],
        "en": [
            ("1. Mic", ["<code>gr.Audio(streaming, stream_every=1.0)</code> → tuple <code>(sr, np.ndarray)</code>",
                        "<code>to_mono_16k()</code>: mono float32 at 16 kHz",
                        "VAD <code>chunk_is_speech()</code>: RMS &gt; 0.02, auto int16 scale"]),
            ("2. Buffer", ["list <code>st[\"samples\"]</code>, appends every chunk",
                           "30 s cap (<code>max_keep</code>), then sliding window",
                           "cursor <code>last_len</code>: what is already transcribed"]),
            ("3. Every 2s", ["fires with 2 s of new audio and 1.5 s since last pass",
                             "<code>run_transcribe()</code> on the whole buffer, peak-normalized to 0.95",
                             "O(n²) cost plus LLM autoregression each pass"]),
            ("4. Chat prompt", ["<code>apply_chat_template(..., add_generation_prompt=True)</code>",
                                "suffix <code>language Italian&lt;asr_text&gt;</code>: pinned language",
                                "<code>Qwen3ASRProcessor(text, audio)</code> → <code>input_ids, input_features, attention_mask</code>"]),
            ("5. LLM generate", ["config rebuilt from <code>thinker_config</code>",
                                 "weight remap <code>thinker.audio_tower.* → model.audio_tower.*</code>, <code>thinker.model.* → model.language_model.*</code>, <code>proj1/2 → multi_modal_projector.linear_1/2</code>",
                                 "<code>.to(\"cuda\", fp16)</code>, <code>generate(max_new_tokens=256)</code>, trim <code>generated[:, input_ids.shape[1]:]</code>"]),
            ("6. Text", ["<code>batch_decode(skip_special_tokens=True)</code>",
                         "<code>last_text</code> rewritten each pass, <code>revs</code> counts them",
                         "<code>t_first</code>: first emission; latency from first sound"]),
        ],
    },
    "openai/whisper-small": {
        "it": [
            ("1. Mic", ["<code>gr.Audio(streaming, stream_every=1.0)</code> → tupla <code>(sr, np.ndarray)</code>",
                        "<code>to_mono_16k()</code>: mono float32 a 16 kHz",
                        "VAD <code>chunk_is_speech()</code>: RMS &gt; 0.02, scala auto int16"]),
            ("2. Buffer", ["lista <code>st[\"samples\"]</code>, accoda ogni chunk",
                           "tetto 30 s (<code>max_keep</code>), poi finestra scorrevole",
                           "cursore <code>last_len</code>: cosa è già stato trascritto"]),
            ("3. Ogni 2s", ["scatta con 2 s di audio nuovo e 1.5 s dall'ultimo giro",
                            "<code>run_transcribe()</code> su tutto il buffer, normalizzato a picco 0.95",
                            "costo O(n²): più parli, più ogni giro costa"]),
            ("4. Log-mel", ["<code>WhisperProcessor</code> → <code>input_features [1,80,3000]</code>",
                            "spostate su device in fp16",
                            "solo feature in input: niente trim del prompt"]),
            ("5. Generate it", ["<code>generate(input_features, language=\"it\", task=\"transcribe\", max_new_tokens=256)</code>",
                                "autoregressivo: un token alla volta",
                                "<code>batch_decode(skip_special_tokens=True)</code>"]),
            ("6. Testo", ["<code>last_text</code> mostrato live, riscritto a ogni giro",
                          "<code>revs</code> conta le riscritture",
                          "<code>t_first</code>: prima emissione; latenza dal primo suono"]),
        ],
        "en": [
            ("1. Mic", ["<code>gr.Audio(streaming, stream_every=1.0)</code> → tuple <code>(sr, np.ndarray)</code>",
                        "<code>to_mono_16k()</code>: mono float32 at 16 kHz",
                        "VAD <code>chunk_is_speech()</code>: RMS &gt; 0.02, auto int16 scale"]),
            ("2. Buffer", ["list <code>st[\"samples\"]</code>, appends every chunk",
                           "30 s cap (<code>max_keep</code>), then sliding window",
                           "cursor <code>last_len</code>: what is already transcribed"]),
            ("3. Every 2s", ["fires with 2 s of new audio and 1.5 s since last pass",
                             "<code>run_transcribe()</code> on the whole buffer, peak-normalized to 0.95",
                             "O(n²) cost: the longer you speak, the pricier each pass"]),
            ("4. Log-mel", ["<code>WhisperProcessor</code> → <code>input_features [1,80,3000]</code>",
                            "moved to device in fp16",
                            "features only in input: no prompt trim"]),
            ("5. Generate it", ["<code>generate(input_features, language=\"it\", task=\"transcribe\", max_new_tokens=256)</code>",
                                "autoregressive: one token at a time",
                                "<code>batch_decode(skip_special_tokens=True)</code>"]),
            ("6. Text", ["<code>last_text</code> shown live, rewritten each pass",
                         "<code>revs</code> counts rewrites",
                         "<code>t_first</code>: first emission; latency from first sound"]),
        ],
    },
    "nvidia/nemotron-3.5-asr-streaming-0.6b": {
        "it": [
            ("1. Mic", ["<code>gr.Audio(streaming, stream_every=1.0)</code> → tupla <code>(sr, np.ndarray)</code>",
                        "<code>to_mono_16k(normalize=False)</code>: scala mic grezza",
                        "<code>_to_float()</code>: /32768 se peak &gt; 1.5, poi VAD RMS &gt; 0.02"]),
            ("2. Feed 0.3s", ["soglia 4800 campioni di audio nuovo per feed",
                              "mel time-major <code>[B,T,128]</code> ricalcolata sul buffer",
                              "consumo a pezzi: primo <code>FIRST_N=25</code> (<code>1+8R</code>), poi <code>NEXT_N=32</code> (<code>8(R+1)</code>), <code>R=3</code>"]),
            ("3. Encoder cache", ["<code>get_audio_features(input_features [1,N,128] fp16, past_key_values, padding_cache, num_lookahead_tokens=3, use_cache=True)</code>",
                                  "<code>prompt_ids=[101]</code> solo sul primo chunk",
                                  "output <code>pooler_output [B,Te,640]</code>, subsampling ×8; cache portate avanti"]),
            ("4. RNNT greedy", ["<code>decoder(last [1,1], cache Nemotron3_5AsrRNNTDecoderCache)</code>: stati LSTM",
                                "<code>joint(dec, enc)</code> → argmax; blank <code>13087</code> = no-op, stato preservato",
                                "max 10 simboli/frame; <code>batch_decode(emitted)</code> + <code>clean_text()</code> via eco <code>&lt;xx&gt;</code>"]),
            ("5. Testo", ["committed: emesso una volta, mai riscritto; <code>revs</code> quasi solo ri-tokenizzazioni BPE",
                          "<code>finish()</code>: zero-pad dell'ultimo pezzo parziale",
                          "validato bit-identico al <code>generate</code> streaming ufficiale"]),
        ],
        "en": [
            ("1. Mic", ["<code>gr.Audio(streaming, stream_every=1.0)</code> → tuple <code>(sr, np.ndarray)</code>",
                        "<code>to_mono_16k(normalize=False)</code>: raw mic scale",
                        "<code>_to_float()</code>: /32768 if peak &gt; 1.5, then VAD RMS &gt; 0.02"]),
            ("2. Feed 0.3s", ["4800 new samples threshold per feed",
                              "time-major mel <code>[B,T,128]</code> recomputed on the buffer",
                              "consumed in pieces: first <code>FIRST_N=25</code> (<code>1+8R</code>), then <code>NEXT_N=32</code> (<code>8(R+1)</code>), <code>R=3</code>"]),
            ("3. Cached encoder", ["<code>get_audio_features(input_features [1,N,128] fp16, past_key_values, padding_cache, num_lookahead_tokens=3, use_cache=True)</code>",
                                   "<code>prompt_ids=[101]</code> on the first chunk only",
                                   "output <code>pooler_output [B,Te,640]</code>, ×8 subsampling; caches carried over"]),
            ("4. RNNT greedy", ["<code>decoder(last [1,1], cache Nemotron3_5AsrRNNTDecoderCache)</code>: LSTM states",
                                 "<code>joint(dec, enc)</code> → argmax; blank <code>13087</code> = no-op, state preserved",
                                 "max 10 symbols/frame; <code>batch_decode(emitted)</code> + <code>clean_text()</code> strips <code>&lt;xx&gt;</code> echo"]),
            ("5. Text", ["committed: emitted once, never rewritten; <code>revs</code> almost only BPE re-tokenizations",
                         "<code>finish()</code>: zero-pads the last partial piece",
                         "validated bit-identical to the official streaming <code>generate</code>"]),
        ],
    },
}

FLOW_CSS = """
<style>
.asrflow { display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; margin: 4px 0; }
a.asrflow-node { border: 1px solid var(--border-color-primary); border-radius: 8px;
  padding: 6px 10px; min-width: 96px; max-width: 150px; background: var(--background-fill-primary);
  text-decoration: none; color: inherit; cursor: pointer; }
a.asrflow-node:hover { border-color: var(--color-accent); }
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
.asrflow-hint { font-size: 12px; opacity: 0.75; }
.asrflow-badge { display: inline-block; font-size: 11px; font-weight: bold; border-radius: 4px;
  padding: 1px 8px; margin-bottom: 4px; }
.asrflow-badge.nat { background: rgba(0,128,0,.15); color: green; }
.asrflow-badge.retx { background: rgba(255,165,0,.18); color: #b26a00; }
.asrflow-deep { display: none; border: 1px solid var(--border-color-primary); border-radius: 8px;
  padding: 8px 12px; margin-top: 6px; font-size: 13px; background: var(--background-fill-primary); }
.asrflow-deep:target { display: block; }
.asrflow-deep code { font-size: 12px; }
.asrflow-deep ul { margin: 4px 0 4px 18px; padding: 0; }
.asrflow-x { float: right; text-decoration: none; font-size: 12px; }
</style>
"""


def model_info(model_id: str, lang: str = "it") -> str:
    """Scheda markdown del modello, con link alla fonte HF."""
    model = next(m for m in MODELS if m.id == model_id)
    lines = [
        f"**{model.name}** · [{T(lang, 'hf_page')}](https://huggingface.co/{model.id})",
        "",
        f"{T(lang, 'mi_license')}: {model.license} | {T(lang, 'mi_params')}: {model.params} | {T(lang, 'mi_size')}: {model.size}",
        f"{T(lang, 'mi_lang')}: {model.languages}",
        f"{T(lang, 'mi_stream')}: {model.streaming_type} — {model.streaming_detail}",
        f"{T(lang, 'mi_feat')}: {model.features}",
    ]
    if model.notes:
        lines.append(f"{T(lang, 'mi_notes')}: {model.notes}")
    return "\n".join(lines)


def flow_html(model_id: str, lang: str = "it") -> str:
    """Diagramma animato + pannelli dettaglio espandibili sotto (CSS :target, niente JS)."""
    spec = FLOW.get(model_id, {"mode": "re-tx"})
    steps = spec.get(lang, spec.get("it", []))
    deep = DEEP.get(model_id, {}).get(lang, DEEP.get(model_id, {}).get("it", []))
    mode = spec["mode"]
    nat = mode == "nativo"
    badge = ("<span class='asrflow-badge nat'>NATIVO</span>"
             if nat else "<span class='asrflow-badge retx'>RE-TX OGNI 2s</span>")
    if lang != "it":
        badge = ("<span class='asrflow-badge nat'>NATIVE</span>"
                 if nat else "<span class='asrflow-badge retx'>RE-TX EVERY 2s</span>")
    parts = [FLOW_CSS, badge, "<div class='asrflow' id='asrflow-top'>"]
    for i, (title, sub) in enumerate(steps):
        cls = "first" if i == 0 else ("last" if i == len(steps) - 1 else "")
        parts.append(f"<a class='asrflow-node {cls}' href='#asrflow-s{i}'><b>{i + 1}. {title}</b><span>{sub}</span></a>")
        if i < len(steps) - 1:
            link_cls = "loop" if (not nat and i == 1) else ""
            parts.append(f"<div class='asrflow-link {link_cls}'><span class='asrflow-dot'></span></div>")
    parts.append("</div>")
    parts.append(f"<div class='asrflow-hint'>{T(lang, 'deep_hint')}</div>")
    for i, (dtitle, bullets) in enumerate(deep):
        lis = "".join(f"<li>{b}</li>" for b in bullets)
        parts.append(f"<div class='asrflow-deep' id='asrflow-s{i}'><b>{dtitle}</b>"
                     f"<a class='asrflow-x' href='#asrflow-top'>{T(lang, 'deep_close')} ✕</a><ul>{lis}</ul></div>")
    parts.append(f"<div class='asrflow-note'>{FLOW_NOTE[mode][lang]}</div>")
    return "".join(parts)
