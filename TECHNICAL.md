# asr-bench — README tecnico

Oggetti, metodi, flusso dati. Prova live singola + tabella sessione.

## 1. Tipi base

- `np.ndarray float32`: audio mono 16k. Da browser: scala -1..1; da chunk int16: scala ±32768 (gestita in VAD).
- `BatchFeature`: dict di tensori. Output dei Processor. Chiavi variabili per modello. Metodo chiave: `.to(device)`.
- `torch.Tensor long`: id token. `torch.Tensor float`: feature audio, logits.
- `ModelInfo`: dataclass in `adapters/base.py`. Campi: id, nome, licenza, lingue, params, size, streaming_type, streaming_detail, features, notes.

## 2. Adapter: interfaccia comune

4 file in `asrbench/adapters/`, tutti registrati in `ADAPTERS` (`app.py`): granite, qwen, whisper, nemotron.

Classe con stessi metodi:

- `__init__(model_id)`: legge `MODELS` da `registry.py`, salva `info`, `model_id`.
- `load() -> None`: scarica Processor + Model da HF, sposta su CUDA, `eval()`.
- `transcribe(audio: np.ndarray, sr=16000) -> str`: Processor -> `generate` / forward -> `batch_decode` -> `strip()`.
- `unload() -> None`: `None` + `torch.cuda.empty_cache()`.

`app.py` tiene un solo modello in memoria: dict `CURRENT = {adapter, id}`. Cambio modello = `unload()` + `load()`. Nomi corti UI in `SHORT` (Granite, Qwen3, Whisper, Nemotron).

## 3. Audio, punteggi, VAD

`asrbench/engine/audio.py`:

- `to_mono_16k(audio, normalize=True) -> (wav, 16000)`: accetta `(sr, np.ndarray)` da Gradio o array puro. Media canali, cast float32, `librosa.resample` se serve, normalizza picco a 0.95 se richiesto.

`asrbench/engine/scores.py` (solo stdlib):

- `norm_it(s)`: minuscole, NFKD senza accenti, apostrofi a spazi, solo `a-z0-9`.
- `wer(ref, hyp)`, `cer(ref, hyp)` (CER su stringa senza spazi): Levenshtein / len(ref). `None` se ref vuoto.
- `fmt(v)`: `12.3%` o `—`.

`app.py`:

- `chunk_is_speech(wav) -> bool`: RMS con scala auto (peak > 1.5 = int16, /32768), soglia `VAD_RMS = 0.02`.

## 4. Modelli: oggetti e metodi invocati (i 4 attivi)

Pattern comune: `load()` = `Processor.from_pretrained` + `Model.from_pretrained(device_map="auto")` + `eval()`; `transcribe()` = processor -> sposta tensori su device (+fp16) -> inferenza -> `batch_decode` -> `strip`/`clean`.

### Granite `granite.py`

- Oggetti: `AutoProcessor`, `AutoModelForCTC` (fp16).
- `processor(audio, sampling_rate, return_tensors="pt")` -> `input_features` + `attention_mask` (spostati su device così come sono).
- `model(**inputs).logits` shape `[1, T, vocab]`. Decode CTC: `torch.argmax(logits, dim=-1)` -> `processor.batch_decode(ids)[0]`.
- Nessun `generate`. Solo forward greedy.
- Path app: load via `_load_new_adapter`, inferenza via `run_transcribe`.

### Whisper `whisper.py`

- Oggetti: `WhisperProcessor`, `WhisperForConditionalGeneration` (fp16).
- `processor(audio, ...)` -> log-mel `input_features` (solo feature passate a `generate`, move esplicito a fp16).
- `model.generate(input_features, language="it", task="transcribe", max_new_tokens=256)` -> id `[1, N]` (taglio prompt non serve).
- `processor.batch_decode(generated, skip_special_tokens=True)[0]`.
- Path app: load via `_load_new_adapter`, inferenza via `run_transcribe`.

### Qwen3-ASR `qwen.py`

- Oggetti: `Qwen3ASRProcessor`, `Qwen3ASRForConditionalGeneration`, `Qwen3ASRConfig` (fp16, `.to("cuda")` esplicito: pretende GPU).
- `load()` ricostruisce config da `thinker_config` e rimappa lo state dict: `thinker.audio_tower.*` -> `model.audio_tower.*`, `thinker.model.*` -> `model.language_model.*`, `proj1/proj2` -> `multi_modal_projector.linear_1/2`, `thinker.lm_head` -> `lm_head`.
- Prompt obbligatorio: `processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)` + suffisso `"language Italian<asr_text>"`.
- `processor(text=text_prompt, audio=audio, sampling_rate, return_tensors)` -> `input_ids, input_features, attention_mask` (spostati su device).
- `model.generate(**inputs, max_new_tokens=256)` -> taglio prompt `generated[:, input_ids.shape[1]:]` -> `batch_decode(..., skip_special_tokens=True)[0]`.
- Path app: load via `_load_new_adapter`, inferenza via `run_transcribe`.

### Nemotron `nemotron.py`

- Oggetti: `Nemotron3_5AsrProcessor`, `Nemotron3_5AsrForRNNT` (fp16, `device_map="auto"`), `Nemotron3_5AsrRNNTDecoderCache(config)`.
- `processor(audio, sampling_rate, return_tensors="pt")` -> mel time-major `input_features [B, T, 128]` + `attention_mask` + `num_lookahead_tokens (=3)` + `prompt_ids` (tensori spostati su device, float in fp16).
- Batch: `model.generate(**inputs, max_new_tokens=256)` -> `NemotronAsrStreamingGenerateOutput` (non tensore: `sequences = out.sequences if hasattr(out, "sequences") else out`).
- `processor.batch_decode(sequences, skip_special_tokens=True)[0]` + `clean_text` (regex via eco prompt `<xx>`).
- **Streaming nativo** via `NemotronStreamSession`:
  - `stream_start()`: reset mel consumate, cache `None`, decoder cache fresca, `last = blank`.
  - `stream_feed(raw_np)`: append raw, mel dal buffer intero, pezzi 25 poi 32 (`fed_mel` avanza solo a pezzo completo), per pezzo `model.get_audio_features(input_features, past_key_values, padding_cache, num_lookahead_tokens=3, prompt_ids[solo primo], use_cache=True)` -> `pooler_output [B, Te, 640]`; loop greedy per frame: `decoder(last, cache)` -> `joint(dec [1,1,1,H], enc [1,1,1,H])` -> `argmax` -> emit se != blank (max 10 simboli/frame), `last` = token emesso **blank inclusi**; ritorna `batch_decode(emitted)`.
  - `stream_finish()`: pad zero-filling dell'ultimo pezzo parziale, decode, ritorna testo. Fallback a `transcribe()` se eccezione.
- Path app: load via `_load_new_adapter`; live via `nemotron_feed`/`nemotron_finish` (decorati `@spaces.GPU(duration=60)`); batch/OFF altri modelli via `run_transcribe`.

## 5. App Gradio (`app.py` + `asrbench/ui/`)

`app.py` è solo entry point (init + `build_demo().launch()`). Codice diviso per responsabilità:

- `asrbench/core/`: `state.py` (`ADAPTERS`, costanti, stato mutabile: mutazioni via `state.X`), `logbuffer.py`, `registry.py`.
- `asrbench/engine/`: `helpers.py`, `service.py` (scope-load a boot in `MODELS_LIVE` via `_preload_all()`; `get_adapter` lookup zero-quota; `preload` seleziona; `unload_model` deseleziona; `run_transcribe(model_id, audio, sr)` GPU con solo tipi picklabili; `nemotron_feed/finish` solo locali), `trials.py` (`new_trial`, `table_view`, `last_card_view`, `write_session_files`, `reset_session`, `live_toggle` + `st` in output, `live_transcribe`; nativo solo se `SPACES_ZERO_GPU != "1"`), `audio.py`, `scores.py`.
- `asrbench/ui/`: `status.py` (badge, `status_html(lang)`, `live_btn_state(lang)`, `model_status_init`), `layout.py` (CSS/JS, `on_select`, `set_verbose`, `ui_clear`, `set_lang`, `build_demo`), `i18n.py` (stringhe IT/EN, `flow_html`, `model_info`, deep-dive).

Layout (dashboard, niente tab):

- Header: titolo + bottone lingua `EN/IT` + bottone `Pannello` (js-only, clicca il toggle nativo della Sidebar).
- Guida: riga fissa in 3 passi (cosa sta facendo l'utente).
- `gr.Sidebar` destra (`asr-sidebar`): tabella 3 colonne (Modello corto, Stato+sub, Uso), log auto 2s, verbose, bottoni.
- Riga setup: GT + dropdown + Attiva/Scarica + status | accordion dettagli + KPI (Lat, RTF, WER).
- Riga prova: mic streaming + Live ON/OFF | trascrizione + stato.
- Riga sessione: Dataframe metriche + card ultima prova | CSV/JSONL/Azzera.

Funzioni:

- `on_select(model_id, lang)`: aggiorna `SELECTED`, ritorna info + diagramma + sidebar + stato bottone Live.
- `preload(model_id, lang)` (semplice, zero quota): seleziona il modello scope-loaded, oppure errore scope.
- `unload_model(lang)`: deseleziona (i pesi scope restano condivisi). Sidebar + bottone aggiornati.
- `_preload_all()`: scope-load a boot di tutti gli adapter (`.to("cuda")` a module-scope = pack ZeroGPU). Errori isolati per modello in `LOAD_ERROR`.
- `run_transcribe(model_id, audio, sr)` (decorato `@spaces.GPU(duration=30)` → ~45s richiesti): lookup del globale scope-loaded nel worker, ritorna `str`. Mai modelli negli arg/return (i tensori CUDA non attraversano il fork).
- `live_toggle(model_id, gt, st, lang)`: un bottone per ON e OFF + `st` in output (11). ON: rifiuta se modello non pronto, altrimenti apre trial (`t_on`, snapshot GT, lock GT+dropdown). Nativo solo locale (`SPACES_ZERO_GPU != "1"`); su Space sempre re-tx. OFF: scarta se <1s, altrimenti trascrizione finale + WER/CER + riga `SESSION` + file + card/KPI + unlock.
- `live_transcribe(model_id, chunk, st, lang)` (generatore): inattivo -> messaggio idle. Attivo: accumula buffer (finestra 30s), VAD per `t_speech`. Se trial nativo (solo locale, Nemotron): feed ogni 0.3s di audio nuovo via `nemotron_feed`, parziali committed, niente ritrascrizione. Altrimenti ritrascrive ogni 2s di audio nuovo. Traccia `t_first`, `infer_s`, `revs` in entrambi i casi.
- `nemotron_feed(adapter, raw)` / `nemotron_finish(adapter)` (decorati `@spaces.GPU(duration=45)`, solo locali: su Space alzano subito e scatta il fallback re-tx).
- `TRIAL_ACTIVE` in `state`: `preload`/`unload_model` rifiutano con warning a prova aperta.
- `set_lang(lang)`: flip IT<->EN, 33 update UI. `lang_state` in ogni callback che emette testo.
- `table_view()`, `last_card_view(lang)`, `reset_session(lang)`, `write_session_files()` (CSV+JSONL per run in `logs/`).

## 6. Trial e metriche

Dict trial: `active, n, mid, gt, samples, last_len, last_text, last_t, t_on, t_first, t_speech, infer_s, revs`.

- Lat = `t_first - (t_speech or t_on)`, clamp >= 0. `None` se nessun testo.
- RTF = `infer_s / audio_s`. WER/CER via `scores.py` solo se GT non vuota.
- Riga sessione: prova, ora, modello, audio_s, gt, testo, wer, cer, lat, infer, rtf, revs.

## 7. ZeroGPU / Space

- `import spaces` con fallback no-op in locale. Decorati: `_load_new_adapter` (180s), `run_transcribe` (60s), `nemotron_feed`/`nemotron_finish` (60s). Resto su CPU.
- `demo.launch()` senza host/porta (bind 0.0.0.0:7860 automatico).
- `gated_hint()`: 401/403/gated/token -> "serve HF_TOKEN con accesso approvato al repo".
- Frontmatter Space in `README.md`. `.gitignore`: `logs/`, cache, check script.
