# asr-bench — README tecnico

Oggetti, metodi, flusso dati. Prova live singola + tabella sessione.

## 1. Tipi base

- `np.ndarray float32`: audio mono 16k. Da browser: scala -1..1; da chunk int16: scala ±32768 (gestita in VAD).
- `BatchFeature`: dict di tensori. Output dei Processor. Chiavi variabili per modello. Metodo chiave: `.to(device)`.
- `torch.Tensor long`: id token. `torch.Tensor float`: feature audio, logits.
- `ModelInfo`: dataclass in `adapters/base.py`. Campi: id, nome, licenza, lingue, params, size, streaming_type, streaming_detail, features, notes.

## 2. Adapter: interfaccia comune

7 file in `asrbench/adapters/`, ma `ADAPTERS` in `app.py` ne registra 4: granite, qwen, whisper, nemotron. Voxtral/moonshine/vibevoice presenti ma esclusi.

Classe con stessi metodi:

- `__init__(model_id)`: legge `MODELS` da `registry.py`, salva `info`, `model_id`.
- `load() -> None`: scarica Processor + Model da HF, sposta su CUDA, `eval()`.
- `transcribe(audio: np.ndarray, sr=16000) -> str`: Processor -> `generate` / forward -> `batch_decode` -> `strip()`.
- `unload() -> None`: `None` + `torch.cuda.empty_cache()`.

`app.py` tiene un solo modello in memoria: dict `CURRENT = {adapter, id}`. Cambio modello = `unload()` + `load()`. Nomi corti UI in `SHORT` (Granite, Qwen3, Whisper, Nemotron).

## 3. Audio, punteggi, VAD

`asrbench/audio.py`:

- `to_mono_16k(audio, normalize=True) -> (wav, 16000)`: accetta `(sr, np.ndarray)` da Gradio o array puro. Media canali, cast float32, `librosa.resample` se serve, normalizza picco a 0.95 se richiesto.
- `duration(wav, sr) -> float`: `len / sr`.

`asrbench/scores.py` (solo stdlib):

- `norm_it(s)`: minuscole, NFKD senza accenti, apostrofi a spazi, solo `a-z0-9`.
- `wer(ref, hyp)`, `cer(ref, hyp)` (CER su stringa senza spazi): Levenshtein / len(ref). `None` se ref vuoto.
- `fmt(v)`: `12.3%` o `—`.

`app.py`:

- `chunk_is_speech(wav) -> bool`: RMS con scala auto (peak > 1.5 = int16, /32768), soglia `VAD_RMS = 0.02`.

## 4. Processor per modello (i 4 attivi)

### Granite `granite.py`

- Oggetti: `AutoProcessor`, `AutoModelForCTC`.
- Forward: `model(**inputs).logits` shape `[1, T, vocab]`. Decode CTC: `argmax` frame-wise -> `batch_decode`.
- Nessun `generate`. Greedy frame-wise.

### Whisper `whisper.py`

- Oggetti: `WhisperProcessor`, `WhisperForConditionalGeneration`.
- `processor(audio, ...)` -> log-mel `[1, 80, 3000]`.
- `model.generate(input_features, language="it", task="transcribe", max_new_tokens=256)` -> taglio non serve (solo feature in input).
- `batch_decode(..., skip_special_tokens=True)[0]`.

### Qwen3-ASR `qwen.py`

- Checkpoint con chiavi `thinker.*`. `load()` ricostruisce config da `thinker_config`, rimappa stato: `thinker.audio_tower.*` -> `model.audio_tower.*`, `thinker.model.*` -> `model.language_model.*`, `proj1/proj2` -> `multi_modal_projector.linear_1/2`, `thinker.lm_head` -> `lm_head`.
- Prompt obbligatorio: `apply_chat_template(...)` + suffisso `"language Italian<asr_text>"`.
- `generate(..., max_new_tokens=256)` -> taglio prompt `generated[:, input_ids.shape[1]:]` -> `batch_decode`.

### Nemotron `nemotron.py`

- Oggetti: `Nemotron3_5AsrProcessor`, `Nemotron3_5AsrForRNNT`.
- `model.generate(...)` -> `NemotronAsrStreamingGenerateOutput`, non tensore: `sequences = out.sequences if hasattr(out, "sequences") else out`.
- `batch_decode(sequences, ...)[0]`. Gated su HF: serve token con accesso approvato.

## 5. App Gradio `app.py`

Layout (dashboard, niente tab):

- Header: titolo + bottone `Pannello` (js-only, clicca il toggle nativo della Sidebar).
- `gr.Sidebar` destra (`asr-sidebar`): tabella 3 colonne (Modello corto, Stato+sub, Uso), log auto 2s, verbose, bottoni.
- Riga setup: GT + dropdown + Attiva/Scarica + status | accordion dettagli + KPI (Lat, RTF, WER).
- Riga prova: mic streaming + Live ON/OFF | trascrizione + stato.
- Riga sessione: Dataframe metriche + card ultima prova | CSV/JSONL/Azzera.

Funzioni:

- `on_select(model_id)`: aggiorna `SELECTED`, ritorna info + sidebar + stato bottone Live.
- `preload(model_id)` (generatore): yield "caricamento" + sidebar, poi `get_adapter`, poi "pronto".
- `unload_model()`: scarica attivo, sidebar + bottone aggiornati.
- `get_adapter(model_id)`: cache-hit diretto; miss -> `_load_new_adapter` (decorato `@spaces.GPU(duration=180)`).
- `run_transcribe(adapter, audio, sr)` (decorato `@spaces.GPU(duration=60)`).
- `live_toggle(model_id, gt, st)`: un bottone per ON e OFF. ON: rifiuta se modello non pronto, altrimenti apre trial (`t_on`, snapshot GT, lock GT+dropdown). OFF: scarta se <1s, altrimenti trascrizione finale + WER/CER + riga `SESSION` + file + card/KPI + unlock.
- `live_transcribe(model_id, chunk, st)` (generatore): inattivo -> messaggio idle. Attivo: accumula buffer (finestra 30s), VAD per `t_speech`, ritrascrive ogni 2s di audio nuovo, traccia `t_first`, `infer_s`, `revs`.
- `table_view()`, `last_card_view()`, `reset_session()`, `write_session_files()` (CSV+JSONL per run in `logs/`).

## 6. Trial e metriche

Dict trial: `active, n, mid, gt, samples, last_len, last_text, last_t, t_on, t_first, t_speech, infer_s, revs`.

- Lat = `t_first - (t_speech or t_on)`, clamp >= 0. `None` se nessun testo.
- RTF = `infer_s / audio_s`. WER/CER via `scores.py` solo se GT non vuota.
- Riga sessione: prova, ora, modello, audio_s, gt, testo, wer, cer, lat, infer, rtf, revs.

## 7. ZeroGPU / Space

- `import spaces` con fallback no-op in locale. Solo `_load_new_adapter` e `run_transcribe` decorati; resto su CPU.
- `demo.launch()` senza host/porta (bind 0.0.0.0:7860 automatico).
- `gated_hint()`: 401/403/gated/token -> "serve HF_TOKEN con accesso approvato al repo".
- Frontmatter Space in `README.md`. `.gitignore`: `logs/`, cache, check script.
