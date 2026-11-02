---
title: asr-bench-it
emoji: 🎙️
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: "3.10"
pinned: false
short_description: "Live Italian ASR bench: WER, latency, RTF."
---

# asr-bench

Banco di prova per ASR italiano in live streaming. Una prova alla volta: scegli il modello, leggi una frase nel microfono, leggi WER, latenza e RTF. Stessa frase su più modelli = confronto equo.

> Come funziona lo streaming qui dentro? Leggi [`STREAMING.md`](STREAMING.md): da zero al perché nessun modello usa la cache incrementale.

## Avvio locale

```bash
pip install -r requirements.txt
python app.py
```

Aprire http://127.0.0.1:7860 con Chrome o Edge (il microfono vuole HTTPS o localhost).

## Flusso di una prova

1. **Frase**: scrivi quello che stai per dire. Senza frase, niente WER. Si blocca durante la prova.
2. **Modello**: seleziona e premi Attiva. Il Live parte solo a modello pronto.
3. **Live ON**: premi il microfono, premi Live ON, leggi la frase, premi Live OFF.
4. **Tabella**: riga con testo, WER/CER, latenza dal primo suono, RTF, durata.
5. Ripeti con un altro modello e la stessa frase. Export CSV/JSONL/log.

La latenza parte dal primo suono rilevato (VAD energetica), non dal click: il silenzio iniziale non conta.

## Metriche

- **WER/CER**: errore su parole/caratteri contro la frase scritta. Normalizzazione italiana: minuscole, via accenti e punteggiatura. `—` se non hai scritto la frase.
- **Lat**: secondi dal primo suono al primo testo non vuoto. Include soglia chunk (2s) + inferenza.
- **RTF**: tempo di inferenza totale / durata audio. Sotto 1 = più veloce del reale, misurato sulla tua GPU.
- **Audio**: durata registrata. Prove sotto 1s scartate.

## Modelli a confronto

|  | Granite 5.0 TurboCTC | Qwen3 ASR 0.6B | Whisper Small | Nemotron 3.5 Streaming |
|---|---|---|---|---|
| **Licenza** | Apache 2.0 | Apache 2.0 | MIT | Apache 2.0 |
| **Parametri** | 470M | 0.6B | 244M | 600M |
| **Peso download** | ~1GB | ~1.2GB | ~460MB | ~1.2GB |
| **Italiano** | Sì (7 lingue) | Sì (52 lingue) | Sì (99 lingue) | Non dichiarato (EN) |
| **Streaming live** | Pseudo (re-tx ogni 2s) | Pseudo in locale (nativo solo via vLLM) | Pseudo (re-tx ogni 2s) | **Nativo qui (cache incrementale, chunk 25/32 mel)** |
| **Punteggiatura** | Nativa (CTC) | — | — | Nativa + maiuscole |
| **Accesso pesi** | Aperto | Aperto | Aperto | Gated (serve HF_TOKEN) |
| **Primo load** | Veloce | Lento (rimappa pesi `thinker.*`) | Veloce | Medio |
| **Profilo** | Il più veloce, greedy CTC | Il più versatile (LID, 52 lingue) | Il più leggero, baseline | L'unico nativo in gara |

Legenda streaming: **pseudo** = accumulo chunk + ritrascrizione integrale ogni 2s (costo O(n²), testo che si riscrive). **Nativo** = encoder con cache, emissione incrementale. Nemotron dichiara chunk 80/160/560/1120ms con cache dei precedenti.

## Come funziona il live (ingegneria)

Stesso percorso per tutti in questa app: chunk da 1s → buffer 30s → VAD → ritrascrizione integrale ogni 2s di audio nuovo, con normalizzazione di picco a 0.95 prima di ogni inferenza. Cambia cosa c'è sotto:

|  | Granite TurboCTC | Qwen3 ASR 0.6B | Whisper Small | Nemotron 3.5 |
|---|---|---|---|---|
| **Classe modello** | `AutoModelForCTC` | `Qwen3ASRForConditionalGeneration` | `WhisperForConditionalGeneration` | `Nemotron3_5AsrForRNNT` |
| **Classe processor** | `AutoProcessor` | `Qwen3ASRProcessor` | `WhisperProcessor` | `Nemotron3_5AsrProcessor` |
| **Input al modello** | `input_features` + `attention_mask` | `input_ids` + `input_features` + `attention_mask` | log-mel `input_features` [1, 80, 3000] | `input_features` + `attention_mask` + `num_lookahead_tokens` + `prompt_ids` |
| **Metodo inferenza** | Forward + `argmax` frame-wise (greedy, no `generate`) | `generate`, taglio prompt in output | `generate` con `language="it"`, `task="transcribe"` | `generate`, ritorna oggetto (non tensore) |
| **Limite output** | = n frame (uno per frame) | `max_new_tokens=256` | `max_new_tokens=256` | `max_new_tokens=256` |
| **Forcing lingua** | No (è nel training CTC) | Prompt chat + suffisso `language Italian<asr_text>` | Parametro `language="it"` | `prompt_ids` del processor |
| **Decode** | `batch_decode` su id frame | `batch_decode` su token dopo il prompt | `batch_decode` con `skip_special_tokens` | `batch_decode` su `.sequences` |
| **Remap checkpoint** | No | Sì (`thinker.*` → `model.*`, vedi `qwen.py`) | No | No (ma estrazione `.sequences`) |
| **Cache streaming nativa** | Non dichiarata | Finestra attenzione dinamica (solo via vLLM) | Non dichiarata | Chunk + cache precedenti (non sfruttata qui) |
| **Costo live qui** | O(n²), forward leggero | O(n²), generate pesante | O(n²), generate medio | O(n²), generate RNNT |

Morale operativa: il collo di bottiglia del live non è il microfono, è la ritrascrizione integrale. Con 30s di buffer, ogni 2s si ricalcola tutto da zero: i modelli con `generate` autoregressivo (Qwen, Whisper, Nemotron) costano più del CTC greedy di Granite a parità di audio. Lo streaming nativo vero (cache incrementale) nessuno dei 4 lo usa in questa app.

## Note operative

- **Un modello attivo alla volta** (VRAM). Cambiare modello = unload + load automatici.
- **Nemotron + italiano**: il repo dichiara inglese. In prova spesso trascrive l'italiano, ma senza garanzia: è la variabile più interessante della tabella.
- **Qwen al primo avvio**: la conversione checkpoint impiega minuti. Dalla seconda volta è veloce (cache disco).
- **Microfono**: se il browser non lo vede, ricarica la pagina e concedi il permesso. Il Live ON senza audio in ingresso resta in `in attesa voce...`.
- **Log**: ogni azione UI scrive in `logs/` (server + CSV + JSONL delle prove). Se qualcosa non torna, il log dice quale click ha prodotto quale backend.

## Spazio Hugging Face

App pronta per ZeroGPU: `spaces.GPU` su load (180s) e inferenza (60s), `launch()` nudo, frontmatter in questo README.

```bash
git add app.py asrbench requirements.txt README.md .gitignore
git commit -m "asr-bench it"
git remote add origin https://huggingface.co/spaces/TUOUSER/asr-bench-it
git push -u origin main
```

Poi Settings → Hardware ZeroGPU, Secrets `HF_TOKEN` (con accesso Nemotron approvato). Primo avvio: build pip + download pesi alla prima Attiva.
