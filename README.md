---
title: asr-bench-it
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: "3.10"
pinned: false
---

# asr-bench

Prova microfono live, una prova alla volta. Confronto italiano fra 4 modelli ASR.

## Avvio

```bash
python app.py
```

Aprire http://127.0.0.1:7860 con Chrome o Edge.

## Flusso prova

1. Scrivi la frase da pronunciare (opzionale, ground truth).
2. Seleziona modello, premi Attiva.
3. Premi Live ON, parla, premi Live OFF.
4. Riga in tabella sessione: testo, WER/CER se c'era GT, latenza primo testo, RTF.
5. Ripeti per ogni modello con la stessa frase. Export CSV/JSONL/log.

Il Live parte solo a modello attivo. GT e modello si bloccano durante la prova.

## Metriche

- **Lat**: secondi dal primo suono rilevato (VAD energetica) al primo testo. Silenzio iniziale escluso.
- **RTF**: tempo di inferenza totale / durata audio. Sotto 1 = più veloce del reale.
- **WER/CER**: errore parole/caratteri vs GT, normalizzazione italiana (minuscole, via accenti/punteggiatura). `—` senza GT.

## Spazio Hugging Face

App pronta per ZeroGPU: `spaces.GPU` su load (180s) e inferenza (60s), `launch()` nudo, frontmatter in questo README.

```bash
git add app.py asrbench requirements.txt README.md .gitignore
git commit -m "asr-bench it"
git remote add origin https://huggingface.co/spaces/TUOUSER/asr-bench-it
git push -u origin main
```

Poi Settings → Hardware ZeroGPU, Secrets `HF_TOKEN` (serve per Nemotron gated).

## Modelli in gara (4)

| Modello | Licenza | Parametri | Dimensione | Streaming | Caratteristiche |
|---|---|---|---|---|---|
| Granite Speech 5.0 470M TurboCTC | Apache 2.0 | 470M | ~1GB | Pseudo | CTC, punteggiatura nativa, veloce |
| Qwen3 ASR 0.6B | Apache 2.0 | 0.6B | ~1.2GB | Pseudo (locale) | LID, allineatore opzionale |
| Whisper Small | MIT | 244M | ~460MB | Pseudo | Baseline robusto |
| Nemotron 3.5 ASR Streaming | Apache 2.0 | 600M | ~1.2GB | Nativo cache-aware | Chunk 80/160/560/1120ms, punteggiatura |

## Note

- GPU fp16/bf16. Un modello attivo alla volta.
- Pseudo-streaming: accumulo chunk e ritrascrizione ogni 2s.
- Nemotron gated: serve HF_TOKEN per il download.
- Log di sessione in `logs/` (server + CSV + JSONL prove).
