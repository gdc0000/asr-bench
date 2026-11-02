# Lo streaming nativo vero (cache incrementale) nessuno dei 4 lo usa in questa app

Report divulgativo: cosa succede tra il microfono e la parola sullo schermo, e perché i quattro modelli generano testi diversi.

## 1. Da zero: cosa significa trascrivere

Un microfono produce numeri: 16.000 campioni al secondo, ognuno l'altezza dell'onda sonora in un istante. Trascrivere significa trasformare questa sequenza lunghissima in una sequenza cortissima: le parole. In mezzo ci sono tre passaggi, uguali per tutti:

1. **Feature**: l'onda viene tagliata a finestrelle (25ms ogni 10ms) e ognuna diventa un vettore di numeri (tipicamente 80 valori log-mel: quanta energia c'è in 80 bande di frequenza). Da 16.000 numeri al secondo a 100 vettori al secondo. Tutti e 4 i modelli mangiano feature, non onda grezza.
2. **Rete neurale**: i vettori diventano ipotesi su cosa è stato detto.
3. **Decode**: le ipotesi diventano testo leggibile.

Le differenze tra i modelli stanno tutte nei punti 2 e 3.

## 2. Le tre famiglie: CTC, autoregressivo, Transducer

### CTC (Granite): un'etichetta per ogni istante

CTC = Connectionist Temporal Classification. L'idea: la rete guarda tutta la finestra audio e per **ogni istante** dice "secondo me qui si sente il fonema X". Output: una matrice `[tempo, vocabolario]`, i logits. Poi si prende, per ogni istante, l'etichetta più probabile (`argmax`) e si comprime: si cancellano le ripetizioni e un simbolo speciale "blank" (silenzio/indecisione). `hhh-eee-lll-lll-ooo` → `hello`.

Ipotesi forte di CTC: **ogni istante è indipendente dagli altri**. La rete non "ricorda" cosa ha scritto un secondo fa. Conseguenze:

- **Greedy** significa proprio questo: scelta golosa istante per istante, senza ripensamenti, senza ricerca. Veloce, ma niente modello di lingua a correggere ("aereo" vs "ereo" decide tutto il frame).
- Perché la punteggiatura è "nativa"? Perché il training include virgole e punti come etichette come le altre. Non c'è un post-processore: escono dalla stessa matrice.
- Cosa ritorna l'inferenza? Un tensore di logits `[1, T, V]`. Non sono parole, non sono nemmeno token: sono punteggi per frame. Il testo nasce dopo, con argmax + collasso.

### Autoregressivo (Whisper, Qwen): una parola tira l'altra

Qui la rete è uno scrittore: genera **un token alla volta, e ogni token dipende da tutti i precedenti**. Per scrivere "gatto" dopo "il", ha riletto "il". Questo è `generate()`: un ciclo che chiede alla rete "prossimo token?", lo accoda all'input, e ricomincia finché non arriva il token di stop o si esaurisce `max_new_tokens`.

Differenze enormi rispetto a CTC:

- C'è un **modello di lingua implicito**: la rete sa che dopo "il" viene un nome, e corregge acustiche ambigue con il contesto. Più accurato, più lento (N passi sequenziali, non parallelizzabili).
- Si può **forzare la lingua**: Whisper riceve token speciali `language="it"`, `task="transcribe"` che condizionano tutta la generazione. Qwen riceve un prompt in formato chat (`apply_chat_template`) con suffisso `language Italian<asr_text>`: l'audio viene proiettato nello spazio degli embedding di un LLM, e l'LLM "risponde" con la trascrizione.
- Cosa ritorna l'inferenza? Un tensore di **id token** `[1, N]`, già parole (o pezzi di parole BPE). Per Qwen, `N` include anche il prompt in input: va tagliato (`generated[:, len(prompt):]`) prima di decodificare, altrimenti ritrascrivi il prompt. Il decode è `batch_decode`: id → testo via vocabolario, con `skip_special_tokens`.

### Transducer / RNNT (Nemotron): nato per lo streaming

RNNT = Recurrent Neural Network Transducer. Tre pezzi: un **encoder** (ascolta l'audio), una **prediction network** (ricorda cosa ha scritto finora, come un mini modello di lingua) e una **joint network** (fonde i due e decide: emetti un token o vai avanti di un frame?). A differenza di CTC, ha memoria del testo; a differenza dell'autoregressivo puro, è **sincrono sui frame**: può emettere mentre l'audio ancora arriva, senza aspettare la fine. Per questo è "streaming-native by design".

Peculiarità ingegneristica vista nel codice: `model.generate()` non ritorna un tensore ma un **oggetto custom** (`NemotronAsrStreamingGenerateOutput`) con campi `.sequences` (gli id), `.durations`, `.attentions`. Una riga di codice in più rispetto agli altri: estrarre `.sequences` prima di `batch_decode`.

## 3. Cosa significa "streaming" davvero

Parola abusata. Scomponiamola:

- **Chunk**: pezzetti di audio che arrivano mentre parli (qui: 1 secondo).
- **Latenza di emissione**: quanto passa dal suono alla parola sullo schermo. Diversa da RTF (velocità media), che può essere ottima con latenza pessima.
- **Lookahead**: per capire un suono serve spesso sentire cosa viene dopo ("notte" vs "notte fonda" cambia poco, ma in generale il contesto destro aiuta). Aspettare contesto destro aggiunge latenza per definizione. Nemotron dichiara `num_lookahead_tokens`: lo paga esplicitamente.
- **Parziali vs finali**: uno streaming serio emette ipotesi instabili (parziali, che possono cambiare) e poi le fissa (finali, spesso a fine frase/VAD). Il lampeggiamento del testo che si riscrive si chiama **flicker**: è normale, si misura (noi contiamo le revisioni).

## 4. I due modi di fare streaming: ricalcolo vs cache

**Modo ingenuo (re-transcription), quello di questa app**: a ogni giro si butta tutto il buffer nel modello e si ritrascrive da zero. Senza stato, senza memoria tra un giro e l'altro. Costo O(n²) sul buffer, flicker garantito, ma funziona con **qualsiasi** modello, anche quelli mai pensati per lo streaming. È il motivo per cui Granite, Whisper e Qwen qui "streammano": non è merito loro, è il wrapper che li interroga in continuazione.

**Modo nativo (cache incrementale)**: il modello tiene lo stato. A un nuovo chunk, riusa i calcoli dei chunk precedenti (cache dell'encoder, KV-cache del decoder) e produce solo il delta. Costo ammortizzato O(n), latenza bassa, flicker contenuto. Richiede architettura e API pensate per questo: encoder causale (non guarda il futuro), cache esplicita, emit parziali. Nemotron ce l'ha nel DNA (RNNT + cache dei chunk), Qwen la dichiara solo via server vLLM, Whisper e Granite non ce l'hanno proprio.

## 5. Cosa fa esattamente questa app, giro per giro

1. Il browser invia chunk da ~1s (16kHz, mono o stereo, float o int16: `to_mono_16k` uniforma tutto).
2. I campioni si accumulano in un buffer (finestra 30s, poi scorre).
3. La VAD energetica (RMS > 0.02, scala auto) segna il primo suono: da lì parte la latenza.
4. Ogni 2s di audio nuovo: normalizzazione di picco a 0.95, `run_transcribe` sull'**intero buffer**, testo aggiornato, revisioni contate.
5. A Live OFF: trascrizione finale + WER/CER contro la frase attesa + RTF + riga in tabella.

Tutto il resto (stati modello, log, export) è osservabilità intorno a questo ciclo.

## 6. Perché i 4 generano testi diversi sulla stessa frase

- **Granite** decide frame per frame senza memoria: velocissimo, ma se un suono è ambiguo non ha un modello di lingua che lo salvi. Punteggiatura gratis dal training.
- **Whisper** rilegge tutto a ogni token con 244M di contesto linguistico in 99 lingue: robusto, ma ogni giro di re-transcription costa un `generate` completo.
- **Qwen** è un LLM che "legge" l'audio: il più versatile (52 lingue, LID), ma paga prompt + 0.6B di autoregressione a ogni giro, e il primo load rimappa il checkpoint.
- **Nemotron** emette durante l'audio senza ricalcolare tutto: encoder con cache (4 frame per chunk, lookahead 3), decoder LSTM con stato portato avanti, emit commitati quasi senza flicker. L'italiano non è dichiarato supportato, ma sul campo trascrive.

## 7. Diario di bordo: come Nemotron è passato al nativo

Tutto verificato sul campo, niente teoria.

**Punto di partenza.** Il `generate` di transformers accetta `input_features` come *generatore* di chunk mel solo per Nemotron (`NemotronAsrStreamingGenerationMixin`). Ma: niente parziali (ritorna solo il testo finale) e niente `streamer`. Per i parziali live serviva replicare il loop greedy a mano.

**Anatomia del chunk.** Lookahead default `R=3`, subsampling `8`: primo chunk 25 frame mel (`1+8R`), poi 32 (`8(R+1)`), ~0.25s/0.32s di audio. Mel time-major `[B,T,128]` (sbagliare asse = primo bug trovato). Soglia di alimentazione in app: 0.3s di audio nuovo.

**Cache portate avanti.** Encoder: `past_key_values` + `padding_cache` (conv causali). Decoder: `Nemotron3_5AsrRNNTDecoderCache` (stati LSTM hidden/cell). Prompt lingua una sola volta sul primo chunk (`prompt_ids=101`, auto). Dtype fp16, `torch.no_grad()` ovunque.

**Il bug che corrompeva tutto.** Il decoder va alimentato con l'ultimo token emesso, *blank inclusi* (il blank è no-op: early return, stato preservato). La prima versione riusava l'ultimo non-blank: rilanciava l'LSTM a ogni frame e il testo divergeva dopo poche parole. Trovato confrontando traccia per traccia (input, hash output) contro il `generate` ufficiale.

**Validazione.** Stesso audio, tre path: `generate` batch, `generate` streaming ufficiale, loop manuale. Loop == ufficiale **bit-identico** (stessi id token). Encoder verificato bit-identico chunk per chunk via hash dei pooler. Entrambi i path deterministici run-to-run.

**Il tag fantasma.** A volte compare `<en-US>` nel testo, a volte no, a parità di input: argmax sul filo del rasoio, flip da nondeterminismo numerico dei kernel. È eco del prompt lingua, non parlato: rimosso via regex in entrambi i path (`clean_text`).

**Numeri veri** (librispeech 5.9s, RTX 3080): streaming ufficiale 0.78s, loop manuale 1.37s (overhead Python per-frame), batch 1.24s. In app: parziali monotoni dal primo chunk, revisioni ~6 (solo ri-tokenizzazioni BPE di coda), RTF 0.34, prima frase ~1s (limite caduta chunk Gradio, non modello).

**Morale aggiornata.** Lo streaming nativo vero in questa app lo usa Nemotron: O(n), commit incrementali, niente ricalcolo. Gli altri tre restano in re-transcription O(n²) con flicker: stesso wrapper, metriche comparabili, costi diversi — ed è esattamente quello che la tabella mostra.
