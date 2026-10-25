# Pronunciation Scoring
https://github.com/user-attachments/assets/bde420f7-0a17-4dc9-acc5-4711c8137fe6

A phoneme-level mispronunciation scoring pipeline: audio + target sentence
in, a 0-2 score per phoneme + below-threshold flags + an articulatory
coaching tip for the worst phone out. No fine-tuned models, no LLM anywhere
in the pipeline, see `REPORT.md` for the full evaluation and the
reasoning behind that constraint.

**v2** (`REPORT_v2.md`) keeps both constraints and raises test correlation
from r = 0.43 to **r = 0.75**. It also ships a local app: record yourself in
the browser and get per-sound feedback, all running on your own machine.









## Run the app locally

```
run_app.bat            # Windows: double-click, or run from a terminal
./run_app.sh           # macOS / Linux
python server.py       # same thing, any OS (--port N, --no-browser)
```

This opens http://127.0.0.1:7860. Type or pick a sentence, press the red
button (or Space), and read it aloud. You get:

- a score for every sound, on a continuous colour ramp (deliberately not
  red/green and not pass/fail)
- the sounds "worth another take", plus which sound the model heard instead
- a hand-written articulation tip for each sound
- word-by-word playback of your own recording, and a reference
  pronunciation from your system's text-to-speech voice
- a per-browser history of your attempts

No microphone? The example recordings work too. Audio never leaves the
machine. The first run downloads the frozen wav2vec2 model (~1.2 GB) from
Hugging Face. The trained v2 head ships in `models/head_v2.joblib` (4 MB),
so no training is needed to run the app.

## Headline numbers

**v2** (shipped in the app; speaker-disjoint test split, see `REPORT_v2.md`):

| | v1 test subset (500 utts) | full test (2,500 utts / 47k phones) |
|---|---|---|
| Pearson r | 0.431 → **0.719** | **0.753** |
| Error-detection AP | 0.439 → **0.622** | **0.642** |
| Flagging (precision / recall) | 0.63 / 0.10 → | **0.69 / 0.40** |

Throughput is 5.9× real time on CPU (2.1× on a single thread).

**v1** (speaker-disjoint test set, 500 utterances / 9,698 phones, from
speechocean762 -- see `REPORT.md` for full methodology)

- **Pearson r = 0.431** (GBM head) vs. **r = 0.378** for a naive
  mean-log-posterior-only baseline -- the trained head adds real signal.
- **AP = 0.439** for phone-error detection (human_score < 2.0), ~2.4x
  random baseline (18.2% positive rate).
- **3.90x real-time** CPU inference throughput (single core, no batching).
- Model substitution required: the originally-specified
  `charsiu/en_w2v2_fc_10ms` has no usable output label set (see
  `REPORT.md` Section 2) -- this pipeline runs on
  `facebook/wav2vec2-lv-60-espeak-cv-ft` instead.

## What's here

```
src/
  data.py          speechocean762 loading
  phone_mapping.py ARPAbet <-> IPA mapping (empirically built, see REPORT.md 2.1)
  acoustic.py       frozen model, frame x phone log-posteriors
  align.py          CTC forced alignment (see REPORT.md 2.2)
  features.py       per-phone GOP features + batch extraction/caching
  train_head.py     ridge/GBM head, metrics, figures
  tips.py           hand-written articulatory coaching tips (v2: every phone + example words)
  text.py           sentence -> ARPAbet words via g2p_en (shared by both apps)
  features_v2.py    v2 features: GOP-SF, GOPT-style vectors, context, pooled hidden states
  head_v2.py        v2 heads: GBM, GOPT-style transformer, ensemble
  scorer.py         v2 inference with word/phone timestamps (used by server.py)
scripts/
  extract_full.py       cache posteriors + hidden states for all 5,000 utterances
  build_features_v2.py  v2 feature table from that cache
  cv_v2.py              speaker-grouped CV on the train split (model selection)
  train_v2.py           final fit, one-time test evaluation, bundle + figures
server.py           local app backend (FastAPI)
web/index.html      local app front end (single file, no build step)
models/head_v2.joblib  trained v2 head (committed)
app.py              original v1 Gradio demo
figures/            v1 PR curve / scatter; v2_* for v2
cache/              cached posteriors, features, v1 head (not committed -- see Setup)
REPORT.md           v1 evaluation writeup
REPORT_v2.md        v2 changes and evaluation
```

## Setup

```
pip install -r requirements.txt
```

Also needs, installed system-wide (not pip-installable):
- **FFmpeg**, "shared" build (ships `avcodec-*.dll`/`.so`) -- required by
  `torchcodec` for audio decoding. On Windows, the plain `winget install
  ffmpeg` (Gyan.FFmpeg) package is static-only; use `Gyan.FFmpeg.Shared`
  instead.
  Only needed for the training/evaluation scripts; the local app decodes
  audio in the browser and doesn't need FFmpeg.
- **espeak-ng** -- needed at runtime too: the model's tokenizer initializes
  a `phonemizer` espeak backend on load (see the note in
  `requirements.txt`). On Windows, `winget install eSpeak-NG.eSpeak-NG`.

The dataset (`mispeech/speechocean762`) is expected at
`datasets/speechocean762/data/{train,test}-00000-of-00001.parquet`
(Hugging Face parquet export).

## Running it

Reproduce end to end (checkpoints match `REPORT.md`'s development order):

```
python src/data.py              # load dataset, print one example + phone inventory
python src/phone_mapping.py     # print the ARPAbet -> IPA mapping table
python src/features.py          # single-utterance GOP feature sanity check
python src/features.py --batch  # batch-extract + cache features for 1000 utterances (~15-30 min CPU)
python src/train_head.py        # fit heads, print metrics, save figures/ + cache/head_model.joblib
python app.py                   # launch the Gradio demo (needs cache/head_model.joblib)
```

v2 pipeline: see `REPORT_v2.md` Section 6.

Everything is seeded (`SEED = 42` throughout) and intermediate artifacts
(model posteriors, features table, trained head) are cached to disk, so any
stage can be re-run without redoing the expensive parts -- see
`src/features.py`'s `extract_corpus_features` for the caching logic.

## Anti-goals

No accent classifier, no accent conversion/voice modification, no
Whisper-transcript-as-pronunciation-score, no chatbot, no speaker
adaptation, no prosody scoring, no database. See `REPORT.md` Section 1 for
why ASR specifically is the wrong tool for this problem.
