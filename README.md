# Pronunciation Scoring

A phoneme-level mispronunciation scoring pipeline: audio + target sentence
in, a 0-2 score per phoneme + below-threshold flags + an articulatory
coaching tip for the worst phone out. No fine-tuned models, no LLM anywhere
in the pipeline -- see `REPORT.md` for the full evaluation and the
reasoning behind that constraint.

## Headline numbers

(speaker-disjoint test set, 500 utterances / 9,698 phones, from
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
  tips.py           hand-written articulatory coaching tips
app.py              Gradio demo
figures/            PR curve, correlation scatter (from train_head.py)
cache/              cached posteriors, features.parquet, head_model.joblib
                     (not committed -- see Setup)
REPORT.md           full evaluation writeup
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
- **espeak-ng** -- only needed to *reproduce* `src/phone_mapping.py`'s
  empirical mapping build (`phonemizer` backend); not required by the
  scoring pipeline itself at runtime.

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

Everything is seeded (`SEED = 42` throughout) and intermediate artifacts
(model posteriors, features table, trained head) are cached to disk, so any
stage can be re-run without redoing the expensive parts -- see
`src/features.py`'s `extract_corpus_features` for the caching logic.

## Anti-goals

No accent classifier, no accent conversion/voice modification, no
Whisper-transcript-as-pronunciation-score, no chatbot, no speaker
adaptation, no prosody scoring, no database. See `REPORT.md` Section 1 for
why ASR specifically is the wrong tool for this problem.
