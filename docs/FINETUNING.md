# Fine-tuning the acoustic model

v2 keeps `facebook/wav2vec2-lv-60-espeak-cv-ft` frozen. This is the optional
next step: fine-tune its top layers so the phone posteriors (and therefore
GOP-SF, the strongest feature) separate correct from mispronounced phones
better, then rebuild the scoring head on top. Optionally, add L2-ARCTIC for
five more first languages.

## Results (2026-11-17)

Two runs, each followed by the full v2 rebuild. Both test sets were scored
once, after training, and every choice was made on validation speakers only.

| | Frozen (v2 shipped) | Fine-tuned, speechocean only (`ft_top8`) | **Fine-tuned + L2-ARCTIC (`ft_l2`)** |
|---|---|---|---|
| speechocean762 test, phone r (full v2 pipeline) | 0.753 | 0.795 | **0.798** |
| speechocean762 test, error AP | 0.642 | 0.672 | **0.672** |
| Standard level: recall / precision | 40% / 69% | 52% / 66% | **54% / 65%** |
| L2-ARCTIC test speakers, r (GOP-SF only) | 0.286 | 0.254 | **0.530** |
| L2-ARCTIC test speakers, AP (GOP-SF only) | 0.323 | 0.295 | **0.555** |
| Fluency / prosody test r | 0.82 / 0.81 | 0.828 / 0.817 | 0.826 / 0.818 |

Per first language on the L2-ARCTIC test speakers (GOP-SF r, one held-out
speaker each, so read these as indicative):

| L1 | Frozen | `ft_top8` | `ft_l2` |
|---|---|---|---|
| Arabic | 0.198 | 0.180 | **0.441** |
| Hindi | 0.149 | 0.131 | **0.409** |
| Korean | 0.221 | 0.185 | **0.468** |
| Mandarin | 0.216 | 0.173 | **0.457** |
| Spanish | 0.202 | 0.193 | **0.525** |
| Vietnamese | 0.474 | 0.429 | **0.688** |

Takeaways:
- Fine-tuning on speechocean762 alone improves speechocean762 but makes
  every other accent *worse* than the frozen model. It specialises in
  Mandarin-L1 children.
- Adding L2-ARCTIC keeps the speechocean762 gain and nearly doubles accuracy
  on unseen speakers of six first languages. `ft_l2` is the better model on
  every row above.
- The L2-ARCTIC rows measure GOP-SF alone, because the v2 head is trained on
  speechocean762 only. Training the head on L2-ARCTIC too is the next step.
- `ft_l2` inherits L2-ARCTIC's **non-commercial** license.

Cost: about $2.75 on Modal for everything (two training runs, two rebuilds,
evaluations, data prep), on A10G/T4 GPUs.

## What it trains

`scripts/finetune_acoustic.py`, loss per batch:

- **CTC on what the speaker actually said**: the canonical phones with the
  annotated mispronunciations applied (substitutions, deletions, multi-phone
  insertions; `X*` distorted phones count as X). Utterances with an
  unidentifiable error (`<unk>`, about 8% of speechocean762) get no CTC
  term. This keeps the model a calibrated phone recognizer.
- **GOP-SF score loss**: GOP-SF computed exactly as in `src/features_v2.py`
  (checked to within 4e-6) but differentiable, mapped to a 0–2 score by a
  4-parameter head, regressed onto the human phone scores. Phones the raters
  marked wrong are weighted 4× (`--err-weight`).

Only the top `--unfreeze` layers (default 8, i.e. layers 16–23) plus the
final layer norm and CTC head train. With the default, the layer-8 and
layer-16 hidden states the head uses do not change.

**Evaluation discipline is unchanged.** The script never reads a test split.
It selects the checkpoint on speaker-disjoint validation sets (10% of
speechocean762 *train* speakers, plus L2-ARCTIC's validation speakers when
used), and first reports the frozen model on the same sets, so every number
in `log.json` is comparable to the baseline. The speechocean762 test split is
scored once, afterwards, by `scripts/train_v2.py`.

## Adding accents: L2-ARCTIC

L2-ARCTIC v5.0 (Zhao et al., 2018) is **CC BY-NC 4.0, non-commercial use
only**. It has 24 speakers: four each with L1 Arabic, Mandarin, Hindi,
Korean, Spanish and Vietnamese. Its 3,599 hand-annotated utterances mark
each phone as correct, substituted, deleted or inserted. There are no graded
scores, so phones become 2 (correct), 0 (wrong) or 1 (distorted version of
itself).

`scripts/l2arctic.py` converts the annotation TextGrids into speechocean762's
row format (3,552 utterances kept, 14.7% of phones marked wrong). The split
is fixed per L1: two speakers train, one validates, and one is held out as
test and never read by training, so each accent can be measured later.

```
python scripts/l2arctic.py inspect <l2arctic_root>   # label stats only
python scripts/l2arctic.py pack <l2arctic_root>      # -> datasets/l2arctic/l2arctic_annotated.parquet
python scripts/finetune_acoustic.py --l2arctic datasets/l2arctic/l2arctic_annotated.parquet ...
```

With `--l2arctic`, validation reports r per set and per L1. The checkpoint is
selected on the mean r across the two sets, so gains on new accents can't
hide a speechocean regression.

## Running it on Modal (what we use)

`scripts/modal_app.py` runs everything on Modal GPUs. A volume `pmd-ft`
holds the datasets, checkpoints (`runs/<name>`, so runs resume), per-run
feature caches and results. GPUs are tried in the order L4, A10G, T4.

```
pip install modal && modal setup
modal run scripts/modal_app.py::prep_l2arctic                  # once, CPU only (~$0.10)
modal run scripts/modal_app.py::train --name smoke --extra "--limit 160 --epochs 1"
modal run --detach scripts/modal_app.py::train --name ft_top8
modal run --detach scripts/modal_app.py::train --name ft_l2 --extra "--l2arctic datasets/l2arctic/l2arctic_annotated.parquet"
modal run --detach scripts/modal_app.py::rebuild --name ft_top8   # one-time test evaluation
modal volume get pmd-ft results/ft_top8 ./runs/
```

`rebuild` runs the README's reproduction steps with `ACOUSTIC_MODEL` set to
the fine-tuned model (`extract_full`, `build_features_v2`, `cv_v2`,
`train_v2`, `calibrate_strictness`, fluency heads). It copies the new heads,
strictness thresholds, metrics and figures to `results/<name>/`.
`extract_full.py` records the model in `cache/v2/utt/MODEL.txt` and refuses
to mix models in one cache.

**Spend guard:** `python scripts/modal_budget.py` checks this month's Modal
spend every 5 minutes, logs it to `runs/modal_credits.log`, and stops every
app at `--limit` (default $25, leaving $5 of the $30 monthly credits).
Modal's report can lag by up to about an hour, so also set a spending limit
in the Modal dashboard as a backstop.

Measured: on an A10G, about 26 utterances/s, 1.5 min per speechocean762
epoch, and about $1.10/hour with the CPUs and memory requested.

## Running it locally (GTX 1650 4 GB and up)

```
python -m venv E:\venvs\pmd-gpu
E:\venvs\pmd-gpu\Scripts\pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
E:\venvs\pmd-gpu\Scripts\pip install -r requirements-finetune.txt
E:\venvs\pmd-gpu\Scripts\python scripts\finetune_acoustic.py --limit 160 --epochs 2 --out runs\smoke
```

Measured on a GTX 1650 with the defaults (`--batch 2`): about 0.5
utterances/s, about 75 minutes per epoch. That's fine for debugging, too
slow for real runs.

fp16 note: the model's positional convolution overflows in fp16 on real
speech, so the script keeps it (and the frozen CNN front end) in fp32 under
autocast. On GPUs with bf16 (L4, A10G, A100) the script uses bf16 instead.

Other clouds: `scripts/lightning_setup.sh` sets up any Ubuntu GPU box
(e.g. Lightning AI). Then run the same commands as `modal_app.py`'s
functions.

## Using a fine-tuned model in the app

Set `ACOUSTIC_MODEL` before starting the server, and use the head that was
trained on that model's features (the shipped head expects the frozen model):

```
$env:ACOUSTIC_MODEL = "E:\models\ft_top8"   # PowerShell
$env:HEAD_BUNDLE = "E:\models\ft_top8\head_v2.joblib"
python server.py
```

To ship it, upload the model folder to the Hugging Face Hub (1.3 GB is too
large for git), point `BASE_MODEL_ID`'s default at it, and commit the new
`models/*.joblib` and `models/strictness.json` together. A model trained
with `--l2arctic` inherits L2-ARCTIC's non-commercial license.

## Hosting the website on Modal

`modal deploy scripts/modal_web.py` serves `server.py` + `web/` on Modal
(CPU, 2 cores / 4 GiB, scales to zero 5 minutes after the last request,
about $0.13 per running hour, at most 2 containers). The first request
after an idle spell takes about 30 s while the model loads.
