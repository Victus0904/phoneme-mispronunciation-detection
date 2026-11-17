# Fine-tuning the acoustic model

v2 keeps `facebook/wav2vec2-lv-60-espeak-cv-ft` frozen. This is the optional
next step: fine-tune its top layers so the phone posteriors (and therefore
GOP-SF, the strongest feature) separate correct from mispronounced phones
better, then rebuild the scoring head on top.

**Status: experimental.** The script runs end to end and its GOP-SF matches
`src/features_v2.gop_sf` to within 4e-6 on the frozen model, but no full run
has been done yet, so it is not known whether it beats v2's test r = 0.753.

## What it trains

`scripts/finetune_acoustic.py`, loss per batch:

- **CTC on what the speaker actually said**: the canonical phones with
  speechocean762's annotated mispronunciations applied (substitutions,
  deletions, multi-phone insertions; `X*` distorted phones count as X).
  Utterances whose annotation is `<unk>` (about 8%) get no CTC term. This
  keeps the model a calibrated phone recognizer.
- **GOP-SF score loss**: GOP-SF computed exactly as in `src/features_v2.py`
  but differentiable, mapped to a 0–2 score by a 4-parameter head, regressed
  onto the human phone scores. Phones the raters marked wrong are weighted
  4× (`--err-weight`), since 81% of phones score 2.0.

Only the top `--unfreeze` layers (default 8, i.e. layers 16–23) plus the
final layer norm and CTC head train. With the default, the layer-8 and
layer-16 hidden states the head uses do not change.

**Evaluation discipline is unchanged.** The script never reads the test
split. It selects the checkpoint on a speaker-disjoint validation set
(10% of *train* speakers, fixed seed), and first reports the frozen model
on that same set, so every number in `log.json` is comparable to the
baseline. Test is scored once, afterwards, by `scripts/train_v2.py`.

## Running it locally (GTX 1650 4 GB and up)

```
python -m venv E:\venvs\pmd-gpu
E:\venvs\pmd-gpu\Scripts\pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
E:\venvs\pmd-gpu\Scripts\pip install -r requirements-finetune.txt
E:\venvs\pmd-gpu\Scripts\python scripts\finetune_acoustic.py --limit 160 --epochs 2 --out runs\smoke
```

Measured on a GTX 1650 with the defaults (`--batch 2`, top 8 layers): about
0.5 utterances/s, so one epoch over the ~2,250 training utterances takes
about 75 minutes, and 10 epochs is an overnight job. Use the cloud for real runs.

fp16 note: the model's positional convolution overflows in fp16 on real
speech, so the script keeps it (and the frozen CNN front end) in fp32 under
autocast. On GPUs with bf16 (L4, A10G, A100) the script uses bf16 instead.

## Running it on Lightning AI

1. Create a Studio, pick a GPU. An **L4 (24 GB)** is a good default; a T4
   works with `--batch 4`. Check the hourly credit cost shown for each GPU.
2. In the Studio terminal:
   ```
   git clone https://github.com/Victus0904/phoneme-mispronunciation-detection.git
   cd phoneme-mispronunciation-detection
   git checkout finetune-acoustic          # until it is merged
   bash scripts/lightning_setup.sh
   ```
3. Measure speed first (2 minutes), then train with a time cap so a mistake
   can't burn the whole budget. Rerunning the same command resumes from
   `last.pt`:
   ```
   python scripts/finetune_acoustic.py --limit 160 --epochs 1 --batch 8 --workers 4 --no-checkpointing --out runs/smoke
   python scripts/finetune_acoustic.py --batch 8 --workers 4 --no-checkpointing --max-hours 3 --out runs/ft_top8
   ```
   If the 24 GB card runs out of memory, drop `--no-checkpointing` or use `--batch 4`.
4. Read `runs/ft_top8/log.json`: `baseline_frozen.r` vs. each epoch's `r`.
   If no epoch beat the baseline, the script saves no `best/`. Stop there
   and try other settings (`--unfreeze 4`, `--lr 1e-5`, `--gop-weight 2`)
   rather than going on to the test split.
5. If it did improve, rebuild the v2 pipeline on the new model. This is the
   one-time test evaluation:
   ```
   export ACOUSTIC_MODEL=runs/ft_top8/best
   python scripts/extract_full.py
   python scripts/build_features_v2.py
   python scripts/train_v2.py                 # prints test r / AP; compare with 0.753 / 0.642
   python scripts/calibrate_strictness.py
   python scripts/build_utt_features.py && python scripts/train_fluency.py
   ```
   `extract_full.py` writes `cache/v2/utt/MODEL.txt` and refuses to mix
   models in one cache.
6. Download `runs/ft_top8/best/` (about 1.3 GB), `models/` and
   `cache/v2/metrics_v2.json`. Stop the Studio so it stops using credits.

## Using a fine-tuned model in the app

Set `ACOUSTIC_MODEL` before starting the server, and use the head that was
trained on that model's features (the shipped head expects the frozen model):

```
$env:ACOUSTIC_MODEL = "E:\models\ft_top8"   # PowerShell
python server.py
```

To ship it, upload the model folder to the Hugging Face Hub (1.3 GB is too
large for git), point `BASE_MODEL_ID`'s default at it, and commit the new
`models/*.joblib` and `models/strictness.json` together.

## Budget

A full run on an L4-class GPU is estimated at 1–2 hours (10 epochs), plus
about 30 minutes for step 5. That estimate is from the GTX 1650 benchmark and
hasn't been measured on an L4. At typical cloud rates, $30 of credits covers
many full runs. Read the actual rate in Lightning before starting, and keep
`--max-hours` set.
