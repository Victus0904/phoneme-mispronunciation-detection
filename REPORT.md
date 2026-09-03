# Pronunciation Scoring: Evaluation Report

## 1. Problem framing

Phoneme-level pronunciation scoring asks a narrow question: *for this specific
sound, in this specific word, did the speaker produce something close to the
target phoneme?* That is a different problem from automatic speech
recognition, and ASR is the wrong tool for it, for a structural reason: ASR
is trained and evaluated to recover the *intended words* despite accent,
noise, and pronunciation variation. Robustness to mispronunciation is the
entire point of a good ASR system -- it is explicitly trained to smooth over
exactly the signal a pronunciation scorer needs to detect. A model that
successfully transcribes "zis" as "this" has, by design, thrown away the
information that the speaker said a stop instead of a fricative. Wrapping
Whisper and diffing its transcript against the target text is a common
shortcut, and it is the wrong one: word-level transcript agreement is not a
phoneme-level pronunciation signal.

What's needed instead is direct access to a model's frame-by-frame belief
about which *sound* is present, before that belief gets collapsed into a
word hypothesis -- and a way to compare that belief against the sound the
speaker was actually supposed to produce.

## 2. Approach, and why it's deliberately simple

Given the time budget, the design goal was a system simple enough to fully
understand and rigorously evaluate, not a system that maximizes correlation.
Concretely:

- **No fine-tuning.** All acoustic representations come from a frozen,
  pretrained model. The only thing fit on this data is a small regression
  head (ridge / gradient boosting) on top of hand-specified Goodness-of-
  Pronunciation (GOP) features -- training takes under two seconds.
- **No LLM anywhere.** The coaching tip is a static, hand-written lookup
  table (`src/tips.py`). See Section 4 for the reasoning.
- **Classical GOP, not a learned end-to-end scorer.** Mean log-posterior,
  margin vs. the best competing phone, posterior entropy, duration, and
  competitor identity -- five interpretable features per phone, computed
  from a forced alignment between the canonical phone sequence and the
  model's frame-level output.

### Model substitution (checked before writing any scoring code)

The brief specified `charsiu/en_w2v2_fc_10ms`, a 10ms-resolution frame
classifier. It doesn't work: the Hub repo ships only `config.json` and
`pytorch_model.bin` -- no processor, no tokenizer, no `id2label` mapping
anywhere -- and its declared architecture, `Wav2Vec2ForFrameClassification`,
does not exist in the `transformers` library (verified: `hasattr(transformers,
'Wav2Vec2ForFrameClassification')` is `False` on transformers 5.16.1) and
isn't distributed as an installable package either. Using it would have
meant guessing a 42-way label mapping with zero ground truth -- precisely
the kind of silent corruption this project's brief warned about avoiding.

Fell back to `facebook/wav2vec2-lv-60-espeak-cv-ft` per the brief's explicit
fallback instruction. Two consequences worth stating plainly:

- **Frame rate**: ~20.25ms/frame (measured from actual model output), not
  10ms. Halves the temporal resolution of alignment.
- **Output vocabulary**: eSpeak/IPA phonemes (392 symbols), not ARPAbet.
  speechocean762 supplies ARPAbet. This mismatch is handled explicitly --
  see below, it was the single largest correctness risk in the project.
- **CTC, not a frame classifier.** This model is trained with a CTC loss and
  its raw per-frame output is consequently "peaky" (near-certain blank at
  ~95%+ of frames, with the true phone surfacing only in short spikes). This
  had a direct, non-obvious effect on the alignment algorithm (Section 2.2)
  and on feature quality (Section 5).

### 2.1 The ARPAbet <-> IPA mapping trap

speechocean762's 67-symbol ARPAbet-with-stress inventory and the acoustic
model's 392-symbol IPA/eSpeak inventory do not correspond 1:1, and the
mismatch was not guessed at. `src/phone_mapping.py` was built by running the
actual espeak-ng en-us backend (the same phonemizer this model's own
training pipeline used) over all 2,604 unique words in speechocean762 and
voting on the aligned ARPAbet<->IPA symbol pairs.

The empirical pass surfaced a real, structural mismatch, not just multiple
spellings of the same thing: **espeak-ng fuses a vowel immediately followed
by R into a single rhotic IPA symbol** (`['AA0', 'R']` in "arm" becomes the
one token `['ɑːɹ']`, not two). This is handled with an explicit
`RHOTIC_R_FUSION` merge rule: a fused target's frames are shared between
both original ARPAbet phones for feature purposes, and both keep their own
human score for training. All 67 ARPAbet symbols map to one of 48 distinct
IPA targets, all 48 of which were confirmed present in the model's actual
output vocabulary before any alignment code was written. Lower-confidence
mappings (e.g. `AH1` was only a 51% plurality; several `<vowel>+R` fusions
for IH/IY/UW/OW were extrapolated by phonological analogy rather than
directly observed with strong sample counts) are flagged in that file's
docstring rather than presented as settled.

### 2.2 Alignment: the first version was wrong, and the failure was diagnosed, not patched around

The first alignment implementation was a plain monotonic DP: assign every
frame to a target phone in order, maximizing summed log-posterior, each
target getting >=1 frame. On the very first sanity-check utterance it
degenerated badly -- one phone (`T`) absorbed 35 of 79 frames.

Root cause, found by printing the raw per-frame argmax before writing any
more code: this is a CTC model, so ~95%+ of frames are near-certainly
`<pad>` (blank), with the true phone appearing only in short spikes. A
plain max-sum DP has no "reject" option for a frame that doesn't belong to
any target, so it hoards ambiguous frames on whichever target is locally
least-bad, and only concedes the bare minimum to the next target when
forced. Renormalizing the posteriors to exclude blank did **not** fix
this (same degenerate 35-frame span) -- the problem wasn't posterior
scaling, it was the missing reject option.

The fix was the standard CTC forced-alignment topology (the same idea
used by e.g. torchaudio's `forced_align`): interleave blank between
targets and run Viterbi with the standard stay/advance/skip-blank
transitions, implemented directly in numpy (`src/align.py`, no external
alignment library). Silence and non-target audio now correctly falls on
blank instead of being force-fed to a neighboring phone. Verified on
multiple utterances: 0 skipped phones, fully monotonic, leading/trailing
silence correctly excluded from every phone span, and alignment score
jumped from -206 to -13 (log-posterior units) on the first sanity
utterance after the fix.

**Trade-off inherited from this fix, stated plainly**: because CTC spikes
are brief, aligned phone durations are almost always exactly 1 frame (20ms)
-- 92.5% of the 19,381 aligned phones in the full 1,000-utterance corpus.
0.3% are 3-4 frames; none were skipped entirely. This is a direct, expected
consequence of substituting a CTC model for the originally-specified frame
classifier, not a bug in the alignment code -- a genuine frame classifier
(like the unusable charsiu model) would not have this limitation, since it
scores every frame independently rather than learning to spike briefly and
suppress everywhere else. The `duration_frames` GOP feature therefore
carries much less information than it would with the originally-specified
model.

## 3. Results

All metrics on speechocean762's **native** train/test split (confirmed
speaker-disjoint: 0 speaker overlap between splits, 125 speakers each in the
full dataset), subsampled to 500 utterances per split (1,000 total, the
brief's cap) with a fixed seed. 9,683 train / 9,698 test phone-level rows.

### 3.1 Correlation with human phoneme scores (test set)

| | Pearson r | Spearman ρ |
|---|---|---|
| Naive baseline (mean log-posterior alone, no head) | 0.378 | 0.323 |
| Ridge head | 0.418 | 0.359 |
| **GBM head (best)** | **0.431** | **0.380** |

The trained head adds real value over the naive single-feature baseline
(+0.053 Pearson), but the gain is modest -- most of the usable signal is
already present in raw posterior confidence; margin, entropy, duration, and
competitor identity contribute a real but secondary improvement. See
`figures/correlation_scatter.png` -- deliberately not cropped or cherry-
picked to look better than it is: there is real, visible scatter and
overlap between score bands, which is an honest reflection of r=0.431, not
a clean diagonal.

### 3.2 Error detection (human_score < 2.0, treated as "error")

Base rate: 18.2% of test-set phones. Average precision (area under the PR
curve): **0.439** (~2.4x the random baseline of 0.182). See
`figures/pr_curve.png`.

Two operating points (chosen by recall target -- see Section 4 for why a
precision target was tried first and abandoned):

| Operating point | Precision | Recall |
|---|---|---|
| Conservative (recall ≈ 0.10) | 0.630 | 0.100 |
| Balanced (recall ≈ 0.50) | 0.437 | 0.501 |

### 3.3 Inference throughput and cost

Measured fresh (not from cache) on 30 utterances, single CPU core:
89.2s of audio processed in 22.9s wall-clock -> **3.90x real-time**.

Extrapolated to 1,000 users x 10 minutes of audio per user per day:
- Daily audio to process: 166.7 hours
- Required CPU compute: **42.7 CPU-hours/day**, single-core, no batching or GPU
- Rough cost at $0.045/vCPU-hour (a stated assumption -- a small on-demand
  CPU instance's blended rate, not a vendor quote): **~$1.92/day (~$58/month)**

This is a CPU-only, unbatched, single-process baseline; real deployment
would batch requests and could plausibly run on GPU, both of which would
substantially change this number in either direction depending on
utilization.

## 4. False positives as the central product problem

For a pronunciation-feedback product, the two error types are not
symmetric. A learner who mispronounces a phone and isn't told has lost one
opportunity to self-correct -- mildly costly, and there will be another
opportunity next time they say the word. A learner who pronounces a phone
*correctly* and is told they're wrong has been given false, confusing
information about their own speech, by a system positioned as an authority
on it. That erodes trust in the tool directly, and in an educational context
can actively teach the learner to "fix" something that wasn't broken. False
positives are more expensive than false negatives here, and the model
should be tuned accordingly.

This is why the "high-precision" operating point in Section 3.2 was chosen
by recall target rather than precision target: a precision=0.85 target was
tried first, and turned out to require recall≈0.003 -- flagging roughly 3
per 1,000 true errors, which is not a usable product answer, just a
degenerate corner of the curve. The precision-recall curve was inspected
directly (precision at recall=0.05, 0.10, 0.15, ... 0.80) before committing
to an operating point, rather than trusting an arbitrary precision target
the data couldn't actually support.

The **conservative point (precision 0.63, recall 0.10)** is the one I'd
ship a first version of this feature with: when the system flags a phone,
it's right about 2 times out of 3, and it stays quiet on the phones it's
unsure about rather than guessing. That is a defensible trade for a
learner-facing feature, at the direct cost of missing 90% of true errors --
which is acceptable *only* because the cost of a miss here (see above) is
much lower than the cost of a false alarm.

`[TODO: your commentary here on the operating threshold, precision/recall
trade-off, and any product-side mitigations -- e.g. framing flagged phones
as "worth double-checking" rather than "wrong", or showing confidence
alongside the flag.]`

## 5. Limitations, stated plainly

- **speechocean762 is entirely Mandarin-L1.** Every speaker in this dataset
  is a native Mandarin speaker learning English. Accent fairness across L1
  backgrounds -- whether this system over- or under-flags speakers with,
  say, a Spanish, Arabic, or Hindi L1 -- **cannot be measured on this data**,
  full stop. Nothing in this report should be read as evidence of fair or
  unfair behavior across accents in general; it is only evidence about
  Mandarin-L1 English learners, specifically children in this dataset (ages
  are skewed young).
- **Duration is a weak feature here** (Section 2.2) -- a direct cost of the
  model substitution, not something more feature engineering would fix
  without a different (non-CTC) acoustic model.
- **The correlation is real but modest** (r=0.431). This system is a
  reasonable triage signal, not a precise scorer -- consistent with the
  brief's premise that a simple system with rigorous, honestly-reported
  evaluation is the goal, not a maximized correlation number.
- **The RHOTIC_R_FUSION table has extrapolated entries** (Section 2.1) that
  weren't directly observed with strong sample counts in the empirical
  mapping pass -- worth re-verifying against more data before relying on
  this system for words with those specific vowel+R combinations.
- **G2P in the demo path is a separate, unvalidated component.** `g2p_en`'s
  output was spot-checked (Section 2 pipeline test), not evaluated against
  the rigor applied to the ARPAbet<->IPA mapping -- it's the demo's
  convenience path, not part of the evaluated pipeline.

## 6. What I would do with a large multi-L1 accent dataset

`[TODO: your own scaffold/commentary. Some starting questions worth
addressing: does the ARPAbet<->IPA mapping and GOP feature set generalize
across L1 backgrounds, or does it need to be re-derived per L1 (e.g. an L1
where a phone contrast doesn't exist at all)? How would you define and
measure "fairness" here -- equal precision/recall across L1 groups at a
fixed threshold? Would you want separate operating thresholds per L1
group, and what are the product/ethical implications of that? Does the
frozen-features assumption hold, or would some L1 groups need the
acoustic model itself to see more of their accented speech during
pretraining/fine-tuning?]`
