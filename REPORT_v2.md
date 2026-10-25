# Pronunciation Scoring v2: What Changed and How Much It Helped

v1 (`REPORT.md`) is kept as-is for reference. v2 keeps v1's two hard
constraints: the acoustic model is **frozen** (no fine-tuning, same
`facebook/wav2vec2-lv-60-espeak-cv-ft`), and there is **no LLM** anywhere.
It also keeps v1's evaluation discipline: speechocean762's native
speaker-disjoint train/test split, every modelling choice made on the train
split only (speaker-grouped 5-fold CV), the test split scored once at the
end, and the flag threshold picked from train out-of-fold predictions, never
from test.

It drops one v1 constraint: the 1,000-utterance cap from the original brief.
v2 trains on all 2,500 train utterances. The table below separates the gain
from more data from the gain from better features, so the cap's effect is
visible.

## 1. Headline numbers (test split, speaker-disjoint)

| Model | v1 test subset* r | v1 subset AP | full test r | full test ρ | full test AP |
|---|---|---|---|---|---|
| Naive: mean log-posterior (v1 baseline) | 0.378 | 0.386 | 0.403 | 0.332 | 0.397 |
| Naive: GOP-SF alone (new single feature) | 0.454 | 0.434 | 0.485 | 0.368 | 0.458 |
| v1 head + v1 features, 500-utt train (v1 as shipped**) | 0.418 | 0.435 | 0.448 | 0.345 | 0.443 |
| v1 head + v1 features, full train | 0.475 | 0.468 | 0.514 | 0.405 | 0.488 |
| **v2 head + v2 features, 500-utt train** | 0.656 | 0.576 | 0.694 | 0.468 | 0.596 |
| **v2 head + v2 features, full train (shipped)** | **0.719** | **0.622** | **0.753** | **0.505** | **0.642** |

\* The exact 500 test utterances (9,698 phones) v1 reported on. "Full test"
is all 2,500 test utterances (47,369 phones).
\** Re-run here with sklearn's `HistGradientBoostingRegressor`; v1's
original sklearn `GradientBoostingRegressor` + one-hot gave r = 0.431 /
AP = 0.439 on the same subset (reproduced exactly before starting v2).

Reading the table:
- **Features, not data, are most of the gain.** With the same 500 training
  utterances v1 had, v2 goes from r = 0.418 to 0.656 on v1's own test
  subset. Using all 2,500 train utterances adds another ~0.06.
- **Error detection**: error base rate 18.7%. AP rises from 0.443 to 0.642.
  Precision at recall 0.10 is 0.95 (v1: 0.63). Precision at recall 0.50 is
  0.63 (v1: 0.44).
- **Spearman ρ stays much lower than Pearson r** (0.505 vs 0.753). 80% of
  phones carry the maximum human score (2.0), so the ranking among
  "correct" phones is mostly ties and noise. Pearson is driven by
  separating the genuine errors from everything else, and that is the part
  the product needs.

### Shipped flag threshold

The threshold is chosen on train OOF predictions as the highest recall with
precision ≥ 0.70: flag when predicted score < 1.75. On test that gives
**precision 0.688, recall 0.404**, and 11.0% of phones are flagged. v1's
conservative point was precision 0.63 at recall 0.10. v2 is now about as
precise as v1 was, while catching 4× as many real errors.

Figures: `figures/v2_pr_curve.png` (v1 vs v2 PR curves with the
operating point), `figures/v2_correlation_scatter.png`.

## 2. What changed

### 2.1 Alignment-free GOP (GOP-SF): the single biggest feature win

v1's diagnosis (REPORT.md §2.2) was that this CTC model is peaky, so 92.5%
of aligned phones get exactly one 20ms frame, and every v1 feature is
read off that single noisy frame. v2 adds features that need no
segmentation at all. Each phone *i* in the canonical sequence is compared
against alternative hypotheses: the same sequence with *i* replaced by each
of the 47 other target phones, or with *i* deleted. Each hypothesis is
scored by the CTC model's full sequence likelihood, which sums over every
possible alignment:

- `gop_sf`: log P(canonical) − logsumexp over {canonical, all substitutes, deletion}
- `sf_sub_llr`: log P(canonical) − log P(best substitute)
- `sf_del_llr`: log P(canonical) − log P(deletion)
- `sf_best_sub`: the most likely substitute (shown in the app as "heard something closer to /x/")

Alone, `gop_sf` reaches r = 0.485 on full test, against 0.403 for v1's mean
log-posterior. It is computed with batched `torch.nn.functional.ctc_loss`
over blank plus the 48 target symbols, roughly 50 hypotheses per phone. It
costs a median 30 ms per utterance (p90 82 ms, max 153 ms on the longest,
34-target utterance; measured on 100 test utterances).

### 2.2 GOPT-style per-phone vectors

These are two 48–49-dim vectors per phone. One is the full substitution LLR
vector from GOP-SF. The other is the windowed (±2 frames) max content
log-posterior of every candidate phone. They give the head *which* phones
the audio resembles, not just how well the target scores.

### 2.3 Context and phone identity

- Phone identity (ARPAbet with stress, base phone, IPA target) as native
  categorical features.
- Neighbour-phone GOP, position in word, and word-level and
  utterance-level GOP statistics, which act as a per-utterance proficiency
  proxy.
- Windowed and blank-aware posteriors around the aligned spike.

Human per-phone scores are strongly correlated within an utterance, and
phone identity is a strong prior. Adding phone identity alone to v1's four
features moves train-CV r from 0.538 to 0.615.

### 2.4 Frozen hidden states

Transformer layer 16 of the same frozen model is mean-pooled over each
phone's aligned span ±1 frame. It is cached alongside layers 8 and 24.
Layer 16 was chosen on CV, but the margin between layers was small.

### 2.5 Head: GBM + GOPT-style transformer ensemble

- `GBMHead`: `HistGradientBoostingRegressor` on the scalar features plus
  PCA(48) of [LLR vector, posterior vector, layer-16 embedding], with
  native categoricals.
- `SeqHead`: a 2-layer, d=96 transformer over the phone sequence of one
  utterance. Each token is [scalar features, projected vectors/embedding,
  phone embeddings]. This is the GOPT recipe (Gong et al. 2022) on frozen
  features.
- The shipped ensemble is 0.5 × GBM + 0.25 × SeqHead (seed 0) + 0.25 ×
  SeqHead (seed 1).

Train-split 5-fold speaker-grouped CV (Pearson / AP):

| Config | CV r | CV AP |
|---|---|---|
| v1 head + v1 features | 0.538 | 0.515 |
| GBM, v2 scalar features | 0.737 | 0.636 |
| GBM + vectors + layer-16 PCA | 0.742 | 0.643 |
| SeqHead (vectors + layer 16), seed 0 | 0.732 | 0.649 |
| SeqHead without embeddings | 0.634 | 0.581 |
| **Ensemble 0.5 / 0.25 / 0.25** | **0.770** | **0.675** |

On their own, the GBM and the transformer are about equal. Their errors are
different enough that averaging them adds ~0.03 r. Test r (0.753) came in
slightly below CV (0.770), which is consistent with an honest, untouched
test split.

## 3. Inference cost

End-to-end per utterance (model forward, alignment, features with GOP-SF,
ensemble head), on the same 30 utterances as v1's benchmark:

- **5.86× real-time** with PyTorch's default threads (v1 measured 3.90× in its benchmark)
- **2.14× real-time** pinned to a single thread

The heavier head costs little: the ensemble scores 47k test phones in 3.6 s.
The wav2vec2 forward pass still dominates.

## 4. Per-phone behaviour (full test, phones with ≥ 60 occurrences)

- Weakest: V (r = 0.56), UW (0.63), L (0.64), TH (0.66), ER (0.67), DH (0.68)
- Strongest: G (0.87), EY (0.84), F (0.83), AY (0.83), S (0.82), M (0.82)

V is still the hardest phone. That is consistent with v1's documented false
positive in `examples/borderline_canberra.wav`: humans scored the V in
"LOVES" 2.0 and v1 predicted 0.24. v2 predicts 1.30, a much smaller error,
but it **still flags it**. The V/W confusion is a real weak spot of this
frozen model for this speaker population.

## 5. Limitations (in addition to REPORT.md §5)

- **Still Mandarin-L1 only.** The phone-identity and context features learn
  priors about *which phones these learners tend to get wrong*. That is part
  of why v2 scores well, and it is exactly the kind of signal that may not
  transfer to other L1 backgrounds. Treat cross-accent use as unvalidated.
- **These numbers are higher than published speechocean762 phone-level
  results I'm aware of** (e.g. GOPT reports r ≈ 0.61). I checked for
  leakage:
  - every feature is computed from audio plus the canonical phone sequence
    (labels are only attached as the target column);
  - the train/test split is speaker-disjoint and asserted;
  - all selection was done on train CV;
  - the test split was scored once.

  I found no leakage. Still, treat the comparison to the literature with
  care: setups differ (acoustic model, phone inventory handling, fused
  vowel+R rows that share features). Speechocean762 prompts also repeat
  across speakers, so the head can learn prompt- and phone-specific priors.
- **Fused vowel+R rows** (REPORT.md §2.1) still share acoustic features
  between the two ARPAbet phones.
- **The demo path's G2P** (`g2p_en`) is still unevaluated, as in v1.

## 6. Reproducing

```
python scripts/extract_full.py          # ~75 min CPU, ~8 GB cache: posteriors + hidden states, all 5,000 utts
python scripts/build_features_v2.py     # ~6 min: cache/v2/features.parquet + per-phone vectors/embeddings
python scripts/cv_v2.py gbm_vec_h16 seq_vec_h16_s0 seq_vec_h16_s1   # train-split CV, writes OOF preds
python scripts/train_v2.py              # final fit, one-time test eval, models/head_v2.joblib, figures
```
