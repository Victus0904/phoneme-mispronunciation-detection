---
title: Pronunciation Coach
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Phoneme-level pronunciation feedback, no LLM
---

# Pronunciation Coach

Read a sentence aloud and get a score for every sound, the sounds worth
another take, what the model heard instead, and a hand-written tip for each.

- Frozen `facebook/wav2vec2-lv-60-espeak-cv-ft` acoustic model (no
  fine-tuning) + alignment-free GOP features + a small GBM/transformer
  ensemble. No LLM anywhere.
- On speechocean762's held-out speakers: Pearson r = 0.75 with expert
  phone scores; flagged sounds are real errors 69% of the time.
- Trained and evaluated only on Mandarin-L1 English learners -- accuracy for
  other accents hasn't been measured.
- Recordings are sent to this Space only to score them and aren't stored.

Source code, full evaluation and a version you can run on your own computer:
https://github.com/Victus0904/phoneme-mispronunciation-detection

Example recordings are from [speechocean762](https://huggingface.co/datasets/mispeech/speechocean762) (CC BY 4.0).
