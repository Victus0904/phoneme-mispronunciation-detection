"""CTC forced alignment of a canonical phone-target sequence to a
frame x vocab log-posterior matrix.

First attempt was a plain monotonic segmental DP (score = sum of the
assigned target's log-posterior per frame, no blank). That degenerated
badly: T grabbed 35/79 frames in a 9-phone sanity-check utterance, because
a CTC model's raw posteriors are ~99%+ <pad> (blank) at nearly every frame
(CTC models are "peaky" -- they spike the true symbol for 1-3 frames and
emit blank everywhere else). With no blank to absorb "doesn't belong to any
of my targets" frames, plain max-sum DP has no reject option: it just
hoards frames on whichever target is locally least-bad, and only concedes
the bare minimum 1 frame to the next target when forced. Excluding blank
and renormalizing over content phones only did NOT fix this (same 35-frame
degenerate span) -- the problem isn't posterior scaling, it's the missing
reject option.

The correct fix is the standard CTC forced-alignment topology (same idea
as torchaudio's forced_align / Kaldi's ctc-segmentation): interleave the
blank symbol between targets, state sequence
  [blank, target_0, blank, target_1, blank, ..., target_{N-1}, blank]
and Viterbi over it with the standard CTC transition rules (stay, advance
one state, or skip a blank between two different labels). Frames that
don't confidently belong to any target now land on blank instead of being
force-fed to a neighboring phone.
"""
import numpy as np

NEG_INF = -1e18


def align(log_posteriors: np.ndarray, target_vocab_indices: list, blank_idx: int) -> list:
    """
    log_posteriors: [T, V] frame x vocab log-posteriors (full vocab, blank included).
    target_vocab_indices: [N] vocab index for each of the N targets, in order.
    blank_idx: vocab index of the CTC blank (<pad>).

    Returns: list of N dicts {'start': int, 'end': int, 'duration': int}
    (end inclusive, frame indices into log_posteriors). A target the
    Viterbi path skipped entirely (0 duration -- valid CTC behavior when a
    phone is reduced away, e.g. fast/careless speech) gets duration=0 and
    start=end set to the single boundary frame where the skip occurred, so
    downstream feature code still has one frame's posterior to read (with
    duration=0 itself carried through as a real, meaningful signal).
    """
    T = log_posteriors.shape[0]
    N = len(target_vocab_indices)
    if N == 0:
        raise ValueError("empty target sequence")

    L = 2 * N + 1
    ext = [blank_idx] * L
    for k, v in enumerate(target_vocab_indices):
        ext[2 * k + 1] = v

    ext_logpost = log_posteriors[:, ext].T  # [L, T]

    cum = np.full((L, T), NEG_INF, dtype=np.float64)
    # pred[s, t] in {0, 1, 2}: predecessor state offset (s, s-1, s-2)
    pred = np.zeros((L, T), dtype=np.int8)

    cum[0, 0] = ext_logpost[0, 0]
    if L > 1:
        cum[1, 0] = ext_logpost[1, 0]

    for t in range(1, T):
        for s in range(min(L - 1, 2 * t + 1) + 1):
            stay = cum[s, t - 1]
            adv = cum[s - 1, t - 1] if s - 1 >= 0 else NEG_INF
            skip = NEG_INF
            if s >= 2 and s % 2 == 1 and ext[s] != ext[s - 2]:
                skip = cum[s - 2, t - 1]

            best, which = stay, 0
            if adv > best:
                best, which = adv, 1
            if skip > best:
                best, which = skip, 2

            if best > NEG_INF:
                cum[s, t] = best + ext_logpost[s, t]
                pred[s, t] = which

    end_state = L - 1 if cum[L - 1, T - 1] >= cum[L - 2, T - 1] else L - 2

    # traceback: recover the state occupied at every frame
    states = np.zeros(T, dtype=np.int64)
    s = end_state
    states[T - 1] = s
    for t in range(T - 1, 0, -1):
        offset = pred[s, t]
        s = s - offset
        states[t - 1] = s

    boundaries = []
    for k in range(N):
        state = 2 * k + 1
        frames = np.where(states == state)[0]
        if len(frames) > 0:
            boundaries.append({"start": int(frames[0]), "end": int(frames[-1]), "duration": int(len(frames))})
        else:
            # skipped entirely: use the frame where the path passed through
            # this state's position in the sequence (nearest neighbor) so
            # there's still one frame of posterior to read.
            later = np.where(states > state)[0]
            boundary_frame = int(later[0]) if len(later) > 0 else T - 1
            boundaries.append({"start": boundary_frame, "end": boundary_frame, "duration": 0})

    return boundaries


def alignment_score(log_posteriors: np.ndarray, target_vocab_indices: list, boundaries: list) -> float:
    total = 0.0
    for b, vocab_idx in zip(boundaries, target_vocab_indices):
        total += log_posteriors[b["start"]:b["end"] + 1, vocab_idx].sum()
    return float(total)


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, ".")
    import numpy as np
    from src.data import load_speechocean762
    from src.phone_mapping import map_word_phones_to_ipa_targets
    from src.acoustic import load_model, get_log_posteriors, trim_silence, vocab_maps

    ds = load_speechocean762(decode_audio=True)
    example = ds["train"][0]
    audio = example["audio"]
    waveform = audio["array"].astype(np.float32)
    sr = audio["sampling_rate"]

    model, processor = load_model()
    sym2idx, idx2sym = vocab_maps(processor)
    blank_idx = sym2idx["<pad>"]
    trimmed = trim_silence(waveform, sr)
    logpost = get_log_posteriors(trimmed, sr, model, processor)

    all_targets = []
    for w in example["words"]:
        all_targets.extend(map_word_phones_to_ipa_targets(w["phones"]))

    target_vocab_indices = [sym2idx[t["ipa"]] for t in all_targets]
    boundaries = align(logpost, target_vocab_indices, blank_idx)
    score = alignment_score(logpost, target_vocab_indices, boundaries)

    ms_per_frame = len(trimmed) / sr / logpost.shape[0] * 1000
    print(f"text: {example['text']}")
    print(f"frames: {logpost.shape[0]}, targets: {len(all_targets)}, total alignment log-score: {score:.1f}")
    print(f"{'arpa':10s} {'ipa':6s} {'frames':12s} {'dur(ms)':8s}")
    for target, b in zip(all_targets, boundaries):
        dur_ms = b["duration"] * ms_per_frame
        flag = "  <- SKIPPED (0 frames)" if b["duration"] == 0 else ""
        print(f"{'+'.join(target['arpa']):10s} {target['ipa']:6s} [{b['start']:3d},{b['end']:3d}]     {dur_ms:6.1f}{flag}")
