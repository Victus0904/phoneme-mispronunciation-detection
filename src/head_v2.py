"""v2 scoring heads, all trained on frozen features (nothing in the
acoustic model is updated):

  GBMHead  -- sklearn HistGradientBoosting on the scalar GOP / GOP-SF /
              context features + native categorical phone identities.
  SeqHead  -- a small GOPT-style transformer over the phone sequence of one
              utterance: each token = [scalar features, projected frozen
              wav2vec2 embedding, phone-identity embedding]; self-attention
              lets a phone's score depend on its neighbours and on the
              speaker's overall level in that utterance.
  EnsembleHead -- weighted average of the above.

Every head exposes .predict(df, emb) where df is the v2 feature table
(src/features_v2.py) for one or more utterances and emb maps hidden layer
-> [n_rows, D] pooled embeddings row-aligned with df.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCALAR_FEATURES = [
    "mean_log_posterior", "margin", "entropy", "duration_frames",
    "full_lp_max_win", "full_lp_lse_win", "content_lp_max_win", "blank_lp_span",
    "gop_sf", "sf_sub_llr", "sf_del_llr",
    "pos_in_word", "word_len", "is_word_initial", "is_word_final", "n_targets", "rel_position",
    "prev_gop_sf", "next_gop_sf", "prev_mlp", "next_mlp",
    "utt_mean_gop_sf", "utt_min_gop_sf", "utt_mean_mlp", "utt_frac_low_gop_sf",
    "frames_per_target", "word_mean_gop_sf", "fused",
]
CAT_FEATURES = ["arpa_phone", "arpa_base", "ipa_target", "sf_best_sub", "top_competitor", "prev_ipa", "next_ipa"]

V1_NUMERIC = ["mean_log_posterior", "margin", "entropy", "duration_frames"]
V1_CAT = ["top_competitor"]


class _Vocab:
    def __init__(self, values):
        self.itos = ["<unk>"] + sorted(set(values))
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def encode(self, values):
        return np.array([self.stoi.get(v, 0) for v in values], dtype=np.int64)


class GBMHead:
    def __init__(self, scalar=SCALAR_FEATURES, cats=CAT_FEATURES, emb_layers=(), n_pca=32, **params):
        self.scalar, self.cats = list(scalar), list(cats)
        self.emb_layers, self.n_pca = tuple(emb_layers), n_pca
        self.params = dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
                           l2_regularization=1.0, random_state=42)
        self.params.update(params)

    def _X(self, df, emb=None):
        X = df[self.scalar].to_numpy(np.float64)
        if self.emb_layers:
            E = np.concatenate([np.asarray(emb[L], dtype=np.float32) for L in self.emb_layers], axis=1)
            X = np.hstack([X, self.pca.transform(E)])
        C = np.stack([self.vocabs[c].encode(df[c].astype(str)) for c in self.cats], axis=1) if self.cats else None
        return np.hstack([X, C]) if C is not None else X

    def fit(self, df, emb=None, y=None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        y = df["human_score"].to_numpy(np.float64) if y is None else y
        self.vocabs = {c: _Vocab(df[c].astype(str)) for c in self.cats}
        n_num = len(self.scalar)
        if self.emb_layers:
            from sklearn.decomposition import PCA
            E = np.concatenate([np.asarray(emb[L], dtype=np.float32) for L in self.emb_layers], axis=1)
            self.pca = PCA(self.n_pca, random_state=42).fit(E)
            n_num += self.n_pca
        cat_mask = [False] * n_num + [True] * len(self.cats)
        self.model = HistGradientBoostingRegressor(categorical_features=cat_mask, **self.params)
        self.model.fit(self._X(df, emb), y)
        return self

    def predict(self, df, emb=None):
        return np.clip(self.model.predict(self._X(df, emb)), 0.0, 2.0)


class _SeqNet(nn.Module):
    def __init__(self, n_scalar, emb_dim, n_phone, n_ipa, d=96, layers=2, heads=4, dropout=0.15, emb_proj=64):
        super().__init__()
        self.scalar_proj = nn.Linear(n_scalar, d)
        self.emb_proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(emb_dim, emb_proj), nn.GELU(), nn.Linear(emb_proj, d)) if emb_dim else None
        self.phone_emb = nn.Embedding(n_phone, d)
        self.ipa_emb = nn.Embedding(n_ipa, d)
        self.pos_emb = nn.Embedding(64, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, xs, xe, ph, ipa, mask):
        B, L = ph.shape
        h = self.scalar_proj(xs) + self.phone_emb(ph) + self.ipa_emb(ipa)
        if self.emb_proj is not None:
            h = h + self.emb_proj(xe)
        h = h + self.pos_emb(torch.arange(L).clamp(max=63)).unsqueeze(0)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.out(h).squeeze(-1)


class SeqHead:
    def __init__(self, layers_used=(16,), scalar=SCALAR_FEATURES, epochs=40, lr=1e-3, batch=32,
                 seed=0, d=96, n_layers=2, weight_decay=1e-2, dropout=0.15, verbose=False):
        self.layers_used = tuple(layers_used)
        self.scalar = list(scalar)
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.d, self.n_layers, self.wd, self.dropout, self.verbose = d, n_layers, weight_decay, dropout, verbose

    def _emb(self, emb, idx):
        if not self.layers_used:
            return None
        return np.concatenate([emb[L][idx].astype(np.float32) for L in self.layers_used], axis=1)

    def _groups(self, df):
        uids = df["uid"].to_numpy()
        # rows of one utterance are contiguous, in phone order
        change = np.r_[0, np.nonzero(uids[1:] != uids[:-1])[0] + 1, len(uids)]
        return [np.arange(change[i], change[i + 1]) for i in range(len(change) - 1)]

    def _tensors(self, df, emb):
        Xs = (df[self.scalar].to_numpy(np.float32) - self.mu) / self.sd
        Xe = self._emb(emb, np.arange(len(df)))
        if Xe is not None:
            Xe = (Xe - self.emu) / self.esd
        ph = self.phone_vocab.encode(df["arpa_phone"].astype(str))
        ipa = self.ipa_vocab.encode(df["ipa_target"].astype(str))
        return Xs, Xe, ph, ipa

    def _batch(self, groups, Xs, Xe, ph, ipa, y=None):
        L = max(len(g) for g in groups)
        B = len(groups)
        xs = np.zeros((B, L, Xs.shape[1]), np.float32)
        xe = np.zeros((B, L, Xe.shape[1]), np.float32) if Xe is not None else np.zeros((B, L, 1), np.float32)
        P = np.zeros((B, L), np.int64); I = np.zeros((B, L), np.int64); M = np.zeros((B, L), bool)
        Y = np.zeros((B, L), np.float32)
        for b, g in enumerate(groups):
            n = len(g)
            xs[b, :n] = Xs[g]
            if Xe is not None:
                xe[b, :n] = Xe[g]
            P[b, :n] = ph[g]; I[b, :n] = ipa[g]; M[b, :n] = True
            if y is not None:
                Y[b, :n] = y[g]
        t = lambda a: torch.from_numpy(a)
        return t(xs), t(xe), t(P), t(I), t(M), t(Y)

    def fit(self, df, emb, y=None):
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)
        y = df["human_score"].to_numpy(np.float32) if y is None else y.astype(np.float32)
        X = df[self.scalar].to_numpy(np.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        E = self._emb(emb, np.arange(len(df)))
        emb_dim = 0
        if E is not None:
            self.emu, self.esd = E.mean(0), E.std(0) + 1e-6
            emb_dim = E.shape[1]
        self.phone_vocab = _Vocab(df["arpa_phone"].astype(str))
        self.ipa_vocab = _Vocab(df["ipa_target"].astype(str))
        self.net = _SeqNet(len(self.scalar), emb_dim, len(self.phone_vocab.itos), len(self.ipa_vocab.itos),
                           d=self.d, layers=self.n_layers, dropout=self.dropout)
        Xs, Xe, ph, ipa = self._tensors(df, emb)
        groups = self._groups(df)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        steps = self.epochs * int(np.ceil(len(groups) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.1)
        self.net.train()
        for ep in range(self.epochs):
            order = rng.permutation(len(groups))
            tot = 0.0
            for b0 in range(0, len(order), self.batch):
                gb = [groups[i] for i in order[b0:b0 + self.batch]]
                xs, xe, P, I, M, Y = self._batch(gb, Xs, Xe, ph, ipa, y)
                pred = self.net(xs, xe if Xe is not None else None, P, I, M)
                loss = ((pred - Y) ** 2)[M].mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); sched.step()
                tot += loss.item() * len(gb)
            if self.verbose and (ep % 10 == 9 or ep == self.epochs - 1):
                print(f"  epoch {ep+1}: train mse {tot/len(groups):.4f}", flush=True)
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, df, emb):
        Xs, Xe, ph, ipa = self._tensors(df, emb)
        out = np.zeros(len(df), np.float32)
        groups = self._groups(df)
        for b0 in range(0, len(groups), 128):
            gb = groups[b0:b0 + 128]
            xs, xe, P, I, M, _ = self._batch(gb, Xs, Xe, ph, ipa)
            pred = self.net(xs, xe if Xe is not None else None, P, I, M).numpy()
            for b, g in enumerate(gb):
                out[g] = pred[b, :len(g)]
        return np.clip(out, 0.0, 2.0)


class EnsembleHead:
    def __init__(self, members, weights=None):
        self.members = members
        self.weights = np.ones(len(members)) / len(members) if weights is None else np.asarray(weights, float)

    def predict(self, df, emb):
        preds = np.stack([m.predict(df, emb) for m in self.members])
        return np.clip((self.weights[:, None] * preds).sum(0) / self.weights.sum(), 0.0, 2.0)
