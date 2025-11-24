#!/usr/bin/env python3
"""
Train (mel + IPA articulatory features) -> Codec2 2400 bits.

- Loads DB rows: id, word, phonemes, start, end, audio_file, codec2_file, features_file
- features_file is npz with keys: mel (T_mel, mel_dim), f0 (T_f0,), rms (T_rms,)
- Upsamples features from 50 Hz -> 150 Hz (factor 3) using linear interpolation
- Uses panphon (if installed) to convert IPA segments -> articulatory feature vectors per segment.
  If panphon is missing, falls back to a simple character embedding.
- Model: Transformer encoder. Predicts 54 bits per codec2 frame (BCEWithLogitsLoss).
"""

import argparse
import sqlite3
from pathlib import Path
import numpy as np
import math
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Try panphon for articulatory features
try:
    import panphon
    FT = panphon.FeatureTable()
    HAVE_PANPHON = True
except Exception:
    HAVE_PANPHON = False

# -----------------------
# Config
# -----------------------
MEL_DIM = 80                # your mel dimension
CODEC2_BITS = 54            # Codec2 2400 bps
UPSAMPLE_FACTOR = 3         # 50 Hz -> 150 Hz
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Utilities
# -----------------------
def load_npz_features(path: Path):
    """
    loads npz produced by your script: keys mel, f0, rms
    returns mel (T_mel, mel_dim), f0 (T_mel or close), rms (T_mel or close)
    """
    d = np.load(path)
    mel = d["mel"]          # shape (T_mel, mel_dim)
    f0 = d.get("f0", None)
    rms = d.get("rms", None)

    # Ensure shapes are 1D/2D
    f0 = np.asarray(f0) if f0 is not None else np.zeros((mel.shape[0],), dtype=np.float32)
    rms = np.asarray(rms) if rms is not None else np.zeros((mel.shape[0],), dtype=np.float32)

    # If f0/rms lengths differ from mel, resample them to mel length using simple interpolation
    def resample_1d(src):
        if src is None:
            return np.zeros((mel.shape[0],), dtype=np.float32)
        if len(src) == mel.shape[0]:
            return np.asarray(src, dtype=np.float32)
        # linear interp over index
        src_idx = np.linspace(0.0, 1.0, num=len(src))
        tgt_idx = np.linspace(0.0, 1.0, num=mel.shape[0])
        return np.interp(tgt_idx, src_idx, src).astype(np.float32)

    f0 = resample_1d(f0)
    rms = resample_1d(rms)

    return mel.astype(np.float32), f0.astype(np.float32), rms.astype(np.float32)


def upsample_features(mel, f0, rms, factor=UPSAMPLE_FACTOR):
    """
    Upsample along time axis by integer factor using linear interpolation.
    Input:
      mel: (T_mel, D)
      f0:  (T_mel,)
      rms: (T_mel,)
    Output:
      mel_u: (T_c2, D)
      f0_u:  (T_c2,)
      rms_u: (T_c2,)
    """
    T = mel.shape[0]
    T_u = T * factor

    src_idx = np.linspace(0.0, 1.0, num=T)
    tgt_idx = np.linspace(0.0, 1.0, num=T_u)

    # mel dims: do per-dim interp (vectorized)
    mel_u = np.empty((T_u, mel.shape[1]), dtype=np.float32)
    for d in range(mel.shape[1]):
        mel_u[:, d] = np.interp(tgt_idx, src_idx, mel[:, d])

    f0_u = np.interp(tgt_idx, src_idx, f0).astype(np.float32)
    rms_u = np.interp(tgt_idx, src_idx, rms).astype(np.float32)

    return mel_u, f0_u, rms_u


# -----------------------
# IPA -> articulatory
# -----------------------
class IPAArticulator:
    def __init__(self):
        self.use_panphon = HAVE_PANPHON
        if self.use_panphon:
            self.ft = FT  # panphon.FeatureTable()
            # panphon outputs arrays of 21 features per segment.
            self.dim = 21
        else:
            # fallback: build char vocab on the fly and use learned embeddings later
            self.char2idx = {"<pad>": 0}
            self.dim = None  # unknown; will be embedding size in model

    def ipa_to_segment_features(self, ipa_str):
        """
        If panphon available:
            returns (N_segments, feat_dim) numpy array
        Else:
            returns list of char indices (to be embedded later)
        """
        if self.use_panphon:
            segs = []
            # attempt to split into segments using panphon's feature table
            for ch in ipa_str:
                # panphon wants segments; using single character segmentation is OK
                try:
                    feats = self.ft.segment_features(ch)
                    if feats is not None:
                        # segment_features returns a dict-like; transform to vector 0/1/-1
                        vec = self.ft.fts_types_vector(feats)
                        # fts_types_vector returns array-like; convert to float
                        segs.append(np.asarray(vec, dtype=np.float32))
                    else:
                        # unknown char -> zero vector
                        segs.append(np.zeros((self.dim,), dtype=np.float32))
                except Exception:
                    segs.append(np.zeros((self.dim,), dtype=np.float32))
            if len(segs) == 0:
                return np.zeros((1, self.dim), dtype=np.float32)
            return np.vstack(segs)
        else:
            # fallback: return char indices array
            ids = []
            for ch in ipa_str:
                if ch not in self.char2idx:
                    self.char2idx[ch] = len(self.char2idx)
                ids.append(self.char2idx[ch])
            if len(ids) == 0:
                ids = [0]
            return np.array(ids, dtype=np.int64)


# -----------------------
# Dataset
# -----------------------
class Codec2DBDataset(Dataset):
    def __init__(self, db_path, articulator: IPAArticulator, max_phonemes=64):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self.cur.execute("SELECT id, phonemes, features_file, codec2_file FROM words")
        self.rows = self.cur.fetchall()
        self.art = articulator
        self.max_phonemes = max_phonemes

    def __len__(self):
        return len(self.rows)

    def load_codec2_bits(self, path: str):
        raw = Path(path).read_bytes()
        bitstr = ''.join(f"{byte:08b}" for byte in raw)
        n_frames = len(bitstr) // CODEC2_BITS
        bitstr = bitstr[: n_frames * CODEC2_BITS]
        arr = np.array([int(b) for b in bitstr], dtype=np.float32).reshape(n_frames, CODEC2_BITS)
        return arr

    def __getitem__(self, idx):
        _id, phonemes, feat_path, codec2_path = self.rows[idx]
        feat_path = Path(feat_path)
        mel, f0, rms = load_npz_features(feat_path)   # T_mel, mel_dim
        mel_u, f0_u, rms_u = upsample_features(mel, f0, rms, factor=UPSAMPLE_FACTOR)  # T_c2, ...

        # build per-frame feature vector: [mel..., f0, rms] -> (T_c2, mel_dim+2)
        feats_u = np.concatenate([mel_u, f0_u[:, None], rms_u[:, None]], axis=1).astype(np.float32)

        # codec2 target bits per codec2 frame
        bits = self.load_codec2_bits(codec2_path)    # (T_c2_bits, CODEC2_BITS)

        # align lengths (trim to shortest)
        T = min(feats_u.shape[0], bits.shape[0])
        feats_u = feats_u[:T]
        bits = bits[:T]

        # IPA articulatory representation
        ipa_repr = self.art.ipa_to_segment_features(phonemes)

        # If panphon used: ipa_repr is (Nseg, feat_dim). We'll tile/repeat to match T_mel then upsample to T_c2
        # If fallback: ipa_repr is a list of char ids; we'll return them directly.
        return ipa_repr, feats_u, bits

# -----------------------
# Collate: we need to convert ipa_repr to fixed-length per-frame conditioning
# -----------------------
def collate_fn(batch, articulator: IPAArticulator):
    """
    batch: list of (ipa_repr, feats(T, D), bits(T, B))
    For panphon:
        ipa_repr: (Nseg, seg_dim) -> we repeat to match original mel frames approx, then upsample similarly
    For fallback:
        ipa_repr: array of char ids -> pad to max chars and return indices
    """
    ipa_list, feats_list, bits_list = zip(*batch)

    # pad time dimension across feats/bits
    maxT = max(f.shape[0] for f in feats_list)
    feat_dim = feats_list[0].shape[1]
    bit_dim = bits_list[0].shape[1]

    feat_batch = torch.zeros((len(batch), maxT, feat_dim), dtype=torch.float32)
    bit_batch = torch.zeros((len(batch), maxT, bit_dim), dtype=torch.float32)

    for i, (f, b) in enumerate(zip(feats_list, bits_list)):
        feat_batch[i, : f.shape[0], :] = torch.from_numpy(f)
        bit_batch[i, : b.shape[0], :] = torch.from_numpy(b)

    # Build ipa conditioning per frame
    if articulator.use_panphon:
        seg_dim = ipa_list[0].shape[1]
        ipa_frame_batch = torch.zeros((len(batch), maxT, seg_dim), dtype=torch.float32)
        for i, (seg_feats, f) in enumerate(zip(ipa_list, feats_list)):
            # seg_feats: (Nseg, seg_dim)
            Nseg = seg_feats.shape[0]
            Tmel = f.shape[0] // UPSAMPLE_FACTOR  # orig mel length (approx)
            if Tmel < 1: Tmel = 1
            # repeat/tiling seg_feats to mel length
            rep = math.ceil(Tmel / Nseg)
            seg_rep = np.tile(seg_feats, (rep, 1))[:Tmel, :]  # (Tmel, seg_dim)
            # upsample seg_rep to T_c2 using linear interp over segment index
            src_idx = np.linspace(0.0, 1.0, num=seg_rep.shape[0])
            tgt_idx = np.linspace(0.0, 1.0, num= f.shape[0])
            seg_up = np.empty((f.shape[0], seg_dim), dtype=np.float32)
            for d in range(seg_dim):
                seg_up[:, d] = np.interp(tgt_idx, src_idx, seg_rep[:, d])
            ipa_frame_batch[i, : f.shape[0], :] = torch.from_numpy(seg_up)
        return ipa_frame_batch.to(DEVICE), feat_batch.to(DEVICE), bit_batch.to(DEVICE)
    else:
        # fallback: we returned arrays of char ids. Pad to max_char_len and return embeddings indices
        max_chars = max(len(a) for a in ipa_list)
        char_batch = torch.zeros((len(batch), max_chars), dtype=torch.long)
        for i, ids in enumerate(ipa_list):
            ids = np.asarray(ids, dtype=np.int64)
            char_batch[i, : len(ids)] = torch.from_numpy(ids)
        return char_batch.to(DEVICE), feat_batch.to(DEVICE), bit_batch.to(DEVICE)

# -----------------------
# Model
# -----------------------
class Codec2PredictorPanphon(nn.Module):
    def __init__(self, feat_dim, seg_feat_dim=None, char_vocab=None, char_emb_dim=64, model_dim=256, n_heads=4, n_layers=4):
        """
        If seg_feat_dim is provided -> panphon path: per-frame articulatory features are concatenated to input features.
        Else -> fallback path: we have char_vocab size and will embed characters and average them into per-frame conditioning vector.
        """
        super().__init__()
        self.use_panphon = seg_feat_dim is not None
        if self.use_panphon:
            self.seg_dim = seg_feat_dim
            self.input_proj = nn.Linear(feat_dim + seg_feat_dim, model_dim)
        else:
            # fallback: build char embedding
            self.char_emb = nn.Embedding(char_vocab, char_emb_dim, padding_idx=0)
            # we'll compute char embedding average then repeat per frame
            self.input_proj = nn.Linear(feat_dim + char_emb_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out = nn.Linear(model_dim, CODEC2_BITS)

    def forward(self, cond, feats):
        """
        If panphon: cond: (B, T, seg_dim), feats: (B, T, feat_dim)
        Else: cond: (B, C) char ids -> compute embedding avg -> expand to (B, T, char_emb_dim), feats: (B,T,feat_dim)
        """
        if self.use_panphon:
            x = torch.cat([feats, cond], dim=-1)  # (B,T, feat+seg)
        else:
            # cond are char ids (B, C)
            emb = self.char_emb(cond)             # (B, C, E)
            emb_avg = emb.mean(dim=1)             # (B, E)
            B, T, _ = feats.shape
            emb_rep = emb_avg.unsqueeze(1).repeat(1, T, 1)  # (B, T, E)
            x = torch.cat([feats, emb_rep], dim=-1)

        x = self.input_proj(x)                    # (B,T,model_dim)
        x = self.encoder(x)                       # (B,T,model_dim)
        logits = self.out(x)                      # (B,T,CODEC2_BITS)
        return logits

# -----------------------
# Train function
# -----------------------
def train(args):
    articulator = IPAArticulator()
    ds = Codec2DBDataset(args.db, articulator)
    # collate needs access to articulator to know panphon vs fallback
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=lambda b: collate_fn(b, articulator))

    # inspect one batch to create model
    sample = next(iter(loader))
    if articulator.use_panphon:
        cond_sample, feats_sample, bits_sample = sample
        seg_dim = cond_sample.shape[-1]
        feat_dim = feats_sample.shape[-1]
        model = Codec2PredictorPanphon(feat_dim=feat_dim, seg_feat_dim=seg_dim).to(DEVICE)
    else:
        char_batch, feats_sample, bits_sample = sample
        char_vocab = len(articulator.char2idx)
        feat_dim = feats_sample.shape[-1]
        model = Codec2PredictorPanphon(feat_dim=feat_dim, seg_feat_dim=None, char_vocab=char_vocab).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            # batch is already moved to DEVICE in collate
            if articulator.use_panphon:
                cond, feats, bits = batch
            else:
                cond, feats, bits = batch  # cond = char ids
            logits = model(cond, feats)
            # trim logits and bits to same time (they already are)
            loss = loss_fn(logits, bits)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            steps += 1

        avg = total_loss / max(1, steps)
        print(f"Epoch {epoch}/{args.epochs} - avg loss: {avg:.6f}")
        ckpt_path = Path(args.out_dir) / f"ckpt_epoch_{epoch}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "articulator_panphon": articulator.use_panphon,
            "articulator_char2idx": getattr(articulator, "char2idx", None)
        }, str(ckpt_path))
    print("Training finished. Final model saved.")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="Path to sqlite DB (words table)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dir", default="./checkpoints")
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)
