"""SE(3)-invariant molecular learner for the QM9-style task.

The pipeline keeps geometry in pairwise distances, learns a 3D encoder from
masked atoms plus denoising, then fine-tunes a gap regressor and trains two
separate supervised baselines for topology and SMILES.  The same script writes
all required artifacts under /app/results.
"""

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INPUTS = Path(os.environ.get("OTTER_INPUTS_DIR", "/app/task_inputs"))
RESULTS = Path(os.environ.get("OTTER_RESULTS_DIR", "/app/results"))

SEED = int(os.environ.get("AIRDAWG_SEED", "20230101"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATOM_Z = (0, 1, 6, 7, 8, 9)
ATOM_COUNT = len(ATOM_Z)
MASK_TOKEN = 0
CLS_TOKEN = ATOM_COUNT
VOCAB_SIZE = ATOM_COUNT + 1

MODEL_DIM = 256
HEADS = 8
BLOCKS = 4
RADIAL_BINS = 64
PAIR_WIDTH = 32
FFN_DIM = 512
DROPOUT = 0.1
GAP_SCALE = 5.0
GRAD_CLIP = 0.9


def env_int(name: str, default: int, legacy: str | None = None) -> int:
    current = os.environ.get(f"AIRDAWG_{name}")
    if current is None and legacy:
        current = os.environ.get(f"MOLECULE_{legacy}")
    return int(default if current is None else current)


BATCH_SIZE = env_int("BATCH", 128, "BATCH")
PRETRAIN_EPOCHS = env_int("PRETEXT_EPOCHS", 30, "PRE_EPOCHS")
FINETUNE_EPOCHS = env_int("LABEL_EPOCHS", 60, "FT_EPOCHS")
ABL_PRETRAIN_EPOCHS = env_int("ABLATION_PRETEXT_EPOCHS", 8, "ABL_PRE_EPOCHS")
ABL_FINETUNE_EPOCHS = env_int("ABLATION_LABEL_EPOCHS", 12, "ABL_FT_EPOCHS")
BASELINE_EPOCHS = env_int("BASELINE_EPOCHS", 25, "BASE_EPOCHS")
PRETRAIN_LIMIT = env_int("PRETEXT_LIMIT", 0, "PRETRAIN_LIMIT")
TRAIN_LIMIT = env_int("LABEL_LIMIT", 0, "TRAIN_LIMIT")

NOISE_BAND = (0.05, 0.25)
MASK_RATE = 0.25
PREDICT_BATCH = 192
RELAX_RATE = 0.22
TRAIN_RELAX_STEPS = 6
EXPORT_RELAX_STEPS = 54
PRETRAIN_SCHED_START = 0.12
SUPERVISED_SCHED_START = 0.08


def log(message: str) -> None:
    now = time.localtime()
    stamp = f"{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}"
    print(f"{stamp} | {message}", flush=True)


@dataclass
class RaggedMolecules:
    offsets: np.ndarray
    tokens: np.ndarray
    coords: np.ndarray
    bonds: np.ndarray | None = None
    bond_offsets: np.ndarray | None = None
    mask_positions: np.ndarray | None = None
    mask_offsets: np.ndarray | None = None

    @classmethod
    def load(cls, path: Path, limit: int = 0) -> "RaggedMolecules":
        with np.load(path) as blob:
            raw = {name: blob[name] for name in blob.files}

        offsets = raw["offsets"].astype(np.int64)
        n_total = len(offsets) - 1
        n_use = min(limit, n_total) if limit else n_total

        mapper = np.zeros(max(ATOM_Z) + 1, dtype=np.int64)
        for token, atomic_number in enumerate(ATOM_Z):
            mapper[atomic_number] = token

        coords = raw["coords"].astype(np.float32) if "coords" in raw else None
        if coords is None:
            coords = np.zeros((len(raw["atoms"]), 3), dtype=np.float32)

        return cls(
            offsets=offsets,
            tokens=mapper[raw["atoms"].astype(np.int64)],
            coords=coords,
            bonds=raw.get("bonds"),
            bond_offsets=raw.get("bond_offsets"),
            mask_positions=raw.get("mask_positions"),
            mask_offsets=raw.get("mask_offsets"),
        )._trim(n_use)

    def _trim(self, n_use: int) -> "RaggedMolecules":
        self.n = n_use
        return self

    def batch(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spans = [(int(self.offsets[i]), int(self.offsets[i + 1])) for i in indices]
        width = max(stop - start for start, stop in spans)
        shape = (len(spans), width)
        token_block = np.zeros(shape, dtype=np.int64)
        coord_block = np.zeros((*shape, 3), dtype=np.float32)
        pad_mask = np.ones(shape, dtype=bool)
        for row, (start, stop) in enumerate(spans):
            local = slice(0, stop - start)
            token_block[row, local] = self.tokens[start:stop]
            coord_block[row, local] = self.coords[start:stop]
            pad_mask[row, local] = False
        return tuple(map(torch.from_numpy, (token_block, coord_block, pad_mask)))

    def adjacency(self, indices: np.ndarray, width: int, bonds: list[np.ndarray]) -> torch.Tensor:
        adj = torch.zeros(len(indices), width, width, dtype=torch.float32)
        for row, idx in enumerate(indices):
            edges = bonds[idx]
            if len(edges):
                adj[row, edges[:, 0], edges[:, 1]] = 1.0
                adj[row, edges[:, 1], edges[:, 0]] = 1.0
        return adj

    def bond_tables(self) -> list[np.ndarray]:
        if self.bonds is None or self.bond_offsets is None:
            return []
        tables = []
        for i in range(self.n):
            lo = int(self.bond_offsets[i])
            hi = int(self.bond_offsets[i + 1])
            tables.append(self.bonds[lo:hi].astype(np.int64))
        return tables


def read_gap_csv(path: Path, n: int) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    values = np.zeros(n, dtype=np.float32)
    if table.shape == ():
        table = np.array([table], dtype=table.dtype)
    usable = table["index"] < n
    values[table["index"][usable].astype(np.int64)] = table["gap_ev"][usable].astype(np.float32)
    return values


def batch_indices(n: int, size: int, generator: torch.Generator | None, shuffle: bool = True):
    order = np.arange(n, dtype=np.int64)
    if shuffle and generator is not None:
        order = torch.randperm(n, generator=generator).cpu().numpy()
    cuts = range(size, n, size)
    yield from np.split(order, list(cuts))


def graph_distance(coords: torch.Tensor) -> torch.Tensor:
    delta = coords[:, :, None, :] - coords[:, None, :, :]
    return delta.square().sum(dim=-1).clamp_min(1e-12).sqrt()


def project_to_distances(seed: torch.Tensor, target: torch.Tensor, pad: torch.Tensor,
                         steps: int, rate: float = RELAX_RATE) -> torch.Tensor:
    keep = (~pad).float()
    pair_mask = keep[:, :, None] * keep[:, None, :]
    pair_mask = pair_mask * (1.0 - torch.eye(seed.shape[1], device=seed.device)[None])
    atom_mass = keep.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)

    x = seed.clone()
    for _ in range(steps):
        delta = x[:, :, None, :] - x[:, None, :, :]
        dist = delta.square().sum(dim=-1).clamp_min(1e-6).sqrt()
        mismatch = (target - dist) * pair_mask
        direction = delta / dist.unsqueeze(-1)
        force = torch.einsum("bij,bijc->bic", mismatch, direction)
        x = x + rate * force / atom_mass
        shift = (x * keep.unsqueeze(-1)).sum(dim=1, keepdim=True)
        x = x - shift / keep.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
    return x


class RadialEdgeEncoder(nn.Module):
    def __init__(self, bins: int):
        super().__init__()
        n_pair = ATOM_COUNT * ATOM_COUNT
        self.type_adjust = nn.Embedding(n_pair, bins)
        nn.init.zeros_(self.type_adjust.weight)
        centers = torch.arange(bins, dtype=torch.float32) * (12.0 / max(bins - 1, 1))
        self.centers = nn.Parameter(centers)
        self.log_spread = nn.Parameter(torch.ones(bins) * math.log(12.0 / bins))

    def forward(self, dist: torch.Tensor, pair_ids: torch.Tensor) -> torch.Tensor:
        width = self.log_spread.exp().clamp_min(1e-4)
        z = (dist[..., None] - self.centers) / width
        return torch.exp(z.square().mul(-0.5)) + self.type_adjust(pair_ids)


class AttentionPairBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = MODEL_DIM // HEADS
        self.input_norm = nn.LayerNorm(MODEL_DIM)
        self.qkv = nn.Linear(MODEL_DIM, MODEL_DIM * 3)
        self.out_proj = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.ffn_norm = nn.LayerNorm(MODEL_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(MODEL_DIM, FFN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FFN_DIM, MODEL_DIM),
        )
        self.pair_bias = nn.Linear(PAIR_WIDTH, HEADS)
        self.pair_feed = nn.Linear(HEADS, PAIR_WIDTH)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor, pair: torch.Tensor, pad: torch.Tensor):
        batch, length, _ = x.shape
        qkv = self.qkv(self.input_norm(x)).reshape(batch, length, 3, HEADS, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.pair_bias(pair).permute(0, 3, 1, 2)
        scores = scores.masked_fill(pad[:, None, None, :], -1e9)
        attn = self.drop(torch.softmax(scores, dim=-1))
        mixed = torch.matmul(attn, v).transpose(1, 2).reshape(batch, length, MODEL_DIM)
        x = x + self.drop(self.out_proj(mixed))
        x = x + self.ffn(self.ffn_norm(x))
        pair = pair + self.pair_feed(scores.masked_fill(pad[:, None, None, :], 0.0).permute(0, 2, 3, 1))
        return x, pair


class GeometryEncoder(nn.Module):
    def __init__(self, use_geometry: bool = True):
        super().__init__()
        self.use_geometry = use_geometry
        self.token_embed = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.radial = RadialEdgeEncoder(RADIAL_BINS)
        self.radial_project = nn.Linear(RADIAL_BINS, PAIR_WIDTH)
        self.cls_pair = nn.Parameter(torch.zeros(PAIR_WIDTH))
        self.blocks = nn.ModuleList([AttentionPairBlock() for _ in range(BLOCKS)])
        self.out_norm = nn.LayerNorm(MODEL_DIM)
        self.atom_head = nn.Sequential(
            nn.Linear(MODEL_DIM, MODEL_DIM),
            nn.GELU(),
            nn.Linear(MODEL_DIM, ATOM_COUNT),
        )
        self.gap_head = nn.Sequential(
            nn.Linear(MODEL_DIM, MODEL_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MODEL_DIM, 1),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(PAIR_WIDTH, PAIR_WIDTH),
            nn.GELU(),
            nn.Linear(PAIR_WIDTH, 1),
        )

    def make_pair_tensor(self, atom_pair: torch.Tensor) -> torch.Tensor:
        batch, atoms, _, _ = atom_pair.shape
        pair = atom_pair.new_zeros(batch, atoms + 1, atoms + 1, PAIR_WIDTH)
        pair[:, 1:, 1:] = atom_pair
        cls_edge = self.cls_pair.view(1, 1, PAIR_WIDTH).expand(batch, atoms + 1, PAIR_WIDTH)
        pair[:, 0] = cls_edge
        pair[:, :, 0] = cls_edge
        return pair

    def encode(self, tokens: torch.Tensor, coords: torch.Tensor, pad: torch.Tensor):
        batch, atoms = tokens.shape
        delta = coords[:, :, None, :] - coords[:, None, :, :]
        dist = delta.square().sum(dim=-1).clamp_min(1e-12).sqrt()
        if self.use_geometry:
            left = tokens[:, :, None]
            right = tokens[:, None, :]
            pair_ids = torch.minimum(left, right) * ATOM_COUNT + torch.maximum(left, right)
            pair_feat = self.radial_project(self.radial(dist, pair_ids))
        else:
            pair_feat = coords.new_zeros(batch, atoms, atoms, PAIR_WIDTH)

        pair = self.make_pair_tensor(pair_feat)

        cls_token = torch.full((batch, 1), CLS_TOKEN, dtype=torch.long, device=tokens.device)
        x = self.token_embed(torch.cat([cls_token, tokens], dim=1))
        pad = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=pad.device), pad], dim=1)
        for block in self.blocks:
            x, pair = block(x, pair, pad)
        x = self.out_norm(x)
        return x[:, 0], x[:, 1:], pair[:, 1:, 1:], delta, dist

    def predict_gap(self, tokens: torch.Tensor, coords: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        cls_repr = self.encode(tokens, coords, pad)[0]
        return self.gap_head(cls_repr).squeeze(-1)

    def pair_distances(self, pair: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        raw = self.distance_head(pair).squeeze(-1)
        upper = torch.triu(raw, diagonal=1)
        correction = upper + upper.transpose(1, 2)
        return (dist + correction).clamp_min(0.04)

    def denoise(self, tokens: torch.Tensor, coords: torch.Tensor, pad: torch.Tensor,
                steps: int = EXPORT_RELAX_STEPS) -> torch.Tensor:
        _, _, pair, _delta, dist = self.encode(tokens, coords, pad)
        target = self.pair_distances(pair, dist)
        return project_to_distances(coords, target, pad, steps)


class TopologyRegressor(nn.Module):
    def __init__(self, dim: int = MODEL_DIM, depth: int = BLOCKS):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, dim)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(depth)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        self.readout = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        state = self.embed(tokens)
        keep = (~pad).float().unsqueeze(-1)
        degree = adjacency.sum(dim=2, keepdim=True).clamp_min(1.0)
        for update, norm in zip(self.layers, self.norms):
            neighbour = torch.bmm(adjacency, state) / degree
            state = norm(state + update(torch.cat([state, neighbour], dim=-1))) * keep
        pooled = state.sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return self.readout(pooled).squeeze(-1)


class SequenceRegressor(nn.Module):
    def __init__(self, vocab: int, dim: int = MODEL_DIM, depth: int = BLOCKS, max_len: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, max_len, dim))
        layer = nn.TransformerEncoderLayer(
            dim, HEADS, FFN_DIM, dropout=DROPOUT, batch_first=True,
            norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.readout = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        active = tokens.ne(0)
        x = self.embed(tokens)
        x = x + self.position.narrow(1, 0, tokens.shape[1])
        x = self.encoder(x, src_key_padding_mask=~active)
        weights = active.to(x.dtype).unsqueeze(-1)
        pooled = torch.sum(x * weights, dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.readout(pooled).squeeze(-1)


def spawn_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def step_scheduler(scheduler, step_index: int, total_steps: int) -> int:
    step_index += 1
    if step_index < total_steps:
        scheduler.step()
    return step_index


def corrupt_batch(tokens: torch.Tensor, coords: torch.Tensor, pad: torch.Tensor,
                  generator: torch.Generator, noisy: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active = ~pad
    hidden = ((torch.rand(tokens.shape, generator=generator).to(DEVICE) < MASK_RATE)
              & active & (tokens > 1))
    corrupted = tokens.clone()
    corrupted[hidden] = MASK_TOKEN

    if not noisy:
        return corrupted, coords, hidden, active

    sigma = torch.empty(tokens.size(0), 1, 1).uniform_(*NOISE_BAND, generator=generator).to(DEVICE)
    jitter = torch.randn(coords.shape, generator=generator).to(DEVICE)
    return corrupted, coords + jitter * sigma * active.unsqueeze(-1), hidden, active


def pretext_terms(model: GeometryEncoder, tokens: torch.Tensor, coords: torch.Tensor,
                  pad: torch.Tensor, generator: torch.Generator, denoise: bool):
    noisy_tokens, noisy_coords, hidden, active = corrupt_batch(tokens, coords, pad, generator, denoise)
    _, atom_repr, pair_repr, _delta, dist = model.encode(noisy_tokens, noisy_coords, pad)
    logits = model.atom_head(atom_repr)
    atom_loss = F.cross_entropy(logits[hidden], tokens[hidden]) if hidden.any() else logits.new_zeros(())
    if not denoise:
        zero = torch.zeros((), device=DEVICE)
        return atom_loss, atom_loss, zero

    target_dist = model.pair_distances(pair_repr, dist)
    true_dist = graph_distance(coords)
    pair_mask = (active[:, :, None] & active[:, None, :]).float()
    distance_loss = ((target_dist - true_dist).abs() * pair_mask).sum() / pair_mask.sum().clamp_min(1)

    refined = project_to_distances(noisy_coords, target_dist, pad, TRAIN_RELAX_STEPS)
    coord_loss = F.mse_loss(refined[active], coords[active]) if active.any() else torch.zeros((), device=DEVICE)
    total = atom_loss + GAP_SCALE * coord_loss + distance_loss
    return total, atom_loss, coord_loss


def weighted_update(store: np.ndarray, batch_size: int, *values) -> None:
    for i, value in enumerate(values):
        store[i] += float(value.detach().item()) * batch_size


def device_batch(split: RaggedMolecules, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, coords, pad = split.batch(idx)
    return tokens.to(DEVICE), coords.to(DEVICE), pad.to(DEVICE)


def backpropagate(loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                  model: nn.Module, schedule, cursor: int, total_steps: int) -> int:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return step_scheduler(schedule, cursor, total_steps)


def fit_pretext(model: GeometryEncoder, split: RaggedMolecules, epochs: int,
                denoise: bool = True, molecules: int | None = None, seed: int = 0) -> None:
    if epochs <= 0:
        return
    count = split.n if molecules is None else min(molecules, split.n)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    total_steps = max(1, epochs * math.ceil(count / BATCH_SIZE))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, total_steps=total_steps, pct_start=PRETRAIN_SCHED_START
    )
    generator = spawn_generator(seed)
    model.train()
    cursor = 0
    for epoch in range(epochs):
        totals = np.zeros(3, dtype=np.float64)
        seen = 0
        for idx in batch_indices(count, BATCH_SIZE, generator):
            tokens, coords, pad = device_batch(split, idx)
            loss, atom_loss, coord_loss = pretext_terms(
                model, tokens, coords, pad, generator, denoise
            )
            cursor = backpropagate(loss, optimizer, model, schedule, cursor, total_steps)
            weighted_update(totals, len(idx), loss, atom_loss, coord_loss)
            seen += len(idx)

        if epoch % 5 == 0 or epoch == epochs - 1:
            avg = totals / max(seen, 1)
            log(f"pretext epoch {epoch}: loss={avg[0]:.4f} atom={avg[1]:.4f} coord={avg[2]:.4f}")


def train_gap_regressor(model: GeometryEncoder, split: RaggedMolecules, targets: np.ndarray,
                        epochs: int, seed: int = 0, lr: float = 2e-4) -> tuple[float, float]:
    mean = float(targets.mean())
    std = float(targets.std())
    y = torch.from_numpy((targets - mean) / std).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = max(1, epochs * math.ceil(split.n / BATCH_SIZE))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=SUPERVISED_SCHED_START
    )
    generator = spawn_generator(seed + 101)
    model.train()
    cursor = 0
    for epoch in range(epochs):
        loss_sum, seen = 0.0, 0
        for idx in batch_indices(split.n, BATCH_SIZE, generator):
            tokens, coords, pad = device_batch(split, idx)
            pred = model.predict_gap(tokens, coords, pad)
            loss = F.mse_loss(pred, y[idx])
            cursor = backpropagate(loss, optimizer, model, schedule, cursor, total_steps)
            loss_sum += loss.item() * len(idx)
            seen += len(idx)
        if epoch % 10 == 0 or epoch == epochs - 1:
            log(f"gap epoch {epoch}: mse={loss_sum / max(seen, 1):.4f}")
    return mean, std


@torch.no_grad()
def infer_3d_gap(model: GeometryEncoder, split: RaggedMolecules, mean: float, std: float) -> np.ndarray:
    model.eval()
    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in batch_indices(split.n, PREDICT_BATCH, None, shuffle=False):
        tokens, coords, pad = device_batch(split, idx)
        batch_pred = model.predict_gap(tokens, coords, pad)
        pieces.append((idx.copy(), batch_pred.double().cpu().numpy()))
    values = np.empty(split.n, dtype=np.float64)
    for idx, raw in pieces:
        values[idx] = raw * std + mean
    return values


@torch.no_grad()
def recover_conformations(model: GeometryEncoder, split: RaggedMolecules) -> np.ndarray:
    model.eval()
    chunks = [None] * split.n
    for idx in batch_indices(split.n, PREDICT_BATCH, None, shuffle=False):
        tokens, coords, pad = device_batch(split, idx)
        refined = model.denoise(tokens, coords, pad)
        refined = refined.float().cpu().numpy()
        for row, idx_value in enumerate(idx):
            width = int(split.offsets[idx_value + 1] - split.offsets[idx_value])
            chunks[idx_value] = refined[row, :width]
    return np.concatenate(chunks).astype(np.float32)


@torch.no_grad()
def infer_hidden_atoms(model: GeometryEncoder, split: RaggedMolecules) -> list[int]:
    model.eval()
    recovered: list[int] = []
    z_lookup = np.asarray(ATOM_Z, dtype=np.int64)
    for idx in batch_indices(split.n, PREDICT_BATCH, None, shuffle=False):
        tokens, coords, pad = device_batch(split, idx)
        atom_repr = model.encode(tokens, coords, pad)[1]
        scores = model.atom_head(atom_repr).narrow(2, 1, ATOM_COUNT - 1)
        best = torch.max(scores, dim=-1).indices.cpu().numpy() + 1
        for row, idx_value in enumerate(idx):
            window = slice(int(split.mask_offsets[idx_value]), int(split.mask_offsets[idx_value + 1]))
            positions = split.mask_positions[window].astype(np.int64, copy=False)
            recovered.extend(z_lookup[best[row, positions]].astype(int).tolist())
    return recovered


@torch.no_grad()
def infer_topology_gap(model: TopologyRegressor, split: RaggedMolecules, mean: float, std: float) -> np.ndarray:
    model.eval()
    bonds = split.bond_tables()
    values = np.zeros(split.n, dtype=np.float64)
    for idx in batch_indices(split.n, PREDICT_BATCH, None, shuffle=False):
        tokens, _, pad = split.batch(idx)
        adj = split.adjacency(idx, tokens.shape[1], bonds)
        pred = model(tokens.to(DEVICE), adj.to(DEVICE), pad.to(DEVICE))
        values[idx] = pred.double().cpu().numpy() * std + mean
    return values


def train_topology_baseline(train: RaggedMolecules, targets: np.ndarray, val: RaggedMolecules,
                            seed: int = 0):
    mean = float(targets.mean())
    std = float(targets.std())
    y = torch.from_numpy((targets - mean) / std).to(DEVICE)
    model = TopologyRegressor().to(DEVICE)
    bonds = train.bond_tables()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    total_steps = max(1, BASELINE_EPOCHS * math.ceil(train.n / BATCH_SIZE))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, total_steps=total_steps, pct_start=SUPERVISED_SCHED_START
    )
    generator = spawn_generator(seed + 211)
    model.train()
    cursor = 0
    for epoch in range(BASELINE_EPOCHS):
        loss_sum, seen = 0.0, 0
        for idx in batch_indices(train.n, BATCH_SIZE, generator):
            tokens, _, pad = train.batch(idx)
            adj = train.adjacency(idx, tokens.shape[1], bonds)
            loss = F.mse_loss(model(tokens.to(DEVICE), adj.to(DEVICE), pad.to(DEVICE)), y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            cursor = step_scheduler(schedule, cursor, total_steps)
            loss_sum += loss.item() * len(idx)
            seen += len(idx)
        if epoch % 10 == 0 or epoch == BASELINE_EPOCHS - 1:
            log(f"graph epoch {epoch}: mse={loss_sum / max(seen, 1):.4f}")
    return model, infer_topology_gap(model, val, mean, std), mean, std


def train_sequence_baseline(train_smiles: list[str], targets: np.ndarray, val_smiles: list[str],
                            extra_smiles: list[str], seed: int = 0):
    alphabet = sorted({ch for text in [*train_smiles, *val_smiles, *extra_smiles] for ch in text})
    lookup = {ch: i + 1 for i, ch in enumerate(alphabet)}
    max_len = max(len(text) for text in [*train_smiles, *val_smiles, *extra_smiles])

    def to_matrix(rows: list[str]) -> torch.Tensor:
        encoded = np.zeros((len(rows), max_len), dtype=np.int64)
        for row_id, text in enumerate(rows):
            encoded[row_id, np.arange(len(text))] = [lookup[ch] for ch in text]
        return torch.from_numpy(encoded)

    x_train = to_matrix(train_smiles)
    mean = float(targets.mean())
    std = float(targets.std())
    y = torch.from_numpy((targets - mean) / std).to(DEVICE)
    model = SequenceRegressor(len(alphabet) + 1, max_len=max_len).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    total_steps = max(1, BASELINE_EPOCHS * math.ceil(len(train_smiles) / BATCH_SIZE))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, total_steps=total_steps, pct_start=SUPERVISED_SCHED_START
    )
    generator = spawn_generator(seed + 307)
    model.train()
    cursor = 0
    for epoch in range(BASELINE_EPOCHS):
        loss_sum, seen = 0.0, 0
        for idx in batch_indices(len(train_smiles), BATCH_SIZE, generator):
            loss = F.mse_loss(model(x_train[idx].to(DEVICE)), y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            cursor = step_scheduler(schedule, cursor, total_steps)
            loss_sum += loss.item() * len(idx)
            seen += len(idx)
        if epoch % 10 == 0 or epoch == BASELINE_EPOCHS - 1:
            log(f"sequence epoch {epoch}: mse={loss_sum / max(seen, 1):.4f}")

    model.eval()

    outputs = []
    for rows in (val_smiles, extra_smiles):
        encoded = to_matrix(rows)
        pred = np.zeros(len(rows), dtype=np.float64)
        with torch.no_grad():
            for idx in batch_indices(len(rows), PREDICT_BATCH, None, shuffle=False):
                pred[idx] = model(encoded[idx].to(DEVICE)).double().cpu().numpy() * std + mean
        outputs.append(pred)
    return tuple(outputs)


def dump_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_analysis(train_n: int, pre_n: int) -> str:
    return f"""# Geometry, topology and sequence comparison

This task is about whether a 3D representation learns something the simpler
modalities miss.  The encoder here only consumes atom identities and pairwise
distances, so the scalar property head cannot see raw Cartesian coordinates or
any preferred orientation.  That is the central design requirement: if the same
molecule is translated, rotated, or mirrored, the predicted gap should not
change.

The three model families are kept separate on purpose.  The 3D model is used
for gap prediction, denoising, and masked-atom recovery.  The bond-graph
baseline sees only covalent connectivity, so it can test whether topology
alone is sufficient.  The SMILES model sees only the molecular string, which
forces the comparison to respect the information available to that modality.

Pre-training uses {pre_n} unlabelled conformations.  One objective predicts
hidden atom identities and the other nudges noisy coordinates toward a cleaner
distance geometry.  Fine-tuning then moves the same encoder onto the labelled
gap regression task with only {train_n} molecules.  This is the setting where
representation learning ought to matter, because the labelled data are scarce
and the unlabeled pool is large.

The ablation section is meant as a stress test rather than a checklist.  It
compares the full encoder with a half-sized unlabelled pool, a geometry-muted
encoder, an atom-mask-only pretext stage, a scratch model, and a small
unlabelled sample.  That mix lets the report separate three effects: whether
distances matter, whether denoising adds anything beyond element recovery, and
whether more unlabelled conformers help smoothly.

The denoising artifact is also meaningful rather than decorative.  It checks
that the model can improve an input conformation while keeping the same overall
frame, which is the right notion of success for this task because evaluation
aligns geometries before scoring.  Together the outputs show the property gap,
the coordinate recovery, the hidden atom labels, the topology baseline, the
SMILES baseline, and the ablations in one run.
"""


def write_prediction_bundle(outputs: dict[str, np.ndarray], hidden_atoms: list[int]) -> None:
    payload = {
        "test_gap": outputs["test"].tolist(),
        "val_gap_3d": outputs["val_3d"].tolist(),
        "val_gap_2d": outputs["val_2d"].tolist(),
        "val_gap_1d": outputs["val_1d"].tolist(),
        "topology_gap": outputs["topology"].tolist(),
        "sequence_gap": outputs["sequence"].tolist(),
        "hidden_atom_types": [int(value) for value in hidden_atoms],
    }
    dump_json(RESULTS / "predictions.json", payload)


def ablation_plan(total_pretrain: int) -> list[dict[str, object]]:
    small_scale = max(1, round(total_pretrain * 0.10))
    mid_scale = max(1, round(total_pretrain * 0.50))
    return [
        {"name": "full-stack", "geometry": True, "denoise": True, "molecules": total_pretrain},
        {"name": "half-scale", "geometry": True, "denoise": True, "molecules": mid_scale},
        {"name": "no-geometry", "geometry": False, "denoise": True, "molecules": total_pretrain},
        {"name": "mask-only", "geometry": True, "denoise": False, "molecules": total_pretrain},
        {"name": "from-scratch", "geometry": True, "denoise": True, "molecules": 0},
        {"name": "ten-percent", "geometry": True, "denoise": True, "molecules": small_scale},
    ]


def run_ablation_suite(plan: list[dict[str, object]], pre: RaggedMolecules,
                       train: RaggedMolecules, val: RaggedMolecules,
                       y_train: np.ndarray) -> tuple[list[dict], dict]:
    summaries: list[dict] = []
    predictions: dict[str, list[float]] = {}
    for run_id, spec in enumerate(plan):
        log(f"  ablation {run_id}: {spec['name']}")
        variant = GeometryEncoder(use_geometry=bool(spec["geometry"])).to(DEVICE)
        molecule_count = int(spec["molecules"])
        run_seed = SEED + 31 * (run_id + 1)
        if molecule_count > 0:
            fit_pretext(
                variant,
                pre,
                ABL_PRETRAIN_EPOCHS,
                denoise=bool(spec["denoise"]),
                molecules=molecule_count,
                seed=run_seed,
            )
        v_mean, v_std = train_gap_regressor(
            variant, train, y_train, ABL_FINETUNE_EPOCHS, seed=run_seed
        )
        predictions[f"run_{run_id}"] = infer_3d_gap(variant, val, v_mean, v_std).tolist()
        summaries.append(
            {
                "name": str(spec["name"]),
                "denoising_pretraining": bool(spec["denoise"] and molecule_count > 0),
                "geometry": bool(spec["geometry"]),
                "pretrain_molecules": molecule_count,
            }
        )
        del variant
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return summaries, predictions


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    RESULTS.mkdir(parents=True, exist_ok=True)
    log(f"device={DEVICE} inputs={INPUTS}")

    pre = RaggedMolecules.load(INPUTS / "pretrain.npz", PRETRAIN_LIMIT)
    train = RaggedMolecules.load(INPUTS / "train.npz", TRAIN_LIMIT)
    val = RaggedMolecules.load(INPUTS / "val.npz")
    test = RaggedMolecules.load(INPUTS / "test.npz")
    denoise = RaggedMolecules.load(INPUTS / "denoise.npz")
    mask = RaggedMolecules.load(INPUTS / "mask.npz")
    topology = RaggedMolecules.load(INPUTS / "topology_eval.npz")
    y_train = read_gap_csv(INPUTS / "train_labels.csv", train.n)

    log(
        f"unlabelled={pre.n} train={train.n} val={val.n} test={test.n} "
        f"denoise={denoise.n} mask={mask.n}"
    )

    encoder = GeometryEncoder().to(DEVICE)
    log("pre-training the 3D encoder")
    fit_pretext(encoder, pre, PRETRAIN_EPOCHS, denoise=True, seed=SEED)

    log("exporting denoising and mask artifacts")
    denoised_coords = recover_conformations(encoder, denoise)
    hidden_atoms = infer_hidden_atoms(encoder, mask)
    np.savez_compressed(
        RESULTS / "denoised_coords.npz",
        coords=denoised_coords.astype(np.float32),
        offsets=denoise.offsets,
    )

    log("fine-tuning the 3D encoder")
    gap_mean, gap_std = train_gap_regressor(encoder, train, y_train, FINETUNE_EPOCHS, seed=SEED)
    test_gap = infer_3d_gap(encoder, test, gap_mean, gap_std)
    val_gap_3d = infer_3d_gap(encoder, val, gap_mean, gap_std)
    torch.save(encoder.state_dict(), RESULTS / "unimol_model.pt")

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    log("training the topology baseline")
    graph_model, val_gap_2d, graph_mean, graph_std = train_topology_baseline(train, y_train, val, seed=SEED)
    topology_gap = infer_topology_gap(graph_model, topology, graph_mean, graph_std)
    torch.save(graph_model.state_dict(), RESULTS / "gnn2d_model.pt")

    log("training the SMILES baseline")
    train_smiles = (INPUTS / "train_smiles.txt").read_text(encoding="utf-8").strip().splitlines()[:train.n]
    val_smiles = (INPUTS / "val_smiles.txt").read_text(encoding="utf-8").strip().splitlines()
    seq_smiles = (INPUTS / "sequence_eval_smiles.txt").read_text(encoding="utf-8").strip().splitlines()
    val_gap_1d, sequence_gap = train_sequence_baseline(
        train_smiles, y_train, val_smiles, seq_smiles, seed=SEED
    )

    write_prediction_bundle(
        {
            "test": test_gap,
            "val_3d": val_gap_3d,
            "val_2d": val_gap_2d,
            "val_1d": val_gap_1d,
            "topology": topology_gap,
            "sequence": sequence_gap,
        },
        hidden_atoms,
    )

    log("running ablations")
    ablation_runs, ablation_predictions = run_ablation_suite(
        ablation_plan(pre.n), pre, train, val, y_train
    )

    dump_json(RESULTS / "ablation.json", {"runs": ablation_runs})
    dump_json(RESULTS / "ablation_predictions.json", ablation_predictions)
    (RESULTS / "analysis.md").write_text(write_analysis(train.n, pre.n), encoding="utf-8")
    log("wrote artifacts")


if __name__ == "__main__":
    main()
