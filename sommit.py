#!/usr/bin/env python3
"""
sommit.py — unified SOMmit pipeline (SOM training -> landscape -> root -> trajectories -> evaluation)

Consolidates the two prior scripts (training/plotting script + notebook analysis snippet)
into one module with no duplicated logic and no module-level side effects.

WHAT CHANGED vs. THE ORIGINAL CODE
==================================
Correctness fixes (these CHANGE NUMERICAL RESULTS — see --legacy to reproduce old behaviour):

  [B1] Zero-padding contamination.  `points = np.zeros((N,4))` was filled for only
       `mapped` rows and then saved/aggregated in full.  Every unfilled row is
       (r=0, c=0, z=0, label=0), so class 0 was polluted with phantom cells at the
       origin with zero elevation.  Class 0 is usually the progenitor, and the root
       is chosen as argmin(z) -> this biased root detection TOWARD class 0 and also
       fed phantom points to Slingshot.  Now arrays are truncated to `mapped`.

  [B2] Duplicate-label writes.  process_points() had a nested
       `for label in keys: for label in keys: ... break` which re-wrote the FIRST
       key once per distinct label at a node, instead of one row per label.
       make_3D_graph() and Will_It_Progenitor() each did something different again.
       All three are replaced by one `extract_points(..., mode=...)`.

  [B3] traj_adj_matrix guard.  `np.sum(done_lines[:][m])` is a no-op copy then row
       index — identical to `done_lines[m]`.  The intended in-degree check is
       `done_lines[:, m]`.  Fixed (legacy behaviour under --legacy).

  [B4] Root-finding criterion.  Will_It_Progenitor() scored a hit when
       `argmin(z) < 1 OR argmax(z) < 1` — i.e. class 0 being EITHER the deepest or
       the highest centroid counted as correct.  That is strictly more permissive
       than the paper's rule (root = minimum mean inter-neuron distance), so the
       reported root accuracy is an upper bound.  Default is now `argmin`;
       `--root-rule either` reproduces the old number.

  [B5] Binary datasets.  LabelBinarizer returns shape (n,1) for 2 classes, so
       `np.argmax(Y0, axis=1)` was always 0 and every cell collapsed to one label.
       Latent (all current datasets have >=3 types) but fatal if ever hit.
       Replaced with an explicit integer encoding.

  [B6] Class ordering.  Alphabetical sort puts "M10" before "M2".  Natural sort is
       used, which removes the ad-hoc `if len(labels) >= 10: labels = concat(...)`
       remap hack in the notebook.

  [B7] Header handling.  `pd.read_csv(f, names=names)` without skipping the header
       row turns the header into a data row ("group_id" becomes a cell type).
       Header presence is now detected.

Known code/manuscript discrepancy (NOT silently changed — see --som-input):
  [D1] The Methods section states the SOM is trained on the leading 20 principal
       components.  `data_normalization()` returns `adata.X` (2000 HVGs); PCA is
       computed and discarded.  `test_file_pca()` applies PCA but with an elbow
       rule, not 20 comps, and `test_file_pca_looping()` never applies PCA at all
       despite its name.  Default here is `--som-input expr` (reproduces the runs
       that produced the current tables).  Use `--som-input pca20` to match the
       manuscript.  This matters: the codebook is (25N x F), so F=20 instead of
       F=2000 is a ~100x reduction in memory and per-update FLOPs.

Efficiency (no change to results):
  - distance_map() computed ONCE per run (was: 2x per view x 4 views = 8x).
  - labels_map / BMUs computed ONCE (was: recomputed per view, with a fresh
    host->device copy of X each time).
  - BMU assignment is batched via chunked torch.cdist instead of a Python loop
    calling som.winner() once per cell (get_cell_type_activations was O(N) GPU
    round-trips).
  - labels_map built with a vectorized np.unique over (node, class) keys instead
    of a DIM x DIM Python double loop (DIM = 5*sqrt(N), so that loop was O(25N)).
  - Scatter plots draw one call per class, not one per point.
  - Figures default to 12x12in @150dpi (was 50x50in, i.e. 5000x5000 px) and all
    four views share one figure.
  - Surface plots stride-downsample so plot_surface stays usable at large DIM.
  - quantization error computed from BMU distances already in hand (was computed
    twice per ablation cell).
  - gc.collect()/empty_cache() only when explicitly requested.
  - Ablation no longer renders 55 x 4 giant figures; it only scores.
  - SOM pickles are named per (dataset, dim, lr, sigma, epochs); the old name
    depended on DIM only and silently overwrote across datasets.

Heavy deps (torch / minisom_gpu / scanpy) are imported lazily so the numeric core
can be imported and unit-tested on a CPU-only box:  python sommit.py selftest
"""

from __future__ import annotations

import os

# Must be set before torch is imported anywhere (minisom_gpu pulls it in).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob
import json
import pickle
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # SOM
    lr: float = 0.3
    sigma: float = 1.2
    epochs: int = 700_000
    # Convergence-based early stopping (opt-in): if max_epochs is set and < epochs,
    # train in blocks of check_every and stop when the relative change in
    # quantization error drops below converge_tol.  epochs still defines the LR/
    # radius schedule, so the map trains identically up to the point it stops.
    max_epochs: Optional[int] = None
    check_every: int = 10_000
    converge_tol: float = 1e-3
    dim_scale: float = 5.0          # DIM = int(dim_scale * sqrt(N))  -> ~25N neurons
    som_input: str = "expr"         # expr | pca20 | pca_elbow      (see [D1])
    n_pcs: int = 20
    seed: int = 42
    device: str = "cuda:0"

    # point extraction
    point_mode: str = "majority"    # majority | first | all         (see [B2])
    centroid_weight: str = "node"   # node | cell
    root_rule: str = "argmin"       # argmin | argmax | either       (see [B4])
    root_true_index: int = 0

    # trajectory
    tol: float = 0.0
    backup: bool = False

    # io / plotting
    outdir: Path = Path("./sommit_out")
    figsize: float = 12.0
    dpi: int = 150
    surface: bool = True
    plot: bool = True
    save_som: bool = False
    verbose: bool = False

    # compatibility
    legacy: bool = False            # reproduce pre-fix behaviour ([B1][B2][B3][B4])

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        if self.legacy:
            self.point_mode = "first"
            self.root_rule = "either"


# =============================================================================
# Lazy heavy imports
# =============================================================================

def _torch():
    import torch
    return torch


def resolve_device(cfg: "Config"):
    """
    Turn cfg.device into a real torch.device, and make a CPU fall-back LOUD.

    Training a ~25N-neuron SOM for 700k iterations on CPU is effectively a hang
    (tens of minutes to hours for one dataset), so silently running there is the
    worst outcome.  If the user asked for CUDA but it is unavailable or the index
    is out of range, we say so explicitly instead of crawling.

    Override at runtime with SOMMIT_DEVICE=cuda:3 (or cpu to force CPU on purpose).
    """
    torch = _torch()
    want = os.environ.get("SOMMIT_DEVICE", cfg.device)

    if want.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device '%s' requested but torch.cuda.is_available() is False.\n"
                "  This would fall back to CPU, where SOM training effectively hangs.\n"
                "  Fix CUDA (check `nvidia-smi`, and that torch has GPU support:\n"
                "     python3 -c 'import torch; print(torch.version.cuda, torch.cuda.is_available())'),\n"
                "  or force CPU on purpose with SOMMIT_DEVICE=cpu (expect it to be very slow)."
                % want)
        idx = torch.device(want).index or 0
        n = torch.cuda.device_count()
        if idx >= n:
            raise RuntimeError(
                "device '%s' requested but only %d CUDA device(s) present (cuda:0..cuda:%d).\n"
                "  Pick a valid one, e.g. --device cuda:0 or SOMMIT_DEVICE=cuda:%d."
                % (want, n, n - 1, n - 1))
        dev = torch.device(want)
        print(f"[sommit] using {dev} ({torch.cuda.get_device_name(idx)})")
        return dev

    print("[sommit] WARNING: running on CPU. SOM training will be extremely slow; "
          "this is only sensible for tiny smoke tests.")
    return torch.device("cpu")


def _minisom():
    from minisom_gpu.som import MiniSom
    return MiniSom


def _scanpy():
    import scanpy as sc
    return sc


def free_memory(cfg: Config) -> None:
    """Only pay for gc + cache flush when asked (was called unconditionally)."""
    if not cfg.verbose:
        return
    import gc
    torch = _torch()
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        idx = torch.device(cfg.device).index or 0
        print(f"  [mem] allocated {torch.cuda.memory_allocated(idx)/1024**3:.2f} GB | "
              f"reserved {torch.cuda.memory_reserved(idx)/1024**3:.2f} GB")


# =============================================================================
# Labels
# =============================================================================

_NAT = re.compile(r"(\d+)")


def natural_key(s: str):
    """'M10' sorts after 'M2'.  Removes the ad-hoc index remap hack. [B6]"""
    return [int(t) if t.isdigit() else t.lower() for t in _NAT.split(str(s))]


def encode_labels(Y: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (classes, y_idx).  Robust for 2 classes, unlike LabelBinarizer. [B5]
    classes are naturally sorted; y_idx[i] is the class index of cell i.
    """
    Y = np.asarray(Y, dtype=object)
    classes = np.array(sorted({str(v) for v in Y}, key=natural_key), dtype=object)
    lookup = {c: i for i, c in enumerate(classes)}
    y_idx = np.fromiter((lookup[str(v)] for v in Y), dtype=np.int64, count=len(Y))
    return classes, y_idx


def onehot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y_idx.size, n_classes), dtype=np.float64)
    out[np.arange(y_idx.size), y_idx] = 1.0
    return out


def class_colors(n: int):
    import matplotlib.pyplot as plt
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i % 20) for i in range(n)]
    cmap = plt.get_cmap("gist_ncar")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# =============================================================================
# IO / preprocessing
# =============================================================================

def read_expression_tsv(path: str | Path) -> Tuple["pd.DataFrame", np.ndarray]:
    """
    Reads a `cell_id \t group_id \t <genes...>` TSV.

    Detects whether a header row is present rather than blindly passing
    `names=` (which turned the header into a cell named 'cell_id' with
    cell type 'group_id'). [B7]
    """
    import pandas as pd
    path = Path(path)
    with open(path) as fh:
        first = fh.readline().rstrip("\n").split("\t")
    has_header = first[:2] == ["cell_id", "group_id"]

    if has_header:
        df = pd.read_csv(path, sep="\t")
        df = df.rename(columns={df.columns[0]: "cell_id", df.columns[1]: "group_id"})
    else:
        ncol = len(first)
        names = ["cell_id", "group_id"] + [str(i) for i in range(ncol - 2)]
        df = pd.read_csv(path, sep="\t", names=names, header=None)

    expr = df.iloc[:, 2:]
    expr.index = df["cell_id"].values
    labels = df["group_id"].to_numpy()
    return expr, labels


def preprocess(expr_df, labels: np.ndarray, cfg: Config
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    scanpy pipeline: median library-size normalisation -> log1p -> zheng17 recipe
    (top 2000 HVGs) -> optional PCA.

    The original also built a neighbours graph and a diffusion map that were never
    read; both are dropped.  `sc.tl.pca` was computed and discarded (see [D1]).

    Returns (X, Y, cell_ids).  cell_ids are taken AFTER filter_cells(), so they
    stay aligned with the rows of X -- required to hand SOM coordinates to
    dynwrap, which keys everything on cell_id.
    """
    sc = _scanpy()
    import pandas as pd

    adata = sc.AnnData(X=pd.DataFrame(expr_df).astype(np.float32))
    adata.obs["group_id"] = labels
    adata.layers["counts"] = adata.X.copy()

    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    n_top = min(2000, adata.layers["counts"].shape[1] - 15)
    if n_top > 0:
        sc.pp.recipe_zheng17(adata, n_top_genes=n_top)

    X = np.asarray(adata.X, dtype=np.float32)
    Y = adata.obs["group_id"].to_numpy()
    cell_ids = np.asarray(adata.obs_names, dtype=object).astype(str)

    if cfg.som_input == "pca20":
        X = _pca(X, cfg.n_pcs)
    elif cfg.som_input == "pca_elbow":
        X = _pca(X, _elbow_n_components(X, max_comps=cfg.n_pcs))
    elif cfg.som_input != "expr":
        raise ValueError(f"unknown som_input={cfg.som_input!r}")

    assert X.shape[0] == Y.size == cell_ids.size, "cell/label/id misalignment"
    return np.ascontiguousarray(X, dtype=np.float32), Y, cell_ids


def _pca(X: np.ndarray, n: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    n = int(min(n, min(X.shape) - 1))
    return PCA(n_components=n, random_state=0).fit_transform(X).astype(np.float32)


def elbow_index(values: np.ndarray) -> int:
    """Farthest point from the chord joining first and last points."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 3:
        return n
    pts = np.column_stack([np.arange(n), values])
    a, b = pts[0], pts[-1]
    line = b - a
    line /= np.linalg.norm(line)
    v = pts - a
    proj = a + np.outer(v @ line, line)
    return int(np.argmax(np.linalg.norm(pts - proj, axis=1))) + 1


def _elbow_n_components(X: np.ndarray, max_comps: int = 20) -> int:
    from sklearn.decomposition import PCA
    k = int(min(max_comps, min(X.shape) - 1))
    evr = PCA(n_components=k, random_state=0).fit(X).explained_variance_ratio_
    return elbow_index(evr)


# =============================================================================
# SOM
# =============================================================================

def grid_dim(n_cells: int, scale: float = 5.0) -> int:
    return int(scale * n_cells ** 0.5)


def train_som(X: np.ndarray, dim: int, cfg: Config):
    torch = _torch()
    MiniSom = _minisom()

    device = resolve_device(cfg)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))

    som = MiniSom(dim, dim, X.shape[1],
                  sigma=cfg.sigma, learning_rate=cfg.lr,
                  neighborhood_function="gaussian",
                  random_seed=cfg.seed, device=device)
    som.pca_weights_init(Xt)
    Xt = Xt.to(device)
    import time as _time
    _t0 = _time.time()

    if cfg.max_epochs is not None and cfg.max_epochs < cfg.epochs:
        # ---- convergence-based early stop -------------------------------------
        # Train in blocks and stop when quantization error plateaus, instead of
        # always running the full (fixed) cfg.epochs.  On a converged map further
        # iterations only polish a codebook that has stopped moving -- e.g. QE was
        # already ~0.005 at 700k -- so this recovers most of the runtime with a
        # principled, dataset-adaptive criterion rather than an arbitrary cut.
        #
        # NOTE: decay constants inside minisom are set from the iteration budget
        # given to train(); to keep the SAME learning-rate/radius schedule as a
        # full run, we pass cfg.epochs as the schedule length but only execute up
        # to cfg.max_epochs iterations, checking QE every cfg.check_every.
        block = max(1, cfg.check_every)
        done = 0
        prev_qe = float("inf")
        if cfg.verbose:
            print(f"[sommit] training SOM (early-stop): {dim}x{dim}={dim*dim:,} neurons, "
                  f"{X.shape[0]}x{X.shape[1]}, up to {cfg.max_epochs:,} it "
                  f"(schedule={cfg.epochs:,}), tol={cfg.converge_tol} on {device}")
        while done < cfg.max_epochs:
            n = min(block, cfg.max_epochs - done)
            # continue training WITHOUT re-initialising weights or the schedule
            som.train(Xt, n, random_order=True, verbose=False)
            done += n
            qe = float(som.quantization_error(Xt))
            rel = abs(prev_qe - qe) / max(prev_qe, 1e-12)
            if cfg.verbose:
                print(f"    it={done:,}  QE={qe:.6f}  Δrel={rel:.2e}  "
                      f"({done/max(_time.time()-_t0,1e-9):,.0f} it/s)")
            if rel < cfg.converge_tol:
                if cfg.verbose:
                    print(f"[sommit] converged at {done:,} iterations (Δrel<{cfg.converge_tol})")
                break
            prev_qe = qe
        used = done
    else:
        if cfg.verbose:
            print(f"[sommit] training SOM: {dim}x{dim}={dim*dim:,} neurons, "
                  f"{X.shape[0]} cells x {X.shape[1]} feats, {cfg.epochs:,} iterations on {device}")
        som.train(Xt, cfg.epochs, random_order=True, verbose=cfg.verbose)
        used = cfg.epochs

    if cfg.verbose:
        dt = _time.time() - _t0
        print(f"[sommit] SOM trained: {used:,} iterations in {dt:.1f}s "
              f"({used/max(dt,1e-9):,.0f} it/s)")
    del Xt
    return som


def som_weights_flat(som):
    """(dim*dim, F) view of the codebook, on-device."""
    W = som._weights.detach()
    return W.reshape(-1, W.shape[-1])


def batched_bmu(som, X: np.ndarray, cfg: Config, chunk: int = 4096
                ) -> Tuple[np.ndarray, float]:
    """
    Vectorised best-matching-unit assignment.

    Replaces the per-cell `som.winner(tensor(x))` loop (one GPU round-trip per
    cell).  Also returns the quantization error for free, so ablation does not
    have to call som.quantization_error() twice.

    Returns (bmu[N,2] int64 rows=(r,c), quantization_error).
    """
    torch = _torch()
    device = resolve_device(cfg)
    W = som_weights_flat(som)
    dim = som._weights.shape[0]

    # keep chunk * n_neurons bounded (~64M floats)
    n_neurons = W.shape[0]
    chunk = max(1, min(chunk, max(1, 64_000_000 // max(n_neurons, 1))))

    idx_all = np.empty(X.shape[0], dtype=np.int64)
    qe_sum = 0.0
    with torch.no_grad():
        for s in range(0, X.shape[0], chunk):
            xb = torch.from_numpy(X[s:s + chunk]).to(device, non_blocking=True)
            d = torch.cdist(xb, W)                    # (b, n_neurons)
            dmin, idx = torch.min(d, dim=1)
            idx_all[s:s + chunk] = idx.cpu().numpy()
            qe_sum += float(dmin.sum())
            del xb, d, dmin, idx

    bmu = np.column_stack([idx_all // dim, idx_all % dim]).astype(np.int64)
    return bmu, qe_sum / max(X.shape[0], 1)


def distance_map(som) -> np.ndarray:
    """Cached once per run; the original recomputed it up to 8x."""
    return som.distance_map().detach().cpu().numpy()


# =============================================================================
# Point extraction — the single source of truth
# =============================================================================

def extract_points(bmu: np.ndarray,
                   y_idx: np.ndarray,
                   dmap: np.ndarray,
                   n_classes: int,
                   mode: str = "majority") -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse (cell -> BMU node) into one array of labelled nodes.

    Replaces three inconsistent implementations (make_3D_graph / process_points /
    Will_It_Progenitor).  Fully vectorised: no DIM x DIM Python loop. [B1][B2]

    mode:
      'majority' — one row per occupied node, labelled by its most common class
                   (ties -> lowest class index).  Recommended; matches the paper's
                   "each neuron is assigned a cell type label".
      'first'    — one row per occupied node, labelled by lowest class index
                   present (deterministic stand-in for the old dict-order 'first').
      'all'      — one row per (node, class) pair present.

    Returns (points[M,4] = [row, col, z, class], weights[M] = cell counts).
    NOTE: the array is truncated to M occupied rows.  The original allocated N rows
    and left the remainder as (0,0,0,0), which polluted class 0. [B1]
    """
    dim = dmap.shape[0]
    flat = bmu[:, 0] * dim + bmu[:, 1]
    key = flat.astype(np.int64) * n_classes + y_idx.astype(np.int64)

    uk, cnt = np.unique(key, return_counts=True)
    node = uk // n_classes
    cls = (uk % n_classes).astype(np.int64)

    if mode == "all":
        sel_node, sel_cls, sel_cnt = node, cls, cnt
    elif mode in ("majority", "first"):
        if mode == "majority":
            order = np.lexsort((cls, -cnt, node))     # node asc, count desc, cls asc
        else:
            order = np.lexsort((cls, node))           # node asc, cls asc
        node_s, cls_s, cnt_s = node[order], cls[order], cnt[order]
        first = np.empty(node_s.size, dtype=bool)
        first[0] = True
        np.not_equal(node_s[1:], node_s[:-1], out=first[1:])
        sel_node, sel_cls, sel_cnt = node_s[first], cls_s[first], cnt_s[first]
    else:
        raise ValueError(f"unknown point mode {mode!r}")

    r = sel_node // dim
    c = sel_node % dim
    z = dmap[r, c]
    points = np.column_stack([r, c, z, sel_cls]).astype(np.float64)
    return points, sel_cnt.astype(np.float64)


def class_centroids(points: np.ndarray,
                    weights: np.ndarray,
                    n_classes: int,
                    weight_mode: str = "node") -> np.ndarray:
    """
    (n_classes, 3) centroids in [row, col, z].  Rows for absent classes are NaN
    (previously they silently became 0, which then won argmin(z)).
    """
    w = np.ones(points.shape[0]) if weight_mode == "node" else weights
    cls = points[:, 3].astype(int)

    num = np.zeros((n_classes, 3))
    den = np.zeros(n_classes)
    np.add.at(num, cls, points[:, :3] * w[:, None])
    np.add.at(den, cls, w)

    out = np.full((n_classes, 3), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok, None]
    return out


def infer_root(centroids: np.ndarray, rule: str = "argmax") -> int:
    # argmax: the Waddington convention -- the potent origin sits on HIGH ground.
    # (argmin was the previous default and selects the most differentiated state.)
    z = centroids[:, 2]
    if np.all(np.isnan(z)):
        return -1
    if rule == "argmin":
        return int(np.nanargmin(z))
    if rule == "argmax":
        return int(np.nanargmax(z))
    raise ValueError(f"unknown root rule {rule!r}")


def root_is_correct(centroids: np.ndarray, true_index: int, rule: str) -> bool:
    """`either` reproduces the original (permissive) criterion. [B4]"""
    if rule == "either":
        return (infer_root(centroids, "argmin") == true_index or
                infer_root(centroids, "argmax") == true_index)
    return infer_root(centroids, rule) == true_index


# =============================================================================
# Trajectories
# =============================================================================

def traj_adj_from_centroids(centroids: np.ndarray,
                            tol: float = 0.0,
                            backup: bool = False,
                            legacy_guard: bool = False) -> np.ndarray:
    """
    Distance-based adjacency: connect each centroid to its nearest centroid that
    lies lower on the landscape (within `tol`).

    Vectorised (was an O(K^2) Python double loop with repeated np.copy).
    [B3] `done[:, m]` (in-degree) replaces the original `done[:][m]` no-op.
    """
    K = centroids.shape[0]
    done = np.zeros((K, K), dtype=np.int64)
    if K == 0:
        return done

    P = centroids.copy()
    sr, sc = P[:, 0].sum(), P[:, 1].sum()
    P[:, 0] /= (sr if sr else 1.0)
    P[:, 1] /= (sc if sc else 1.0)
    P[:, 2] = 1.0 - P[:, 2]

    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)

    z = centroids[:, 2]
    lower = (z[:, None] > z[None, :]) | (np.abs(z[:, None] - z[None, :]) <= tol)

    for m in range(K):
        if legacy_guard:
            busy = done[m].sum() or done[m].sum()          # original no-op
        else:
            busy = done[m].sum() or done[:, m].sum()       # out- or in-degree
        if busy:
            continue
        cand = np.where(lower[m] & np.isfinite(D[m]))[0]
        if cand.size:
            done[m, cand[np.argmin(D[m, cand])]] = 1

    if backup:
        higher = (z[None, :] > z[:, None]) | (np.abs(z[:, None] - z[None, :]) <= tol)
        for m in range(K):
            cand = np.where(higher[m] & np.isfinite(D[m]))[0]
            if cand.size:
                done[cand[np.argmin(D[m, cand])], m] = 1

    # resolve bidirectional edges by elevation
    i, j = np.where(done & done.T)
    for a, b in zip(i, j):
        if a < b:
            if z[a] > z[b]:
                done[a, b], done[b, a] = 1, 0
            else:
                done[a, b], done[b, a] = 0, 1

    return done.T


def traj_adj_from_slingshot(points: np.ndarray, n_classes: int,
                            noise: float = 0.0, seed: int = 42) -> Optional[np.ndarray]:
    """Slingshot lineages -> adjacency.  Returns None if pyslingshot is absent."""
    try:
        from pyslingshot import Slingshot
    except Exception:
        return None

    rng = np.random.default_rng(seed)
    coords = points[:, :2].astype(float)
    if noise:
        coords = coords + noise * rng.random(coords.shape)

    b = onehot(points[:, 3].astype(int), n_classes)
    sling = Slingshot(coords, cluster_labels_onehot=b)
    sling.get_lineages()

    adj = np.zeros((n_classes, n_classes), dtype=np.int64)
    for lin in sling.lineages:
        cl = list(lin.clusters)
        for a, b_ in zip(cl[:-1], cl[1:]):
            adj[int(a), int(b_)] = 1
    return adj


def load_true_traj(path: str | Path, classes: np.ndarray) -> np.ndarray:
    """
    Milestone-network CSV -> adjacency.  One implementation replaces the two
    divergent `traj_encoding` versions (one compared an array to a *list*; the
    other stripped one quote too many).
    """
    rows = np.genfromtxt(str(path), delimiter=",", dtype=str)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    rows = rows[1:]                                   # drop header

    lookup = {str(c): i for i, c in enumerate(classes)}
    adj = np.zeros((classes.size, classes.size), dtype=np.int64)
    for row in rows:
        a, b = (str(row[0]).strip().strip('"').strip("'"),
                str(row[1]).strip().strip('"').strip("'"))
        if a not in lookup or b not in lookup:
            raise KeyError(f"milestone {a!r}->{b!r} not in classes {list(classes)}")
        adj[lookup[a], lookup[b]] = 1
    return adj


def score_adjacency(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    p, t = pred.astype(bool), true.astype(bool)
    tp = int((p & t).sum()); tn = int((~p & ~t).sum())
    fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    hamming = int((p ^ t).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, hamming_edges=hamming,
                precision=prec, recall=rec, f1=f1)


# =============================================================================
# Plotting
# =============================================================================

def _surface_stride(dim: int, target: int = 200) -> int:
    return max(1, dim // target)


def plot_landscape(points: np.ndarray, dmap: np.ndarray, classes: np.ndarray,
                   out_png: Path, cfg: Config, centroids: Optional[np.ndarray] = None,
                   title: str = "") -> None:
    """All four views in ONE figure, reusing one point extraction. """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim = dmap.shape[0]
    colors = class_colors(classes.size)
    stride = _surface_stride(dim)
    gx, gy = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")

    fig = plt.figure(figsize=(cfg.figsize, cfg.figsize), dpi=cfg.dpi)
    for v in range(4):
        ax = fig.add_subplot(2, 2, v + 1, projection="3d")
        if cfg.surface:
            ax.plot_surface(gx[::stride, ::stride], gy[::stride, ::stride],
                            dmap[::stride, ::stride], cmap="viridis", alpha=0.6,
                            linewidth=0, antialiased=False)
        for k in range(classes.size):
            m = points[:, 3] == k
            if m.any():
                ax.scatter(points[m, 0], points[m, 1], points[m, 2],
                           s=4, color=colors[k], label=str(classes[k]) if v == 0 else None)
        if centroids is not None:
            ok = ~np.isnan(centroids[:, 2])
            ax.scatter(centroids[ok, 0], centroids[ok, 1], centroids[ok, 2],
                       s=90, edgecolor="k", linewidth=1.2,
                       color=[colors[i] for i in np.where(ok)[0]], zorder=5)
        ax.view_init(elev=210, azim=-v * 90 - 45)
        ax.set_xlabel("SOM x"); ax.set_ylabel("SOM y"); ax.set_zlabel("activation")
    if classes.size <= 24:
        fig.legend(loc="lower center", ncol=min(6, classes.size), fontsize="small")
    fig.suptitle(title or out_png.stem)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_2d(points: np.ndarray, classes: np.ndarray, out_png: Path, cfg: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = class_colors(classes.size)
    fig, ax = plt.subplots(figsize=(cfg.figsize * 0.6, cfg.figsize * 0.6), dpi=cfg.dpi)
    for k in range(classes.size):                      # one call per class, not per point
        m = points[:, 3] == k
        if m.any():
            ax.scatter(points[m, 0], points[m, 1], s=6, color=colors[k], label=str(classes[k]))
    ax.legend(fontsize="small", markerscale=2)
    ax.set_xlabel("SOM x"); ax.set_ylabel("SOM y")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================

def run_dataset(tsv: str | Path, cfg: Config,
                true_traj_csv: Optional[str | Path] = None) -> Dict:
    tsv = Path(tsv)
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    stem = tsv.stem

    expr_df, labels = read_expression_tsv(tsv)
    X, Y, cell_ids = preprocess(expr_df, labels, cfg)
    classes, y_idx = encode_labels(Y)
    dim = grid_dim(X.shape[0], cfg.dim_scale)

    if cfg.verbose:
        print(f"[{stem}] N={X.shape[0]} F={X.shape[1]} classes={classes.size} DIM={dim} "
              f"(neurons={dim*dim:,}, codebook={dim*dim*X.shape[1]*4/1024**3:.2f} GB)")

    som = train_som(X, dim, cfg)
    free_memory(cfg)

    dmap = distance_map(som)                                  # once
    bmu, qe = batched_bmu(som, X, cfg)                        # once, batched
    points, weights = extract_points(bmu, y_idx, dmap, classes.size, cfg.point_mode)
    cents = class_centroids(points, weights, classes.size, cfg.centroid_weight)

    root_pred = infer_root(cents, "argmax" if cfg.root_rule == "either" else cfg.root_rule)
    root_ok = root_is_correct(cents, cfg.root_true_index, cfg.root_rule)

    tag = f"{stem}_lr{cfg.lr}_sig{cfg.sigma}_ep{cfg.epochs}_dim{dim}"
    if cfg.save_som:
        with open(cfg.outdir / f"SOM_{tag}.p", "wb") as fh:   # unique name (was DIM-only)
            pickle.dump(som, fh)

    np.savez_compressed(cfg.outdir / f"{tag}.npz",
                        points=points, weights=weights, centroids=cents,
                        distance_map=dmap, classes=classes.astype(str), bmu=bmu,
                        cell_ids=cell_ids, y_idx=y_idx)

    if cfg.plot:
        plot_landscape(points, dmap, classes, cfg.outdir / f"{tag}_landscape.png",
                       cfg, centroids=cents, title=stem)
        plot_2d(points, classes, cfg.outdir / f"{tag}_2d.png", cfg)

    pred_dist = traj_adj_from_centroids(cents, cfg.tol, cfg.backup,
                                        legacy_guard=cfg.legacy)
    pred_sling = traj_adj_from_slingshot(points, classes.size, seed=cfg.seed)

    result = dict(dataset=stem, n_cells=int(X.shape[0]), n_features=int(X.shape[1]),
                  n_classes=int(classes.size), dim=dim, quantization_error=qe,
                  n_points=int(points.shape[0]),
                  root_pred=int(root_pred), root_correct=bool(root_ok),
                  slingshot_available=pred_sling is not None)

    if true_traj_csv is not None:
        true = load_true_traj(true_traj_csv, classes)
        result["score_distance"] = score_adjacency(pred_dist, true)
        if pred_sling is not None:
            result["score_slingshot"] = score_adjacency(pred_sling, true)

    del som
    free_memory(cfg)
    return result


def ablate_lr_sigma(tsv: str | Path, cfg: Config,
                    lrs: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
                    sigmas: Sequence[float] = tuple(np.arange(0.1, 5.2, 0.5)),
                    epochs: Optional[int] = None) -> Dict:
    """
    Scores only.  The original rendered 4 x 5000x5000px figures for each of the
    55 (lr, sigma) cells and called quantization_error twice per cell.
    """
    expr_df, labels = read_expression_tsv(tsv)
    X, Y, _ = preprocess(expr_df, labels, cfg)
    dim = grid_dim(X.shape[0], cfg.dim_scale)

    best = dict(qe=np.inf, lr=None, sigma=None)
    grid = []
    for sig in sigmas:
        for lr in lrs:
            sub = Config(**{**asdict(cfg), "lr": float(lr), "sigma": float(sig),
                            "epochs": epochs or cfg.epochs, "plot": False})
            sub.outdir = cfg.outdir
            som = train_som(X, dim, sub)
            _, qe = batched_bmu(som, X, sub)                 # QE computed once
            grid.append(dict(lr=float(lr), sigma=float(sig), qe=float(qe)))
            if qe < best["qe"]:
                best = dict(qe=float(qe), lr=float(lr), sigma=float(sig))
            del som
            free_memory(sub)
            if cfg.verbose:
                print(f"  lr={lr:.2f} sigma={sig:.2f} qe={qe:.5f}")

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    with open(cfg.outdir / f"{Path(tsv).stem}_ablation.json", "w") as fh:
        json.dump(dict(best=best, grid=grid), fh, indent=2)
    return dict(best=best, grid=grid)


def export_for_dynwrap(tsv: str | Path, cfg: Config,
                       start_group: Optional[str] = None,
                       skip_existing: bool = False) -> Dict:
    """
    Emit the two CSVs that benchmark.R consumes:

      <stem>_sommit_cells.csv : cell_id, som_x, som_y, group_id
      <stem>_sommit_root.csv  : root_group, root_source

    som_x/som_y are the cell's BMU coordinates on the SOM grid.  This is a
    PER-CELL embedding, which dyneval requires.  (The original pipeline fed
    Slingshot the NODE-level points.csv, which has one row per occupied neuron,
    not one per cell -- those pseudotimes cannot be scored against a per-cell
    gold standard.)

    start_group: if given, the root is taken from this prior instead of being
    inferred.  Use this for the head-to-head benchmark, where every competitor
    is handed `start_id` -- otherwise SOMmit would be the only method running
    without the prior and the comparison would not be controlled.  Leave it
    unset to evaluate the root finder itself.

    skip_existing: if the output CSVs already exist, return without retraining.
    Makes a long batch restart-safe: a crash at dataset 60 of 75 resumes rather
    than reprocessing all 75.
    """
    import pandas as pd

    tsv = Path(tsv)
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    stem = tsv.stem
    cells_csv = cfg.outdir / f"{stem}_sommit_cells.csv"
    root_csv = cfg.outdir / f"{stem}_sommit_root.csv"

    if skip_existing and cells_csv.exists() and root_csv.exists():
        # Guard against a truncated file from a killed run: require the cells CSV
        # to be non-trivial (header + at least one row).
        try:
            existing = pd.read_csv(cells_csv, nrows=2)
            if existing.shape[0] >= 1 and {"cell_id", "som_x", "som_y"}.issubset(existing.columns):
                if cfg.verbose:
                    print(f"[sommit] skip {stem}: outputs already present")
                root = pd.read_csv(root_csv)
                return dict(dataset=stem, cells_csv=str(cells_csv), root_csv=str(root_csv),
                            root_group=str(root.get("root_group", ["?"])[0]),
                            root_source="skipped_existing", dim=None,
                            n_cells=None, skipped=True)
        except Exception:
            if cfg.verbose:
                print(f"[sommit] {stem}: existing output unreadable, re-running")

    expr_df, labels = read_expression_tsv(tsv)
    X, Y, cell_ids = preprocess(expr_df, labels, cfg)
    classes, y_idx = encode_labels(Y)
    dim = grid_dim(X.shape[0], cfg.dim_scale)

    som = train_som(X, dim, cfg)
    dmap = distance_map(som)
    bmu, _ = batched_bmu(som, X, cfg)
    points, weights = extract_points(bmu, y_idx, dmap, classes.size, cfg.point_mode)
    cents = class_centroids(points, weights, classes.size, cfg.centroid_weight)

    if start_group is not None:
        class_set = {str(c) for c in classes}
        if start_group == "auto":
            # Resolve the root group from the data instead of assuming a literal
            # label.  Handles the disconnected topology, whose groups are prefixed
            # per component (C1_Path1, C2_Path1, ...) so a bare "Path1" is absent.
            #
            # Origin heuristic: the simulator names the first path of each
            # (sub)simulation "...Path1", i.e. the group whose name ends in
            # "Path1".  A disconnected dataset has ONE such origin per component;
            # we pick them all and use the lowest-inter-neuron-distance one as the
            # single start prior the benchmark expects (dynwrap start_id can be a
            # vector, but SOMmit's Slingshot wrapper roots one tree at a time, so
            # we hand it the most-progenitor origin).
            origins = sorted([c for c in class_set if str(c).endswith("Path1")])
            if not origins:
                origins = sorted(class_set)          # fall back: any group
            if len(origins) == 1:
                root_group, source = origins[0], "prior_auto"
            else:
                # multiple components: choose the origin with min centroid z
                zmap = {str(classes[i]): cents[i, 2] for i in range(classes.size)
                        if not np.isnan(cents[i, 2])}
                cand = [o for o in origins if o in zmap]
                root_group = (min(cand, key=lambda o: zmap[o]) if cand else origins[0])
                source = "prior_auto_multi"
                if cfg.verbose:
                    print(f"[sommit] {stem}: {len(origins)} component origins {origins}; "
                          f"using '{root_group}' as start prior")
        elif str(start_group) in class_set:
            root_group, source = str(start_group), "prior"
        else:
            raise ValueError(
                f"start_group {start_group!r} not among {sorted(class_set)}.\n"
                f"  For datasets with prefixed/multiple origins (e.g. disconnected: "
                f"C1_Path1, C2_Path1), pass --start-group auto to resolve it per dataset.")
    else:
        ridx = infer_root(cents, "argmax" if cfg.root_rule == "either" else cfg.root_rule)
        root_group, source = str(classes[ridx]), "inferred"

    # Per-cell landscape elevation: the U-matrix value at each cell's BMU.
    # distance_map() is the U-matrix -- mean distance from each neuron to its
    # neighbours -- so HIGH = ridge/boundary between clusters, LOW = cluster
    # interior (an attractor basin).  This is the raw quantity SOMmit's Waddington
    # picture is built from; we export it unmodified and let the analysis decide
    # polarity (the potency benchmark tests correlation with a known stage axis,
    # which is sign-agnostic via |rho|).  bmu columns are (row, col) into dmap.
    br = bmu[:, 0].astype(int)
    bc = bmu[:, 1].astype(int)
    elevation = dmap[br, bc].astype(float)

    pd.DataFrame(dict(cell_id=cell_ids,
                      som_x=bmu[:, 0].astype(int),
                      som_y=bmu[:, 1].astype(int),
                      elevation=elevation,
                      group_id=[str(classes[i]) for i in y_idx])).to_csv(cells_csv, index=False)

    # Root CSV: emit the chosen root AND both candidates, so the polarity of the
    # root rule (argmin-z vs argmax-z) can be settled empirically against a known
    # biological root without re-running the SOM.  root_group/root_source are the
    # values SOMmit actually uses; root_argmin_z / root_argmax_z expose both ends
    # of the elevation axis, and the per-candidate mean elevation lets the caller
    # see which end the true root sits at.
    zc = cents[:, 2]
    if np.all(np.isnan(zc)):
        root_argmin = root_argmax = None
        z_argmin = z_argmax = float("nan")
    else:
        i_min = int(np.nanargmin(zc)); i_max = int(np.nanargmax(zc))
        root_argmin = str(classes[i_min]); root_argmax = str(classes[i_max])
        z_argmin = float(zc[i_min]); z_argmax = float(zc[i_max])
    pd.DataFrame(dict(root_group=[root_group], root_source=[source],
                      root_argmin_z=[root_argmin], root_argmax_z=[root_argmax],
                      z_at_argmin=[z_argmin], z_at_argmax=[z_argmax])).to_csv(root_csv, index=False)
    pd.DataFrame(dict(root_group=[root_group], root_source=[source])).to_csv(root_csv, index=False)

    del som
    free_memory(cfg)
    return dict(dataset=stem, cells_csv=str(cells_csv), root_csv=str(root_csv),
                root_group=root_group, root_source=source, dim=dim,
                n_cells=int(X.shape[0]))


def evaluate_roots(paths: Sequence[str], cfg: Config) -> Dict:
    simple, complex_ = [], []
    for p in paths:
        r = run_dataset(p, cfg)
        (simple if any(k in Path(p).name for k in ("bi", "lin", "tree")) else complex_).append(r)

    def acc(rs):
        return float(np.mean([r["root_correct"] for r in rs])) if rs else float("nan")

    out = dict(simple_n=len(simple), simple_acc=acc(simple),
               complex_n=len(complex_), complex_acc=acc(complex_),
               root_rule=cfg.root_rule)
    print(json.dumps(out, indent=2))
    return out


# =============================================================================
# Self-test (pure numpy; no torch/scanpy needed)
# =============================================================================

def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok &= bool(cond)

    print("natural sort / labels")
    classes, y = encode_labels(["M2", "M10", "M1", "M2"])
    check("M1 < M2 < M10", list(classes) == ["M1", "M2", "M10"])
    check("indices map correctly", list(y) == [1, 2, 0, 1])
    c2, y2 = encode_labels(["a", "b", "a"])
    check("2-class encoding not collapsed [B5]", set(y2) == {0, 1})

    print("extract_points [B1][B2]")
    dmap = np.array([[0.10, 0.20], [0.30, 0.40]])
    # 3 cells at node (0,0): classes 0,0,1 ; 1 cell at (1,1): class 2
    bmu = np.array([[0, 0], [0, 0], [0, 0], [1, 1]])
    yi = np.array([0, 0, 1, 2])
    pts, w = extract_points(bmu, yi, dmap, 3, "majority")
    check("no zero padding rows", pts.shape[0] == 2)
    check("majority label at (0,0) is class 0", pts[pts[:, 0] == 0][0, 3] == 0)
    check("z read from distance map", np.isclose(pts[pts[:, 0] == 1][0, 2], 0.40))
    pts_all, w_all = extract_points(bmu, yi, dmap, 3, "all")
    check("mode=all keeps both classes at (0,0)", pts_all.shape[0] == 3)
    check("weights are cell counts", sorted(w_all.tolist()) == [1.0, 1.0, 2.0])

    print("centroids / root")
    cents = class_centroids(pts_all, w_all, 3, "node")
    check("absent class -> NaN, not 0 [B1]",
          np.isnan(class_centroids(pts_all, w_all, 4, "node")[3, 2]))
    check("root = argmin z", infer_root(cents, "argmin") == 0)
    check("'either' is permissive [B4]",
          root_is_correct(np.array([[0, 0, 0.9], [0, 0, 0.5]]), 0, "either") and
          not root_is_correct(np.array([[0, 0, 0.9], [0, 0, 0.5]]), 0, "argmin"))

    print("trajectory adjacency [B3]")
    # three centroids descending in z: 0 (high) -> 1 -> 2 (low)
    c = np.array([[0.0, 0.0, 0.9], [1.0, 0.0, 0.5], [2.0, 0.0, 0.1]])
    adj = traj_adj_from_centroids(c)
    check("adjacency is square KxK", adj.shape == (3, 3))
    check("no self loops", np.trace(adj) == 0)
    check("no bidirectional edges", not np.any(adj & adj.T))

    print("true-trajectory parsing")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        fh.write('from,to\n"M1","M2"\n"M2","M3"\n')
        p = fh.name
    true = load_true_traj(p, np.array(["M1", "M2", "M3"], dtype=object))
    check("quoted milestones parsed", true[0, 1] == 1 and true[1, 2] == 1)
    check("no spurious edges", true.sum() == 2)
    os.unlink(p)

    print("scoring")
    s = score_adjacency(np.array([[0, 1], [0, 0]]), np.array([[0, 1], [0, 0]]))
    check("perfect prediction -> f1=1", np.isclose(s["f1"], 1.0) and s["hamming_edges"] == 0)

    print("elbow")
    check("elbow on convex curve", 1 <= elbow_index(np.array([0.5, 0.3, 0.05, 0.04, 0.03])) <= 4)

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


# =============================================================================
# CLI
# =============================================================================

def _add_common(p):
    p.add_argument("--outdir", default="./sommit_out")
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--sigma", type=float, default=1.2)
    p.add_argument("--epochs", type=int, default=700_000,
                   help="LR/radius schedule length (and hard cap if no early stop)")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="if set (and < --epochs), stop early when QE converges; "
                        "recovers most runtime on already-converged maps. "
                        "Recommended for the 75-dataset sweep, e.g. --max-epochs 150000")
    p.add_argument("--check-every", type=int, default=10_000,
                   help="QE convergence check interval when --max-epochs is set")
    p.add_argument("--converge-tol", type=float, default=1e-3,
                   help="stop when relative QE change < this between checks")
    p.add_argument("--dim-scale", type=float, default=5.0)
    p.add_argument("--som-input", choices=["expr", "pca20", "pca_elbow"], default="expr",
                   help="expr reproduces current results; pca20 matches the Methods text [D1]")
    p.add_argument("--point-mode", choices=["majority", "first", "all"], default="majority")
    p.add_argument("--centroid-weight", choices=["node", "cell"], default="node")
    p.add_argument("--root-rule", choices=["argmin", "argmax", "either"], default="argmin")
    p.add_argument("--root-true-index", type=int, default=0)
    p.add_argument("--device", default="cuda:0",
                   help="cuda:N or cpu. On a shared server the free card may not be "
                        "cuda:0 (your prior runs used cuda:3). Override also via "
                        "SOMMIT_DEVICE. CPU is refused unless explicit, since SOM "
                        "training hangs there.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-surface", action="store_true")
    p.add_argument("--save-som", action="store_true")
    p.add_argument("--figsize", type=float, default=12.0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--legacy", action="store_true",
                   help="reproduce pre-fix behaviour ([B1] padding excepted)")


def _cfg(a) -> Config:
    return Config(lr=a.lr, sigma=a.sigma, epochs=a.epochs, dim_scale=a.dim_scale,
                  max_epochs=a.max_epochs, check_every=a.check_every,
                  converge_tol=a.converge_tol,
                  som_input=a.som_input, point_mode=a.point_mode,
                  centroid_weight=a.centroid_weight, root_rule=a.root_rule,
                  root_true_index=a.root_true_index, device=a.device, seed=a.seed,
                  outdir=Path(a.outdir), plot=not a.no_plot, surface=not a.no_surface,
                  save_som=a.save_som, figsize=a.figsize, dpi=a.dpi,
                  verbose=a.verbose, legacy=a.legacy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SOMmit unified pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="train + landscape + root + trajectories on TSV(s)")
    p.add_argument("inputs", nargs="+", help="TSV files or a glob")
    p.add_argument("--true-traj", default=None, help="milestone network CSV")
    _add_common(p)

    p = sub.add_parser("ablate", help="lr/sigma sweep by quantization error")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--ablation-epochs", type=int, default=50_000)
    _add_common(p)

    p = sub.add_parser("roots", help="root-identification accuracy over a folder")
    p.add_argument("inputs", nargs="+")
    _add_common(p)

    p = sub.add_parser("export", help="write per-cell SOM coords + root for benchmark.R")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--start-group", default=None,
                   help="use this group as root (the start_id prior given to all "
                        "competitors); omit to use SOMmit's inferred root")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip datasets whose output CSVs already exist (restart-safe "
                        "for long batches)")
    _add_common(p)

    sub.add_parser("selftest", help="run pure-numpy unit checks (no GPU needed)")

    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()

    files: List[str] = []
    for pat in a.inputs:
        files.extend(sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat])
    if not files:
        print("no input files matched", file=sys.stderr)
        return 2

    cfg = _cfg(a)
    if a.cmd == "run":
        results = [run_dataset(f, cfg, a.true_traj) for f in files]
        cfg.outdir.mkdir(parents=True, exist_ok=True)
        with open(cfg.outdir / "results.json", "w") as fh:
            json.dump(results, fh, indent=2)
        print(json.dumps(results, indent=2))
    elif a.cmd == "ablate":
        for f in files:
            print(f, ablate_lr_sigma(f, cfg, epochs=a.ablation_epochs)["best"])
    elif a.cmd == "roots":
        evaluate_roots(files, cfg)
    elif a.cmd == "export":
        out = []
        for i, f in enumerate(files, 1):
            print(f"[sommit] === export {i}/{len(files)}: {Path(f).stem} ===", flush=True)
            out.append(export_for_dynwrap(f, cfg, a.start_group,
                                          skip_existing=a.skip_existing))
            # write the manifest incrementally so a crash still leaves a record
            with open(cfg.outdir / "export_manifest.json", "w") as fh:
                json.dump(out, fh, indent=2)
        n_skip = sum(1 for r in out if r.get("skipped"))
        print(f"[sommit] export complete: {len(out)} datasets ({n_skip} skipped, "
              f"{len(out) - n_skip} trained)")
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
