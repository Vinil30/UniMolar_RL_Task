"""Deterministic checks for the Uni-Mol 3D representation task.

Only raw measured values go into ``metrics``; the thresholds for those live in
human_scores.json and are applied by the fixed pass_fail.py harness.

What gets re-derived here rather than trusted:
  - the HOMO-LUMO gap error, the denoising RMSD and the hidden-atom accuracy
    are all computed against answers the agent never receives;
  - rotation/translation invariance is measured behaviourally.  Three hundred
    molecules occur twice in the test split under independent rigid motions
    and atom permutations, and nothing marks them; a model that reads raw
    coordinates instead of a genuine SE(3)-invariant encoding cannot give the
    two rows the same answer;
  - every ablation run is re-scored from the predictions that run submitted,
    so the reported configuration has to match what was actually trained;
  - the 3D, 2D and 1D representations are compared on held-out labels rather
    than on numbers the agent reports about itself.
"""

import csv
import json
import os
import traceback
from typing import Any

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_default_task_dir = "/app" if os.path.isdir("/app") else os.path.dirname(SCRIPT_DIR)
TASK_DIR = os.environ.get("TASK_DIR", _default_task_dir)

RESULTS_DIR = os.environ.get("OTTER_RESULTS_DIR", os.path.join(TASK_DIR, "results"))
INPUTS_DIR = os.environ.get("OTTER_INPUTS_DIR", os.path.join(TASK_DIR, "task_inputs"))
ANSWERS_DIR = os.environ.get("OTTER_ANSWERS_DIR", SCRIPT_DIR)

PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "predictions.json")
DENOISED_PATH = os.path.join(RESULTS_DIR, "denoised_coords.npz")
UNIMOL_CKPT_PATH = os.path.join(RESULTS_DIR, "unimol_model.pt")
BASELINE_CKPT_PATH = os.path.join(RESULTS_DIR, "gnn2d_model.pt")
ABLATION_PATH = os.path.join(RESULTS_DIR, "ablation.json")
ABLATION_PREDS_PATH = os.path.join(RESULTS_DIR, "ablation_predictions.json")
ANALYSIS_PATH = os.path.join(RESULTS_DIR, "analysis.md")

VALID_ELEMENTS = (1, 6, 7, 8, 9)

# These exist only to reject a stub - an empty state dict, a single bias
# vector - not to dictate capacity. A bond-graph network sized sensibly for a
# thousand labels can land near 40k parameters and must not be failed for it,
# so the floors sit well under any working model rather than near the
# reference implementation's 2.2M and 0.9M.
MIN_UNIMOL_PARAMETERS = 20_000
MIN_BASELINE_PARAMETERS = 5_000
MIN_WEIGHT_MATRICES = 2
# a molecule and its rigid motion must land on the same number; the encoder
# reaches this to ~1e-6 eV, so the tolerance only excludes coordinate readers
TWIN_TOLERANCE_EV = 0.02
MIN_PREDICTION_SPREAD_EV = 0.20
MAX_REFINEMENT_SHIFT = 1.5  # angstrom, after rigid alignment to the input
MIN_DENOISE_GAIN = 0.02  # angstrom below the untouched input
MIN_MODEL_DIFFERENCE_EV = 0.05
MIN_ABLATION_DIFFERENCE_EV = 0.02
MIN_GEOMETRY_GAIN_EV = 0.05
BASELINE_CEILING_EV = {"2d": 0.90, "1d": 0.95}
THREE_D_MARGIN_EV = 0.01
MIN_ABLATION_RUNS = 6
MIN_DISTINCT_SCALES = 3
MIN_ANALYSIS_WORDS = 250
# only the comparisons the instruction actually asks the write-up to cover
ANALYSIS_TOPICS = (
    ("smiles",),
    ("bond graph", "topology", "topological", "2d"),
    ("ablation", "ablate"),
)

N_TEST = 3000
N_VAL = 2000
N_TOPOLOGY = 1000
N_SEQUENCE = 1000
# A bond-graph model gets no coordinates on the topology split and a sequence
# model gets nothing but the string, so clearing these is proof that those two
# models exist rather than being the 3D encoder under another name.
MODALITY_CEILING_EV = {"topology": 0.85, "sequence": 0.75}
MIN_PRETRAINING_GAIN_EV = 0.05


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _read_label_column(path: str) -> np.ndarray:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([float(r["gap_ev"]) for r in rows], dtype=np.float64)


def _load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {k: handle[k] for k in handle.files}


_cache: dict[str, Any] = {}


def _get(name: str, loader):
    if name not in _cache:
        _cache[name] = loader()
    return _cache[name]


def test_labels() -> np.ndarray:
    return _get("test_labels", lambda: _read_label_column(
        os.path.join(ANSWERS_DIR, "test_labels.csv")))


def val_labels() -> np.ndarray:
    return _get("val_labels", lambda: _read_label_column(
        os.path.join(ANSWERS_DIR, "val_labels.csv")))


def modality_labels(name: str) -> np.ndarray:
    return _get(f"{name}_labels", lambda: _read_label_column(
        os.path.join(ANSWERS_DIR, f"{name}_eval_labels.csv")))


def twin_pairs() -> np.ndarray:
    return _get("twins", lambda: np.load(
        os.path.join(ANSWERS_DIR, "twin_pairs.npy")))


def clean_coords() -> dict[str, np.ndarray]:
    return _get("clean", lambda: _load_npz(
        os.path.join(ANSWERS_DIR, "denoise_clean.npz")))


def mask_answers() -> dict[str, np.ndarray]:
    return _get("mask_answers", lambda: _load_npz(
        os.path.join(ANSWERS_DIR, "mask_answers.npz")))


def noisy_inputs() -> dict[str, np.ndarray]:
    return _get("noisy", lambda: _load_npz(
        os.path.join(INPUTS_DIR, "denoise.npz")))


def predictions() -> dict[str, Any]:
    def load():
        with open(PREDICTIONS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return _get("predictions", load)


def ablation() -> dict[str, Any]:
    def load():
        with open(ABLATION_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return _get("ablation", load)


def ablation_predictions() -> dict[str, Any]:
    def load():
        with open(ABLATION_PREDS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return _get("ablation_preds", load)


def _vector(name: str, expected: int) -> np.ndarray:
    values = predictions()[name]
    if not isinstance(values, list):
        raise TypeError(f"'{name}' must be a list")
    if len(values) != expected:
        raise AssertionError(
            f"'{name}' has {len(values)} entries, expected {expected}")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise AssertionError(f"'{name}' contains non-finite values")
    return array


def _state_dict(path: str):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise TypeError(f"{os.path.basename(path)} is not a state dict")
    tensors = {k: v for k, v in blob.items() if hasattr(v, "shape")}
    if not tensors:
        raise AssertionError(f"{os.path.basename(path)} holds no tensors")
    return tensors


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _kabsch_displacement(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Displacement of pred from truth after the optimal rigid alignment."""
    p = pred - pred.mean(axis=0)
    q = truth - truth.mean(axis=0)
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return p @ rotation.T - q


def _pooled_rmsd(pred_flat: np.ndarray, truth_flat: np.ndarray,
                 offsets: np.ndarray) -> float:
    squared, count = 0.0, 0
    for i in range(len(offsets) - 1):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if hi - lo < 2:
            continue
        disp = _kabsch_displacement(pred_flat[lo:hi].astype(np.float64),
                                    truth_flat[lo:hi].astype(np.float64))
        squared += float((disp ** 2).sum())
        count += hi - lo
    return float(np.sqrt(squared / max(count, 1)))


# ---------------------------------------------------------------------------
# measured quantities
# ---------------------------------------------------------------------------

def gap_mae() -> float:
    return float(np.mean(np.abs(_vector("test_gap", N_TEST) - test_labels())))


def denoise_rmsd() -> float:
    data = _load_npz(DENOISED_PATH)
    truth = clean_coords()
    return _pooled_rmsd(data["coords"], truth["coords"], truth["offsets"])


def identity_rmsd() -> float:
    truth = clean_coords()
    return _get("identity", lambda: _pooled_rmsd(
        noisy_inputs()["coords"], truth["coords"], truth["offsets"]))


def mask_accuracy() -> float:
    truth = mask_answers()["answers"].astype(np.int64)
    guess = np.asarray(predictions()["hidden_atom_types"], dtype=np.int64)
    if len(guess) != len(truth):
        raise AssertionError(
            f"'hidden_atom_types' has {len(guess)} entries, "
            f"expected {len(truth)}")
    return float(np.mean(guess == truth))


def _val_mae(name: str) -> float:
    return float(np.mean(np.abs(_vector(name, N_VAL) - val_labels())))


def _ablation_mae(key: str) -> float:
    values = ablation_predictions()[key]
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (N_VAL,):
        raise AssertionError(
            f"ablation '{key}' has {array.shape} predictions, "
            f"expected ({N_VAL},)")
    return float(np.mean(np.abs(array - val_labels())))


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------

def check_predictions_wellformed(context=None) -> str:
    test = _vector("test_gap", N_TEST)
    for name in ("val_gap_3d", "val_gap_2d", "val_gap_1d"):
        _vector(name, N_VAL)
    _vector("topology_gap", N_TOPOLOGY)
    _vector("sequence_gap", N_SEQUENCE)
    guesses = np.asarray(predictions()["hidden_atom_types"], dtype=np.int64)
    expected = len(mask_answers()["answers"])
    assert len(guesses) == expected, (
        f"'hidden_atom_types' has {len(guesses)} entries, expected {expected}")
    bad = sorted(set(guesses.tolist()) - set(VALID_ELEMENTS))
    assert not bad, f"hidden atom guesses contain non-elements {bad}"
    spread = float(test.std())
    assert spread >= MIN_PREDICTION_SPREAD_EV, (
        f"test predictions are nearly constant (std {spread:.4f} eV < "
        f"{MIN_PREDICTION_SPREAD_EV})")
    return (f"all prediction vectors present and well-formed "
            f"(test std {spread:.3f} eV, {expected} hidden atoms)")


def check_denoised_wellformed(context=None) -> str:
    data = _load_npz(DENOISED_PATH)
    inputs = noisy_inputs()
    assert "coords" in data and "offsets" in data, (
        "denoised_coords.npz needs 'coords' and 'offsets'")
    assert np.array_equal(data["offsets"].astype(np.int64),
                          inputs["offsets"].astype(np.int64)), (
        "denoised_coords.npz offsets do not match denoise.npz")
    assert data["coords"].shape == inputs["coords"].shape, (
        f"denoised coords {data['coords'].shape} != input "
        f"{inputs['coords'].shape}")
    assert np.all(np.isfinite(data["coords"])), "denoised coords not finite"
    shift = _pooled_rmsd(data["coords"], inputs["coords"], inputs["offsets"])
    assert shift <= MAX_REFINEMENT_SHIFT, (
        f"refined geometry sits {shift:.3f} A from the input, beyond the "
        f"{MAX_REFINEMENT_SHIFT} A a refinement should move")
    return (f"refined coordinates aligned with the denoising split "
            f"({shift:.3f} A from the input)")


def check_ablation_wellformed(context=None) -> str:
    runs = ablation()["runs"]
    assert len(runs) >= MIN_ABLATION_RUNS, (
        f"{len(runs)} ablation runs, need {MIN_ABLATION_RUNS}")
    preds = ablation_predictions()
    for k in range(len(runs)):
        assert f"run_{k}" in preds, f"ablation_predictions missing run_{k}"
        _ablation_mae(f"run_{k}")
    denoising = {bool(r["denoising_pretraining"]) for r in runs}
    pair = {bool(r["geometry"]) for r in runs}
    scales = {int(r["pretrain_molecules"]) for r in runs}
    assert denoising == {True, False}, (
        "ablation must cover 3D denoising pre-training both on and off")
    assert pair == {True, False}, (
        "ablation must cover the pair/spatial encoding both on and off")
    assert len(scales) >= MIN_DISTINCT_SCALES, (
        f"ablation covers {len(scales)} pre-training scales, "
        f"need {MIN_DISTINCT_SCALES}")
    return (f"{len(runs)} runs covering denoising on/off, geometry "
            f"used/withheld and {len(scales)} pre-training scales")


def _signature(tensors) -> tuple:
    return tuple(sorted((k, tuple(int(d) for d in t.shape))
                        for k, t in tensors.items()))


def check_unimol_checkpoint(context=None) -> str:
    """A trained network of substance was saved.

    Deliberately shallow: what the encoder must *do* is measured behaviourally
    - invariance across the hidden rigid-motion pairs, geometry recovered on
    the denoising split, hidden atoms identified, and the ablation arm that
    withholds geometry - rather than by dictating its internal layout.
    """
    tensors = _state_dict(UNIMOL_CKPT_PATH)
    total = int(sum(int(np.prod(tuple(t.shape))) for t in tensors.values()))
    assert total >= MIN_UNIMOL_PARAMETERS, (
        f"checkpoint holds {total} parameters, expected at least "
        f"{MIN_UNIMOL_PARAMETERS}")
    matrices = [t for t in tensors.values() if t.dim() >= 2]
    assert len(matrices) >= MIN_WEIGHT_MATRICES, (
        f"checkpoint holds {len(matrices)} weight matrices, expected at least "
        f"{MIN_WEIGHT_MATRICES} for a trained encoder")
    return (f"encoder checkpoint valid ({total} parameters, "
            f"{len(matrices)} weight matrices)")


def check_baseline_checkpoint(context=None) -> str:
    tensors = _state_dict(BASELINE_CKPT_PATH)
    total = int(sum(int(np.prod(tuple(t.shape))) for t in tensors.values()))
    assert total >= MIN_BASELINE_PARAMETERS, (
        f"baseline holds {total} parameters, expected at least "
        f"{MIN_BASELINE_PARAMETERS}")
    assert _signature(tensors) != _signature(_state_dict(UNIMOL_CKPT_PATH)), (
        "the 2D baseline checkpoint is the same model as the 3D encoder; the "
        "comparison needs two separately trained models")
    return f"2D baseline checkpoint valid and distinct ({total} parameters)"


def check_models_distinct(context=None) -> str:
    vectors = {name: _vector(f"val_gap_{name}", N_VAL)
               for name in ("3d", "2d", "1d")}
    details = []
    for a, b in (("3d", "2d"), ("3d", "1d"), ("2d", "1d")):
        difference = float(np.mean(np.abs(vectors[a] - vectors[b])))
        details.append(f"{a}/{b}={difference:.3f}")
        assert difference >= MIN_MODEL_DIFFERENCE_EV, (
            f"{a} and {b} predictions differ by only {difference:.4f} eV; "
            f"they must be genuinely different representations")
    return ("representations produce distinct predictions ("
            + ", ".join(details) + ")")


def check_ablation_informative(context=None) -> str:
    runs = ablation()["runs"]
    scores = {k: _ablation_mae(f"run_{k}") for k in range(len(runs))}
    vectors = {k: np.asarray(ablation_predictions()[f"run_{k}"],
                             dtype=np.float64) for k in range(len(runs))}
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            difference = float(np.mean(np.abs(vectors[a] - vectors[b])))
            assert difference >= MIN_ABLATION_DIFFERENCE_EV, (
                f"runs {a} ({runs[a]['name']}) and {b} ({runs[b]['name']}) "
                f"differ by only {difference:.4f} eV - separate "
                f"configurations should not give near-identical predictions")
    with_pair = [k for k in range(len(runs)) if runs[k]["geometry"]]
    without_pair = [k for k in range(len(runs)) if not runs[k]["geometry"]]
    best_with = min(scores[k] for k in with_pair)
    best_without = min(scores[k] for k in without_pair)
    assert best_with <= best_without - MIN_GEOMETRY_GAIN_EV, (
        f"removing the pair/spatial encoding changed validation MAE from "
        f"{best_with:.4f} to {best_without:.4f} eV; a model that truly uses "
        f"3D structure should degrade by at least "
        f"{MIN_GEOMETRY_GAIN_EV} eV without it")
    listing = ", ".join(f"{runs[k]['name']}={scores[k]:.3f}"
                        for k in range(len(runs)))
    return f"ablation runs re-scored on held-out labels ({listing})"


def check_se3_invariance(context=None) -> str:
    values = _vector("test_gap", N_TEST)
    pairs = twin_pairs()
    assert len(pairs) > 0, "no twin pairs available"
    gaps = np.abs(values[pairs[:, 0]] - values[pairs[:, 1]])
    worst = float(gaps.max())
    assert worst <= TWIN_TOLERANCE_EV, (
        f"{int((gaps > TWIN_TOLERANCE_EV).sum())} of {len(pairs)} molecules "
        f"that appear twice under a rigid motion received different "
        f"predictions (worst {worst:.4f} eV): the representation is not "
        f"invariant to rotation and translation")
    return (f"predictions invariant across {len(pairs)} hidden rigid-motion "
            f"pairs (worst disagreement {worst:.2e} eV)")


def check_denoising_beats_input(context=None) -> str:
    model, identity = denoise_rmsd(), identity_rmsd()
    assert model <= identity - MIN_DENOISE_GAIN, (
        f"refined geometry RMSD {model:.4f} A barely improves on the "
        f"perturbed input at {identity:.4f} A")
    return (f"refinement improves geometry {identity:.4f} -> {model:.4f} A "
            f"(input baseline aligned the same way)")


def check_modality_baselines(context=None) -> str:
    """Each baseline works on a split that only carries its own modality."""
    details = []
    for name, ceiling in MODALITY_CEILING_EV.items():
        expected = N_TOPOLOGY if name == "topology" else N_SEQUENCE
        predicted = _vector(f"{name}_gap", expected)
        score = float(np.mean(np.abs(predicted - modality_labels(name))))
        details.append(f"{name}={score:.3f}")
        assert score <= ceiling, (
            f"the {name}-only model reaches {score:.4f} eV MAE on the split "
            f"that carries nothing but {name}, above the {ceiling} eV ceiling; "
            f"a working {name}-only model is what makes the comparison mean "
            f"something")
    return ("modality-restricted baselines competent (" + ", ".join(details)
            + ")")


def check_pretraining_matters(context=None) -> str:
    """Skipping pre-training has to cost accuracy, or it was never needed."""
    runs = ablation()["runs"]
    scores = {k: _ablation_mae(f"run_{k}") for k in range(len(runs))}
    pretrained = [k for k in range(len(runs))
                  if int(runs[k]["pretrain_molecules"]) > 0]
    scratch = [k for k in range(len(runs))
               if int(runs[k]["pretrain_molecules"]) == 0]
    assert pretrained and scratch, (
        "the ablation must contain at least one pre-trained run and one "
        "trained from scratch")
    best_pretrained = min(scores[k] for k in pretrained)
    best_scratch = min(scores[k] for k in scratch)
    assert best_pretrained <= best_scratch - MIN_PRETRAINING_GAIN_EV, (
        f"pre-training changed validation MAE from {best_scratch:.4f} to "
        f"{best_pretrained:.4f} eV; with this few labels a representation "
        f"learned from the unlabelled conformations should be worth at least "
        f"{MIN_PRETRAINING_GAIN_EV} eV")
    return (f"pre-training earns its place ({best_scratch:.3f} -> "
            f"{best_pretrained:.3f} eV validation MAE)")


def check_baselines_competent(context=None) -> str:
    details = []
    for name, ceiling in BASELINE_CEILING_EV.items():
        score = _val_mae(f"val_gap_{name}")
        details.append(f"{name}={score:.3f}")
        assert score <= ceiling, (
            f"the {name} baseline reaches only {score:.4f} eV MAE, above the "
            f"{ceiling} eV a trained-in-good-faith baseline should clear")
    return "representation baselines trained competently (" + \
        ", ".join(details) + ")"


def check_three_d_advantage(context=None) -> str:
    three = _val_mae("val_gap_3d")
    two = _val_mae("val_gap_2d")
    one = _val_mae("val_gap_1d")
    assert three <= two - THREE_D_MARGIN_EV, (
        f"3D MAE {three:.4f} eV does not improve on the 2D baseline "
        f"{two:.4f} eV")
    assert three <= one - THREE_D_MARGIN_EV, (
        f"3D MAE {three:.4f} eV does not improve on the 1D baseline "
        f"{one:.4f} eV")
    return (f"3D representation ahead of both baselines on held-out labels "
            f"(3D={three:.3f}, 2D={two:.3f}, 1D={one:.3f} eV)")


def check_analysis_written(context=None) -> str:
    with open(ANALYSIS_PATH, encoding="utf-8") as fh:
        text = fh.read()
    words = len(text.split())
    assert words >= MIN_ANALYSIS_WORDS, (
        f"analysis.md holds {words} words, expected at least "
        f"{MIN_ANALYSIS_WORDS}")
    lowered = text.lower()
    missing = [topic[0] for topic in ANALYSIS_TOPICS
               if not any(word in lowered for word in topic)]
    assert not missing, f"analysis.md never discusses {missing}"
    return f"analysis covers the required comparisons ({words} words)"


# ---------------------------------------------------------------------------

def _safe(fn, default=0.0) -> float:
    try:
        value = float(fn())
    except Exception:  # noqa: BLE001 - a broken artifact scores, never crashes
        return default
    return value if np.isfinite(value) else default


def compute_metrics() -> dict[str, float]:
    """Measured values only; thresholds are applied by the fixed harness."""
    return {
        "gap_mae": _safe(gap_mae, 99.0),
        "denoise_rmsd": _safe(denoise_rmsd, 99.0),
        "mask_accuracy": _safe(mask_accuracy, 0.0),
    }


def _criterion(criterion_id: str, fn, milestone_id: str) -> dict[str, Any]:
    try:
        detail = fn()
        return {
            "id": criterion_id,
            "passed": True,
            "detail": detail or "passed",
            "milestone_id": milestone_id,
        }
    except Exception:  # noqa: BLE001 - report the failure, never crash the run
        return {
            "id": criterion_id,
            "passed": False,
            "detail": traceback.format_exc(limit=6),
            "milestone_id": milestone_id,
        }


def evaluate(context: dict | None = None) -> dict[str, Any]:
    criteria = [
        _criterion("predictions-wellformed",
                   lambda: check_predictions_wellformed(context),
                   "data-and-setup"),
        _criterion("denoised-coords-wellformed",
                   lambda: check_denoised_wellformed(context),
                   "data-and-setup"),
        _criterion("ablation-wellformed",
                   lambda: check_ablation_wellformed(context),
                   "data-and-setup"),
        _criterion("unimol-checkpoint",
                   lambda: check_unimol_checkpoint(context),
                   "model-development"),
        _criterion("baseline-checkpoint",
                   lambda: check_baseline_checkpoint(context),
                   "model-development"),
        _criterion("models-distinct", lambda: check_models_distinct(context),
                   "model-development"),
        _criterion("ablation-informative",
                   lambda: check_ablation_informative(context),
                   "model-development"),
        _criterion("se3-invariance", lambda: check_se3_invariance(context),
                   "evaluation-and-reporting"),
        _criterion("denoising-beats-input",
                   lambda: check_denoising_beats_input(context),
                   "evaluation-and-reporting"),
        _criterion("pretraining-matters",
                   lambda: check_pretraining_matters(context),
                   "model-development"),
        _criterion("baselines-competent",
                   lambda: check_baselines_competent(context),
                   "evaluation-and-reporting"),
        _criterion("modality-baselines-competent",
                   lambda: check_modality_baselines(context),
                   "evaluation-and-reporting"),
        _criterion("three-d-advantage",
                   lambda: check_three_d_advantage(context),
                   "evaluation-and-reporting"),
        _criterion("analysis-written", lambda: check_analysis_written(context),
                   "evaluation-and-reporting"),
    ]
    return {"criteria": criteria, "metrics": compute_metrics()}
