"""cup.well.wavelet_consensus: generate a global wavelet from candidates.

This module owns the low-dimensional wavelet-shape algorithm.  It does not read
LAS, time-depth tables, seismic files, or auto-tie artifacts; callers provide
already aligned candidate wavelets and an evaluator callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from cup.well.wavelet import wavelet_l2_normalize


# Evaluators may return a scalar score or a metric mapping containing
# ``ConsensusSearchPolicy.score_key``; the default key is ``"score"``.
WaveletEvaluator = Callable[[np.ndarray], Mapping[str, float] | float]


@dataclass(frozen=True)
class WaveletBasis:
    """PCA basis built from aligned, L2-normalized candidate wavelets."""

    mean_wavelet: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    coefficient_lower: np.ndarray
    coefficient_upper: np.ndarray
    candidate_coefficients: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.mean_wavelet.size)

    def project(self, values: np.ndarray) -> np.ndarray:
        values = _as_wavelet_vector(values, n_samples=self.n_samples)
        if self.n_components == 0:
            return np.empty(0, dtype=np.float64)
        return (values - self.mean_wavelet) @ self.components.T


@dataclass(frozen=True)
class ConsensusSearchPolicy:
    """Search controls for global wavelet generation."""

    random_trials: int = 512
    max_refine_iters: int = 120
    top_refine_count: int = 3
    seed: int | None = 20260529
    score_key: str = "score"


@dataclass(frozen=True)
class WaveletGenerationTrial:
    """One evaluated point in PCA coefficient space."""

    trial_id: int
    coefficients: tuple[float, ...]
    score: float
    metrics: dict[str, float]
    status: str
    reason: str = ""
    selected: bool = False

    def to_row(self) -> dict[str, float | int | str | bool]:
        row: dict[str, float | int | str | bool] = {
            "trial_id": self.trial_id,
            "score": self.score,
            "status": self.status,
            "reason": self.reason,
            "selected": self.selected,
        }
        for index, value in enumerate(self.coefficients):
            row[f"coef_{index}"] = float(value)
        row.update({key: float(value) for key, value in self.metrics.items() if _is_finite_number(value)})
        return row


@dataclass(frozen=True)
class WaveletGenerationResult:
    """Best generated wavelet and the search trace that produced it."""

    wavelet: np.ndarray
    coefficients: np.ndarray
    score: float
    metrics: dict[str, float]
    trials: list[WaveletGenerationTrial]
    selection_mode: str

    def summary(self) -> dict[str, object]:
        return {
            "score": float(self.score),
            "coefficients": [float(value) for value in self.coefficients],
            "metrics": dict(self.metrics),
            "selection_mode": self.selection_mode,
            "n_trials": len(self.trials),
        }


def build_wavelet_pca_basis(
    wavelets: np.ndarray,
    *,
    n_components: int = 4,
    coefficient_bounds: str = "quantile",
    coefficient_quantiles: tuple[float, float] = (0.05, 0.95),
) -> WaveletBasis:
    """Build a PCA basis from candidate wavelet amplitudes using SVD."""
    matrix = _as_wavelet_matrix(wavelets)
    if int(n_components) < 0:
        raise ValueError(f"n_components must be non-negative, got {n_components}.")
    lower_q, upper_q = (float(coefficient_quantiles[0]), float(coefficient_quantiles[1]))
    if not 0.0 <= lower_q <= upper_q <= 1.0:
        raise ValueError(f"coefficient_quantiles must satisfy 0 <= low <= high <= 1, got {coefficient_quantiles}.")

    mean_wavelet = np.mean(matrix, axis=0)
    centered = matrix - mean_wavelet
    max_components = max(0, min(int(n_components), matrix.shape[0] - 1, matrix.shape[1]))
    if max_components == 0:
        return WaveletBasis(
            mean_wavelet=mean_wavelet.astype(np.float64, copy=True),
            components=np.empty((0, matrix.shape[1]), dtype=np.float64),
            explained_variance_ratio=np.empty(0, dtype=np.float64),
            coefficient_lower=np.empty(0, dtype=np.float64),
            coefficient_upper=np.empty(0, dtype=np.float64),
            candidate_coefficients=np.empty((matrix.shape[0], 0), dtype=np.float64),
        )

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:max_components].astype(np.float64, copy=True)
    variances = singular_values[:max_components] ** 2
    total_variance = float(np.sum(singular_values**2))
    explained = variances / total_variance if total_variance > 0.0 else np.zeros_like(variances)
    coefficients = centered @ components.T

    if coefficient_bounds == "quantile":
        lower = np.quantile(coefficients, lower_q, axis=0)
        upper = np.quantile(coefficients, upper_q, axis=0)
    elif coefficient_bounds == "minmax":
        lower = np.min(coefficients, axis=0)
        upper = np.max(coefficients, axis=0)
    else:
        raise ValueError(f"Unsupported coefficient_bounds: {coefficient_bounds}")

    equal = np.isclose(lower, upper, rtol=0.0, atol=1e-12)
    if np.any(equal):
        lower = lower.copy()
        upper = upper.copy()
        lower[equal] -= 1e-6
        upper[equal] += 1e-6

    return WaveletBasis(
        mean_wavelet=mean_wavelet.astype(np.float64, copy=True),
        components=components,
        explained_variance_ratio=explained.astype(np.float64, copy=False),
        coefficient_lower=lower.astype(np.float64, copy=False),
        coefficient_upper=upper.astype(np.float64, copy=False),
        candidate_coefficients=coefficients.astype(np.float64, copy=False),
    )


def generate_consensus_wavelet(basis: WaveletBasis, coefficients: np.ndarray) -> np.ndarray:
    """Generate one L2-normalized wavelet from PCA coefficients."""
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coefficients.size != basis.n_components:
        raise ValueError(f"Expected {basis.n_components} coefficients, got {coefficients.size}.")
    values = basis.mean_wavelet.copy()
    if basis.n_components:
        values = values + coefficients @ basis.components
    return wavelet_l2_normalize(values)[0]


def optimize_consensus_wavelet(
    basis: WaveletBasis,
    evaluator: WaveletEvaluator,
    *,
    policy: ConsensusSearchPolicy | None = None,
) -> WaveletGenerationResult:
    """Search PCA coefficient space for the best evaluator score."""
    policy = policy or ConsensusSearchPolicy()
    trials: list[WaveletGenerationTrial] = []
    rng = np.random.default_rng(policy.seed)

    def add_trial(coefficients: np.ndarray, reason: str = "") -> WaveletGenerationTrial:
        trial = _evaluate_coefficients(
            trial_id=len(trials),
            basis=basis,
            coefficients=coefficients,
            evaluator=evaluator,
            score_key=policy.score_key,
            reason=reason,
        )
        trials.append(trial)
        return trial

    zero = np.zeros(basis.n_components, dtype=np.float64)
    add_trial(zero, reason="mean_wavelet")
    for coefficients in basis.candidate_coefficients:
        add_trial(coefficients, reason="candidate_projection")

    if basis.n_components > 0 and policy.random_trials > 0:
        lower = basis.coefficient_lower
        upper = basis.coefficient_upper
        for _ in range(int(policy.random_trials)):
            add_trial(rng.uniform(lower, upper), reason="random")

    if basis.n_components > 0 and policy.max_refine_iters > 0:
        _refine_best_trials(basis, evaluator, policy, trials)

    best_index = _best_trial_index(trials)
    selected_trials = []
    for index, trial in enumerate(trials):
        selected_trials.append(
            WaveletGenerationTrial(
                trial_id=trial.trial_id,
                coefficients=trial.coefficients,
                score=trial.score,
                metrics=trial.metrics,
                status=trial.status,
                reason=trial.reason,
                selected=index == best_index,
            )
        )
    best = selected_trials[best_index]
    best_coefficients = np.asarray(best.coefficients, dtype=np.float64)
    return WaveletGenerationResult(
        wavelet=generate_consensus_wavelet(basis, best_coefficients),
        coefficients=best_coefficients,
        score=float(best.score),
        metrics=dict(best.metrics),
        trials=selected_trials,
        selection_mode="optimized_consensus" if basis.n_components > 0 else "mean_wavelet",
    )


def _as_wavelet_matrix(wavelets: np.ndarray) -> np.ndarray:
    matrix = np.asarray(wavelets, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"wavelets must be a 2-D matrix [n_candidates, n_samples], got {matrix.shape}.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError(f"wavelets matrix is too small: {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("wavelets matrix contains non-finite values.")
    return matrix


def _as_wavelet_vector(values: np.ndarray, *, n_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != int(n_samples):
        raise ValueError(f"Expected {n_samples} wavelet samples, got {values.size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("wavelet contains non-finite values.")
    return values


def _evaluate_coefficients(
    *,
    trial_id: int,
    basis: WaveletBasis,
    coefficients: np.ndarray,
    evaluator: WaveletEvaluator,
    score_key: str,
    reason: str,
) -> WaveletGenerationTrial:
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if basis.n_components:
        coefficients = np.clip(coefficients, basis.coefficient_lower, basis.coefficient_upper)
    try:
        wavelet = generate_consensus_wavelet(basis, coefficients)
        raw_metrics = evaluator(wavelet)
        if isinstance(raw_metrics, Mapping):
            metrics = {str(key): float(value) for key, value in raw_metrics.items()}
            score = float(metrics.get(score_key, np.nan))
        else:
            score = float(raw_metrics)
            metrics = {score_key: score}
        status = "ok" if np.isfinite(score) else "failed"
        trial_reason = reason if status == "ok" else f"{reason};non_finite_score"
    except Exception as exc:  # pragma: no cover - caller needs failed trial records for reports
        metrics = {}
        score = float("-inf")
        status = "failed"
        trial_reason = f"{reason};{type(exc).__name__}:{exc}"
    return WaveletGenerationTrial(
        trial_id=int(trial_id),
        coefficients=tuple(float(value) for value in coefficients),
        score=score,
        metrics=metrics,
        status=status,
        reason=trial_reason.strip(";"),
    )


def _best_trial_index(trials: list[WaveletGenerationTrial]) -> int:
    scores = np.asarray([trial.score for trial in trials], dtype=np.float64)
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise ValueError("No finite consensus wavelet trial score was produced.")
    scores[~finite] = float("-inf")
    return int(np.argmax(scores))


def _refine_best_trials(
    basis: WaveletBasis,
    evaluator: WaveletEvaluator,
    policy: ConsensusSearchPolicy,
    trials: list[WaveletGenerationTrial],
) -> None:
    try:
        from scipy.optimize import minimize
    except Exception:
        return

    ok_trials = [trial for trial in trials if trial.status == "ok" and np.isfinite(trial.score)]
    ok_trials = sorted(ok_trials, key=lambda trial: trial.score, reverse=True)[: max(1, int(policy.top_refine_count))]
    bounds = list(zip(basis.coefficient_lower, basis.coefficient_upper))

    def objective(coefficients: np.ndarray) -> float:
        trial = _evaluate_coefficients(
            trial_id=-1,
            basis=basis,
            coefficients=coefficients,
            evaluator=evaluator,
            score_key=policy.score_key,
            reason="refine_probe",
        )
        return -float(trial.score) if np.isfinite(trial.score) else 1e12

    for trial in ok_trials:
        result = minimize(
            objective,
            np.asarray(trial.coefficients, dtype=np.float64),
            method="Powell",
            bounds=bounds,
            options={"maxiter": int(policy.max_refine_iters), "disp": False},
        )
        coefficients = np.asarray(result.x, dtype=np.float64)
        refined = _evaluate_coefficients(
            trial_id=len(trials),
            basis=basis,
            coefficients=coefficients,
            evaluator=evaluator,
            score_key=policy.score_key,
            reason="refined",
        )
        trials.append(refined)


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
