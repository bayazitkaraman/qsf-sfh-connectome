"""
QRC-v2 Source Coordinate System.

This experiment is closest to the original idea:

1. Choose a small set of source/anchor regions.
2. Describe every other region by its relational response to those sources.
3. Test whether those source responses recover coordinates and hemisphere.

Two decoding modes are compared:
- barycentric: no learning, just response-weighted source coordinates
- global_ridge: train on subjects, evaluate on held-out subjects

The script uses repeated balanced random anchor sets and compares QRC against
classical heat diffusion, shortest-path geometry, and direct edge weights.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain_quantum.linear_decoder import ridge_coefficients

from brain_quantum.analysis_suite import (  # noqa: E402
    all_pairs_shortest_path,
    anchor_metrics,
    barycentric_predict,
    compute_walk_stack,
    finite_position_mask,
    mean_by_group,
    position_extent,
    response_coordinate_moments,
)
from brain_quantum.qrc_connectome import (  # noqa: E402
    Connectome,
    normalized_laplacian,
    parse_graphml,
    parse_times,
    transform_weights,
    write_csv,
)


@dataclass
class SubjectRecord:
    connectome: Connectome
    weights: np.ndarray
    shortest_paths: np.ndarray
    spectral_embedding: np.ndarray
    q_real: list[np.ndarray]
    q_imag: list[np.ndarray]
    q_prob: list[np.ndarray]
    c_heat: list[np.ndarray]
    extent: float
    c_heat_raw: list[np.ndarray] | None = None


@dataclass
class RidgeModel:
    mean: np.ndarray
    std: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray


def mean_coordinate_template(records: list[SubjectRecord]) -> np.ndarray:
    """Average node coordinates over finite observations in training records."""
    if not records:
        raise ValueError("At least one training record is required.")
    positions = np.stack(
        [np.asarray(record.connectome.positions, dtype=float) for record in records],
        axis=0,
    )
    finite = np.all(np.isfinite(positions), axis=2)
    counts = finite.sum(axis=0)[:, None]
    sums = np.where(finite[:, :, None], positions, 0.0).sum(axis=0)
    template = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=template, where=counts > 0)
    return template


def eligible_source_indices(records: list[SubjectRecord]) -> np.ndarray:
    """Return nodes with connectivity and a coordinate in the training data."""
    if not records:
        raise ValueError("At least one training record is required.")
    ever_connected = np.zeros(records[0].weights.shape[0], dtype=bool)
    ever_positioned = np.zeros(records[0].weights.shape[0], dtype=bool)
    for record in records:
        ever_connected |= record.weights.sum(axis=1) > 0
        ever_positioned |= finite_position_mask(record.connectome.positions)
    return np.where(ever_connected & ever_positioned)[0].astype(int)


def cache_subject(path: Path, times: list[float], weight_transform: str) -> SubjectRecord:
    connectome = parse_graphml(path)
    weights = transform_weights(connectome.adjacency, weight_transform).astype(np.float32)
    laplacian = normalized_laplacian(weights)
    walk = compute_walk_stack(laplacian, times)
    shortest_paths = all_pairs_shortest_path(weights).astype(np.float32)
    spectral_embedding = laplacian_eigenmap(walk["evecs"]).astype(np.float32)
    return SubjectRecord(
        connectome=connectome,
        weights=weights,
        shortest_paths=shortest_paths,
        spectral_embedding=spectral_embedding,
        q_real=[arr.astype(np.float32) for arr in walk["q_real"]],
        q_imag=[arr.astype(np.float32) for arr in walk["q_imag"]],
        q_prob=[arr.astype(np.float32) for arr in walk["q_prob"]],
        c_heat=[arr.astype(np.float32) for arr in walk["c_heat"]],
        extent=position_extent(connectome.positions),
        c_heat_raw=[arr.astype(np.float32) for arr in walk["c_heat_raw"]],
    )


def choose_balanced_anchors(hemispheres: list[str], count: int, rng: np.random.Generator) -> np.ndarray:
    left = np.array([idx for idx, hemi in enumerate(hemispheres) if hemi.startswith("left")], dtype=int)
    right = np.array([idx for idx, hemi in enumerate(hemispheres) if hemi.startswith("right")], dtype=int)
    unknown = np.array([idx for idx, hemi in enumerate(hemispheres) if not (hemi.startswith("left") or hemi.startswith("right"))], dtype=int)

    left_count = min(len(left), count // 2)
    right_count = min(len(right), count - left_count)
    chosen = []
    if left_count:
        chosen.extend(rng.choice(left, size=left_count, replace=False).tolist())
    if right_count:
        chosen.extend(rng.choice(right, size=right_count, replace=False).tolist())

    while len(chosen) < count:
        pool = np.array([idx for idx in range(len(hemispheres)) if idx not in chosen], dtype=int)
        if len(pool) == 0:
            break
        chosen.append(int(rng.choice(pool)))

    # Occasionally include midline/unknown structures like brain stem if a count
    # is large enough; they can act as useful coordinate anchors.
    if len(unknown) and count >= 8 and rng.random() < 0.35:
        replace_idx = int(rng.integers(0, len(chosen)))
        chosen[replace_idx] = int(rng.choice(unknown))

    return np.array(sorted(set(chosen)), dtype=int)


def response_scale(matrix: np.ndarray) -> float:
    finite = matrix[np.isfinite(matrix)]
    if len(finite) == 0:
        return 1.0
    scale = float(np.median(finite))
    return max(scale, 1e-6)


def laplacian_eigenmap(evecs: np.ndarray, max_dims: int = 64) -> np.ndarray:
    dims = min(max_dims, max(0, evecs.shape[1] - 1))
    if dims == 0:
        return np.zeros((evecs.shape[0], 1), dtype=float)
    return evecs[:, 1 : dims + 1]


def spectral_source_response(record: SubjectRecord, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dims = min(record.spectral_embedding.shape[1], max(8, len(anchors)))
    embedding = record.spectral_embedding[:, :dims]
    anchor_embedding = embedding[anchors]
    diffs = embedding[:, None, :] - anchor_embedding[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    scale = response_scale(distances)
    response = np.exp(-distances / scale)
    features = np.concatenate([distances / scale, response], axis=1)
    return features, response


def anchor_features(
    record: SubjectRecord,
    anchors: np.ndarray,
    method: str,
    qrc_variant: str = "full",
    feature_mode: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    if method == "qrc":
        channel_sets = {
            "full": {"real", "imag", "prob"},
            "real": {"real"},
            "imag": {"imag"},
            "prob": {"prob"},
            "real_imag": {"real", "imag"},
            "real_prob": {"real", "prob"},
            "imag_prob": {"imag", "prob"},
        }
        if qrc_variant not in channel_sets:
            raise ValueError(f"Unknown QRC variant: {qrc_variant}")
        selected_channels = channel_sets[qrc_variant]
        blocks = []
        for real, imag, prob in zip(record.q_real, record.q_imag, record.q_prob):
            if "real" in selected_channels:
                blocks.append(real[:, anchors])
            if "imag" in selected_channels:
                blocks.append(imag[:, anchors])
            if "prob" in selected_channels:
                blocks.append(prob[:, anchors])
        base = np.concatenate(blocks, axis=1)
        response = np.mean([prob[:, anchors] for prob in record.q_prob], axis=0)
    elif method in {"classical", "classical_raw"}:
        heat_stack = record.c_heat if method == "classical" else record.c_heat_raw
        if heat_stack is None:
            raise ValueError("Raw heat kernels were not cached for this record.")
        base = np.concatenate([heat[:, anchors] for heat in heat_stack], axis=1)
        response = np.mean([heat[:, anchors] for heat in heat_stack], axis=0)
    elif method == "shortest_path":
        sp = record.shortest_paths[:, anchors]
        scale = response_scale(sp)
        response = np.exp(-sp / scale)
        base = np.concatenate([sp / scale, response], axis=1)
    elif method == "direct_weight":
        direct = record.weights[:, anchors]
        response = direct
        base = direct
    elif method == "spectral_source":
        base, response = spectral_source_response(record, anchors)
    else:
        raise ValueError(f"Unknown method: {method}")

    if feature_mode not in {"full", "dynamic", "moments"}:
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    bary, spread = response_coordinate_moments(response, record.connectome.positions[anchors])
    moments = np.concatenate([bary, spread], axis=1).astype(np.float32)
    if feature_mode == "dynamic":
        features = base
    elif feature_mode == "moments":
        features = moments
    else:
        features = np.concatenate([base, moments], axis=1)
    return features.astype(np.float32), response.astype(np.float32)


def non_anchor_mask(n: int, anchors: np.ndarray) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[anchors] = False
    return mask


def fit_global_ridge(
    records: list[SubjectRecord],
    anchors: np.ndarray,
    method: str,
    alpha: float,
    qrc_variant: str = "full",
) -> RidgeModel:
    x_parts = []
    y_parts = []
    for record in records:
        features, _ = anchor_features(record, anchors, method, qrc_variant=qrc_variant)
        mask = non_anchor_mask(features.shape[0], anchors)
        mask &= finite_position_mask(record.connectome.positions)
        x_parts.append(features[mask])
        y_parts.append(record.connectome.positions[mask].astype(np.float32))

    if not x_parts or sum(part.shape[0] for part in x_parts) == 0:
        raise ValueError("No finite coordinate targets available for ridge fitting.")

    x = np.nan_to_num(np.vstack(x_parts).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(np.vstack(y_parts).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    y_mean = y.mean(axis=0, keepdims=True)
    yc = y - y_mean

    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    coef = ridge_coefficients(xs, yc, alpha)
    return RidgeModel(mean=mean, std=std, y_mean=y_mean, coef=coef)


def predict_ridge(model: RidgeModel, features: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    xs = (clean - model.mean) / model.std
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    return xs @ model.coef + model.y_mean


def normal_approx_p_from_t(t_value: float) -> float:
    # Large paired samples make the normal approximation acceptable for this
    # first report. p = 2 * survival(|z|).
    return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def paired_tests(rows: list[dict], baseline_method: str = "qrc") -> list[dict]:
    index = {}
    for row in rows:
        key = (
            row["sample_id"],
            row["subject_id"],
            row["anchor_count"],
            row["mode"],
            row["method"],
        )
        index[key] = row

    methods = sorted({row["method"] for row in rows if row["method"] != baseline_method})
    modes = sorted({row["mode"] for row in rows})
    counts = sorted({row["anchor_count"] for row in rows}, key=int)
    metrics = {
        "normalized_rmse": "lower",
        "hemisphere_accuracy": "higher",
        "distance_corr": "higher",
    }
    tests = []
    for count in counts:
        for mode in modes:
            for other in methods:
                for metric, direction in metrics.items():
                    diffs = []
                    wins = 0
                    total = 0
                    for row in rows:
                        if row["anchor_count"] != count or row["mode"] != mode or row["method"] != baseline_method:
                            continue
                        other_key = (row["sample_id"], row["subject_id"], count, mode, other)
                        if other_key not in index:
                            continue
                        q_val = float(row[metric])
                        o_val = float(index[other_key][metric])
                        if not (np.isfinite(q_val) and np.isfinite(o_val)):
                            continue
                        diff = q_val - o_val
                        diffs.append(diff)
                        if direction == "lower":
                            wins += int(q_val < o_val)
                        else:
                            wins += int(q_val > o_val)
                        total += 1
                    if total < 3:
                        continue
                    arr = np.array(diffs, dtype=float)
                    mean = float(arr.mean())
                    std = float(arr.std(ddof=1))
                    t_value = mean / (std / math.sqrt(total)) if std > 0 else 0.0
                    tests.append(
                        {
                            "anchor_count": count,
                            "mode": mode,
                            "metric": metric,
                            "qrc_compared_to": other,
                            "direction": direction,
                            "n": total,
                            "mean_delta_qrc_minus_other": mean,
                            "paired_t": t_value,
                            "normal_approx_p": normal_approx_p_from_t(t_value),
                            "qrc_win_rate": wins / total,
                        }
                    )
    return tests


def write_v2_summary(output_dir: Path, summary_rows: list[dict], test_rows: list[dict], pair_rows: list[dict]) -> None:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def table(rows: list[dict], fields: list[str], limit: int = 20) -> str:
        header = "| " + " | ".join(fields) + " |"
        sep = "| " + " | ".join(["---"] * len(fields)) + " |"
        body = []
        for row in rows[:limit]:
            body.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
        return "\n".join([header, sep, *body])

    low_error = sorted(summary_rows, key=lambda row: float(row["normalized_rmse_mean"]))
    high_hemi = sorted(summary_rows, key=lambda row: -float(row["hemisphere_accuracy_mean"]))
    qrc_rows = [row for row in low_error if row["method"] == "qrc"]
    qrc_tests = [
        row
        for row in pair_rows
        if row["qrc_compared_to"] == "classical"
        and row["metric"] in {"normalized_rmse", "hemisphere_accuracy"}
    ]
    qrc_tests = sorted(qrc_tests, key=lambda row: (int(row["anchor_count"]), row["mode"], row["metric"]))

    text = f"""# QRC-v2 Source Coordinate Summary

Rows evaluated: {len(test_rows)}

## Lowest Coordinate Error

{table(low_error, ["anchor_count", "method", "mode", "normalized_rmse_mean", "hemisphere_accuracy_mean", "distance_corr_mean"], 16)}

## Highest Hemisphere Recovery

{table(high_hemi, ["anchor_count", "method", "mode", "hemisphere_accuracy_mean", "normalized_rmse_mean", "distance_corr_mean"], 16)}

## Best QRC Rows

{table(qrc_rows, ["anchor_count", "method", "mode", "normalized_rmse_mean", "hemisphere_accuracy_mean", "distance_corr_mean"], 12)}

## QRC vs Classical Paired Tests

`mean_delta_qrc_minus_other` is QRC minus the compared method. For RMSE, lower
is better, so negative means QRC is better. For hemisphere accuracy, positive
means QRC is better.

{table(qrc_tests, ["anchor_count", "mode", "metric", "mean_delta_qrc_minus_other", "qrc_win_rate", "normal_approx_p"], 16)}

## Interpretation

This v2 experiment uses repeated balanced random source regions and evaluates on
held-out subjects. It tests whether a source-defined relational coordinate
system generalizes across brains.
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QRC-v2 source coordinate experiments.")
    parser.add_argument("--input-dir", type=Path, default=Path("research/data/brain_graph_hcp_86_nodes/graphml"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/outputs/qrc_v2_source_coordinates"))
    parser.add_argument("--subjects", type=int, default=1064)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--anchor-counts", type=parse_ints, default=parse_ints("4,8,16,32"))
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--times", type=parse_times, default=parse_times("0.25,0.5,1,2,4,8"))
    parser.add_argument("--weight-transform", choices=["raw", "log1p", "binary"], default="log1p")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-alpha", type=float, default=25.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = sorted(args.input_dir.glob("*.graphml"))[: args.subjects]
    if not files:
        raise SystemExit(f"No GraphML files found in {args.input_dir}")

    shuffled = np.array(files, dtype=object)
    rng.shuffle(shuffled)
    train_n = max(1, int(len(shuffled) * args.train_fraction))
    train_files = list(shuffled[:train_n])
    test_files = list(shuffled[train_n:])

    print(f"Caching {len(train_files)} train subjects and {len(test_files)} test subjects")
    train_records = [cache_subject(path, args.times, args.weight_transform) for path in train_files]
    test_records = [cache_subject(path, args.times, args.weight_transform) for path in test_files]

    reference = train_records[0].connectome
    methods = ["qrc", "classical", "shortest_path", "direct_weight", "spectral_source"]
    rows = []
    anchor_rows = []

    sample_id = 0
    for count in args.anchor_counts:
        for repeat in range(args.repeats):
            sample_id += 1
            anchors = choose_balanced_anchors(reference.hemispheres, count, rng)
            anchor_names = [reference.node_names[idx] for idx in anchors]
            anchor_rows.append(
                {
                    "sample_id": sample_id,
                    "anchor_count": len(anchors),
                    "repeat": repeat,
                    "anchor_ids": " ".join(reference.node_ids[idx] for idx in anchors),
                    "anchor_names": " | ".join(anchor_names),
                }
            )
            print(f"Sample {sample_id}: {len(anchors)} anchors")

            ridge_models = {
                method: fit_global_ridge(train_records, anchors, method, alpha=args.ridge_alpha)
                for method in methods
            }

            for record in test_records:
                for method in methods:
                    features, response = anchor_features(record, anchors, method)
                    bary_pred = barycentric_predict(response, record.connectome.positions[anchors])
                    metric = anchor_metrics(
                        record.connectome,
                        method,
                        bary_pred,
                        anchors,
                        "random_balanced",
                        len(anchors),
                        "barycentric",
                        record.extent,
                    )
                    metric["sample_id"] = sample_id
                    rows.append(metric)

                    ridge_pred = predict_ridge(ridge_models[method], features)
                    metric = anchor_metrics(
                        record.connectome,
                        method,
                        ridge_pred,
                        anchors,
                        "random_balanced",
                        len(anchors),
                        "global_ridge",
                        record.extent,
                    )
                    metric["sample_id"] = sample_id
                    rows.append(metric)

    fields = [
        "sample_id",
        "subject_id",
        "anchor_strategy",
        "anchor_count",
        "method",
        "mode",
        "normalized_rmse",
        "mean_error",
        "x_corr",
        "y_corr",
        "z_corr",
        "distance_corr",
        "hemisphere_accuracy",
    ]
    write_csv(args.output_dir / "v2_source_coordinate_metrics.csv", rows, fields)
    write_csv(args.output_dir / "v2_anchor_sets.csv", anchor_rows, ["sample_id", "anchor_count", "repeat", "anchor_ids", "anchor_names"])

    summary = mean_by_group(
        rows,
        ["anchor_count", "method", "mode"],
        ["normalized_rmse", "hemisphere_accuracy", "distance_corr", "x_corr", "y_corr", "z_corr"],
    )
    summary_fields = [
        "anchor_count",
        "method",
        "mode",
        "normalized_rmse_mean",
        "normalized_rmse_std",
        "hemisphere_accuracy_mean",
        "hemisphere_accuracy_std",
        "distance_corr_mean",
        "distance_corr_std",
        "x_corr_mean",
        "x_corr_std",
        "y_corr_mean",
        "y_corr_std",
        "z_corr_mean",
        "z_corr_std",
        "n",
    ]
    write_csv(args.output_dir / "v2_summary.csv", summary, summary_fields)

    pair_rows = paired_tests(rows)
    write_csv(
        args.output_dir / "v2_pairwise_tests.csv",
        pair_rows,
        [
            "anchor_count",
            "mode",
            "metric",
            "qrc_compared_to",
            "direction",
            "n",
            "mean_delta_qrc_minus_other",
            "paired_t",
            "normal_approx_p",
            "qrc_win_rate",
        ],
    )
    write_v2_summary(args.output_dir, summary, rows, pair_rows)
    print(f"Wrote QRC-v2 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
