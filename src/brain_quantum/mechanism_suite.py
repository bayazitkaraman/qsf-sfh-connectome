"""Mechanistic analyses for QRC-v2.

This suite asks what the QRC-v2 decoder is using, not just whether it works:

1. Which single source regions are most informative?
2. Do nested source sets behave like a coarse-to-fine hierarchy?
3. Which real/imaginary/probability channel combinations carry the advantage?
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain_quantum.analysis_suite import (  # noqa: E402
    anchor_metrics,
    farthest_spread_anchors,
    finite_position_mask,
    mean_by_group,
    pairwise_distance_corr_mask,
    response_coordinate_moments,
    source_informed_hemi_accuracy,
)
from brain_quantum.qrc_connectome import node_influence, parse_times, pearson_abs, write_csv  # noqa: E402
from brain_quantum.linear_decoder import ridge_coefficients
from brain_quantum.source_coordinate_v2 import (  # noqa: E402
    RidgeModel,
    SubjectRecord,
    anchor_features,
    cache_subject,
    choose_balanced_anchors,
    fit_global_ridge,
    mean_coordinate_template,
    non_anchor_mask,
    predict_ridge,
)


CHANNEL_VARIANTS = {
    "heat_classical": ("heat",),
    "q_real": ("real",),
    "q_imag": ("imag",),
    "q_prob": ("prob",),
    "phase_real_imag": ("real", "imag"),
    "real_prob": ("real", "prob"),
    "imag_prob": ("imag", "prob"),
    "full_qrc": ("real", "imag", "prob"),
    "qprob_minus_heat": ("contrast",),
    "full_qrc_plus_heat": ("real", "imag", "prob", "heat"),
}


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def clean_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict], fields: list[str], limit: int = 12) -> str:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = []
    for row in rows[:limit]:
        body.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join([header, sep, *body])


def train_test_split(
    records: list[SubjectRecord],
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[SubjectRecord], list[SubjectRecord]]:
    indices = np.arange(len(records))
    rng.shuffle(indices)
    train_n = max(1, int(len(indices) * train_fraction))
    train_idx = indices[:train_n]
    test_idx = indices[train_n:]
    if len(test_idx) == 0:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return [records[i] for i in train_idx], [records[i] for i in test_idx]


def node_descriptors(records: list[SubjectRecord]) -> list[dict]:
    reference = records[0].connectome
    n = len(reference.node_ids)
    influence_values = np.zeros((len(records), n), dtype=float)
    strength_values = np.zeros((len(records), n), dtype=float)
    cross_hemi_values = np.zeros((len(records), n), dtype=float)
    entropy_values = np.zeros((len(records), n), dtype=float)

    for r_idx, record in enumerate(records):
        q_avg = np.mean(np.stack(record.q_prob, axis=0), axis=0)
        influence_values[r_idx] = node_influence(q_avg, record.connectome.hemispheres)
        strength_values[r_idx] = record.weights.sum(axis=1)
        for source in range(n):
            p = q_avg[:, source].astype(float)
            p = p / max(float(p.sum()), 1e-12)
            positive = p > 0
            entropy_values[r_idx, source] = -float(np.sum(p[positive] * np.log(p[positive]))) / math.log(n)
            cross_hemi_values[r_idx, source] = float(
                sum(
                    p[target]
                    for target in range(n)
                    if record.connectome.hemispheres[target] != record.connectome.hemispheres[source]
                )
            )

    rows = []
    positions = mean_coordinate_template(records)
    for idx in range(n):
        rows.append(
            {
                "node_index": idx,
                "node_id": reference.node_ids[idx],
                "node_name": reference.node_names[idx],
                "region": reference.regions[idx],
                "hemisphere": reference.hemispheres[idx],
                "x": float(positions[idx, 0]),
                "y": float(positions[idx, 1]),
                "z": float(positions[idx, 2]),
                "mean_qrc_influence": float(np.mean(influence_values[:, idx])),
                "mean_weighted_strength": float(np.mean(strength_values[:, idx])),
                "mean_response_entropy": float(np.mean(entropy_values[:, idx])),
                "mean_cross_hemisphere_response": float(np.mean(cross_hemi_values[:, idx])),
            }
        )
    return rows


def evaluate_anchor_model(
    train_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    anchors: np.ndarray,
    method: str,
    alpha: float,
    strategy: str,
    sample_id: str,
    qrc_variant: str = "full",
) -> list[dict]:
    model = fit_global_ridge(train_records, anchors, method, alpha=alpha, qrc_variant=qrc_variant)
    rows = []
    for record in test_records:
        features, _ = anchor_features(record, anchors, method, qrc_variant=qrc_variant)
        pred = predict_ridge(model, features)
        row = anchor_metrics(
            record.connectome,
            method,
            pred,
            anchors,
            strategy,
            len(anchors),
            "global_ridge",
            record.extent,
        )
        row["sample_id"] = sample_id
        row["qrc_variant"] = qrc_variant if method == "qrc" else "na"
        rows.append(row)
    return rows


def masked_anchor_metrics(
    record: SubjectRecord,
    method_name: str,
    predicted: np.ndarray,
    anchors: np.ndarray,
    strategy: str,
    count: int,
    mode: str,
    target_mask: np.ndarray,
) -> dict:
    mask = np.array(target_mask, dtype=bool)
    if np.any(mask[anchors]):
        mask[anchors] = False
    mask &= finite_position_mask(record.connectome.positions)
    if int(mask.sum()) == 0:
        return {
            "subject_id": record.connectome.subject_id,
            "anchor_strategy": strategy,
            "anchor_count": count,
            "method": method_name,
            "mode": mode,
            "normalized_rmse": float("nan"),
            "mean_error": float("nan"),
            "x_corr": float("nan"),
            "y_corr": float("nan"),
            "z_corr": float("nan"),
            "distance_corr": float("nan"),
            "hemisphere_accuracy": float("nan"),
        }
    errors = np.linalg.norm(predicted[mask] - record.connectome.positions[mask], axis=1)
    return {
        "subject_id": record.connectome.subject_id,
        "anchor_strategy": strategy,
        "anchor_count": count,
        "method": method_name,
        "mode": mode,
        "normalized_rmse": float(np.sqrt(np.mean(errors**2)) / record.extent),
        "mean_error": float(np.mean(errors)),
        "x_corr": pearson_abs(predicted[mask, 0], record.connectome.positions[mask, 0]),
        "y_corr": pearson_abs(predicted[mask, 1], record.connectome.positions[mask, 1]),
        "z_corr": pearson_abs(predicted[mask, 2], record.connectome.positions[mask, 2]),
        "distance_corr": pairwise_distance_corr_mask(predicted, record.connectome.positions, mask),
        "hemisphere_accuracy": source_informed_hemi_accuracy(
            predicted[:, 0],
            record.connectome.hemispheres,
            mask,
            record.connectome.positions,
            anchors,
        ),
    }


def evaluate_anchor_model_fixed_targets(
    train_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    anchors: np.ndarray,
    target_indices: np.ndarray,
    alpha: float,
    strategy: str,
    sample_id: str,
) -> list[dict]:
    model = fit_global_ridge(train_records, anchors, "qrc", alpha=alpha, qrc_variant="full")
    target_mask = np.zeros(len(train_records[0].connectome.node_ids), dtype=bool)
    target_mask[target_indices] = True
    rows = []
    for record in test_records:
        features, _ = anchor_features(record, anchors, "qrc", qrc_variant="full")
        pred = predict_ridge(model, features)
        row = masked_anchor_metrics(
            record,
            "qrc",
            pred,
            anchors,
            strategy,
            len(anchors),
            "global_ridge_fixed_targets",
            target_mask,
        )
        row["sample_id"] = sample_id
        row["qrc_variant"] = "full"
        row["target_count"] = int(target_mask.sum())
        rows.append(row)
    return rows


def summarize_source_importance(
    source_metric_rows: list[dict],
    descriptors: list[dict],
) -> list[dict]:
    grouped = defaultdict(list)
    for row in source_metric_rows:
        grouped[(int(row["source_index"]), row["method"])].append(row)

    descriptor_by_index = {int(row["node_index"]): row for row in descriptors}
    out = []
    for idx, descriptor in descriptor_by_index.items():
        qrc_rows = grouped.get((idx, "qrc"), [])
        classical_rows = grouped.get((idx, "classical"), [])

        def mean_metric(rows: list[dict], metric: str) -> float:
            values = [clean_float(row[metric]) for row in rows]
            values = [value for value in values if np.isfinite(value)]
            return float(np.mean(values)) if values else float("nan")

        qrc_rmse = mean_metric(qrc_rows, "normalized_rmse")
        classical_rmse = mean_metric(classical_rows, "normalized_rmse")
        out.append(
            {
                **descriptor,
                "single_qrc_rmse_mean": qrc_rmse,
                "single_classical_rmse_mean": classical_rmse,
                "single_qrc_minus_classical_rmse": qrc_rmse - classical_rmse,
                "single_qrc_hemi_mean": mean_metric(qrc_rows, "hemisphere_accuracy"),
                "single_qrc_distance_corr_mean": mean_metric(qrc_rows, "distance_corr"),
                "single_classical_distance_corr_mean": mean_metric(classical_rows, "distance_corr"),
                "single_source_n": len(qrc_rows),
            }
        )

    out.sort(key=lambda row: row["single_qrc_rmse_mean"])
    for rank, row in enumerate(out, start=1):
        row["single_qrc_rank"] = rank
    return out


def signed_pearson(x_values: list[float], y_values: list[float]) -> float:
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask] - float(np.mean(x[mask]))
    y = y[mask] - float(np.mean(y[mask]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def source_descriptor_correlations(source_summary: list[dict]) -> list[dict]:
    descriptors = [
        "mean_qrc_influence",
        "mean_weighted_strength",
        "mean_response_entropy",
        "mean_cross_hemisphere_response",
    ]
    targets = [
        "single_qrc_rmse_mean",
        "single_qrc_distance_corr_mean",
        "single_qrc_minus_classical_rmse",
    ]
    rows = []
    for descriptor in descriptors:
        for target in targets:
            rows.append(
                {
                    "descriptor": descriptor,
                    "target": target,
                    "signed_pearson": signed_pearson(
                        [clean_float(row[descriptor]) for row in source_summary],
                        [clean_float(row[target]) for row in source_summary],
                    ),
                }
            )
    return rows


def run_source_importance(
    train_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    descriptors: list[dict],
    alpha: float,
) -> tuple[list[dict], list[dict]]:
    reference = train_records[0].connectome
    metric_rows = []
    for idx in range(len(reference.node_ids)):
        anchors = np.array([idx], dtype=int)
        for method in ("qrc", "classical"):
            metric_rows.extend(
                {
                    **row,
                    "source_index": idx,
                    "source_node_id": reference.node_ids[idx],
                    "source_node_name": reference.node_names[idx],
                    "source_region": reference.regions[idx],
                    "source_hemisphere": reference.hemispheres[idx],
                }
                for row in evaluate_anchor_model(
                    train_records,
                    test_records,
                    anchors,
                    method=method,
                    alpha=alpha,
                    strategy="single_source",
                    sample_id=f"single-{idx}-{method}",
                )
            )
    summary_rows = summarize_source_importance(metric_rows, descriptors)
    return metric_rows, summary_rows


def balanced_ranked_anchors(
    ranked_indices: list[int],
    hemispheres: list[str],
    count: int,
) -> np.ndarray:
    left = [idx for idx in ranked_indices if hemispheres[idx].startswith("left")]
    right = [idx for idx in ranked_indices if hemispheres[idx].startswith("right")]
    unknown = [idx for idx in ranked_indices if not (hemispheres[idx].startswith("left") or hemispheres[idx].startswith("right"))]

    selected: list[int] = []
    left_target = count // 2
    right_target = count - left_target
    selected.extend(left[:left_target])
    selected.extend(right[:right_target])
    for idx in ranked_indices:
        if len(selected) >= count:
            break
        if idx not in selected:
            selected.append(idx)
    for idx in unknown:
        if len(selected) >= count:
            break
        if idx not in selected:
            selected.append(idx)
    return np.array(sorted(selected[:count]), dtype=int)


def top_ranked_anchors(ranked_indices: list[int], count: int) -> np.ndarray:
    return np.array(sorted(ranked_indices[:count]), dtype=int)


def run_hierarchy_analysis(
    train_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    source_summary: list[dict],
    descriptors: list[dict],
    counts: list[int],
    random_repeats: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict]]:
    reference = train_records[0].connectome
    training_positions = mean_coordinate_template(train_records)
    n = len(reference.node_ids)
    counts = [count for count in counts if 1 <= count < n]
    ranked_by_single = [int(row["node_index"]) for row in sorted(source_summary, key=lambda row: row["single_qrc_rmse_mean"])]
    ranked_by_influence = [
        int(row["node_index"]) for row in sorted(descriptors, key=lambda row: -float(row["mean_qrc_influence"]))
    ]
    ranked_by_strength = [
        int(row["node_index"]) for row in sorted(descriptors, key=lambda row: -float(row["mean_weighted_strength"]))
    ]

    metric_rows = []
    anchor_rows = []

    for count in counts:
        strategies = {
            "single_source_ranked": top_ranked_anchors(ranked_by_single, count),
            "single_source_ranked_balanced": balanced_ranked_anchors(ranked_by_single, reference.hemispheres, count),
            "qrc_influence_ranked": top_ranked_anchors(ranked_by_influence, count),
            "strength_hubs": top_ranked_anchors(ranked_by_strength, count),
            "spatial_spread": farthest_spread_anchors(training_positions, count),
        }
        for strategy, anchors in strategies.items():
            sample_id = f"{strategy}-{count}"
            anchor_rows.append(anchor_record(reference, anchors, sample_id, strategy, count))
            metric_rows.extend(
                evaluate_anchor_model(
                    train_records,
                    test_records,
                    anchors,
                    method="qrc",
                    alpha=alpha,
                    strategy=strategy,
                    sample_id=sample_id,
                )
            )
        for repeat in range(random_repeats):
            anchors = choose_balanced_anchors(reference.hemispheres, count, rng)
            sample_id = f"random_balanced-{count}-{repeat + 1}"
            anchor_rows.append(anchor_record(reference, anchors, sample_id, "random_balanced", count))
            metric_rows.extend(
                evaluate_anchor_model(
                    train_records,
                    test_records,
                    anchors,
                    method="qrc",
                    alpha=alpha,
                    strategy="random_balanced",
                    sample_id=sample_id,
                )
            )

    summary_rows = mean_by_group(
        metric_rows,
        ["anchor_strategy", "anchor_count"],
        ["normalized_rmse", "hemisphere_accuracy", "distance_corr", "x_corr", "y_corr", "z_corr"],
    )
    increments = hierarchy_increments(summary_rows)
    return metric_rows, summary_rows, anchor_rows + increments


def anchor_record(reference, anchors: np.ndarray, sample_id: str, strategy: str, count: int) -> dict:
    return {
        "sample_id": sample_id,
        "anchor_strategy": strategy,
        "anchor_count": count,
        "anchor_ids": " ".join(reference.node_ids[idx] for idx in anchors),
        "anchor_names": " | ".join(reference.node_names[idx] for idx in anchors),
        "anchor_regions": " | ".join(reference.regions[idx] for idx in anchors),
        "anchor_hemispheres": " | ".join(reference.hemispheres[idx] for idx in anchors),
    }


def hierarchy_increments(summary_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["anchor_strategy"]].append(row)
    out = []
    for strategy, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["anchor_count"]))
        previous = None
        for row in rows:
            current = {
                "sample_id": f"increment-{strategy}-{row['anchor_count']}",
                "anchor_strategy": f"{strategy}_increment",
                "anchor_count": row["anchor_count"],
                "anchor_ids": "",
                "anchor_names": "",
                "anchor_regions": "",
                "anchor_hemispheres": "",
                "rmse_gain_from_previous": "",
                "distance_corr_gain_from_previous": "",
                "x_corr_gain_from_previous": "",
                "y_corr_gain_from_previous": "",
                "z_corr_gain_from_previous": "",
            }
            if previous is not None:
                current["rmse_gain_from_previous"] = clean_float(previous["normalized_rmse_mean"]) - clean_float(
                    row["normalized_rmse_mean"]
                )
                current["distance_corr_gain_from_previous"] = clean_float(row["distance_corr_mean"]) - clean_float(
                    previous["distance_corr_mean"]
                )
                current["x_corr_gain_from_previous"] = clean_float(row["x_corr_mean"]) - clean_float(previous["x_corr_mean"])
                current["y_corr_gain_from_previous"] = clean_float(row["y_corr_mean"]) - clean_float(previous["y_corr_mean"])
                current["z_corr_gain_from_previous"] = clean_float(row["z_corr_mean"]) - clean_float(previous["z_corr_mean"])
            out.append(current)
            previous = row
    return out


def subset_farthest_spread_anchors(positions: np.ndarray, candidates: list[int], count: int) -> np.ndarray:
    candidate_arr = np.array(candidates, dtype=int)
    finite = finite_position_mask(positions[candidate_arr])
    finite_candidates = candidate_arr[finite]
    if len(finite_candidates) == 0:
        return np.array(sorted(candidate_arr[:count].tolist()), dtype=int)
    candidate_positions = positions[finite_candidates]
    first_local = int(np.argmin(candidate_positions[:, 0]))
    selected_local = [first_local]
    while len(selected_local) < min(count, len(finite_candidates)):
        selected_positions = candidate_positions[np.array(selected_local, dtype=int)]
        diffs = candidate_positions[:, None, :] - selected_positions[None, :, :]
        nearest = np.sqrt(np.sum(diffs * diffs, axis=2)).min(axis=1)
        nearest[np.array(selected_local, dtype=int)] = -1.0
        selected_local.append(int(np.argmax(nearest)))
    selected = finite_candidates[selected_local].tolist()
    if len(selected) < count:
        selected_set = set(selected)
        selected.extend(int(idx) for idx in candidate_arr if int(idx) not in selected_set)
    return np.array(sorted(selected[:count]), dtype=int)


def choose_balanced_from_pool(
    hemispheres: list[str],
    candidates: list[int],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate_set = set(candidates)
    left = np.array([idx for idx in candidates if hemispheres[idx].startswith("left")], dtype=int)
    right = np.array([idx for idx in candidates if hemispheres[idx].startswith("right")], dtype=int)
    unknown = np.array(
        [idx for idx in candidates if not (hemispheres[idx].startswith("left") or hemispheres[idx].startswith("right"))],
        dtype=int,
    )
    left_count = min(len(left), count // 2)
    right_count = min(len(right), count - left_count)
    chosen = []
    if left_count:
        chosen.extend(rng.choice(left, size=left_count, replace=False).tolist())
    if right_count:
        chosen.extend(rng.choice(right, size=right_count, replace=False).tolist())
    while len(chosen) < count:
        pool = np.array(sorted(candidate_set - set(chosen)), dtype=int)
        if len(pool) == 0:
            break
        chosen.append(int(rng.choice(pool)))
    if len(unknown) and count >= 8 and rng.random() < 0.35:
        replace_idx = int(rng.integers(0, len(chosen)))
        chosen[replace_idx] = int(rng.choice(unknown))
    return np.array(sorted(set(chosen)), dtype=int)


def run_fixed_target_hierarchy(
    train_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    source_summary: list[dict],
    descriptors: list[dict],
    counts: list[int],
    random_repeats: int,
    target_count: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    reference = train_records[0].connectome
    training_positions = mean_coordinate_template(train_records)
    n = len(reference.node_ids)
    ranked_by_single = [int(row["node_index"]) for row in sorted(source_summary, key=lambda row: row["single_qrc_rmse_mean"])]
    ranked_by_influence = [
        int(row["node_index"]) for row in sorted(descriptors, key=lambda row: -float(row["mean_qrc_influence"]))
    ]
    ranked_by_strength = [
        int(row["node_index"]) for row in sorted(descriptors, key=lambda row: -float(row["mean_weighted_strength"]))
    ]
    target_indices = np.array(sorted(ranked_by_single[-target_count:]), dtype=int)
    target_set = set(target_indices.tolist())
    candidate_ranked = [idx for idx in ranked_by_single if idx not in target_set]
    candidate_influence = [idx for idx in ranked_by_influence if idx not in target_set]
    candidate_strength = [idx for idx in ranked_by_strength if idx not in target_set]
    candidate_all = [idx for idx in range(n) if idx not in target_set]
    counts = [count for count in counts if 1 <= count <= len(candidate_all)]

    target_rows = [
        {
            "target_index": idx,
            "node_id": reference.node_ids[idx],
            "node_name": reference.node_names[idx],
            "region": reference.regions[idx],
            "hemisphere": reference.hemispheres[idx],
        }
        for idx in target_indices
    ]

    metric_rows = []
    anchor_rows = []
    for count in counts:
        strategies = {
            "fixed_single_source_ranked": top_ranked_anchors(candidate_ranked, count),
            "fixed_single_source_ranked_balanced": balanced_ranked_anchors(candidate_ranked, reference.hemispheres, count),
            "fixed_qrc_influence_ranked": top_ranked_anchors(candidate_influence, count),
            "fixed_strength_hubs": top_ranked_anchors(candidate_strength, count),
            "fixed_spatial_spread": subset_farthest_spread_anchors(training_positions, candidate_all, count),
        }
        for strategy, anchors in strategies.items():
            sample_id = f"{strategy}-{count}"
            anchor_rows.append(anchor_record(reference, anchors, sample_id, strategy, count))
            metric_rows.extend(
                evaluate_anchor_model_fixed_targets(
                    train_records,
                    test_records,
                    anchors,
                    target_indices,
                    alpha=alpha,
                    strategy=strategy,
                    sample_id=sample_id,
                )
            )
        for repeat in range(random_repeats):
            anchors = choose_balanced_from_pool(reference.hemispheres, candidate_all, count, rng)
            sample_id = f"fixed_random_balanced-{count}-{repeat + 1}"
            anchor_rows.append(anchor_record(reference, anchors, sample_id, "fixed_random_balanced", count))
            metric_rows.extend(
                evaluate_anchor_model_fixed_targets(
                    train_records,
                    test_records,
                    anchors,
                    target_indices,
                    alpha=alpha,
                    strategy="fixed_random_balanced",
                    sample_id=sample_id,
                )
            )

    summary_rows = mean_by_group(
        metric_rows,
        ["anchor_strategy", "anchor_count", "target_count"],
        ["normalized_rmse", "hemisphere_accuracy", "distance_corr", "x_corr", "y_corr", "z_corr"],
    )
    return metric_rows, summary_rows, anchor_rows, target_rows


def custom_channel_features(record: SubjectRecord, anchors: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    channels = CHANNEL_VARIANTS[variant]
    blocks = []
    for real, imag, prob, heat in zip(record.q_real, record.q_imag, record.q_prob, record.c_heat):
        for channel in channels:
            if channel == "real":
                blocks.append(real[:, anchors])
            elif channel == "imag":
                blocks.append(imag[:, anchors])
            elif channel == "prob":
                blocks.append(prob[:, anchors])
            elif channel == "heat":
                blocks.append(heat[:, anchors])
            elif channel == "contrast":
                blocks.append(prob[:, anchors] - heat[:, anchors])
            else:
                raise ValueError(f"Unknown channel: {channel}")

    base = np.concatenate(blocks, axis=1).astype(np.float32)
    if variant == "heat_classical":
        response = np.mean([heat[:, anchors] for heat in record.c_heat], axis=0)
    elif variant == "qprob_minus_heat":
        response = np.maximum(
            np.mean([prob[:, anchors] for prob in record.q_prob], axis=0)
            - np.mean([heat[:, anchors] for heat in record.c_heat], axis=0),
            0.0,
        )
    else:
        response = np.mean([prob[:, anchors] for prob in record.q_prob], axis=0)

    if float(np.sum(response)) <= 1e-12:
        response = np.mean([prob[:, anchors] for prob in record.q_prob], axis=0)

    bary, spread = response_coordinate_moments(
        response,
        record.connectome.positions[anchors],
    )
    features = np.concatenate([base, bary.astype(np.float32), spread.astype(np.float32)], axis=1)
    return features.astype(np.float32), response.astype(np.float32)


def fit_channel_ridge(
    records: list[SubjectRecord],
    anchors: np.ndarray,
    variant: str,
    alpha: float,
) -> RidgeModel:
    x_parts = []
    y_parts = []
    for record in records:
        features, _ = custom_channel_features(record, anchors, variant)
        mask = non_anchor_mask(features.shape[0], anchors)
        mask &= finite_position_mask(record.connectome.positions)
        x_parts.append(features[mask])
        y_parts.append(record.connectome.positions[mask].astype(np.float32))

    x = np.nan_to_num(np.vstack(x_parts).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(np.vstack(y_parts).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    y_mean = y.mean(axis=0, keepdims=True)
    yc = y - y_mean
    coef = ridge_coefficients(xs, yc, alpha)
    return RidgeModel(mean=mean, std=std, y_mean=y_mean, coef=coef)


def evaluate_channel_model(
    train_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    anchors: np.ndarray,
    variant: str,
    alpha: float,
    strategy: str,
    sample_id: str,
) -> list[dict]:
    model = fit_channel_ridge(train_records, anchors, variant=variant, alpha=alpha)
    rows = []
    for record in test_records:
        features, _ = custom_channel_features(record, anchors, variant)
        pred = predict_ridge(model, features)
        row = anchor_metrics(
            record.connectome,
            variant,
            pred,
            anchors,
            strategy,
            len(anchors),
            "global_ridge",
            record.extent,
        )
        row["sample_id"] = sample_id
        row["channel_variant"] = variant
        rows.append(row)
    return rows


def run_phase_analysis(
    train_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    source_summary: list[dict],
    counts: list[int],
    random_repeats: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict]]:
    reference = train_records[0].connectome
    ranked_by_single = [int(row["node_index"]) for row in sorted(source_summary, key=lambda row: row["single_qrc_rmse_mean"])]
    metric_rows = []
    anchor_rows = []
    phase_counts = [count for count in counts if 1 <= count < len(reference.node_ids)]

    for count in phase_counts:
        source_sets = [
            ("single_source_ranked_balanced", balanced_ranked_anchors(ranked_by_single, reference.hemispheres, count)),
        ]
        for repeat in range(random_repeats):
            source_sets.append((f"random_balanced_{repeat + 1}", choose_balanced_anchors(reference.hemispheres, count, rng)))

        for strategy, anchors in source_sets:
            anchor_rows.append(anchor_record(reference, anchors, f"phase-{strategy}-{count}", strategy, count))
            for variant in CHANNEL_VARIANTS:
                metric_rows.extend(
                    evaluate_channel_model(
                        train_records,
                        test_records,
                        anchors,
                        variant=variant,
                        alpha=alpha,
                        strategy=strategy,
                        sample_id=f"phase-{strategy}-{count}-{variant}",
                    )
                )

    summary_rows = mean_by_group(
        metric_rows,
        ["anchor_strategy", "anchor_count", "channel_variant"],
        ["normalized_rmse", "hemisphere_accuracy", "distance_corr", "x_corr", "y_corr", "z_corr"],
    )
    synergy_rows = phase_synergy(summary_rows)
    return metric_rows, summary_rows, anchor_rows + synergy_rows


def phase_synergy(summary_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in summary_rows:
        grouped[(row["anchor_strategy"], row["anchor_count"])][row["channel_variant"]] = row

    out = []
    for (strategy, count), by_variant in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        full = by_variant.get("full_qrc")
        if not full:
            continue
        singles = [by_variant[name] for name in ("q_real", "q_imag", "q_prob") if name in by_variant]
        pairs = [
            by_variant[name]
            for name in ("phase_real_imag", "real_prob", "imag_prob")
            if name in by_variant
        ]
        heat = by_variant.get("heat_classical")
        contrast = by_variant.get("qprob_minus_heat")
        plus_heat = by_variant.get("full_qrc_plus_heat")
        full_rmse = clean_float(full["normalized_rmse_mean"])

        def best_rmse(rows: list[dict]) -> float:
            values = [clean_float(row["normalized_rmse_mean"]) for row in rows]
            values = [value for value in values if np.isfinite(value)]
            return min(values) if values else float("nan")

        row = {
            "sample_id": f"phase-synergy-{strategy}-{count}",
            "anchor_strategy": f"{strategy}_phase_synergy",
            "anchor_count": count,
            "anchor_ids": "",
            "anchor_names": "",
            "anchor_regions": "",
            "anchor_hemispheres": "",
            "full_qrc_rmse": full_rmse,
            "best_single_channel_rmse": best_rmse(singles),
            "best_pair_channel_rmse": best_rmse(pairs),
            "heat_classical_rmse": clean_float(heat["normalized_rmse_mean"]) if heat else float("nan"),
            "qprob_minus_heat_rmse": clean_float(contrast["normalized_rmse_mean"]) if contrast else float("nan"),
            "full_qrc_plus_heat_rmse": clean_float(plus_heat["normalized_rmse_mean"]) if plus_heat else float("nan"),
        }
        row["gain_vs_best_single"] = row["best_single_channel_rmse"] - full_rmse
        row["gain_vs_best_pair"] = row["best_pair_channel_rmse"] - full_rmse
        row["gain_vs_heat"] = row["heat_classical_rmse"] - full_rmse
        row["plus_heat_minus_full"] = row["full_qrc_plus_heat_rmse"] - full_rmse
        out.append(row)
    return out


def write_bar_svg(path: Path, rows: list[dict], title: str, label_field: str, value_field: str, limit: int = 12) -> None:
    rows = rows[:limit]
    width = 980
    height = 120 + 34 * len(rows)
    left = 260
    right = 40
    values = np.array([clean_float(row[value_field]) for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(finite)) if len(finite) else 1.0
    vmin = float(np.min(finite)) if len(finite) else 0.0
    span = max(vmax - vmin, 1e-9)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="32" y="42" font-family="Arial" font-size="24" fill="#111827">{escape_svg(title)}</text>',
    ]
    for i, row in enumerate(rows):
        y = 82 + i * 34
        label = str(row[label_field])
        value = clean_float(row[value_field])
        bar_w = 30 + (value - vmin) / span * (width - left - right - 40)
        lines.append(f'<text x="32" y="{y + 18}" font-family="Arial" font-size="13" fill="#334155">{escape_svg(label)}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="22" rx="4" fill="#2563eb"/>')
        lines.append(
            f'<text x="{left + bar_w + 8:.2f}" y="{y + 16}" font-family="Arial" font-size="12" fill="#111827">{value:.4f}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_line_svg(path: Path, rows: list[dict], title: str, strategies: list[str]) -> None:
    width = 980
    height = 560
    left, top, right, bottom = 78, 78, 34, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#2563eb", "#0f766e", "#b45309", "#7c3aed", "#dc2626", "#475569"]
    selected = [row for row in rows if row["anchor_strategy"] in strategies]
    counts = sorted({int(row["anchor_count"]) for row in selected})
    values = [clean_float(row["normalized_rmse_mean"]) for row in selected]
    values = [value for value in values if np.isfinite(value)]
    ymin = min(values) if values else 0.0
    ymax = max(values) if values else 1.0
    yspan = max(ymax - ymin, 1e-9)

    def sx(count: int) -> float:
        if len(counts) == 1:
            return left + plot_w / 2
        return left + counts.index(count) / (len(counts) - 1) * plot_w

    def sy(value: float) -> float:
        return top + (ymax - value) / yspan * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="32" y="42" font-family="Arial" font-size="24" fill="#111827">{escape_svg(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#94a3b8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#94a3b8"/>',
    ]
    for count in counts:
        x = sx(count)
        lines.append(f'<text x="{x - 10}" y="{top + plot_h + 30}" font-family="Arial" font-size="12" fill="#334155">{count}</text>')
    for tick in range(5):
        value = ymin + tick / 4 * yspan
        y = sy(value)
        lines.append(f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        lines.append(f'<text x="24" y="{y + 4:.2f}" font-family="Arial" font-size="11" fill="#475569">{value:.3f}</text>')

    for s_idx, strategy in enumerate(strategies):
        strategy_rows = sorted(
            [row for row in selected if row["anchor_strategy"] == strategy],
            key=lambda row: int(row["anchor_count"]),
        )
        points = [(sx(int(row["anchor_count"])), sy(clean_float(row["normalized_rmse_mean"]))) for row in strategy_rows]
        if len(points) >= 2:
            path_data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            lines.append(f'<polyline points="{path_data}" fill="none" stroke="{colors[s_idx % len(colors)]}" stroke-width="3"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{colors[s_idx % len(colors)]}"/>')
        legend_y = 74 + s_idx * 22
        lines.append(f'<rect x="690" y="{legend_y - 11}" width="14" height="14" fill="{colors[s_idx % len(colors)]}"/>')
        lines.append(
            f'<text x="712" y="{legend_y}" font-family="Arial" font-size="12" fill="#111827">{escape_svg(strategy)}</text>'
        )
    lines.append('<text x="430" y="535" font-family="Arial" font-size="13" fill="#334155">Number of source regions</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def escape_svg(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_summary(
    output_dir: Path,
    docs_path: Path | None,
    subject_count: int,
    train_count: int,
    test_count: int,
    source_summary: list[dict],
    source_correlation_rows: list[dict],
    hierarchy_summary: list[dict],
    fixed_hierarchy_summary: list[dict],
    phase_summary: list[dict],
    synergy_rows: list[dict],
) -> None:
    top_sources = sorted(source_summary, key=lambda row: row["single_qrc_rmse_mean"])[:15]
    strongest_gain = sorted(source_summary, key=lambda row: row["single_qrc_minus_classical_rmse"])[:15]
    hierarchy_focus = [
        row
        for row in sorted(hierarchy_summary, key=lambda row: (row["anchor_strategy"], int(row["anchor_count"])))
        if row["anchor_strategy"] in {"single_source_ranked_balanced", "random_balanced", "spatial_spread", "strength_hubs"}
    ]
    fixed_hierarchy_focus = [
        row
        for row in sorted(fixed_hierarchy_summary, key=lambda row: (row["anchor_strategy"], int(row["anchor_count"])))
        if row["anchor_strategy"]
        in {"fixed_single_source_ranked_balanced", "fixed_random_balanced", "fixed_spatial_spread", "fixed_strength_hubs"}
    ]
    phase_focus = [
        row
        for row in sorted(phase_summary, key=lambda row: (row["anchor_strategy"], int(row["anchor_count"]), row["normalized_rmse_mean"]))
        if row["anchor_strategy"] == "single_source_ranked_balanced"
    ]
    synergy_focus = [
        row for row in synergy_rows if row["anchor_strategy"] == "single_source_ranked_balanced_phase_synergy"
    ]

    text = f"""# QRC-v2 Mechanism Analysis

Dataset: BrainGraph HCP 86-node structural connectomes

Subjects analyzed: {subject_count} ({train_count} train, {test_count} test)

## 1. Source Importance

Each node was tested as the only known source. Lower RMSE means that one source
gave the global decoder more useful relational information about all other
nodes.

{markdown_table(top_sources, ["single_qrc_rank", "node_name", "hemisphere", "region", "single_qrc_rmse_mean", "single_qrc_distance_corr_mean", "mean_qrc_influence", "mean_weighted_strength"], 15)}

Sources where QRC most improved over the matched classical heat channel:

{markdown_table(strongest_gain, ["node_name", "hemisphere", "single_qrc_minus_classical_rmse", "single_qrc_rmse_mean", "single_classical_rmse_mean"], 15)}

Source-descriptor correlations. For RMSE, negative means the descriptor is
associated with better single-source performance.

{markdown_table(source_correlation_rows, ["descriptor", "target", "signed_pearson"], 12)}

## 2. Hierarchical Source Sets

Nested source sets were formed from the single-source ranking and compared with
balanced random sets, spatially spread sources, and weighted-strength hubs.

{markdown_table(hierarchy_focus, ["anchor_strategy", "anchor_count", "normalized_rmse_mean", "hemisphere_accuracy_mean", "distance_corr_mean", "x_corr_mean", "y_corr_mean", "z_corr_mean"], 32)}

Because ordinary anchor metrics remove the source nodes from the evaluation set,
the target nodes can change as the hierarchy grows. The fixed-target check below
holds out the same low-ranked target nodes at every source count.

{markdown_table(fixed_hierarchy_focus, ["anchor_strategy", "anchor_count", "target_count", "normalized_rmse_mean", "hemisphere_accuracy_mean", "distance_corr_mean", "x_corr_mean", "y_corr_mean", "z_corr_mean"], 32)}

## 3. Phase / Interference Channels

The same source hierarchy was decoded with separate real, imaginary,
probability, pairwise-combined, full-QRC, and quantum-minus-classical contrast
features.

{markdown_table(phase_focus, ["anchor_count", "channel_variant", "normalized_rmse_mean", "hemisphere_accuracy_mean", "distance_corr_mean"], 36)}

Synergy summary. Positive gain means full QRC was better than the comparison.

{markdown_table(synergy_focus, ["anchor_count", "full_qrc_rmse", "best_single_channel_rmse", "best_pair_channel_rmse", "heat_classical_rmse", "gain_vs_best_single", "gain_vs_best_pair", "gain_vs_heat", "plus_heat_minus_full"], 12)}

## Working Interpretation

The source-importance ranking tests whether some regions are unusually good
coordinate anchors. The hierarchy analysis tests whether a nested source set
adds information in stages: hemisphere and left-right structure first, then
finer spatial axes and pairwise geometry. The channel analysis tests whether the
advantage comes from a single QRC component or from combining phase amplitude
with probability-like response.

These results are exploratory mechanism evidence. They should be treated as the
next hypothesis generator before being moved into the paper.
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")
    if docs_path is not None:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QRC-v2 source, hierarchy, and phase mechanism analyses.")
    parser.add_argument("--input-dir", type=Path, default=Path("research/data/brain_graph_hcp_86_nodes/graphml"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/outputs/qrc_v2_mechanism"))
    parser.add_argument("--docs-path", type=Path, default=Path("docs/qrc_v2_mechanism_2026-06-18.md"))
    parser.add_argument("--subjects", type=int, default=1064)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--hierarchy-counts", type=parse_ints, default=parse_ints("2,4,8,16,32,64"))
    parser.add_argument("--phase-counts", type=parse_ints, default=parse_ints("8,16,32"))
    parser.add_argument("--random-repeats", type=int, default=4)
    parser.add_argument("--phase-random-repeats", type=int, default=2)
    parser.add_argument("--fixed-target-count", type=int, default=16)
    parser.add_argument("--times", type=parse_times, default=parse_times("0.25,0.5,1,2,4,8"))
    parser.add_argument("--weight-transform", choices=["raw", "log1p", "binary"], default="log1p")
    parser.add_argument("--ridge-alpha", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    files = sorted(args.input_dir.glob("*.graphml"))[: args.subjects]
    if not files:
        raise SystemExit(f"No GraphML files found in {args.input_dir}")

    print(f"Caching {len(files)} subjects")
    records = [cache_subject(path, args.times, args.weight_transform) for path in files]
    train_records, test_records = train_test_split(records, args.train_fraction, rng)
    print(f"Train subjects: {len(train_records)}; test subjects: {len(test_records)}")

    print("Computing node descriptors")
    descriptors = node_descriptors(records)
    write_csv(
        args.output_dir / "node_descriptors.csv",
        descriptors,
        [
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "x",
            "y",
            "z",
            "mean_qrc_influence",
            "mean_weighted_strength",
            "mean_response_entropy",
            "mean_cross_hemisphere_response",
        ],
    )

    print("Running single-source importance")
    source_metric_rows, source_summary = run_source_importance(
        train_records,
        test_records,
        descriptors,
        alpha=args.ridge_alpha,
    )
    write_csv(
        args.output_dir / "source_importance_metrics.csv",
        source_metric_rows,
        [
            "sample_id",
            "source_index",
            "source_node_id",
            "source_node_name",
            "source_region",
            "source_hemisphere",
            "subject_id",
            "anchor_strategy",
            "anchor_count",
            "method",
            "qrc_variant",
            "mode",
            "normalized_rmse",
            "mean_error",
            "x_corr",
            "y_corr",
            "z_corr",
            "distance_corr",
            "hemisphere_accuracy",
        ],
    )
    write_csv(
        args.output_dir / "source_importance_summary.csv",
        source_summary,
        [
            "single_qrc_rank",
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "x",
            "y",
            "z",
            "mean_qrc_influence",
            "mean_weighted_strength",
            "mean_response_entropy",
            "mean_cross_hemisphere_response",
            "single_qrc_rmse_mean",
            "single_classical_rmse_mean",
            "single_qrc_minus_classical_rmse",
            "single_qrc_hemi_mean",
            "single_qrc_distance_corr_mean",
            "single_classical_distance_corr_mean",
            "single_source_n",
        ],
    )
    source_correlation_rows = source_descriptor_correlations(source_summary)
    write_csv(
        args.output_dir / "source_descriptor_correlations.csv",
        source_correlation_rows,
        ["descriptor", "target", "signed_pearson"],
    )

    print("Running hierarchical source analysis")
    hierarchy_metric_rows, hierarchy_summary, hierarchy_aux_rows = run_hierarchy_analysis(
        train_records,
        test_records,
        source_summary,
        descriptors,
        counts=args.hierarchy_counts,
        random_repeats=args.random_repeats,
        alpha=args.ridge_alpha,
        rng=rng,
    )
    write_csv(
        args.output_dir / "hierarchy_metrics.csv",
        hierarchy_metric_rows,
        [
            "sample_id",
            "subject_id",
            "anchor_strategy",
            "anchor_count",
            "method",
            "qrc_variant",
            "mode",
            "normalized_rmse",
            "mean_error",
            "x_corr",
            "y_corr",
            "z_corr",
            "distance_corr",
            "hemisphere_accuracy",
        ],
    )
    write_csv(
        args.output_dir / "hierarchy_summary.csv",
        hierarchy_summary,
        [
            "anchor_strategy",
            "anchor_count",
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
        ],
    )
    write_csv(
        args.output_dir / "hierarchy_anchor_sets_and_increments.csv",
        hierarchy_aux_rows,
        [
            "sample_id",
            "anchor_strategy",
            "anchor_count",
            "anchor_ids",
            "anchor_names",
            "anchor_regions",
            "anchor_hemispheres",
            "rmse_gain_from_previous",
            "distance_corr_gain_from_previous",
            "x_corr_gain_from_previous",
            "y_corr_gain_from_previous",
            "z_corr_gain_from_previous",
        ],
    )

    print("Running fixed-target hierarchical source analysis")
    fixed_metric_rows, fixed_summary, fixed_anchor_rows, fixed_target_rows = run_fixed_target_hierarchy(
        train_records,
        test_records,
        source_summary,
        descriptors,
        counts=args.hierarchy_counts,
        random_repeats=args.random_repeats,
        target_count=args.fixed_target_count,
        alpha=args.ridge_alpha,
        rng=rng,
    )
    write_csv(
        args.output_dir / "fixed_target_hierarchy_metrics.csv",
        fixed_metric_rows,
        [
            "sample_id",
            "subject_id",
            "anchor_strategy",
            "anchor_count",
            "target_count",
            "method",
            "qrc_variant",
            "mode",
            "normalized_rmse",
            "mean_error",
            "x_corr",
            "y_corr",
            "z_corr",
            "distance_corr",
            "hemisphere_accuracy",
        ],
    )
    write_csv(
        args.output_dir / "fixed_target_hierarchy_summary.csv",
        fixed_summary,
        [
            "anchor_strategy",
            "anchor_count",
            "target_count",
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
        ],
    )
    write_csv(
        args.output_dir / "fixed_target_hierarchy_anchor_sets.csv",
        fixed_anchor_rows,
        [
            "sample_id",
            "anchor_strategy",
            "anchor_count",
            "anchor_ids",
            "anchor_names",
            "anchor_regions",
            "anchor_hemispheres",
        ],
    )
    write_csv(
        args.output_dir / "fixed_target_nodes.csv",
        fixed_target_rows,
        ["target_index", "node_id", "node_name", "region", "hemisphere"],
    )

    print("Running phase/interference channel analysis")
    phase_metric_rows, phase_summary, phase_aux_rows = run_phase_analysis(
        train_records,
        test_records,
        source_summary,
        counts=args.phase_counts,
        random_repeats=args.phase_random_repeats,
        alpha=args.ridge_alpha,
        rng=rng,
    )
    synergy_rows = [row for row in phase_aux_rows if str(row["anchor_strategy"]).endswith("_phase_synergy")]
    write_csv(
        args.output_dir / "phase_channel_metrics.csv",
        phase_metric_rows,
        [
            "sample_id",
            "subject_id",
            "anchor_strategy",
            "anchor_count",
            "method",
            "mode",
            "channel_variant",
            "normalized_rmse",
            "mean_error",
            "x_corr",
            "y_corr",
            "z_corr",
            "distance_corr",
            "hemisphere_accuracy",
        ],
    )
    write_csv(
        args.output_dir / "phase_channel_summary.csv",
        phase_summary,
        [
            "anchor_strategy",
            "anchor_count",
            "channel_variant",
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
        ],
    )
    write_csv(
        args.output_dir / "phase_anchor_sets_and_synergy.csv",
        phase_aux_rows,
        [
            "sample_id",
            "anchor_strategy",
            "anchor_count",
            "anchor_ids",
            "anchor_names",
            "anchor_regions",
            "anchor_hemispheres",
            "full_qrc_rmse",
            "best_single_channel_rmse",
            "best_pair_channel_rmse",
            "heat_classical_rmse",
            "qprob_minus_heat_rmse",
            "full_qrc_plus_heat_rmse",
            "gain_vs_best_single",
            "gain_vs_best_pair",
            "gain_vs_heat",
            "plus_heat_minus_full",
        ],
    )

    write_bar_svg(
        args.output_dir / "source_importance_top.svg",
        sorted(source_summary, key=lambda row: row["single_qrc_rmse_mean"]),
        "Best Single QRC Sources",
        "node_name",
        "single_qrc_rmse_mean",
        limit=15,
    )
    write_line_svg(
        args.output_dir / "hierarchy_rmse.svg",
        hierarchy_summary,
        "Hierarchical Source Sets",
        ["single_source_ranked_balanced", "random_balanced", "spatial_spread", "strength_hubs"],
    )
    write_line_svg(
        args.output_dir / "fixed_target_hierarchy_rmse.svg",
        fixed_summary,
        "Fixed-Target Hierarchical Source Sets",
        [
            "fixed_single_source_ranked_balanced",
            "fixed_random_balanced",
            "fixed_spatial_spread",
            "fixed_strength_hubs",
        ],
    )
    phase_ranked = [
        row
        for row in sorted(phase_summary, key=lambda row: clean_float(row["normalized_rmse_mean"]))
        if row["anchor_strategy"] == "single_source_ranked_balanced" and int(row["anchor_count"]) == 16
    ]
    write_bar_svg(
        args.output_dir / "phase_channels_16_sources.svg",
        phase_ranked,
        "Phase Channel Comparison, 16 Ranked Sources",
        "channel_variant",
        "normalized_rmse_mean",
        limit=12,
    )
    write_summary(
        args.output_dir,
        args.docs_path,
        subject_count=len(records),
        train_count=len(train_records),
        test_count=len(test_records),
        source_summary=source_summary,
        source_correlation_rows=source_correlation_rows,
        hierarchy_summary=hierarchy_summary,
        fixed_hierarchy_summary=fixed_summary,
        phase_summary=phase_summary,
        synergy_rows=synergy_rows,
    )
    print(f"Wrote mechanism outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
