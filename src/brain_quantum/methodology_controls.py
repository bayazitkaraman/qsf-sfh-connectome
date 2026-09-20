"""Focused controls for the QSF source-coordinate validation.

The controls reuse saved subject splits and source sets so that methodological
questions can be answered without changing the original benchmark:

- separate dynamic response features from coordinate-moment features;
- compare with privileged atlas-identity and source-alignment references;
- test generalization to region labels omitted from ridge training; and
- aggregate paired differences by unique subject before bootstrapping.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain_quantum.linear_decoder import ridge_coefficients

from brain_quantum.analysis_suite import (  # noqa: E402
    finite_position_mask,
    pairwise_distance_corr_mask,
    source_informed_hemi_accuracy,
)
from brain_quantum.qrc_connectome import parse_times, write_csv  # noqa: E402
from brain_quantum.source_coordinate_v2 import (  # noqa: E402
    RidgeModel,
    SubjectRecord,
    anchor_features,
    cache_subject,
    non_anchor_mask,
    predict_ridge,
)


METRIC_FIELDS = ["normalized_rmse", "distance_corr", "hemisphere_accuracy"]

CHANNEL_SPECS = [
    ("qsf_full", "qrc", "full", "dynamic"),
    ("qsf_real", "qrc", "real", "dynamic"),
    ("qsf_imaginary", "qrc", "imag", "dynamic"),
    ("qsf_probability", "qrc", "prob", "dynamic"),
    ("heat", "classical", "full", "dynamic"),
    ("heat_raw", "classical_raw", "full", "dynamic"),
]

PAIR_CHANNEL_SPECS = [
    ("qsf_real_imag", "qrc", "real_imag", "dynamic"),
    ("qsf_real_probability", "qrc", "real_prob", "dynamic"),
    ("qsf_imaginary_probability", "qrc", "imag_prob", "dynamic"),
]

TUNED_DECODER_SPECS = [
    ("qsf_full_features", "qrc", "full", "full"),
    ("heat_full_features", "classical", "full", "full"),
    ("heat_raw_full_features", "classical_raw", "full", "full"),
    *CHANNEL_SPECS,
    *PAIR_CHANNEL_SPECS,
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fit_ridge_arrays(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeModel:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] == 0 or x.shape[0] != y.shape[0]:
        raise ValueError("Ridge fitting requires nonempty, row-aligned 2D feature and target arrays.")
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    xs = np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    y_mean = y.mean(axis=0, keepdims=True)
    yc = y - y_mean
    coef = ridge_coefficients(xs, yc, alpha)
    return RidgeModel(mean=mean, std=std, y_mean=y_mean, coef=coef)


def fit_control_model(
    records: list[SubjectRecord],
    anchors: np.ndarray,
    method: str,
    feature_mode: str,
    alpha: float,
    target_builder: Callable[[SubjectRecord], np.ndarray] | None = None,
    allowed_nodes: np.ndarray | None = None,
    qrc_variant: str = "full",
) -> RidgeModel:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for record in records:
        features, _ = anchor_features(
            record,
            anchors,
            method,
            qrc_variant=qrc_variant,
            feature_mode=feature_mode,
        )
        mask = non_anchor_mask(features.shape[0], anchors)
        mask &= finite_position_mask(record.connectome.positions)
        if allowed_nodes is not None:
            mask &= allowed_nodes
        target = record.connectome.positions if target_builder is None else target_builder(record)
        x_parts.append(features[mask])
        y_parts.append(target[mask])
    return fit_ridge_arrays(np.vstack(x_parts), np.vstack(y_parts), alpha)


def validation_rmse_summary(
    records: list[SubjectRecord],
    anchors: np.ndarray,
    method: str,
    feature_mode: str,
    qrc_variant: str,
    model: RidgeModel,
) -> tuple[float, float]:
    values = []
    mask = non_anchor_mask(len(records[0].connectome.node_names), anchors)
    for record in records:
        features, _ = anchor_features(
            record,
            anchors,
            method,
            qrc_variant=qrc_variant,
            feature_mode=feature_mode,
        )
        predicted = predict_ridge(model, features)
        values.append(evaluate_prediction(record, predicted, mask, anchors)["normalized_rmse"])
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("inf"), float("inf")
    array = np.asarray(finite, dtype=float)
    standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return float(np.mean(array)), standard_error


def select_decoder_alpha(
    validation_scores: list[tuple[float, float, float]],
    rule: str,
) -> float:
    """Select ridge strength by minimum mean or the one-standard-error rule."""
    if not validation_scores:
        raise ValueError("At least one decoder validation score is required.")
    best_mean, best_se, best_alpha = min(validation_scores, key=lambda row: (row[0], row[2]))
    if rule == "min_mean":
        return float(best_alpha)
    if rule != "one_se":
        raise ValueError(f"Unknown decoder selection rule: {rule}")
    threshold = best_mean + best_se
    eligible = [alpha for mean, _, alpha in validation_scores if mean <= threshold]
    return float(max(eligible))


def inner_subject_split(
    records: list[SubjectRecord],
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[SubjectRecord], list[SubjectRecord]]:
    indices = rng.permutation(len(records))
    validation_count = max(1, int(round(len(records) * validation_fraction)))
    validation_count = min(validation_count, len(records) - 1)
    validation_indices = set(int(index) for index in indices[:validation_count])
    inner_train = [record for index, record in enumerate(records) if index not in validation_indices]
    inner_validation = [record for index, record in enumerate(records) if index in validation_indices]
    return inner_train, inner_validation


def similarity_align(atlas: np.ndarray, subject: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    valid = anchors[
        np.all(np.isfinite(atlas[anchors]), axis=1)
        & np.all(np.isfinite(subject[anchors]), axis=1)
    ]
    if len(valid) < 3:
        return atlas.copy()
    source = atlas[valid]
    target = subject[valid]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    u, singular, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
        # The scale must use the same proper-rotation constraint as the rotation.
        singular[-1] *= -1
    scale = float(singular.sum() / max(float(np.sum(source_centered**2)), 1e-12))
    return scale * (atlas - source_mean) @ rotation + target_mean


def evaluate_prediction(
    record: SubjectRecord,
    predicted: np.ndarray,
    mask: np.ndarray,
    anchors: np.ndarray,
) -> dict[str, float]:
    mask = mask & finite_position_mask(record.connectome.positions)
    errors = np.linalg.norm(predicted[mask] - record.connectome.positions[mask], axis=1)
    return {
        "normalized_rmse": float(np.sqrt(np.mean(errors**2)) / record.extent),
        "distance_corr": pairwise_distance_corr_mask(
            predicted,
            record.connectome.positions,
            mask,
        ),
        "hemisphere_accuracy": source_informed_hemi_accuracy(
            predicted[:, 0],
            record.connectome.hemispheres,
            mask,
            record.connectome.positions,
            anchors,
        ),
    }


def balanced_node_folds(
    candidates: np.ndarray,
    hemispheres: list[str],
    fold_count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    groups = [
        [int(idx) for idx in candidates if hemispheres[int(idx)].startswith("left")],
        [int(idx) for idx in candidates if hemispheres[int(idx)].startswith("right")],
        [
            int(idx)
            for idx in candidates
            if not (
                hemispheres[int(idx)].startswith("left")
                or hemispheres[int(idx)].startswith("right")
            )
        ],
    ]
    offset = 0
    for group in groups:
        shuffled = rng.permutation(group).tolist()
        for pos, node in enumerate(shuffled):
            folds[(offset + pos) % fold_count].append(int(node))
        offset = (offset + len(shuffled)) % fold_count
    return [np.array(sorted(fold), dtype=int) for fold in folds]


def mean_summary(
    rows: list[dict[str, object]],
    group_fields: list[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, object]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        record = dict(zip(group_fields, key))
        for metric in METRIC_FIELDS:
            finite = [float(row[metric]) for row in values if np.isfinite(float(row[metric]))]
            record[f"{metric}_mean"] = float(np.mean(finite)) if finite else float("nan")
            record[f"{metric}_std"] = float(np.std(finite)) if finite else float("nan")
        record["n"] = len(values)
        output.append(record)
    return output


def clustered_pair_summary(
    rows: list[dict[str, object]],
    group_fields: list[str],
    method_a: str,
    method_b: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    key_fields = ["split_id", "sample_id"]
    if any("fold_id" in row for row in rows):
        key_fields.append("fold_id")
    key_fields.append("subject_id")
    subject_position = key_fields.index("subject_id")
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    output: list[dict[str, object]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        index = {
            tuple(row[field] for field in key_fields) + (str(row["method"]),): row
            for row in group_rows
        }
        paired_keys = sorted(
            {
                tuple(row[field] for field in key_fields)
                for row in group_rows
                if str(row["method"]) == method_a
                and tuple(row[field] for field in key_fields) + (method_b,) in index
            }
        )
        for metric in METRIC_FIELDS:
            subject_deltas: dict[str, list[float]] = defaultdict(list)
            for pair_key in paired_keys:
                a = float(index[pair_key + (method_a,)][metric])
                b = float(index[pair_key + (method_b,)][metric])
                if np.isfinite(a) and np.isfinite(b):
                    subject_deltas[str(pair_key[subject_position])].append(a - b)
            values = np.array(
                [float(np.mean(deltas)) for deltas in subject_deltas.values()],
                dtype=float,
            )
            if len(values):
                bootstrap = np.empty(bootstrap_samples, dtype=float)
                for sample in range(bootstrap_samples):
                    bootstrap[sample] = float(
                        np.mean(rng.choice(values, size=len(values), replace=True))
                    )
                lower, upper = np.quantile(bootstrap, [0.025, 0.975])
                higher_is_better = metric != "normalized_rmse"
                wins = values > 0 if higher_is_better else values < 0
                mean_delta = float(np.mean(values))
                win_rate = float(np.mean(wins))
            else:
                lower = upper = mean_delta = win_rate = float("nan")
            record = dict(zip(group_fields, group_key))
            record.update(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "unique_subjects": len(values),
                    "mean_delta_a_minus_b": mean_delta,
                    "cluster_bootstrap_ci95_low": float(lower),
                    "cluster_bootstrap_ci95_high": float(upper),
                    "subject_win_rate_a": win_rate,
                }
            )
            output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused QSF methodology controls.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("research/data/brain_graph_hcp_86_nodes/graphml"),
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("research/outputs/qrc_v2_validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/outputs/methodology_controls_hcp83"),
    )
    parser.add_argument("--subjects", type=int, default=1064)
    parser.add_argument("--times", type=parse_times, default=parse_times("0.25,0.5,1,2,4,8"))
    parser.add_argument("--weight-transform", default="log1p")
    parser.add_argument("--ridge-alpha", type=float, default=25.0)
    parser.add_argument(
        "--decoder-alpha-grid",
        default="0,0.001,0.01,0.1,1,10,25,100,1000",
    )
    parser.add_argument(
        "--decoder-selection-rule",
        choices=["one_se", "min_mean"],
        default="one_se",
    )
    parser.add_argument("--inner-validation-fraction", type=float, default=0.20)
    parser.add_argument("--tuned-decoder-count", type=int, default=32)
    parser.add_argument("--tuned-decoder-repeats", default="0")
    parser.add_argument("--feature-repeats", default="0")
    parser.add_argument("--node-holdout-count", type=int, default=32)
    parser.add_argument("--node-holdout-repeat", type=int, default=0)
    parser.add_argument("--node-folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repeats = {int(value) for value in args.feature_repeats.split(",") if value.strip()}
    decoder_alphas = [
        float(value) for value in args.decoder_alpha_grid.split(",") if value.strip()
    ]
    tuned_repeats = {
        int(value) for value in args.tuned_decoder_repeats.split(",") if value.strip()
    }
    if not decoder_alphas or any(alpha < 0 for alpha in decoder_alphas):
        raise SystemExit("Decoder alpha grid must contain nonnegative values.")
    metrics = read_csv(args.validation_dir / "validation_metrics.csv")
    anchor_sets = read_csv(args.validation_dir / "validation_anchor_sets.csv")
    selected_sets = [row for row in anchor_sets if int(row["repeat"]) in repeats]
    files = sorted(args.input_dir.glob("*.graphml"))[: args.subjects]
    if not files:
        raise SystemExit(f"No GraphML files found in {args.input_dir}")

    print(f"Caching {len(files)} subjects", flush=True)
    records = [cache_subject(path, args.times, args.weight_transform) for path in files]
    by_subject = {record.connectome.subject_id: record for record in records}
    reference = records[0]
    name_to_index = {
        name: index for index, name in enumerate(reference.connectome.node_names)
    }

    missing_frequency = np.zeros(len(reference.connectome.node_names), dtype=int)
    finite_counts = []
    for record in records:
        finite = finite_position_mask(record.connectome.positions)
        finite_counts.append(int(finite.sum()))
        missing_frequency += (~finite).astype(int)
    inventory = [
        {
            "subjects": len(records),
            "graph_nodes": len(reference.connectome.node_names),
            "minimum_finite_coordinates": min(finite_counts),
            "median_finite_coordinates": float(np.median(finite_counts)),
            "maximum_finite_coordinates": max(finite_counts),
            "consistent_node_order": all(
                record.connectome.node_names == reference.connectome.node_names
                for record in records
            ),
        }
    ]
    write_csv(
        args.output_dir / "dataset_inventory.csv",
        inventory,
        list(inventory[0]),
    )
    missing_rows = [
        {
            "node_index": index,
            "node_name": reference.connectome.node_names[index],
            "missing_subjects": int(count),
            "missing_fraction": float(count / len(records)),
        }
        for index, count in enumerate(missing_frequency)
        if count > 0
    ]
    write_csv(
        args.output_dir / "missing_coordinate_frequency.csv",
        missing_rows,
        ["node_index", "node_name", "missing_subjects", "missing_fraction"],
    )

    test_subjects: dict[int, set[str]] = defaultdict(set)
    for row in metrics:
        if (
            row["method"] == "qrc"
            and row["qrc_variant"] == "full"
            and row["mode"] == "global_ridge"
        ):
            test_subjects[int(row["split_id"])].add(row["subject_id"])

    decomposition_rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    split_cache: dict[int, tuple[list[SubjectRecord], list[SubjectRecord], np.ndarray]] = {}
    for split_id in sorted(test_subjects):
        test_ids = test_subjects[split_id]
        train = [record for record in records if record.connectome.subject_id not in test_ids]
        test = [by_subject[subject] for subject in sorted(test_ids)]
        atlas = np.nanmean(
            np.stack([record.connectome.positions for record in train], axis=0),
            axis=0,
        )
        split_cache[split_id] = (train, test, atlas)
        split_sets = [row for row in selected_sets if int(row["split_id"]) == split_id]
        for set_number, source_set in enumerate(split_sets, start=1):
            anchors = np.array(
                [name_to_index[name] for name in source_set["anchor_names"].split(" | ")],
                dtype=int,
            )
            count = int(source_set["anchor_count"])
            print(
                f"Decomposition split {split_id}: set {set_number}/{len(split_sets)}, "
                f"sources={count}, repeat={source_set['repeat']}",
                flush=True,
            )
            models = {
                (method, feature_mode): fit_control_model(
                    train,
                    anchors,
                    method,
                    feature_mode,
                    args.ridge_alpha,
                )
                for method in ["qrc", "classical"]
                for feature_mode in ["full", "dynamic", "moments"]
            }
            channel_models = {}
            for label, method, qrc_variant, feature_mode in CHANNEL_SPECS:
                if label == "qsf_full":
                    channel_models[label] = models[("qrc", "dynamic")]
                elif label == "heat":
                    channel_models[label] = models[("classical", "dynamic")]
                else:
                    channel_models[label] = fit_control_model(
                        train,
                        anchors,
                        method,
                        feature_mode,
                        args.ridge_alpha,
                        qrc_variant=qrc_variant,
                    )
            non_source = non_anchor_mask(len(reference.connectome.node_names), anchors)
            for record in test:
                for method in ["qrc", "classical"]:
                    for feature_mode in ["full", "dynamic", "moments"]:
                        features, _ = anchor_features(
                            record,
                            anchors,
                            method,
                            qrc_variant="full",
                            feature_mode=feature_mode,
                        )
                        predicted = predict_ridge(models[(method, feature_mode)], features)
                        row: dict[str, object] = {
                            "split_id": split_id,
                            "sample_id": source_set["sample_id"],
                            "repeat": int(source_set["repeat"]),
                            "subject_id": record.connectome.subject_id,
                            "anchor_count": count,
                            "method": method,
                            "feature_mode": feature_mode,
                        }
                        row.update(evaluate_prediction(record, predicted, non_source, anchors))
                        decomposition_rows.append(row)

                for label, method, qrc_variant, feature_mode in CHANNEL_SPECS:
                    features, _ = anchor_features(
                        record,
                        anchors,
                        method,
                        qrc_variant=qrc_variant,
                        feature_mode=feature_mode,
                    )
                    predicted = predict_ridge(channel_models[label], features)
                    channel_row: dict[str, object] = {
                        "split_id": split_id,
                        "sample_id": source_set["sample_id"],
                        "repeat": int(source_set["repeat"]),
                        "subject_id": record.connectome.subject_id,
                        "anchor_count": count,
                        "method": label,
                        "qrc_variant": qrc_variant if method == "qrc" else "not_applicable",
                        "feature_mode": feature_mode,
                        "feature_count": int(features.shape[1]),
                    }
                    channel_row.update(evaluate_prediction(record, predicted, non_source, anchors))
                    channel_rows.append(channel_row)

                atlas_row: dict[str, object] = {
                    "split_id": split_id,
                    "sample_id": source_set["sample_id"],
                    "repeat": int(source_set["repeat"]),
                    "subject_id": record.connectome.subject_id,
                    "anchor_count": count,
                    "method": "atlas_identity_mean",
                }
                atlas_row.update(evaluate_prediction(record, atlas, non_source, anchors))
                geometry_rows.append(atlas_row)
                aligned = similarity_align(atlas, record.connectome.positions, anchors)
                aligned_row = dict(atlas_row)
                aligned_row["method"] = "source_similarity_alignment"
                aligned_row.update(evaluate_prediction(record, aligned, non_source, anchors))
                geometry_rows.append(aligned_row)

    metric_fields = [
        "split_id",
        "sample_id",
        "repeat",
        "subject_id",
        "anchor_count",
        "method",
        "feature_mode",
        *METRIC_FIELDS,
    ]
    write_csv(args.output_dir / "feature_decomposition_metrics.csv", decomposition_rows, metric_fields)
    decomposition_summary = mean_summary(
        decomposition_rows,
        ["anchor_count", "method", "feature_mode"],
    )
    write_csv(
        args.output_dir / "feature_decomposition_summary.csv",
        decomposition_summary,
        list(decomposition_summary[0]),
    )
    write_csv(
        args.output_dir / "geometry_reference_metrics.csv",
        geometry_rows,
        [
            "split_id",
            "sample_id",
            "repeat",
            "subject_id",
            "anchor_count",
            "method",
            *METRIC_FIELDS,
        ],
    )
    geometry_summary = mean_summary(geometry_rows, ["anchor_count", "method"])
    write_csv(
        args.output_dir / "geometry_reference_summary.csv",
        geometry_summary,
        list(geometry_summary[0]),
    )

    cluster_rng = np.random.default_rng(args.seed + 1)
    clustered = clustered_pair_summary(
        decomposition_rows,
        ["anchor_count", "feature_mode"],
        "qrc",
        "classical",
        args.bootstrap_samples,
        cluster_rng,
    )
    write_csv(
        args.output_dir / "feature_decomposition_clustered_pairs.csv",
        clustered,
        list(clustered[0]),
    )

    channel_fields = [
        "split_id",
        "sample_id",
        "repeat",
        "subject_id",
        "anchor_count",
        "method",
        "qrc_variant",
        "feature_mode",
        "feature_count",
        *METRIC_FIELDS,
    ]
    write_csv(args.output_dir / "channel_capacity_metrics.csv", channel_rows, channel_fields)
    channel_summary = mean_summary(channel_rows, ["anchor_count", "method", "feature_count"])
    write_csv(
        args.output_dir / "channel_capacity_summary.csv",
        channel_summary,
        list(channel_summary[0]),
    )
    channel_clustered: list[dict[str, object]] = []
    for heat_method in ["heat", "heat_raw"]:
        for qsf_method in ["qsf_full", "qsf_real", "qsf_imaginary", "qsf_probability"]:
            channel_clustered.extend(
                clustered_pair_summary(
                    channel_rows,
                    ["anchor_count"],
                    qsf_method,
                    heat_method,
                    args.bootstrap_samples,
                    np.random.default_rng(args.seed + 100 + len(channel_clustered)),
                )
            )
    write_csv(
        args.output_dir / "channel_capacity_clustered_pairs.csv",
        channel_clustered,
        list(channel_clustered[0]),
    )

    tuned_grid_rows: list[dict[str, object]] = []
    tuned_metric_rows: list[dict[str, object]] = []
    inner_splits: dict[int, tuple[list[SubjectRecord], list[SubjectRecord]]] = {}
    tuned_sets = [
        row
        for row in selected_sets
        if int(row["anchor_count"]) == args.tuned_decoder_count
        and int(row["repeat"]) in tuned_repeats
    ]
    for source_set in tuned_sets:
        split_id = int(source_set["split_id"])
        train, test, _ = split_cache[split_id]
        if split_id not in inner_splits:
            inner_splits[split_id] = inner_subject_split(
                train,
                args.inner_validation_fraction,
                np.random.default_rng(args.seed + 10000 + split_id),
            )
        inner_train, inner_validation = inner_splits[split_id]
        anchors = np.array(
            [name_to_index[name] for name in source_set["anchor_names"].split(" | ")],
            dtype=int,
        )
        non_source = non_anchor_mask(len(reference.connectome.node_names), anchors)
        print(
            f"Nested decoder tuning split {split_id}: "
            f"sources={args.tuned_decoder_count}, repeat={source_set['repeat']}",
            flush=True,
        )
        for label, method, qrc_variant, feature_mode in TUNED_DECODER_SPECS:
            validation_scores = []
            for alpha in decoder_alphas:
                candidate = fit_control_model(
                    inner_train,
                    anchors,
                    method,
                    feature_mode,
                    alpha,
                    qrc_variant=qrc_variant,
                )
                validation_rmse, validation_rmse_se = validation_rmse_summary(
                    inner_validation,
                    anchors,
                    method,
                    feature_mode,
                    qrc_variant,
                    candidate,
                )
                validation_scores.append((validation_rmse, validation_rmse_se, alpha))
                tuned_grid_rows.append(
                    {
                        "split_id": split_id,
                        "sample_id": source_set["sample_id"],
                        "repeat": int(source_set["repeat"]),
                        "anchor_count": args.tuned_decoder_count,
                        "method": label,
                        "qrc_variant": qrc_variant if method == "qrc" else "not_applicable",
                        "feature_mode": feature_mode,
                        "alpha": alpha,
                        "inner_validation_normalized_rmse": validation_rmse,
                        "inner_validation_normalized_rmse_se": validation_rmse_se,
                    }
                )
            selected_alpha = select_decoder_alpha(
                validation_scores,
                args.decoder_selection_rule,
            )
            final_model = fit_control_model(
                train,
                anchors,
                method,
                feature_mode,
                selected_alpha,
                qrc_variant=qrc_variant,
            )
            for record in test:
                features, _ = anchor_features(
                    record,
                    anchors,
                    method,
                    qrc_variant=qrc_variant,
                    feature_mode=feature_mode,
                )
                predicted = predict_ridge(final_model, features)
                tuned_row: dict[str, object] = {
                    "split_id": split_id,
                    "sample_id": source_set["sample_id"],
                    "repeat": int(source_set["repeat"]),
                    "subject_id": record.connectome.subject_id,
                    "anchor_count": args.tuned_decoder_count,
                    "method": label,
                    "qrc_variant": qrc_variant if method == "qrc" else "not_applicable",
                    "feature_mode": feature_mode,
                    "feature_count": int(features.shape[1]),
                    "selected_alpha": selected_alpha,
                    "decoder_selection_rule": args.decoder_selection_rule,
                }
                tuned_row.update(evaluate_prediction(record, predicted, non_source, anchors))
                tuned_metric_rows.append(tuned_row)

    if tuned_metric_rows:
        write_csv(
            args.output_dir / "nested_decoder_validation_grid.csv",
            tuned_grid_rows,
            list(tuned_grid_rows[0]),
        )
        write_csv(
            args.output_dir / "nested_decoder_metrics.csv",
            tuned_metric_rows,
            list(tuned_metric_rows[0]),
        )
        tuned_summary = mean_summary(
            tuned_metric_rows,
            ["anchor_count", "method", "feature_count"],
        )
        write_csv(
            args.output_dir / "nested_decoder_summary.csv",
            tuned_summary,
            list(tuned_summary[0]),
        )
        tuned_clustered: list[dict[str, object]] = []
        for method_a, method_b in [
            ("qsf_full_features", "heat_full_features"),
            ("qsf_full_features", "heat_raw_full_features"),
            ("qsf_full", "heat"),
            ("qsf_full", "heat_raw"),
            ("qsf_real", "heat"),
            ("qsf_real", "heat_raw"),
            ("qsf_imaginary", "heat"),
            ("qsf_imaginary", "heat_raw"),
            ("qsf_probability", "heat"),
            ("qsf_probability", "heat_raw"),
            ("qsf_real_imag", "qsf_full"),
            ("qsf_real_probability", "qsf_full"),
            ("qsf_imaginary_probability", "qsf_full"),
            ("qsf_real_imag", "heat_raw"),
        ]:
            tuned_clustered.extend(
                clustered_pair_summary(
                    tuned_metric_rows,
                    ["anchor_count"],
                    method_a,
                    method_b,
                    args.bootstrap_samples,
                    np.random.default_rng(args.seed + 200 + len(tuned_clustered)),
                )
            )
        write_csv(
            args.output_dir / "nested_decoder_clustered_pairs.csv",
            tuned_clustered,
            list(tuned_clustered[0]),
        )

    node_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    atlas_residual_rows: list[dict[str, object]] = []
    holdout_sets = [
        row
        for row in anchor_sets
        if int(row["anchor_count"]) == args.node_holdout_count
        and int(row["repeat"]) == args.node_holdout_repeat
    ]
    for source_set in holdout_sets:
        split_id = int(source_set["split_id"])
        train, test, atlas = split_cache[split_id]
        anchors = np.array(
            [name_to_index[name] for name in source_set["anchor_names"].split(" | ")],
            dtype=int,
        )
        non_source_nodes = np.array(
            [
                index
                for index in range(len(reference.connectome.node_names))
                if index not in set(anchors.tolist())
            ],
            dtype=int,
        )
        fold_rng = np.random.default_rng(args.seed + split_id)
        folds = balanced_node_folds(
            non_source_nodes,
            reference.connectome.hemispheres,
            args.node_folds,
            fold_rng,
        )
        for fold_id, held_nodes in enumerate(folds, start=1):
            allowed = np.ones(len(reference.connectome.node_names), dtype=bool)
            allowed[held_nodes] = False
            models = {
                method: fit_control_model(
                    train,
                    anchors,
                    method,
                    "full",
                    args.ridge_alpha,
                    allowed_nodes=allowed,
                )
                for method in ["qrc", "classical"]
            }
            held_mask = np.zeros(len(reference.connectome.node_names), dtype=bool)
            held_mask[held_nodes] = True
            for record in test:
                for method, model in models.items():
                    features, _ = anchor_features(record, anchors, method, feature_mode="full")
                    predicted = predict_ridge(model, features)
                    row = {
                        "split_id": split_id,
                        "sample_id": source_set["sample_id"],
                        "subject_id": record.connectome.subject_id,
                        "anchor_count": args.node_holdout_count,
                        "fold_id": fold_id,
                        "held_node_count": len(held_nodes),
                        "method": method,
                    }
                    row.update(evaluate_prediction(record, predicted, held_mask, anchors))
                    node_rows.append(row)

        def aligned_residual(record: SubjectRecord) -> np.ndarray:
            aligned = similarity_align(atlas, record.connectome.positions, anchors)
            return record.connectome.positions - aligned

        def atlas_residual(record: SubjectRecord) -> np.ndarray:
            return record.connectome.positions - atlas

        residual_models = {
            method: fit_control_model(
                train,
                anchors,
                method,
                "full",
                args.ridge_alpha,
                target_builder=aligned_residual,
            )
            for method in ["qrc", "classical"]
        }
        atlas_residual_models = {
            method: fit_control_model(
                train,
                anchors,
                method,
                "full",
                args.ridge_alpha,
                target_builder=atlas_residual,
            )
            for method in ["qrc", "classical"]
        }
        non_source_mask = non_anchor_mask(len(reference.connectome.node_names), anchors)
        for record in test:
            aligned = similarity_align(atlas, record.connectome.positions, anchors)
            base_row = {
                "split_id": split_id,
                "sample_id": source_set["sample_id"],
                "subject_id": record.connectome.subject_id,
                "anchor_count": args.node_holdout_count,
                "method": "source_similarity_alignment",
            }
            base_row.update(evaluate_prediction(record, aligned, non_source_mask, anchors))
            residual_rows.append(base_row)
            atlas_base_row = dict(base_row)
            atlas_base_row["method"] = "atlas_identity_mean"
            atlas_base_row.update(evaluate_prediction(record, atlas, non_source_mask, anchors))
            atlas_residual_rows.append(atlas_base_row)
            for method, model in residual_models.items():
                features, _ = anchor_features(record, anchors, method, feature_mode="full")
                predicted = aligned + predict_ridge(model, features)
                row = dict(base_row)
                row["method"] = method
                row.update(evaluate_prediction(record, predicted, non_source_mask, anchors))
                residual_rows.append(row)
            for method, model in atlas_residual_models.items():
                features, _ = anchor_features(record, anchors, method, feature_mode="full")
                predicted = atlas + predict_ridge(model, features)
                row = dict(atlas_base_row)
                row["method"] = method
                row.update(evaluate_prediction(record, predicted, non_source_mask, anchors))
                atlas_residual_rows.append(row)

    node_fields = [
        "split_id",
        "sample_id",
        "subject_id",
        "anchor_count",
        "fold_id",
        "held_node_count",
        "method",
        *METRIC_FIELDS,
    ]
    write_csv(args.output_dir / "node_holdout_metrics.csv", node_rows, node_fields)
    node_summary = mean_summary(node_rows, ["anchor_count", "method"])
    write_csv(args.output_dir / "node_holdout_summary.csv", node_summary, list(node_summary[0]))
    node_clustered = clustered_pair_summary(
        node_rows,
        ["anchor_count"],
        "qrc",
        "classical",
        args.bootstrap_samples,
        np.random.default_rng(args.seed + 2),
    )
    write_csv(
        args.output_dir / "node_holdout_clustered_pairs.csv",
        node_clustered,
        list(node_clustered[0]),
    )

    write_csv(
        args.output_dir / "aligned_residual_metrics.csv",
        residual_rows,
        [
            "split_id",
            "sample_id",
            "subject_id",
            "anchor_count",
            "method",
            *METRIC_FIELDS,
        ],
    )
    residual_summary = mean_summary(residual_rows, ["anchor_count", "method"])
    write_csv(
        args.output_dir / "aligned_residual_summary.csv",
        residual_summary,
        list(residual_summary[0]),
    )
    write_csv(
        args.output_dir / "atlas_residual_metrics.csv",
        atlas_residual_rows,
        [
            "split_id",
            "sample_id",
            "subject_id",
            "anchor_count",
            "method",
            *METRIC_FIELDS,
        ],
    )
    atlas_residual_summary = mean_summary(
        atlas_residual_rows,
        ["anchor_count", "method"],
    )
    write_csv(
        args.output_dir / "atlas_residual_summary.csv",
        atlas_residual_summary,
        list(atlas_residual_summary[0]),
    )
    atlas_residual_clustered = clustered_pair_summary(
        atlas_residual_rows,
        ["anchor_count"],
        "qrc",
        "classical",
        args.bootstrap_samples,
        np.random.default_rng(args.seed + 3),
    )
    atlas_residual_clustered.extend(
        clustered_pair_summary(
            atlas_residual_rows,
            ["anchor_count"],
            "qrc",
            "atlas_identity_mean",
            args.bootstrap_samples,
            np.random.default_rng(args.seed + 4),
        )
    )
    write_csv(
        args.output_dir / "atlas_residual_clustered_pairs.csv",
        atlas_residual_clustered,
        list(atlas_residual_clustered[0]),
    )
    print(f"Wrote methodology controls to {args.output_dir}")


if __name__ == "__main__":
    main()
