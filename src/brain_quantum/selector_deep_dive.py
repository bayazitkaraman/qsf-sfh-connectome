"""Deep-dive source selector analyses for QRC source-coordinate recovery.

This module treats source selection itself as the research object:

1. Paired statistics for existing QRC-v3 selector outputs.
2. Hybrid selectors that combine weighted strength with strength-residual QRC
   influence/entropy.
3. Sensitivity of the hybrid selector to strength/QRC/redundancy weights.
4. Source-discovery null controls using coordinate shuffles and degree
   preserving rewires.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain_quantum.analysis_suite import mean_by_group  # noqa: E402
from brain_quantum.mechanism_suite import (  # noqa: E402
    anchor_record,
    clean_float,
    evaluate_anchor_model,
    markdown_table,
    node_descriptors,
)
from brain_quantum.null_controls import (  # noqa: E402
    coordinate_shuffle_records,
    degree_rewired_records,
)
from brain_quantum.qrc_connectome import parse_times, write_csv  # noqa: E402
from brain_quantum.source_coordinate_v2 import SubjectRecord, cache_subject  # noqa: E402
from brain_quantum.source_discovery_suite import (  # noqa: E402
    anchors_for_strategy,
    descriptor_rankings,
    greedy_redundancy_ranking,
    residualize_against_strength,
    source_profile_similarity,
    split_records_three_way,
    zscore,
)


METRIC_DIRECTIONS = {
    "normalized_rmse": "lower",
    "distance_corr": "higher",
    "hemisphere_accuracy": "higher",
}

HYBRID_CONFIGS = [
    {
        "strategy": "hybrid_s50_q50",
        "strength_weight": 0.50,
        "resid_influence_weight": 0.25,
        "resid_entropy_weight": 0.25,
        "cross_weight": 0.00,
        "greedy": False,
        "redundancy_weight": 0.45,
        "balance_weight": 0.30,
    },
    {
        "strategy": "hybrid_s50_q50_greedy",
        "strength_weight": 0.50,
        "resid_influence_weight": 0.25,
        "resid_entropy_weight": 0.25,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.45,
        "balance_weight": 0.30,
    },
    {
        "strategy": "hybrid_s30_q70_greedy",
        "strength_weight": 0.30,
        "resid_influence_weight": 0.35,
        "resid_entropy_weight": 0.35,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.45,
        "balance_weight": 0.30,
    },
    {
        "strategy": "hybrid_s70_q30_greedy",
        "strength_weight": 0.70,
        "resid_influence_weight": 0.15,
        "resid_entropy_weight": 0.15,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.45,
        "balance_weight": 0.30,
    },
]

SENSITIVITY_CONFIGS = [
    {
        "strategy": "sens_s30_q70_red015",
        "strength_weight": 0.30,
        "resid_influence_weight": 0.35,
        "resid_entropy_weight": 0.35,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.15,
        "balance_weight": 0.30,
    },
    {
        "strategy": "sens_s30_q70_red045",
        "strength_weight": 0.30,
        "resid_influence_weight": 0.35,
        "resid_entropy_weight": 0.35,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.45,
        "balance_weight": 0.30,
    },
    {
        "strategy": "sens_s30_q70_red075",
        "strength_weight": 0.30,
        "resid_influence_weight": 0.35,
        "resid_entropy_weight": 0.35,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.75,
        "balance_weight": 0.30,
    },
    {
        "strategy": "sens_s50_q50_red045",
        "strength_weight": 0.50,
        "resid_influence_weight": 0.25,
        "resid_entropy_weight": 0.25,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.45,
        "balance_weight": 0.30,
    },
    {
        "strategy": "sens_s70_q30_red045",
        "strength_weight": 0.70,
        "resid_influence_weight": 0.15,
        "resid_entropy_weight": 0.15,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.45,
        "balance_weight": 0.30,
    },
]


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normal_approx_p_from_t(t_value: float) -> float:
    return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def split_indices_three_way(
    n: int,
    train_fraction: float,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(n)
    rng.shuffle(indices)
    train_n = max(1, int(n * train_fraction))
    val_n = max(1, int(n * val_fraction))
    if train_n + val_n >= n:
        val_n = max(1, n - train_n - 1)
    train_idx = indices[:train_n]
    val_idx = indices[train_n : train_n + val_n]
    test_idx = indices[train_n + val_n :]
    if len(test_idx) == 0:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]
    return train_idx, val_idx, test_idx


def take(records: list[SubjectRecord], indices: np.ndarray) -> list[SubjectRecord]:
    return [records[int(idx)] for idx in indices]


def aggregate_strategy_rows(rows: list[dict], dataset_label: str = "") -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        dataset = row.get("dataset_label", dataset_label)
        control = row.get("control", "")
        key = (
            dataset,
            control,
            int(row["split_id"]),
            str(row["subject_id"]),
            int(row["anchor_count"]),
            str(row["selection_strategy"]),
        )
        groups[key].append(row)

    out = []
    for (dataset, control, split_id, subject_id, count, strategy), group_rows in sorted(groups.items()):
        record = {
            "dataset_label": dataset,
            "control": control,
            "split_id": split_id,
            "subject_id": subject_id,
            "anchor_count": count,
            "selection_strategy": strategy,
            "n_samples": len(group_rows),
        }
        for metric in METRIC_DIRECTIONS:
            values = [clean_float(row.get(metric)) for row in group_rows]
            values = [value for value in values if np.isfinite(value)]
            record[metric] = float(np.mean(values)) if values else float("nan")
        out.append(record)
    return out


def paired_selector_tests(
    rows: list[dict],
    comparisons: list[tuple[str, str]],
    dataset_label: str = "",
) -> list[dict]:
    aggregated = aggregate_strategy_rows(rows, dataset_label)
    index = {
        (
            row["dataset_label"],
            int(row["split_id"]),
            row["subject_id"],
            int(row["anchor_count"]),
            row["selection_strategy"],
        ): row
        for row in aggregated
    }
    datasets = sorted({row["dataset_label"] for row in aggregated})
    counts = sorted({int(row["anchor_count"]) for row in aggregated})
    out = []
    for dataset in datasets:
        dataset_rows = [row for row in aggregated if row["dataset_label"] == dataset]
        for count in counts:
            cases = [
                (int(row["split_id"]), row["subject_id"])
                for row in dataset_rows
                if int(row["anchor_count"]) == count
            ]
            cases = sorted(set(cases))
            for baseline, comparator in comparisons:
                for metric, direction in METRIC_DIRECTIONS.items():
                    diffs = []
                    signed_improvements = []
                    base_values = []
                    comp_values = []
                    wins = 0
                    for split_id, subject_id in cases:
                        base = index.get((dataset, split_id, subject_id, count, baseline))
                        comp = index.get((dataset, split_id, subject_id, count, comparator))
                        if base is None or comp is None:
                            continue
                        base_value = clean_float(base[metric])
                        comp_value = clean_float(comp[metric])
                        if not (np.isfinite(base_value) and np.isfinite(comp_value)):
                            continue
                        delta = comp_value - base_value
                        improvement = -delta if direction == "lower" else delta
                        diffs.append(delta)
                        signed_improvements.append(improvement)
                        base_values.append(base_value)
                        comp_values.append(comp_value)
                        wins += int(comp_value < base_value) if direction == "lower" else int(comp_value > base_value)
                    n = len(diffs)
                    if n < 3:
                        continue
                    arr = np.array(diffs, dtype=float)
                    std = float(arr.std(ddof=1))
                    t_value = float(arr.mean() / (std / np.sqrt(n))) if std > 0 else 0.0
                    out.append(
                        {
                            "dataset_label": dataset,
                            "baseline_strategy": baseline,
                            "comparator_strategy": comparator,
                            "anchor_count": count,
                            "metric": metric,
                            "direction": direction,
                            "n": n,
                            "baseline_mean": float(np.mean(base_values)),
                            "comparator_mean": float(np.mean(comp_values)),
                            "mean_delta_comparator_minus_baseline": float(np.mean(diffs)),
                            "mean_signed_improvement": float(np.mean(signed_improvements)),
                            "paired_t": t_value,
                            "normal_approx_p": normal_approx_p_from_t(t_value),
                            "comparator_win_rate": wins / n,
                        }
                    )
    return out


def rank_by_score(
    indices: np.ndarray,
    score: np.ndarray,
    candidate_indices: np.ndarray | None = None,
) -> list[int]:
    candidates = indices if candidate_indices is None else np.asarray(candidate_indices, dtype=int)
    order = candidates[np.argsort(-np.asarray(score, dtype=float)[candidates])].astype(int).tolist()
    selected = set(order)
    order.extend(int(index) for index in indices if int(index) not in selected)
    return order


def candidate_zscore(values: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    output[candidate_indices] = zscore(np.asarray(values, dtype=float)[candidate_indices])
    return output


def build_hybrid_rankings(
    descriptors: list[dict],
    train_records: list[SubjectRecord],
    split_id: int,
    configs: list[dict],
    candidate_indices: np.ndarray | None = None,
) -> tuple[dict[str, list[int]], list[dict]]:
    sorted_descriptors = sorted(descriptors, key=lambda row: int(row["node_index"]))
    indices = np.array([int(row["node_index"]) for row in sorted_descriptors], dtype=int)
    influence = np.array([clean_float(row["mean_qrc_influence"]) for row in sorted_descriptors])
    entropy = np.array([clean_float(row["mean_response_entropy"]) for row in sorted_descriptors])
    strength = np.array([clean_float(row["mean_weighted_strength"]) for row in sorted_descriptors])
    cross = np.array([clean_float(row["mean_cross_hemisphere_response"]) for row in sorted_descriptors])

    candidates = indices if candidate_indices is None else np.asarray(candidate_indices, dtype=int)
    residual_influence = residualize_against_strength(
        influence,
        strength,
        candidate_indices=candidates,
    )
    residual_entropy = residualize_against_strength(
        entropy,
        strength,
        candidate_indices=candidates,
    )
    similarity = source_profile_similarity(train_records, target_indices=candidates)
    hemispheres = [str(row["hemisphere"]) for row in sorted_descriptors]

    rankings = {}
    score_rows = []
    for config in configs:
        score = (
            float(config["strength_weight"]) * candidate_zscore(strength, candidates)
            + float(config["resid_influence_weight"]) * candidate_zscore(residual_influence, candidates)
            + float(config["resid_entropy_weight"]) * candidate_zscore(residual_entropy, candidates)
            + float(config["cross_weight"]) * candidate_zscore(cross, candidates)
        )
        strategy = str(config["strategy"])
        if config["greedy"]:
            ranking = greedy_redundancy_ranking(
                score,
                similarity,
                hemispheres,
                redundancy_weight=float(config["redundancy_weight"]),
                balance_weight=float(config["balance_weight"]),
                candidate_indices=candidates,
            )
        else:
            ranking = rank_by_score(indices, score, candidate_indices=candidates)
        rankings[strategy] = ranking
        rank_lookup = {idx: rank for rank, idx in enumerate(ranking, start=1)}
        for i, descriptor in enumerate(sorted_descriptors):
            idx = int(descriptor["node_index"])
            score_rows.append(
                {
                    "split_id": split_id,
                    "strategy": strategy,
                    "rank": rank_lookup[idx],
                    "node_index": idx,
                    "node_id": descriptor["node_id"],
                    "node_name": descriptor["node_name"],
                    "region": descriptor["region"],
                    "hemisphere": descriptor["hemisphere"],
                    "hybrid_score": float(score[i]),
                    "strength_weight": float(config["strength_weight"]),
                    "resid_influence_weight": float(config["resid_influence_weight"]),
                    "resid_entropy_weight": float(config["resid_entropy_weight"]),
                    "cross_weight": float(config["cross_weight"]),
                    "redundancy_weight": float(config["redundancy_weight"]),
                    "balance_weight": float(config["balance_weight"]),
                    "greedy": "yes" if config["greedy"] else "no",
                }
            )
    return rankings, score_rows


def evaluate_rankings(
    train_records: list[SubjectRecord],
    val_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    rankings: dict[str, list[int]],
    counts: list[int],
    split_id: int,
    alpha: float,
    dataset_label: str,
    control: str = "real",
) -> tuple[list[dict], list[dict]]:
    reference = train_records[0].connectome
    final_train = train_records + val_records
    metric_rows = []
    anchor_rows = []
    for count in counts:
        for strategy, ranking in rankings.items():
            anchors = anchors_for_strategy(strategy, ranking, reference.hemispheres, count)
            sample_id = f"{dataset_label}-{control}-{split_id}-{strategy}-{count}"
            anchor_rows.append(
                anchor_record(reference, anchors, sample_id, strategy, count)
                | {
                    "dataset_label": dataset_label,
                    "control": control,
                    "split_id": split_id,
                }
            )
            metric_rows.extend(
                {
                    **row,
                    "dataset_label": dataset_label,
                    "control": control,
                    "split_id": split_id,
                    "selection_strategy": strategy,
                    "selection_uses_coordinates": "no",
                    "sample_id": sample_id,
                }
                for row in evaluate_anchor_model(
                    final_train,
                    test_records,
                    anchors,
                    method="qrc",
                    alpha=alpha,
                    strategy=strategy,
                    sample_id=sample_id,
                )
            )
    return metric_rows, anchor_rows


def add_summary_deltas(summary_rows: list[dict], baseline_strategy: str) -> list[dict]:
    by_key = {
        (row.get("control", "real"), int(row["anchor_count"]), row["selection_strategy"]): row
        for row in summary_rows
    }
    out = []
    for row in summary_rows:
        control = row.get("control", "real")
        count = int(row["anchor_count"])
        baseline = by_key.get((control, count, baseline_strategy))
        baseline_rmse = clean_float(baseline["normalized_rmse_mean"]) if baseline else float("nan")
        baseline_dist = clean_float(baseline["distance_corr_mean"]) if baseline else float("nan")
        out.append(
            {
                **row,
                f"delta_rmse_vs_{baseline_strategy}": clean_float(row["normalized_rmse_mean"]) - baseline_rmse,
                f"delta_distance_corr_vs_{baseline_strategy}": clean_float(row["distance_corr_mean"]) - baseline_dist,
            }
        )
    return out


def load_existing_discovery_metrics(args: argparse.Namespace) -> list[dict]:
    rows = []
    for dataset_label, input_dir in [
        ("hcp86_qrc_v3", args.paired_input_86),
        ("hcp129_qrc_v3", args.paired_input_129),
    ]:
        path = input_dir / "discovery_test_metrics.csv"
        if not path.exists():
            print(f"Skipping paired stats for missing {path}")
            continue
        for row in read_csv(path):
            row["dataset_label"] = dataset_label
            rows.append(row)
    return rows


def write_paired_existing(args: argparse.Namespace) -> list[dict]:
    rows = load_existing_discovery_metrics(args)
    comparisons = [
        ("random_balanced", "unsup_strength"),
        ("random_balanced", "unsup_composite"),
        ("random_balanced", "qrc_v3_greedy_residual"),
        ("random_balanced", "supervised_val_ranked"),
        ("unsup_strength", "qrc_resid_influence"),
        ("unsup_strength", "qrc_resid_entropy"),
        ("unsup_strength", "qrc_v3_residual_composite"),
        ("unsup_strength", "qrc_v3_greedy_residual"),
        ("unsup_strength", "unsup_composite"),
        ("qrc_v3_greedy_residual", "supervised_val_ranked"),
    ]
    pair_rows = paired_selector_tests(rows, comparisons)
    fields = [
        "dataset_label",
        "baseline_strategy",
        "comparator_strategy",
        "anchor_count",
        "metric",
        "direction",
        "n",
        "baseline_mean",
        "comparator_mean",
        "mean_delta_comparator_minus_baseline",
        "mean_signed_improvement",
        "paired_t",
        "normal_approx_p",
        "comparator_win_rate",
    ]
    write_csv(args.output_dir / "paired_selector_tests_existing_qrc_v3.csv", pair_rows, fields)
    return pair_rows


def run_hybrid_analysis(records: list[SubjectRecord], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(args.seed + 200)
    metric_rows = []
    anchor_rows = []
    score_rows = []
    descriptor_rows = []
    strategy_names = [
        "unsup_strength",
        "unsup_composite",
        "qrc_v3_greedy_residual",
    ]

    for split_id in range(1, args.hybrid_splits + 1):
        train_records, val_records, test_records = split_records_three_way(
            records,
            args.train_fraction,
            args.val_fraction,
            rng,
        )
        print(
            f"Hybrid split {split_id}/{args.hybrid_splits}: "
            f"train={len(train_records)} val={len(val_records)} test={len(test_records)}"
        )
        descriptors = node_descriptors(train_records)
        base_rankings, split_descriptor_rows = descriptor_rankings(descriptors, split_id, train_records)
        hybrid_rankings, split_score_rows = build_hybrid_rankings(
            descriptors,
            train_records,
            split_id,
            HYBRID_CONFIGS,
        )
        rankings = {name: base_rankings[name] for name in strategy_names}
        rankings.update(hybrid_rankings)
        split_metrics, split_anchors = evaluate_rankings(
            train_records,
            val_records,
            test_records,
            rankings,
            args.anchor_counts,
            split_id,
            args.ridge_alpha,
            dataset_label="hcp86_hybrid",
        )
        metric_rows.extend(split_metrics)
        anchor_rows.extend(split_anchors)
        descriptor_rows.extend(split_descriptor_rows)
        score_rows.extend(split_score_rows)

    summary = mean_by_group(
        metric_rows,
        ["selection_strategy", "anchor_count"],
        ["normalized_rmse", "distance_corr", "hemisphere_accuracy", "x_corr", "y_corr", "z_corr"],
    )
    summary = add_summary_deltas(add_summary_deltas(summary, "unsup_strength"), "qrc_v3_greedy_residual")
    pair_rows = paired_selector_tests(
        metric_rows,
        [
            ("unsup_strength", "hybrid_s50_q50"),
            ("unsup_strength", "hybrid_s50_q50_greedy"),
            ("unsup_strength", "hybrid_s30_q70_greedy"),
            ("unsup_strength", "hybrid_s70_q30_greedy"),
            ("qrc_v3_greedy_residual", "hybrid_s50_q50_greedy"),
            ("qrc_v3_greedy_residual", "hybrid_s30_q70_greedy"),
            ("qrc_v3_greedy_residual", "hybrid_s70_q30_greedy"),
        ],
        dataset_label="hcp86_hybrid",
    )

    fields = [
        "dataset_label",
        "control",
        "split_id",
        "sample_id",
        "selection_strategy",
        "selection_uses_coordinates",
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
    ]
    write_csv(args.output_dir / "hybrid_selector_metrics.csv", metric_rows, fields)
    summary_fields = [
        "selection_strategy",
        "anchor_count",
        "normalized_rmse_mean",
        "normalized_rmse_std",
        "distance_corr_mean",
        "distance_corr_std",
        "hemisphere_accuracy_mean",
        "hemisphere_accuracy_std",
        "x_corr_mean",
        "y_corr_mean",
        "z_corr_mean",
        "n",
        "delta_rmse_vs_unsup_strength",
        "delta_distance_corr_vs_unsup_strength",
        "delta_rmse_vs_qrc_v3_greedy_residual",
        "delta_distance_corr_vs_qrc_v3_greedy_residual",
    ]
    write_csv(args.output_dir / "hybrid_selector_summary.csv", summary, summary_fields)
    write_csv(
        args.output_dir / "hybrid_selector_paired_tests.csv",
        pair_rows,
        [
            "dataset_label",
            "baseline_strategy",
            "comparator_strategy",
            "anchor_count",
            "metric",
            "direction",
            "n",
            "baseline_mean",
            "comparator_mean",
            "mean_delta_comparator_minus_baseline",
            "mean_signed_improvement",
            "paired_t",
            "normal_approx_p",
            "comparator_win_rate",
        ],
    )
    write_csv(
        args.output_dir / "hybrid_selector_anchor_sets.csv",
        anchor_rows,
        [
            "dataset_label",
            "control",
            "split_id",
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
        args.output_dir / "hybrid_selector_scores.csv",
        score_rows,
        [
            "split_id",
            "strategy",
            "rank",
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "hybrid_score",
            "strength_weight",
            "resid_influence_weight",
            "resid_entropy_weight",
            "cross_weight",
            "redundancy_weight",
            "balance_weight",
            "greedy",
        ],
    )
    write_csv(
        args.output_dir / "hybrid_unsupervised_descriptors.csv",
        descriptor_rows,
        [
            "split_id",
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "unsup_qrc_influence_score",
            "unsup_response_entropy_score",
            "unsup_weighted_strength_score",
            "unsup_cross_hemisphere_score",
            "unsup_composite_score",
            "strength_residual_qrc_influence_score",
            "strength_residual_response_entropy_score",
            "qrc_v3_residual_composite_score",
        ],
    )
    return summary, pair_rows


def run_sensitivity_analysis(records: list[SubjectRecord], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(args.seed + 300)
    metric_rows = []
    score_rows = []
    for split_id in range(1, args.sensitivity_splits + 1):
        train_records, val_records, test_records = split_records_three_way(
            records,
            args.train_fraction,
            args.val_fraction,
            rng,
        )
        print(
            f"Sensitivity split {split_id}/{args.sensitivity_splits}: "
            f"train={len(train_records)} val={len(val_records)} test={len(test_records)}"
        )
        descriptors = node_descriptors(train_records)
        base_rankings, _ = descriptor_rankings(descriptors, split_id, train_records)
        sensitivity_rankings, split_score_rows = build_hybrid_rankings(
            descriptors,
            train_records,
            split_id,
            SENSITIVITY_CONFIGS,
        )
        rankings = {
            "unsup_strength": base_rankings["unsup_strength"],
            "qrc_v3_greedy_residual": base_rankings["qrc_v3_greedy_residual"],
            **sensitivity_rankings,
        }
        split_metrics, _ = evaluate_rankings(
            train_records,
            val_records,
            test_records,
            rankings,
            args.sensitivity_counts,
            split_id,
            args.ridge_alpha,
            dataset_label="hcp86_selector_sensitivity",
        )
        metric_rows.extend(split_metrics)
        score_rows.extend(split_score_rows)

    summary = mean_by_group(
        metric_rows,
        ["selection_strategy", "anchor_count"],
        ["normalized_rmse", "distance_corr", "hemisphere_accuracy"],
    )
    summary = add_summary_deltas(add_summary_deltas(summary, "unsup_strength"), "qrc_v3_greedy_residual")
    pair_rows = paired_selector_tests(
        metric_rows,
        [
            ("unsup_strength", "sens_s30_q70_red015"),
            ("unsup_strength", "sens_s30_q70_red045"),
            ("unsup_strength", "sens_s30_q70_red075"),
            ("unsup_strength", "sens_s50_q50_red045"),
            ("unsup_strength", "sens_s70_q30_red045"),
            ("qrc_v3_greedy_residual", "sens_s30_q70_red015"),
            ("qrc_v3_greedy_residual", "sens_s30_q70_red045"),
            ("qrc_v3_greedy_residual", "sens_s30_q70_red075"),
            ("qrc_v3_greedy_residual", "sens_s50_q50_red045"),
            ("qrc_v3_greedy_residual", "sens_s70_q30_red045"),
        ],
        dataset_label="hcp86_selector_sensitivity",
    )

    write_csv(
        args.output_dir / "selector_sensitivity_metrics.csv",
        metric_rows,
        [
            "dataset_label",
            "control",
            "split_id",
            "sample_id",
            "selection_strategy",
            "selection_uses_coordinates",
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
        args.output_dir / "selector_sensitivity_summary.csv",
        summary,
        [
            "selection_strategy",
            "anchor_count",
            "normalized_rmse_mean",
            "normalized_rmse_std",
            "distance_corr_mean",
            "distance_corr_std",
            "hemisphere_accuracy_mean",
            "hemisphere_accuracy_std",
            "n",
            "delta_rmse_vs_unsup_strength",
            "delta_distance_corr_vs_unsup_strength",
            "delta_rmse_vs_qrc_v3_greedy_residual",
            "delta_distance_corr_vs_qrc_v3_greedy_residual",
        ],
    )
    write_csv(
        args.output_dir / "selector_sensitivity_paired_tests.csv",
        pair_rows,
        [
            "dataset_label",
            "baseline_strategy",
            "comparator_strategy",
            "anchor_count",
            "metric",
            "direction",
            "n",
            "baseline_mean",
            "comparator_mean",
            "mean_delta_comparator_minus_baseline",
            "mean_signed_improvement",
            "paired_t",
            "normal_approx_p",
            "comparator_win_rate",
        ],
    )
    write_csv(
        args.output_dir / "selector_sensitivity_scores.csv",
        score_rows,
        [
            "split_id",
            "strategy",
            "rank",
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "hybrid_score",
            "strength_weight",
            "resid_influence_weight",
            "resid_entropy_weight",
            "cross_weight",
            "redundancy_weight",
            "balance_weight",
            "greedy",
        ],
    )
    return summary, pair_rows


def paired_null_tests(rows: list[dict]) -> list[dict]:
    aggregated = aggregate_strategy_rows(rows, "hcp86_selector_null")
    index = {
        (
            row["control"],
            int(row["split_id"]),
            row["subject_id"],
            int(row["anchor_count"]),
            row["selection_strategy"],
        ): row
        for row in aggregated
    }
    controls = sorted({row["control"] for row in aggregated if row.get("control") != "real"})
    counts = sorted({int(row["anchor_count"]) for row in aggregated})
    strategies = sorted({row["selection_strategy"] for row in aggregated})
    out = []
    for control in controls:
        for strategy in strategies:
            for count in counts:
                cases = [
                    (int(row["split_id"]), row["subject_id"])
                    for row in aggregated
                    if row["control"] == "real"
                    and row["selection_strategy"] == strategy
                    and int(row["anchor_count"]) == count
                ]
                for metric, direction in METRIC_DIRECTIONS.items():
                    diffs = []
                    signed = []
                    real_values = []
                    control_values = []
                    wins = 0
                    for split_id, subject_id in cases:
                        real = index.get(("real", split_id, subject_id, count, strategy))
                        null = index.get((control, split_id, subject_id, count, strategy))
                        if real is None or null is None:
                            continue
                        real_value = clean_float(real[metric])
                        null_value = clean_float(null[metric])
                        if not (np.isfinite(real_value) and np.isfinite(null_value)):
                            continue
                        delta = real_value - null_value
                        improvement = -delta if direction == "lower" else delta
                        diffs.append(delta)
                        signed.append(improvement)
                        real_values.append(real_value)
                        control_values.append(null_value)
                        wins += int(real_value < null_value) if direction == "lower" else int(real_value > null_value)
                    n = len(diffs)
                    if n < 3:
                        continue
                    arr = np.array(diffs, dtype=float)
                    std = float(arr.std(ddof=1))
                    t_value = float(arr.mean() / (std / np.sqrt(n))) if std > 0 else 0.0
                    out.append(
                        {
                            "control": control,
                            "selection_strategy": strategy,
                            "anchor_count": count,
                            "metric": metric,
                            "direction": direction,
                            "n": n,
                            "real_mean": float(np.mean(real_values)),
                            "control_mean": float(np.mean(control_values)),
                            "mean_delta_real_minus_control": float(np.mean(diffs)),
                            "mean_signed_real_improvement": float(np.mean(signed)),
                            "paired_t": t_value,
                            "normal_approx_p": normal_approx_p_from_t(t_value),
                            "real_win_rate": wins / n,
                        }
                    )
    return out


def run_selector_null_controls(records: list[SubjectRecord], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    null_records = records[: min(args.null_subjects, len(records))]
    print(f"Building selector null controls with {len(null_records)} subjects")
    coord_records = coordinate_shuffle_records(null_records, np.random.default_rng(args.seed + 401))
    rewired_records = degree_rewired_records(null_records, args.times, np.random.default_rng(args.seed + 402))
    controls = {
        "real": null_records,
        "coord_shuffle": coord_records,
        "degree_rewire": rewired_records,
    }
    rng = np.random.default_rng(args.seed + 400)
    metric_rows = []
    anchor_rows = []
    for split_id in range(1, args.null_splits + 1):
        train_idx, val_idx, test_idx = split_indices_three_way(
            len(null_records),
            args.train_fraction,
            args.val_fraction,
            rng,
        )
        print(
            f"Null split {split_id}/{args.null_splits}: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )
        for control, control_records in controls.items():
            train_records = take(control_records, train_idx)
            val_records = take(control_records, val_idx)
            test_records = take(control_records, test_idx)
            descriptors = node_descriptors(train_records)
            base_rankings, _ = descriptor_rankings(descriptors, split_id, train_records)
            hybrid_rankings, _ = build_hybrid_rankings(
                descriptors,
                train_records,
                split_id,
                [HYBRID_CONFIGS[1], HYBRID_CONFIGS[2], HYBRID_CONFIGS[3]],
            )
            rankings = {
                "unsup_strength": base_rankings["unsup_strength"],
                "qrc_v3_greedy_residual": base_rankings["qrc_v3_greedy_residual"],
                "hybrid_s50_q50_greedy": hybrid_rankings["hybrid_s50_q50_greedy"],
                "hybrid_s30_q70_greedy": hybrid_rankings["hybrid_s30_q70_greedy"],
                "hybrid_s70_q30_greedy": hybrid_rankings["hybrid_s70_q30_greedy"],
            }
            split_metrics, split_anchors = evaluate_rankings(
                train_records,
                val_records,
                test_records,
                rankings,
                args.null_counts,
                split_id,
                args.ridge_alpha,
                dataset_label="hcp86_selector_null",
                control=control,
            )
            metric_rows.extend(split_metrics)
            anchor_rows.extend(split_anchors)

    summary = mean_by_group(
        metric_rows,
        ["control", "selection_strategy", "anchor_count"],
        ["normalized_rmse", "distance_corr", "hemisphere_accuracy"],
    )
    pair_rows = paired_null_tests(metric_rows)
    write_csv(
        args.output_dir / "selector_null_metrics.csv",
        metric_rows,
        [
            "dataset_label",
            "control",
            "split_id",
            "sample_id",
            "selection_strategy",
            "selection_uses_coordinates",
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
        args.output_dir / "selector_null_summary.csv",
        summary,
        [
            "control",
            "selection_strategy",
            "anchor_count",
            "normalized_rmse_mean",
            "normalized_rmse_std",
            "distance_corr_mean",
            "distance_corr_std",
            "hemisphere_accuracy_mean",
            "hemisphere_accuracy_std",
            "n",
        ],
    )
    write_csv(
        args.output_dir / "selector_null_pairwise_tests.csv",
        pair_rows,
        [
            "control",
            "selection_strategy",
            "anchor_count",
            "metric",
            "direction",
            "n",
            "real_mean",
            "control_mean",
            "mean_delta_real_minus_control",
            "mean_signed_real_improvement",
            "paired_t",
            "normal_approx_p",
            "real_win_rate",
        ],
    )
    write_csv(
        args.output_dir / "selector_null_anchor_sets.csv",
        anchor_rows,
        [
            "dataset_label",
            "control",
            "split_id",
            "sample_id",
            "anchor_strategy",
            "anchor_count",
            "anchor_ids",
            "anchor_names",
            "anchor_regions",
            "anchor_hemispheres",
        ],
    )
    return summary, pair_rows


def write_markdown_summary(
    args: argparse.Namespace,
    existing_pair_rows: list[dict],
    hybrid_summary: list[dict],
    hybrid_pair_rows: list[dict],
    sensitivity_summary: list[dict],
    null_summary: list[dict],
    null_pair_rows: list[dict],
) -> None:
    focus_existing = [
        row
        for row in existing_pair_rows
        if row["metric"] == "normalized_rmse"
        and row["baseline_strategy"] == "unsup_strength"
        and row["comparator_strategy"]
        in {"qrc_v3_greedy_residual", "qrc_v3_residual_composite", "unsup_composite"}
    ]
    focus_existing.sort(key=lambda row: (row["dataset_label"], int(row["anchor_count"]), row["comparator_strategy"]))

    hybrid_focus = sorted(
        [
            row
            for row in hybrid_summary
            if row["selection_strategy"]
            in {
                "unsup_strength",
                "qrc_v3_greedy_residual",
                "hybrid_s50_q50_greedy",
                "hybrid_s30_q70_greedy",
                "hybrid_s70_q30_greedy",
            }
        ],
        key=lambda row: (int(row["anchor_count"]), clean_float(row["normalized_rmse_mean"])),
    )
    hybrid_pairs_focus = [
        row
        for row in hybrid_pair_rows
        if row["metric"] == "normalized_rmse"
        and row["baseline_strategy"] in {"unsup_strength", "qrc_v3_greedy_residual"}
    ]
    hybrid_pairs_focus.sort(key=lambda row: (int(row["anchor_count"]), row["baseline_strategy"], row["comparator_strategy"]))

    sensitivity_focus = sorted(
        sensitivity_summary,
        key=lambda row: (int(row["anchor_count"]), clean_float(row["normalized_rmse_mean"])),
    )

    null_focus = sorted(
        null_summary,
        key=lambda row: (row["control"], int(row["anchor_count"]), clean_float(row["normalized_rmse_mean"])),
    )
    null_pairs_focus = [
        row
        for row in null_pair_rows
        if row["metric"] == "normalized_rmse" and row["selection_strategy"] in {"unsup_strength", "hybrid_s50_q50_greedy"}
    ]
    null_pairs_focus.sort(key=lambda row: (row["control"], int(row["anchor_count"]), row["selection_strategy"]))

    text = f"""# Selector Deep Dive

This analysis asks whether the QRC source-coordinate method can discover useful
source hierarchies, not merely decode coordinates from hand-chosen sources.

## 1. Paired Selector Statistics for Existing QRC-v3 Runs

Negative RMSE deltas mean the comparator beat the baseline on the same split,
subject, and source count. The existing full 86-node and 129-node runs show a
mixed but informative pattern: strength hubs are hard to beat at medium source
counts, while residual-QRC selectors become competitive at larger source sets.

{markdown_table(focus_existing, ["dataset_label", "baseline_strategy", "comparator_strategy", "anchor_count", "baseline_mean", "comparator_mean", "mean_delta_comparator_minus_baseline", "comparator_win_rate", "normal_approx_p"], 64)}

## 2. Hybrid Selector Results

Hybrid selectors combine weighted strength with strength-residual QRC influence
and entropy. This tests the practical hypothesis that the best brain source
hierarchy may need both hub-like reach and non-hub QRC relational structure.

{markdown_table(hybrid_focus, ["selection_strategy", "anchor_count", "normalized_rmse_mean", "distance_corr_mean", "delta_rmse_vs_unsup_strength", "delta_rmse_vs_qrc_v3_greedy_residual", "n"], 48)}

Paired hybrid comparisons:

{markdown_table(hybrid_pairs_focus, ["baseline_strategy", "comparator_strategy", "anchor_count", "baseline_mean", "comparator_mean", "mean_delta_comparator_minus_baseline", "comparator_win_rate", "normal_approx_p"], 48)}

## 3. Selector Sensitivity

The sensitivity grid varies the strength/QRC weighting and the redundancy
penalty. Stable improvements here are more credible than a single tuned
selector.

{markdown_table(sensitivity_focus, ["selection_strategy", "anchor_count", "normalized_rmse_mean", "distance_corr_mean", "delta_rmse_vs_unsup_strength", "delta_rmse_vs_qrc_v3_greedy_residual", "n"], 48)}

## 4. Source-Discovery Null Controls

The null controls evaluate source discovery after independently shuffling
coordinate labels across subjects or degree-preserving graph rewiring. Real
performance should be much better than both nulls if the source hierarchy uses
real anatomical graph structure.

{markdown_table(null_focus, ["control", "selection_strategy", "anchor_count", "normalized_rmse_mean", "distance_corr_mean", "hemisphere_accuracy_mean", "n"], 60)}

Paired real-vs-null RMSE comparisons:

{markdown_table(null_pairs_focus, ["control", "selection_strategy", "anchor_count", "real_mean", "control_mean", "mean_delta_real_minus_control", "real_win_rate", "normal_approx_p"], 32)}

## Working Interpretation

The important signal is nuanced. QRC descriptors are not a magic replacement for
weighted strength: source hubs remain powerful. The more interesting result is
that strength-residual QRC information helps at larger source sets and can be
combined with strength in hybrid selectors. That supports the source-coordinate
idea as a graph algorithm: a useful hierarchy may begin with high-reach hubs
and then add QRC-residual sources that cover relational modes not explained by
strength alone.
"""
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    if args.docs_path is not None:
        args.docs_path.parent.mkdir(parents=True, exist_ok=True)
        args.docs_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QRC selector deep-dive analyses.")
    parser.add_argument("--input-dir", type=Path, default=Path("research/data/brain_graph_hcp_86_nodes/graphml"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/outputs/selector_deep_dive"))
    parser.add_argument("--docs-path", type=Path, default=Path("docs/selector_deep_dive_2026-06-18.md"))
    parser.add_argument("--paired-input-86", type=Path, default=Path("research/outputs/qrc_v3_source_discovery_86_nodes"))
    parser.add_argument("--paired-input-129", type=Path, default=Path("research/outputs/qrc_v3_source_discovery_129_nodes"))
    parser.add_argument("--subjects", type=int, default=1064)
    parser.add_argument("--hybrid-splits", type=int, default=5)
    parser.add_argument("--sensitivity-splits", type=int, default=3)
    parser.add_argument("--null-subjects", type=int, default=320)
    parser.add_argument("--null-splits", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--anchor-counts", type=parse_ints, default=parse_ints("8,16,32"))
    parser.add_argument("--sensitivity-counts", type=parse_ints, default=parse_ints("16,32"))
    parser.add_argument("--null-counts", type=parse_ints, default=parse_ints("16,32"))
    parser.add_argument("--times", type=parse_times, default=parse_times("0.25,0.5,1,2,4,8"))
    parser.add_argument("--weight-transform", choices=["raw", "log1p", "binary"], default="log1p")
    parser.add_argument("--ridge-alpha", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--skip-paired", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-nulls", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing_pair_rows: list[dict] = []
    if not args.skip_paired:
        print("Writing paired selector statistics for existing QRC-v3 runs")
        existing_pair_rows = write_paired_existing(args)

    records: list[SubjectRecord] = []
    if not (args.skip_hybrid and args.skip_sensitivity and args.skip_nulls):
        files = sorted(args.input_dir.glob("*.graphml"))[: args.subjects]
        if not files:
            raise SystemExit(f"No GraphML files found in {args.input_dir}")
        print(f"Caching {len(files)} 86-node subjects for selector deep dive")
        records = [cache_subject(path, args.times, args.weight_transform) for path in files]

    hybrid_summary: list[dict] = []
    hybrid_pair_rows: list[dict] = []
    if not args.skip_hybrid:
        hybrid_summary, hybrid_pair_rows = run_hybrid_analysis(records, args)

    sensitivity_summary: list[dict] = []
    if not args.skip_sensitivity:
        sensitivity_summary, _ = run_sensitivity_analysis(records, args)

    null_summary: list[dict] = []
    null_pair_rows: list[dict] = []
    if not args.skip_nulls:
        null_summary, null_pair_rows = run_selector_null_controls(records, args)

    write_markdown_summary(
        args,
        existing_pair_rows,
        hybrid_summary,
        hybrid_pair_rows,
        sensitivity_summary,
        null_summary,
        null_pair_rows,
    )
    print(f"Wrote selector deep-dive outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
