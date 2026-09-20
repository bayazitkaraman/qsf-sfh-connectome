"""Strict source-discovery analyses for QRC-v2.

This suite moves from exploratory mechanism analysis to a cleaner algorithmic
test:

1. Rank source regions on train/validation data only.
2. Evaluate nested source sets once on held-out test subjects.
3. Compare supervised validation ranking with graph-only unsupervised rules.
4. Measure hard-node failure modes and source stability across splits.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain_quantum.analysis_suite import anchor_metrics, farthest_spread_anchors, mean_by_group  # noqa: E402
from brain_quantum.mechanism_suite import (  # noqa: E402
    anchor_record,
    balanced_ranked_anchors,
    choose_balanced_from_pool,
    clean_float,
    evaluate_anchor_model,
    evaluate_anchor_model_fixed_targets,
    fmt,
    markdown_table,
    node_descriptors,
    subset_farthest_spread_anchors,
    top_ranked_anchors,
    write_bar_svg,
    write_line_svg,
)
from brain_quantum.qrc_connectome import parse_times, write_csv  # noqa: E402
from brain_quantum.source_coordinate_v2 import (  # noqa: E402
    SubjectRecord,
    anchor_features,
    cache_subject,
    choose_balanced_anchors,
    fit_global_ridge,
    mean_coordinate_template,
    predict_ridge,
)


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def split_records_three_way(
    records: list[SubjectRecord],
    train_fraction: float,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[SubjectRecord], list[SubjectRecord], list[SubjectRecord]]:
    indices = np.arange(len(records))
    rng.shuffle(indices)
    train_n = max(1, int(len(indices) * train_fraction))
    val_n = max(1, int(len(indices) * val_fraction))
    if train_n + val_n >= len(indices):
        val_n = max(1, len(indices) - train_n - 1)
    train_idx = indices[:train_n]
    val_idx = indices[train_n : train_n + val_n]
    test_idx = indices[train_n + val_n :]
    if len(test_idx) == 0:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]
    return [records[i] for i in train_idx], [records[i] for i in val_idx], [records[i] for i in test_idx]


def summarize_single_source(
    metric_rows: list[dict],
    descriptors: list[dict],
    split_id: int,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in metric_rows:
        grouped[int(row["source_index"])].append(row)
    descriptor_by_index = {int(row["node_index"]): row for row in descriptors}

    rows = []
    for idx, group_rows in grouped.items():
        descriptor = descriptor_by_index[idx]

        def mean_metric(name: str) -> float:
            values = [clean_float(row[name]) for row in group_rows]
            values = [value for value in values if np.isfinite(value)]
            return float(np.mean(values)) if values else float("nan")

        rows.append(
            {
                "split_id": split_id,
                "node_index": idx,
                "node_id": descriptor["node_id"],
                "node_name": descriptor["node_name"],
                "region": descriptor["region"],
                "hemisphere": descriptor["hemisphere"],
                "val_single_qrc_rmse_mean": mean_metric("normalized_rmse"),
                "val_single_qrc_hemi_mean": mean_metric("hemisphere_accuracy"),
                "val_single_qrc_distance_corr_mean": mean_metric("distance_corr"),
                "mean_qrc_influence": descriptor["mean_qrc_influence"],
                "mean_weighted_strength": descriptor["mean_weighted_strength"],
                "mean_response_entropy": descriptor["mean_response_entropy"],
                "mean_cross_hemisphere_response": descriptor["mean_cross_hemisphere_response"],
            }
        )
    rows.sort(key=lambda row: row["val_single_qrc_rmse_mean"])
    for rank, row in enumerate(rows, start=1):
        row["val_single_source_rank"] = rank
    return rows


def rank_single_sources_on_validation(
    train_records: list[SubjectRecord],
    val_records: list[SubjectRecord],
    descriptors: list[dict],
    split_id: int,
    alpha: float,
) -> tuple[list[dict], list[dict]]:
    reference = train_records[0].connectome
    metric_rows = []
    for idx in range(len(reference.node_ids)):
        anchors = np.array([idx], dtype=int)
        model = fit_global_ridge(train_records, anchors, "qrc", alpha=alpha, qrc_variant="full")
        for record in val_records:
            features, _ = anchor_features(record, anchors, "qrc", qrc_variant="full")
            pred = predict_ridge(model, features)
            row = anchor_metrics(
                record.connectome,
                "qrc",
                pred,
                anchors,
                "validation_single_source",
                1,
                "global_ridge",
                record.extent,
            )
            row.update(
                {
                    "split_id": split_id,
                    "source_index": idx,
                    "source_node_id": reference.node_ids[idx],
                    "source_node_name": reference.node_names[idx],
                    "source_region": reference.regions[idx],
                    "source_hemisphere": reference.hemispheres[idx],
                }
            )
            metric_rows.append(row)
    return metric_rows, summarize_single_source(metric_rows, descriptors, split_id)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if std < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def residualize_against_strength(
    values: np.ndarray,
    strength: np.ndarray,
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    candidates = (
        np.arange(len(strength), dtype=int)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=int)
    )
    x = np.column_stack([np.ones(len(strength)), strength.astype(float)])
    y = values.astype(float)
    coef = np.linalg.lstsq(x[candidates], y[candidates], rcond=None)[0]
    return y - x @ coef


def source_profile_similarity(
    records: list[SubjectRecord],
    target_indices: np.ndarray | None = None,
) -> np.ndarray:
    n = len(records[0].connectome.node_ids)
    profiles = np.zeros((n, n), dtype=float)
    for record in records:
        q_avg = np.mean(np.stack(record.q_prob, axis=0), axis=0)
        col_sums = q_avg.sum(axis=0, keepdims=True)
        profiles += q_avg / np.maximum(col_sums, 1e-12)
    profiles /= max(len(records), 1)
    targets = (
        np.arange(n, dtype=int)
        if target_indices is None
        else np.asarray(target_indices, dtype=int)
    )
    active_profiles = profiles[targets]
    centered = active_profiles - active_profiles.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0, keepdims=True)
    normalized = centered / np.maximum(norms, 1e-12)
    similarity = np.abs(normalized.T @ normalized)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def greedy_redundancy_ranking(
    scores: np.ndarray,
    similarity: np.ndarray,
    hemispheres: list[str],
    redundancy_weight: float = 0.45,
    balance_weight: float = 0.30,
    candidate_indices: np.ndarray | None = None,
) -> list[int]:
    n = len(scores)
    selected: list[int] = []
    candidates = (
        np.arange(n, dtype=int)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=int)
    )
    remaining = set(int(index) for index in candidates)
    left_count = 0
    right_count = 0
    scaled_scores = np.zeros(n, dtype=float)
    scaled_scores[candidates] = zscore(np.asarray(scores, dtype=float)[candidates])
    while remaining:
        best_idx = None
        best_value = -float("inf")
        for idx in remaining:
            redundancy = float(np.max(similarity[idx, selected])) if selected else 0.0
            hemi = hemispheres[idx]
            new_left = left_count + int(hemi.startswith("left"))
            new_right = right_count + int(hemi.startswith("right"))
            balance_penalty = max(abs(new_left - new_right) - 1, 0)
            value = scaled_scores[idx] - redundancy_weight * redundancy - balance_weight * balance_penalty
            if value > best_value:
                best_value = value
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
        left_count += int(hemispheres[best_idx].startswith("left"))
        right_count += int(hemispheres[best_idx].startswith("right"))
    selected.extend(index for index in range(n) if index not in set(selected))
    return selected


def descriptor_rankings(
    descriptors: list[dict],
    split_id: int,
    train_records: list[SubjectRecord],
) -> tuple[dict[str, list[int]], list[dict]]:
    rows = []
    indices = np.array([int(row["node_index"]) for row in descriptors], dtype=int)
    influence = np.array([clean_float(row["mean_qrc_influence"]) for row in descriptors])
    entropy = np.array([clean_float(row["mean_response_entropy"]) for row in descriptors])
    strength = np.array([clean_float(row["mean_weighted_strength"]) for row in descriptors])
    cross = np.array([clean_float(row["mean_cross_hemisphere_response"]) for row in descriptors])
    composite = 0.40 * zscore(influence) + 0.40 * zscore(entropy) + 0.20 * zscore(strength)
    residual_influence = residualize_against_strength(influence, strength)
    residual_entropy = residualize_against_strength(entropy, strength)
    residual_composite = 0.45 * zscore(residual_influence) + 0.45 * zscore(residual_entropy) + 0.10 * zscore(cross)
    similarity = source_profile_similarity(train_records)
    hemispheres = [str(row["hemisphere"]) for row in sorted(descriptors, key=lambda row: int(row["node_index"]))]
    greedy_residual = greedy_redundancy_ranking(residual_composite, similarity, hemispheres)
    for i, row in enumerate(descriptors):
        rows.append(
            {
                "split_id": split_id,
                "node_index": int(row["node_index"]),
                "node_id": row["node_id"],
                "node_name": row["node_name"],
                "region": row["region"],
                "hemisphere": row["hemisphere"],
                "unsup_qrc_influence_score": float(influence[i]),
                "unsup_response_entropy_score": float(entropy[i]),
                "unsup_weighted_strength_score": float(strength[i]),
                "unsup_cross_hemisphere_score": float(cross[i]),
                "unsup_composite_score": float(composite[i]),
                "strength_residual_qrc_influence_score": float(residual_influence[i]),
                "strength_residual_response_entropy_score": float(residual_entropy[i]),
                "qrc_v3_residual_composite_score": float(residual_composite[i]),
            }
        )

    def rank_by(values: np.ndarray) -> list[int]:
        order = np.argsort(-values)
        return indices[order].astype(int).tolist()

    rankings = {
        "unsup_qrc_influence": rank_by(influence),
        "unsup_response_entropy": rank_by(entropy),
        "unsup_strength": rank_by(strength),
        "unsup_composite": rank_by(composite),
        "qrc_resid_influence": rank_by(residual_influence),
        "qrc_resid_entropy": rank_by(residual_entropy),
        "qrc_v3_residual_composite": rank_by(residual_composite),
        "qrc_v3_greedy_residual": greedy_residual,
    }
    return rankings, rows


def ranking_from_source_summary(source_summary: list[dict]) -> list[int]:
    return [int(row["node_index"]) for row in sorted(source_summary, key=lambda row: row["val_single_qrc_rmse_mean"])]


def anchors_for_strategy(strategy: str, ranking: list[int], hemispheres: list[str], count: int) -> np.ndarray:
    if "greedy" in strategy:
        return top_ranked_anchors(ranking, count)
    return balanced_ranked_anchors(ranking, hemispheres, count)


def evaluate_discovered_sources(
    train_records: list[SubjectRecord],
    val_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    source_summary: list[dict],
    unsup_rankings: dict[str, list[int]],
    anchor_counts: list[int],
    random_repeats: int,
    split_id: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict[str, list[int]]]:
    reference = train_records[0].connectome
    training_positions = mean_coordinate_template(train_records)
    final_train = train_records + val_records
    supervised_ranking = ranking_from_source_summary(source_summary)
    strategy_rankings = {
        "supervised_val_ranked": supervised_ranking,
        **unsup_rankings,
    }

    metric_rows = []
    anchor_rows = []
    selected_by_strategy: dict[str, list[int]] = {}
    for count in anchor_counts:
        for strategy, ranking in strategy_rankings.items():
            anchors = anchors_for_strategy(strategy, ranking, reference.hemispheres, count)
            sample_id = f"{split_id}-{strategy}-{count}"
            anchor_rows.append(anchor_record(reference, anchors, sample_id, strategy, count) | {"split_id": split_id})
            selected_by_strategy[f"{strategy}:{count}"] = anchors.tolist()
            metric_rows.extend(
                {
                    **row,
                    "split_id": split_id,
                    "selection_strategy": strategy,
                    "selection_uses_coordinates": "yes" if strategy == "supervised_val_ranked" else "no",
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

        spatial = farthest_spread_anchors(training_positions, count)
        sample_id = f"{split_id}-atlas_spatial_spread-{count}"
        anchor_rows.append(anchor_record(reference, spatial, sample_id, "atlas_spatial_spread", count) | {"split_id": split_id})
        metric_rows.extend(
            {
                **row,
                "split_id": split_id,
                "selection_strategy": "atlas_spatial_spread",
                "selection_uses_coordinates": "atlas_only",
                "sample_id": sample_id,
            }
            for row in evaluate_anchor_model(
                final_train,
                test_records,
                spatial,
                method="qrc",
                alpha=alpha,
                strategy="atlas_spatial_spread",
                sample_id=sample_id,
            )
        )

        for repeat in range(random_repeats):
            anchors = choose_balanced_anchors(reference.hemispheres, count, rng)
            sample_id = f"{split_id}-random_balanced-{count}-{repeat + 1}"
            anchor_rows.append(anchor_record(reference, anchors, sample_id, "random_balanced", count) | {"split_id": split_id})
            metric_rows.extend(
                {
                    **row,
                    "split_id": split_id,
                    "selection_strategy": "random_balanced",
                    "selection_uses_coordinates": "no",
                    "sample_id": sample_id,
                }
                for row in evaluate_anchor_model(
                    final_train,
                    test_records,
                    anchors,
                    method="qrc",
                    alpha=alpha,
                    strategy="random_balanced",
                    sample_id=sample_id,
                )
            )
    return metric_rows, anchor_rows, strategy_rankings


def evaluate_fixed_targets(
    train_records: list[SubjectRecord],
    val_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    source_summary: list[dict],
    unsup_rankings: dict[str, list[int]],
    anchor_counts: list[int],
    random_repeats: int,
    target_count: int,
    split_id: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict]]:
    reference = train_records[0].connectome
    training_positions = mean_coordinate_template(train_records)
    final_train = train_records + val_records
    supervised_ranking = ranking_from_source_summary(source_summary)
    target_indices = np.array(sorted(supervised_ranking[-target_count:]), dtype=int)
    target_set = set(target_indices.tolist())
    all_candidates = [idx for idx in range(len(reference.node_ids)) if idx not in target_set]

    target_rows = [
        {
            "split_id": split_id,
            "target_index": idx,
            "node_id": reference.node_ids[idx],
            "node_name": reference.node_names[idx],
            "region": reference.regions[idx],
            "hemisphere": reference.hemispheres[idx],
        }
        for idx in target_indices
    ]

    strategy_rankings = {
        "fixed_supervised_val_ranked": [idx for idx in supervised_ranking if idx not in target_set],
        "fixed_unsup_composite": [idx for idx in unsup_rankings["unsup_composite"] if idx not in target_set],
        "fixed_unsup_qrc_influence": [idx for idx in unsup_rankings["unsup_qrc_influence"] if idx not in target_set],
        "fixed_unsup_strength": [idx for idx in unsup_rankings["unsup_strength"] if idx not in target_set],
        "fixed_qrc_resid_influence": [idx for idx in unsup_rankings["qrc_resid_influence"] if idx not in target_set],
        "fixed_qrc_resid_entropy": [idx for idx in unsup_rankings["qrc_resid_entropy"] if idx not in target_set],
        "fixed_qrc_v3_residual_composite": [
            idx for idx in unsup_rankings["qrc_v3_residual_composite"] if idx not in target_set
        ],
        "fixed_qrc_v3_greedy_residual": [idx for idx in unsup_rankings["qrc_v3_greedy_residual"] if idx not in target_set],
    }

    metric_rows = []
    anchor_rows = []
    for count in anchor_counts:
        for strategy, ranking in strategy_rankings.items():
            anchors = anchors_for_strategy(strategy, ranking, reference.hemispheres, count)
            sample_id = f"{split_id}-{strategy}-{count}"
            anchor_rows.append(anchor_record(reference, anchors, sample_id, strategy, count) | {"split_id": split_id})
            metric_rows.extend(
                {
                    **row,
                    "split_id": split_id,
                    "selection_strategy": strategy,
                    "sample_id": sample_id,
                }
                for row in evaluate_anchor_model_fixed_targets(
                    final_train,
                    test_records,
                    anchors,
                    target_indices,
                    alpha=alpha,
                    strategy=strategy,
                    sample_id=sample_id,
                )
            )

        spatial = subset_farthest_spread_anchors(training_positions, all_candidates, count)
        sample_id = f"{split_id}-fixed_atlas_spatial_spread-{count}"
        anchor_rows.append(anchor_record(reference, spatial, sample_id, "fixed_atlas_spatial_spread", count) | {"split_id": split_id})
        metric_rows.extend(
            {
                **row,
                "split_id": split_id,
                "selection_strategy": "fixed_atlas_spatial_spread",
                "sample_id": sample_id,
            }
            for row in evaluate_anchor_model_fixed_targets(
                final_train,
                test_records,
                spatial,
                target_indices,
                alpha=alpha,
                strategy="fixed_atlas_spatial_spread",
                sample_id=sample_id,
            )
        )

        for repeat in range(random_repeats):
            anchors = choose_balanced_from_pool(reference.hemispheres, all_candidates, count, rng)
            sample_id = f"{split_id}-fixed_random_balanced-{count}-{repeat + 1}"
            anchor_rows.append(anchor_record(reference, anchors, sample_id, "fixed_random_balanced", count) | {"split_id": split_id})
            metric_rows.extend(
                {
                    **row,
                    "split_id": split_id,
                    "selection_strategy": "fixed_random_balanced",
                    "sample_id": sample_id,
                }
                for row in evaluate_anchor_model_fixed_targets(
                    final_train,
                    test_records,
                    anchors,
                    target_indices,
                    alpha=alpha,
                    strategy="fixed_random_balanced",
                    sample_id=sample_id,
                )
            )
    return metric_rows, anchor_rows, target_rows


def collect_node_errors(
    train_records: list[SubjectRecord],
    val_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    source_summary: list[dict],
    unsup_rankings: dict[str, list[int]],
    counts: list[int],
    split_id: int,
    alpha: float,
) -> list[dict]:
    reference = train_records[0].connectome
    final_train = train_records + val_records
    supervised_ranking = ranking_from_source_summary(source_summary)
    strategies = {
        "supervised_val_ranked": supervised_ranking,
        "unsup_composite": unsup_rankings["unsup_composite"],
        "qrc_v3_greedy_residual": unsup_rankings["qrc_v3_greedy_residual"],
    }
    rows = []
    for count in counts:
        for strategy, ranking in strategies.items():
            anchors = balanced_ranked_anchors(ranking, reference.hemispheres, count)
            anchor_set = set(anchors.tolist())
            model = fit_global_ridge(final_train, anchors, "qrc", alpha=alpha, qrc_variant="full")
            for record in test_records:
                features, _ = anchor_features(record, anchors, "qrc", qrc_variant="full")
                pred = predict_ridge(model, features)
                errors = pred - record.connectome.positions
                for idx in range(len(record.connectome.node_ids)):
                    if idx in anchor_set or not np.all(np.isfinite(record.connectome.positions[idx])):
                        continue
                    err = errors[idx]
                    rows.append(
                        {
                            "split_id": split_id,
                            "selection_strategy": strategy,
                            "anchor_count": count,
                            "subject_id": record.connectome.subject_id,
                            "node_index": idx,
                            "node_id": record.connectome.node_ids[idx],
                            "node_name": record.connectome.node_names[idx],
                            "region": record.connectome.regions[idx],
                            "hemisphere": record.connectome.hemispheres[idx],
                            "normalized_error": float(np.linalg.norm(err) / record.extent),
                            "x_abs_error": float(abs(err[0]) / record.extent),
                            "y_abs_error": float(abs(err[1]) / record.extent),
                            "z_abs_error": float(abs(err[2]) / record.extent),
                        }
                    )
    return rows


def summarize_node_errors(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    node_summary = mean_by_group(
        rows,
        ["selection_strategy", "anchor_count", "node_index", "node_id", "node_name", "region", "hemisphere"],
        ["normalized_error", "x_abs_error", "y_abs_error", "z_abs_error"],
    )
    for strategy in sorted({row["selection_strategy"] for row in node_summary}):
        for count in sorted({int(row["anchor_count"]) for row in node_summary if row["selection_strategy"] == strategy}):
            subset = [
                row
                for row in node_summary
                if row["selection_strategy"] == strategy and int(row["anchor_count"]) == count
            ]
            subset.sort(key=lambda row: -clean_float(row["normalized_error_mean"]))
            for rank, row in enumerate(subset, start=1):
                row["hard_node_rank"] = rank

    region_summary = mean_by_group(
        rows,
        ["selection_strategy", "anchor_count", "region", "hemisphere"],
        ["normalized_error", "x_abs_error", "y_abs_error", "z_abs_error"],
    )
    return node_summary, region_summary


def stability_rows(
    split_rankings: list[dict],
    reference,
    top_ks: list[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    strategies = sorted({item["strategy"] for item in split_rankings})
    node_rows = []
    jaccard_rows = []
    rank_rows = []

    for strategy in strategies:
        rankings = [item for item in split_rankings if item["strategy"] == strategy]
        for idx in range(len(reference.node_ids)):
            ranks = []
            for item in rankings:
                ranks.append(item["ranking"].index(idx) + 1)
            row = {
                "strategy": strategy,
                "node_index": idx,
                "node_id": reference.node_ids[idx],
                "node_name": reference.node_names[idx],
                "region": reference.regions[idx],
                "hemisphere": reference.hemispheres[idx],
                "mean_rank": float(np.mean(ranks)),
                "rank_std": float(np.std(ranks)),
            }
            for top_k in top_ks:
                row[f"top_{top_k}_frequency"] = int(sum(rank <= top_k for rank in ranks))
            node_rows.append(row)

        for top_k in top_ks:
            sets = {
                item["split_id"]: set(item["ranking"][:top_k])
                for item in rankings
            }
            for a, b in combinations(sorted(sets), 2):
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                jaccard_rows.append(
                    {
                        "strategy": strategy,
                        "top_k": top_k,
                        "split_a": a,
                        "split_b": b,
                        "jaccard": inter / max(union, 1),
                    }
                )
        for item in rankings:
            for rank, idx in enumerate(item["ranking"], start=1):
                rank_rows.append(
                    {
                        "split_id": item["split_id"],
                        "strategy": strategy,
                        "rank": rank,
                        "node_index": idx,
                        "node_id": reference.node_ids[idx],
                        "node_name": reference.node_names[idx],
                        "region": reference.regions[idx],
                        "hemisphere": reference.hemispheres[idx],
                    }
                )
    node_rows.sort(key=lambda row: (row["strategy"], row["mean_rank"]))
    return node_rows, jaccard_rows, rank_rows


def strength_control_rows(discovery_summary: list[dict]) -> list[dict]:
    by_key = {
        (row["selection_strategy"], int(row["anchor_count"])): row
        for row in discovery_summary
    }
    strategies = [
        "unsup_strength",
        "qrc_resid_influence",
        "qrc_resid_entropy",
        "qrc_v3_residual_composite",
        "qrc_v3_greedy_residual",
        "unsup_qrc_influence",
        "unsup_composite",
        "supervised_val_ranked",
    ]
    counts = sorted({int(row["anchor_count"]) for row in discovery_summary})
    rows = []
    for count in counts:
        strength = by_key.get(("unsup_strength", count))
        random = by_key.get(("random_balanced", count))
        for strategy in strategies:
            row = by_key.get((strategy, count))
            if not row:
                continue
            rmse = clean_float(row["normalized_rmse_mean"])
            strength_rmse = clean_float(strength["normalized_rmse_mean"]) if strength else float("nan")
            random_rmse = clean_float(random["normalized_rmse_mean"]) if random else float("nan")
            rows.append(
                {
                    "selection_strategy": strategy,
                    "anchor_count": count,
                    "normalized_rmse_mean": rmse,
                    "distance_corr_mean": clean_float(row["distance_corr_mean"]),
                    "delta_rmse_vs_strength": rmse - strength_rmse,
                    "delta_rmse_vs_random": rmse - random_rmse,
                }
            )
    return rows


def write_summary(
    output_dir: Path,
    docs_path: Path | None,
    subject_count: int,
    split_count: int,
    discovery_summary: list[dict],
    fixed_summary: list[dict],
    hard_node_summary: list[dict],
    region_summary: list[dict],
    stability_summary: list[dict],
    jaccard_summary: list[dict],
    strength_control_summary: list[dict],
    dataset_label: str,
) -> None:
    focus_strategies = {
        "supervised_val_ranked",
        "unsup_composite",
        "unsup_qrc_influence",
        "unsup_response_entropy",
        "unsup_strength",
        "qrc_resid_influence",
        "qrc_resid_entropy",
        "qrc_v3_residual_composite",
        "qrc_v3_greedy_residual",
        "random_balanced",
        "atlas_spatial_spread",
    }
    discovery_focus = [
        row
        for row in sorted(discovery_summary, key=lambda row: (row["selection_strategy"], int(row["anchor_count"])))
        if row["selection_strategy"] in focus_strategies
    ]
    fixed_focus = [
        row
        for row in sorted(fixed_summary, key=lambda row: (row["selection_strategy"], int(row["anchor_count"])))
        if row["selection_strategy"]
        in {
            "fixed_supervised_val_ranked",
            "fixed_unsup_composite",
            "fixed_unsup_qrc_influence",
            "fixed_unsup_strength",
            "fixed_qrc_resid_influence",
            "fixed_qrc_resid_entropy",
            "fixed_qrc_v3_residual_composite",
            "fixed_qrc_v3_greedy_residual",
            "fixed_random_balanced",
            "fixed_atlas_spatial_spread",
        }
    ]
    hard_focus = [
        row
        for row in sorted(hard_node_summary, key=lambda row: (row["selection_strategy"], int(row["anchor_count"]), int(row.get("hard_node_rank", 9999))))
        if row["selection_strategy"] == "qrc_v3_greedy_residual" and int(row["anchor_count"]) == 16 and int(row.get("hard_node_rank", 9999)) <= 15
    ]
    region_focus = sorted(
        [
            row
            for row in region_summary
            if row["selection_strategy"] == "qrc_v3_greedy_residual" and int(row["anchor_count"]) == 16
        ],
        key=lambda row: -clean_float(row["normalized_error_mean"]),
    )[:12]
    stability_focus = []
    for strategy in [
        "supervised_val_ranked",
        "unsup_strength",
        "qrc_resid_influence",
        "qrc_resid_entropy",
        "qrc_v3_greedy_residual",
        "unsup_composite",
    ]:
        strategy_rows = [
            row
            for row in stability_summary
            if row["strategy"] == strategy and int(row.get("top_16_frequency", 0)) > 0
        ]
        strategy_rows.sort(key=lambda row: (-int(row.get("top_16_frequency", 0)), clean_float(row["mean_rank"])))
        stability_focus.extend(strategy_rows[:8])
    jaccard_focus = sorted(jaccard_summary, key=lambda row: (row["strategy"], int(row["top_k"])))
    strength_focus = [
        row
        for row in strength_control_summary
        if row["selection_strategy"]
        in {
            "unsup_strength",
            "qrc_resid_influence",
            "qrc_resid_entropy",
            "qrc_v3_residual_composite",
            "qrc_v3_greedy_residual",
            "unsup_qrc_influence",
            "unsup_composite",
        }
    ]

    text = f"""# QRC-v3 Source Discovery Analysis

Dataset: {dataset_label}

Subjects analyzed: {subject_count}

Splits: {split_count}. Each split ranked sources without using the held-out test
subjects, then evaluated nested source sets on the test subjects.

## 1. Held-Out Nested Source Discovery

`supervised_val_ranked` uses only train/validation coordinate recovery to rank
sources. The `unsup_*` rules use only graph/QRC descriptors from training
subjects, not coordinate labels.

{markdown_table(discovery_focus, ["selection_strategy", "anchor_count", "normalized_rmse_mean", "hemisphere_accuracy_mean", "distance_corr_mean", "x_corr_mean", "y_corr_mean", "z_corr_mean", "n"], 64)}

## 2. Strength-Controlled QRC Selectors

The residual strategies remove the linear effect of weighted strength from QRC
influence/entropy before ranking. The v3 greedy selector then adds a redundancy
penalty between source-response profiles and a soft hemisphere-balance penalty.
Negative `delta_rmse_vs_strength` means a selector beats ordinary strength hubs.

{markdown_table(strength_focus, ["selection_strategy", "anchor_count", "normalized_rmse_mean", "distance_corr_mean", "delta_rmse_vs_strength", "delta_rmse_vs_random"], 48)}

## 3. Fixed-Target Hard-Node Hierarchy

For each split, the hardest validation-ranked target nodes were held out as a
constant target set. This tests whether adding source regions improves
prediction on the same difficult nodes.

{markdown_table(fixed_focus, ["selection_strategy", "anchor_count", "target_count", "normalized_rmse_mean", "hemisphere_accuracy_mean", "distance_corr_mean", "n"], 64)}

## 4. Failure / Hard-Region Analysis

Hardest test nodes for the graph-only `qrc_v3_greedy_residual` rule with 16
sources:

{markdown_table(hard_focus, ["hard_node_rank", "node_name", "hemisphere", "region", "normalized_error_mean", "x_abs_error_mean", "y_abs_error_mean", "z_abs_error_mean"], 15)}

Hardest broad region/hemisphere groups:

{markdown_table(region_focus, ["region", "hemisphere", "normalized_error_mean", "x_abs_error_mean", "y_abs_error_mean", "z_abs_error_mean", "n"], 12)}

## 5. Source Stability Across Splits

Stable high-ranking sources across split-specific source rankings:

{markdown_table(stability_focus, ["strategy", "node_name", "hemisphere", "region", "mean_rank", "rank_std", "top_8_frequency", "top_16_frequency", "top_32_frequency"], 32)}

Mean pairwise Jaccard overlap of top-k source sets:

{markdown_table(jaccard_focus, ["strategy", "top_k", "jaccard_mean", "jaccard_std", "n"], 24)}

## Working Interpretation

The main test is whether a source hierarchy can be discovered without test
leakage. If the unsupervised QRC descriptor rules approach the validation-ranked
rule, then QRC-v2 is moving from a decoder into a source-discovery algorithm:
the connectome itself identifies useful relational anchors.
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")
    if docs_path is not None:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict QRC-v2 source discovery analyses.")
    parser.add_argument("--input-dir", type=Path, default=Path("research/data/brain_graph_hcp_86_nodes/graphml"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/outputs/qrc_v2_source_discovery"))
    parser.add_argument("--docs-path", type=Path, default=Path("docs/qrc_v2_source_discovery_2026-06-18.md"))
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--subjects", type=int, default=1064)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--anchor-counts", type=parse_ints, default=parse_ints("2,4,8,16,32"))
    parser.add_argument("--node-error-counts", type=parse_ints, default=parse_ints("16,32"))
    parser.add_argument("--fixed-target-count", type=int, default=16)
    parser.add_argument("--random-repeats", type=int, default=4)
    parser.add_argument("--times", type=parse_times, default=parse_times("0.25,0.5,1,2,4,8"))
    parser.add_argument("--weight-transform", choices=["raw", "log1p", "binary"], default="log1p")
    parser.add_argument("--ridge-alpha", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=59)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_rng = np.random.default_rng(args.seed)
    source_rng = np.random.default_rng(args.seed + 1)
    files = sorted(args.input_dir.glob("*.graphml"))[: args.subjects]
    if not files:
        raise SystemExit(f"No GraphML files found in {args.input_dir}")

    print(f"Caching {len(files)} subjects")
    records = [cache_subject(path, args.times, args.weight_transform) for path in files]
    reference = records[0].connectome

    validation_metric_rows = []
    validation_summary_rows = []
    descriptor_rows = []
    discovery_metric_rows = []
    discovery_anchor_rows = []
    fixed_metric_rows = []
    fixed_anchor_rows = []
    fixed_target_rows = []
    node_error_rows = []
    split_rankings = []

    for split_id in range(1, args.splits + 1):
        train_records, val_records, test_records = split_records_three_way(
            records,
            args.train_fraction,
            args.val_fraction,
            split_rng,
        )
        print(
            f"Split {split_id}/{args.splits}: "
            f"train={len(train_records)} val={len(val_records)} test={len(test_records)}"
        )

        descriptors = node_descriptors(train_records)
        rankings, split_descriptor_rows = descriptor_rankings(descriptors, split_id, train_records)
        descriptor_rows.extend(split_descriptor_rows)

        val_metrics, val_summary = rank_single_sources_on_validation(
            train_records,
            val_records,
            descriptors,
            split_id=split_id,
            alpha=args.ridge_alpha,
        )
        validation_metric_rows.extend(val_metrics)
        validation_summary_rows.extend(val_summary)

        discovered_metrics, anchor_rows, strategy_rankings = evaluate_discovered_sources(
            train_records,
            val_records,
            test_records,
            val_summary,
            rankings,
            anchor_counts=args.anchor_counts,
            random_repeats=args.random_repeats,
            split_id=split_id,
            alpha=args.ridge_alpha,
            rng=source_rng,
        )
        discovery_metric_rows.extend(discovered_metrics)
        discovery_anchor_rows.extend(anchor_rows)

        split_rankings.append(
            {
                "split_id": split_id,
                "strategy": "supervised_val_ranked",
                "ranking": strategy_rankings["supervised_val_ranked"],
            }
        )
        for strategy in (
            "unsup_composite",
            "unsup_qrc_influence",
            "unsup_response_entropy",
            "unsup_strength",
            "qrc_resid_influence",
            "qrc_resid_entropy",
            "qrc_v3_residual_composite",
            "qrc_v3_greedy_residual",
        ):
            split_rankings.append({"split_id": split_id, "strategy": strategy, "ranking": rankings[strategy]})

        fixed_metrics, fixed_anchors, split_targets = evaluate_fixed_targets(
            train_records,
            val_records,
            test_records,
            val_summary,
            rankings,
            anchor_counts=args.anchor_counts,
            random_repeats=args.random_repeats,
            target_count=args.fixed_target_count,
            split_id=split_id,
            alpha=args.ridge_alpha,
            rng=source_rng,
        )
        fixed_metric_rows.extend(fixed_metrics)
        fixed_anchor_rows.extend(fixed_anchors)
        fixed_target_rows.extend(split_targets)

        node_error_rows.extend(
            collect_node_errors(
                train_records,
                val_records,
                test_records,
                val_summary,
                rankings,
                counts=args.node_error_counts,
                split_id=split_id,
                alpha=args.ridge_alpha,
            )
        )

    write_csv(
        args.output_dir / "validation_single_source_metrics.csv",
        validation_metric_rows,
        [
            "split_id",
            "source_index",
            "source_node_id",
            "source_node_name",
            "source_region",
            "source_hemisphere",
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
        ],
    )
    write_csv(
        args.output_dir / "validation_source_rankings.csv",
        validation_summary_rows,
        [
            "split_id",
            "val_single_source_rank",
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "val_single_qrc_rmse_mean",
            "val_single_qrc_hemi_mean",
            "val_single_qrc_distance_corr_mean",
            "mean_qrc_influence",
            "mean_weighted_strength",
            "mean_response_entropy",
            "mean_cross_hemisphere_response",
        ],
    )
    write_csv(
        args.output_dir / "unsupervised_source_descriptors.csv",
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
    write_csv(
        args.output_dir / "discovery_test_metrics.csv",
        discovery_metric_rows,
        [
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
        args.output_dir / "discovery_anchor_sets.csv",
        discovery_anchor_rows,
        [
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

    discovery_summary = mean_by_group(
        discovery_metric_rows,
        ["selection_strategy", "anchor_count"],
        ["normalized_rmse", "hemisphere_accuracy", "distance_corr", "x_corr", "y_corr", "z_corr"],
    )
    strength_summary = strength_control_rows(discovery_summary)
    write_csv(
        args.output_dir / "discovery_test_summary.csv",
        discovery_summary,
        [
            "selection_strategy",
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
        args.output_dir / "strength_control_summary.csv",
        strength_summary,
        [
            "selection_strategy",
            "anchor_count",
            "normalized_rmse_mean",
            "distance_corr_mean",
            "delta_rmse_vs_strength",
            "delta_rmse_vs_random",
        ],
    )

    write_csv(
        args.output_dir / "fixed_target_metrics.csv",
        fixed_metric_rows,
        [
            "split_id",
            "sample_id",
            "selection_strategy",
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
        args.output_dir / "fixed_target_anchor_sets.csv",
        fixed_anchor_rows,
        [
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
        args.output_dir / "fixed_target_nodes.csv",
        fixed_target_rows,
        ["split_id", "target_index", "node_id", "node_name", "region", "hemisphere"],
    )
    fixed_summary = mean_by_group(
        fixed_metric_rows,
        ["selection_strategy", "anchor_count", "target_count"],
        ["normalized_rmse", "hemisphere_accuracy", "distance_corr", "x_corr", "y_corr", "z_corr"],
    )
    write_csv(
        args.output_dir / "fixed_target_summary.csv",
        fixed_summary,
        [
            "selection_strategy",
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

    node_error_summary, region_error_summary = summarize_node_errors(node_error_rows)
    write_csv(
        args.output_dir / "node_error_metrics.csv",
        node_error_rows,
        [
            "split_id",
            "selection_strategy",
            "anchor_count",
            "subject_id",
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "normalized_error",
            "x_abs_error",
            "y_abs_error",
            "z_abs_error",
        ],
    )
    write_csv(
        args.output_dir / "node_error_summary.csv",
        node_error_summary,
        [
            "selection_strategy",
            "anchor_count",
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "normalized_error_mean",
            "normalized_error_std",
            "x_abs_error_mean",
            "x_abs_error_std",
            "y_abs_error_mean",
            "y_abs_error_std",
            "z_abs_error_mean",
            "z_abs_error_std",
            "n",
            "hard_node_rank",
        ],
    )
    write_csv(
        args.output_dir / "region_error_summary.csv",
        region_error_summary,
        [
            "selection_strategy",
            "anchor_count",
            "region",
            "hemisphere",
            "normalized_error_mean",
            "normalized_error_std",
            "x_abs_error_mean",
            "x_abs_error_std",
            "y_abs_error_mean",
            "y_abs_error_std",
            "z_abs_error_mean",
            "z_abs_error_std",
            "n",
        ],
    )

    stability_node_rows, stability_jaccard_rows, stability_rank_rows = stability_rows(
        split_rankings,
        reference,
        top_ks=[8, 16, 32],
    )
    jaccard_summary = mean_by_group(stability_jaccard_rows, ["strategy", "top_k"], ["jaccard"])
    write_csv(
        args.output_dir / "source_stability_summary.csv",
        stability_node_rows,
        [
            "strategy",
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "mean_rank",
            "rank_std",
            "top_8_frequency",
            "top_16_frequency",
            "top_32_frequency",
        ],
    )
    write_csv(
        args.output_dir / "source_stability_rankings_by_split.csv",
        stability_rank_rows,
        ["split_id", "strategy", "rank", "node_index", "node_id", "node_name", "region", "hemisphere"],
    )
    write_csv(
        args.output_dir / "source_stability_jaccard.csv",
        stability_jaccard_rows,
        ["strategy", "top_k", "split_a", "split_b", "jaccard"],
    )
    write_csv(
        args.output_dir / "source_stability_jaccard_summary.csv",
        jaccard_summary,
        ["strategy", "top_k", "jaccard_mean", "jaccard_std", "n"],
    )

    write_line_svg(
        args.output_dir / "discovery_rmse.svg",
        [row | {"anchor_strategy": row["selection_strategy"]} for row in discovery_summary],
        "Held-Out Source Discovery",
        ["supervised_val_ranked", "qrc_v3_greedy_residual", "qrc_resid_influence", "unsup_strength", "random_balanced"],
    )
    write_line_svg(
        args.output_dir / "fixed_target_rmse.svg",
        [row | {"anchor_strategy": row["selection_strategy"]} for row in fixed_summary],
        "Fixed-Target Hard-Node Hierarchy",
        [
            "fixed_supervised_val_ranked",
            "fixed_qrc_v3_greedy_residual",
            "fixed_qrc_resid_influence",
            "fixed_unsup_strength",
            "fixed_random_balanced",
        ],
    )
    hard_bar_rows = [
        row | {"node_label": f"{row['node_name']} ({row['hemisphere']})"}
        for row in sorted(
            [
                row
                for row in node_error_summary
                if row["selection_strategy"] == "qrc_v3_greedy_residual"
                and int(row["anchor_count"]) == 16
                and int(row.get("hard_node_rank", 9999)) <= 15
            ],
            key=lambda row: int(row["hard_node_rank"]),
        )
    ]
    write_bar_svg(
        args.output_dir / "hard_nodes_qrc_v3_greedy_residual_16.svg",
        hard_bar_rows,
        "Hardest Nodes, QRC-v3 Greedy Residual 16 Sources",
        "node_label",
        "normalized_error_mean",
        limit=15,
    )
    stable_bar_rows = [
        row | {"node_label": f"{row['node_name']} ({row['hemisphere']})"}
        for row in sorted(
            [row for row in stability_node_rows if row["strategy"] == "supervised_val_ranked"],
            key=lambda row: (-int(row["top_16_frequency"]), clean_float(row["mean_rank"])),
        )[:15]
    ]
    write_bar_svg(
        args.output_dir / "source_stability_supervised_top16.svg",
        stable_bar_rows,
        "Stable Validation-Discovered Sources",
        "node_label",
        "top_16_frequency",
        limit=15,
    )

    write_summary(
        args.output_dir,
        args.docs_path,
        subject_count=len(records),
        split_count=args.splits,
        discovery_summary=discovery_summary,
        fixed_summary=fixed_summary,
        hard_node_summary=node_error_summary,
        region_summary=region_error_summary,
        stability_summary=stability_node_rows,
        jaccard_summary=jaccard_summary,
        strength_control_summary=strength_summary,
        dataset_label=args.dataset_label or f"BrainGraph HCP {len(reference.node_ids)}-node structural connectomes",
    )
    print(f"Wrote source discovery outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
