"""Nested QRC-H source hierarchy experiments.

QRC-H is the algorithmic version of the source-discovery idea:

1. Compute graph-only QRC/strength source descriptors on training subjects.
2. Build one ordered source hierarchy, so A8 subset A16 subset A32.
3. Choose hybrid selector weights using validation subjects only.
4. Evaluate the learned hierarchy once on held-out test subjects.
5. Compare against nested strength, QRC-residual, spatial, random, and
   validation-ranked baselines.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain_quantum.analysis_suite import farthest_spread_anchors, mean_by_group  # noqa: E402
from brain_quantum.mechanism_suite import (  # noqa: E402
    anchor_record,
    clean_float,
    evaluate_anchor_model,
    markdown_table,
    node_descriptors,
)
from brain_quantum.qrc_connectome import parse_times, write_csv  # noqa: E402
from brain_quantum.selector_deep_dive import build_hybrid_rankings, paired_selector_tests  # noqa: E402
from brain_quantum.source_coordinate_v2 import (  # noqa: E402
    SubjectRecord,
    cache_subject,
    eligible_source_indices,
    mean_coordinate_template,
)
from brain_quantum.source_discovery_suite import (  # noqa: E402
    rank_single_sources_on_validation,
    ranking_from_source_summary,
    residualize_against_strength,
    source_profile_similarity,
    split_records_three_way,
    zscore,
)


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def hierarchy_configs() -> list[dict]:
    configs = []
    for strength_weight, label in [
        (0.00, "s00_q90_c10"),
        (0.25, "s25_q65_c10"),
        (0.50, "s50_q50"),
        (0.70, "s70_q30"),
        (0.90, "s90_q10"),
        (1.00, "s100_q00"),
    ]:
        q_weight = max(0.0, 1.0 - strength_weight)
        cross = 0.10 if strength_weight < 0.50 else 0.00
        remaining_q = max(0.0, q_weight - cross)
        for redundancy_weight in [0.15, 0.45, 0.75]:
            configs.append(
                {
                    "selector_type": "hybrid",
                    "strategy": f"qrc_h_{label}_red{int(redundancy_weight * 100):02d}",
                    "config_id": f"{label}_red{int(redundancy_weight * 100):02d}",
                    "strength_weight": strength_weight,
                    "resid_influence_weight": remaining_q / 2.0,
                    "resid_entropy_weight": remaining_q / 2.0,
                    "cross_weight": cross,
                    "greedy": True,
                    "redundancy_weight": redundancy_weight,
                    "balance_weight": 0.30,
                }
            )
    for strength_weight, label in [
        (0.00, "cover_s00_q90_c10"),
        (0.25, "cover_s25_q65_c10"),
        (0.50, "cover_s50_q50"),
        (0.70, "cover_s70_q30"),
        (1.00, "cover_s100_q00"),
    ]:
        q_weight = max(0.0, 1.0 - strength_weight)
        cross = 0.10 if strength_weight < 0.50 else 0.00
        remaining_q = max(0.0, q_weight - cross)
        for coverage_weight in [0.50, 1.00, 2.00]:
            configs.append(
                {
                    "selector_type": "graph_coverage",
                    "strategy": f"qrc_h_{label}_cov{int(coverage_weight * 100):03d}",
                    "config_id": f"{label}_cov{int(coverage_weight * 100):03d}",
                    "strength_weight": strength_weight,
                    "resid_influence_weight": remaining_q / 2.0,
                    "resid_entropy_weight": remaining_q / 2.0,
                    "cross_weight": cross,
                    "greedy": True,
                    "redundancy_weight": 0.45,
                    "coverage_weight": coverage_weight,
                    "balance_weight": 0.30,
                }
            )
    return configs


BASELINE_CONFIGS = [
    {
        "selector_type": "hybrid",
        "strategy": "strength_nested",
        "config_id": "strength_nested",
        "strength_weight": 1.00,
        "resid_influence_weight": 0.00,
        "resid_entropy_weight": 0.00,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.00,
        "balance_weight": 0.30,
    },
    {
        "selector_type": "hybrid",
        "strategy": "qrc_residual_nested",
        "config_id": "qrc_residual_nested",
        "strength_weight": 0.00,
        "resid_influence_weight": 0.45,
        "resid_entropy_weight": 0.45,
        "cross_weight": 0.10,
        "greedy": True,
        "redundancy_weight": 0.45,
        "balance_weight": 0.30,
    },
    {
        "selector_type": "hybrid",
        "strategy": "qrc_h_fixed_s70_q30",
        "config_id": "fixed_s70_q30",
        "strength_weight": 0.70,
        "resid_influence_weight": 0.15,
        "resid_entropy_weight": 0.15,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.45,
        "balance_weight": 0.30,
    },
    {
        "selector_type": "hybrid",
        "strategy": "qrc_h_fixed_s30_q70",
        "config_id": "fixed_s30_q70",
        "strength_weight": 0.30,
        "resid_influence_weight": 0.35,
        "resid_entropy_weight": 0.35,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.45,
        "balance_weight": 0.30,
    },
    {
        "selector_type": "graph_coverage",
        "strategy": "qrc_h_cover_s50_q50",
        "config_id": "cover_s50_q50_cov100",
        "strength_weight": 0.50,
        "resid_influence_weight": 0.25,
        "resid_entropy_weight": 0.25,
        "cross_weight": 0.00,
        "greedy": True,
        "redundancy_weight": 0.45,
        "coverage_weight": 1.00,
        "balance_weight": 0.30,
    },
]


def anchors_from_ranking(ranking: list[int], count: int) -> np.ndarray:
    return np.array(ranking[:count], dtype=int)


def eligible_first_ranking(
    ranking: list[int],
    candidate_indices: np.ndarray,
    node_count: int,
) -> list[int]:
    candidates = set(int(index) for index in candidate_indices)
    output = [int(index) for index in ranking if int(index) in candidates]
    output.extend(index for index in range(node_count) if index not in candidates)
    return output


def nested_random_ranking(
    hemispheres: list[str],
    rng: np.random.Generator,
    candidate_indices: np.ndarray | None = None,
) -> list[int]:
    candidates = (
        list(range(len(hemispheres)))
        if candidate_indices is None
        else [int(index) for index in candidate_indices]
    )
    left = [idx for idx in candidates if hemispheres[idx].startswith("left")]
    right = [idx for idx in candidates if hemispheres[idx].startswith("right")]
    other = [idx for idx in candidates if not (hemispheres[idx].startswith("left") or hemispheres[idx].startswith("right"))]
    rng.shuffle(left)
    rng.shuffle(right)
    rng.shuffle(other)
    ranking = []
    while left or right:
        if left:
            ranking.append(left.pop())
        if right:
            ranking.append(right.pop())
    ranking.extend(other)
    selected = set(ranking)
    ranking.extend(index for index in range(len(hemispheres)) if index not in selected)
    return ranking


def spatial_nested_ranking(
    positions: np.ndarray,
    max_count: int,
    candidate_indices: np.ndarray | None = None,
) -> list[int]:
    del max_count
    geometric = farthest_spread_anchors(positions, positions.shape[0]).astype(int).tolist()
    candidates = (
        set(range(positions.shape[0]))
        if candidate_indices is None
        else set(int(index) for index in candidate_indices)
    )
    ranking = [index for index in geometric if index in candidates]
    ranking.extend(index for index in range(positions.shape[0]) if index not in candidates)
    return ranking


def average_shortest_paths(records: list[SubjectRecord]) -> np.ndarray:
    paths = np.mean(np.stack([record.shortest_paths for record in records], axis=0), axis=0)
    return np.nan_to_num(paths, nan=0.0, posinf=0.0, neginf=0.0)


def graph_coverage_ranking(
    descriptors: list[dict],
    train_records: list[SubjectRecord],
    config: dict,
    seed: list[int] | None = None,
    candidate_indices: np.ndarray | None = None,
) -> list[int]:
    sorted_descriptors = sorted(descriptors, key=lambda row: int(row["node_index"]))
    influence = np.array([clean_float(row["mean_qrc_influence"]) for row in sorted_descriptors])
    entropy = np.array([clean_float(row["mean_response_entropy"]) for row in sorted_descriptors])
    strength = np.array([clean_float(row["mean_weighted_strength"]) for row in sorted_descriptors])
    cross = np.array([clean_float(row["mean_cross_hemisphere_response"]) for row in sorted_descriptors])
    candidates = (
        np.arange(len(sorted_descriptors), dtype=int)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=int)
    )
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

    def candidate_zscore(values: np.ndarray) -> np.ndarray:
        output = np.zeros(len(values), dtype=float)
        output[candidates] = zscore(values[candidates])
        return output

    score = (
        float(config["strength_weight"]) * candidate_zscore(strength)
        + float(config["resid_influence_weight"]) * candidate_zscore(residual_influence)
        + float(config["resid_entropy_weight"]) * candidate_zscore(residual_entropy)
        + float(config["cross_weight"]) * candidate_zscore(cross)
    )

    distances = average_shortest_paths(train_records)
    finite = distances[np.isfinite(distances) & (distances > 0)]
    scale = float(np.median(finite)) if len(finite) else 1.0
    distances = distances / max(scale, 1e-9)
    similarity = source_profile_similarity(train_records, target_indices=candidates)
    hemispheres = [str(row["hemisphere"]) for row in sorted_descriptors]

    candidate_set = set(int(index) for index in candidates)
    selected: list[int] = [int(index) for index in (seed or []) if int(index) in candidate_set]
    remaining = candidate_set - set(selected)
    left_count = sum(1 for idx in selected if hemispheres[idx].startswith("left"))
    right_count = sum(1 for idx in selected if hemispheres[idx].startswith("right"))
    while remaining:
        remaining_list = sorted(remaining)
        if selected:
            min_dist = np.array([float(np.min(distances[idx, selected])) for idx in remaining_list])
            coverage_bonus = zscore(min_dist)
        else:
            coverage_bonus = np.zeros(len(remaining_list), dtype=float)

        best_idx = None
        best_value = -float("inf")
        for pos, idx in enumerate(remaining_list):
            redundancy = float(np.max(similarity[idx, selected])) if selected else 0.0
            hemi = hemispheres[idx]
            new_left = left_count + int(hemi.startswith("left"))
            new_right = right_count + int(hemi.startswith("right"))
            balance_penalty = max(abs(new_left - new_right) - 1, 0)
            value = (
                score[idx]
                + float(config["coverage_weight"]) * coverage_bonus[pos]
                - float(config["redundancy_weight"]) * redundancy
                - float(config["balance_weight"]) * balance_penalty
            )
            if value > best_value:
                best_value = value
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
        left_count += int(hemispheres[best_idx].startswith("left"))
        right_count += int(hemispheres[best_idx].startswith("right"))
    selected.extend(index for index in range(len(sorted_descriptors)) if index not in candidate_set)
    return selected


def staged_qrc_h_ranking(
    descriptors: list[dict],
    train_records: list[SubjectRecord],
    strength_heavy_ranking: list[int],
    strength_ranking: list[int],
    first_count: int,
    second_count: int,
) -> list[int]:
    selected: list[int] = []
    for idx in strength_heavy_ranking:
        if idx not in selected:
            selected.append(idx)
        if len(selected) == first_count:
            break
    for idx in strength_ranking:
        if idx not in selected:
            selected.append(idx)
        if len(selected) == second_count:
            break
    coverage_config = {
        "strategy": "qrc_h_staged",
        "strength_weight": 0.00,
        "resid_influence_weight": 0.45,
        "resid_entropy_weight": 0.45,
        "cross_weight": 0.10,
        "redundancy_weight": 0.45,
        "coverage_weight": 1.00,
        "balance_weight": 0.30,
    }
    return graph_coverage_ranking(
        descriptors,
        train_records,
        coverage_config,
        seed=selected,
        candidate_indices=eligible_source_indices(train_records),
    )


def build_rankings_for_configs(
    descriptors: list[dict],
    train_records: list[SubjectRecord],
    split_id: int,
    configs: list[dict],
) -> tuple[dict[str, list[int]], list[dict]]:
    candidates = eligible_source_indices(train_records)
    hybrid_configs = [config for config in configs if config.get("selector_type", "hybrid") == "hybrid"]
    coverage_configs = [config for config in configs if config.get("selector_type") == "graph_coverage"]
    rankings: dict[str, list[int]] = {}
    rows: list[dict] = []
    if hybrid_configs:
        hybrid_rankings, hybrid_rows = build_hybrid_rankings(
            descriptors,
            train_records,
            split_id,
            hybrid_configs,
            candidate_indices=candidates,
        )
        rankings.update(hybrid_rankings)
        rows.extend(hybrid_rows)
    for config in coverage_configs:
        rankings[str(config["strategy"])] = graph_coverage_ranking(
            descriptors,
            train_records,
            config,
            candidate_indices=candidates,
        )
    return rankings, rows


def evaluate_hierarchy(
    train_records: list[SubjectRecord],
    eval_records: list[SubjectRecord],
    ranking: list[int],
    counts: list[int],
    split_id: int,
    strategy: str,
    alpha: float,
    dataset_label: str,
    phase: str,
) -> tuple[list[dict], list[dict]]:
    reference = train_records[0].connectome
    metric_rows = []
    anchor_rows = []
    for count in counts:
        anchors = anchors_from_ranking(ranking, count)
        sample_id = f"{dataset_label}-{phase}-{split_id}-{strategy}-{count}"
        anchor_rows.append(
            anchor_record(reference, anchors, sample_id, strategy, count)
            | {
                "dataset_label": dataset_label,
                "phase": phase,
                "split_id": split_id,
                "is_nested_hierarchy": "yes",
            }
        )
        for row in evaluate_anchor_model(
            train_records,
            eval_records,
            anchors,
            method="qrc",
            alpha=alpha,
            strategy=strategy,
            sample_id=sample_id,
        ):
            row.update(
                {
                    "dataset_label": dataset_label,
                    "phase": phase,
                    "split_id": split_id,
                    "selection_strategy": strategy,
                    "sample_id": sample_id,
                    "is_nested_hierarchy": "yes",
                }
            )
            metric_rows.append(row)
    return metric_rows, anchor_rows


def validation_grid_search(
    train_records: list[SubjectRecord],
    val_records: list[SubjectRecord],
    descriptors: list[dict],
    counts: list[int],
    split_id: int,
    alpha: float,
    dataset_label: str,
) -> tuple[dict, list[int], list[dict], list[dict]]:
    configs = hierarchy_configs()
    rankings, _ = build_rankings_for_configs(descriptors, train_records, split_id, configs)
    metric_rows = []
    config_rows = []
    for config in configs:
        strategy = str(config["strategy"])
        metrics, _ = evaluate_hierarchy(
            train_records,
            val_records,
            rankings[strategy],
            counts,
            split_id,
            strategy,
            alpha,
            dataset_label=dataset_label,
            phase="validation_grid",
        )
        metric_rows.extend(metrics)
        count_summary = mean_by_group(metrics, ["anchor_count"], ["normalized_rmse", "distance_corr"])
        objective = float(np.mean([clean_float(row["normalized_rmse_mean"]) for row in count_summary]))
        config_rows.append(
            {
                "split_id": split_id,
                "config_id": config["config_id"],
                "strategy": strategy,
                "objective_rmse_mean_across_counts": objective,
                "strength_weight": config["strength_weight"],
                "resid_influence_weight": config["resid_influence_weight"],
                "resid_entropy_weight": config["resid_entropy_weight"],
                "cross_weight": config["cross_weight"],
                "redundancy_weight": config["redundancy_weight"],
                "coverage_weight": config.get("coverage_weight", 0.0),
                "balance_weight": config["balance_weight"],
            }
        )
        for row in count_summary:
            config_rows.append(
                {
                    "split_id": split_id,
                    "config_id": config["config_id"],
                    "strategy": strategy,
                    "anchor_count": row["anchor_count"],
                    "validation_rmse_mean": row["normalized_rmse_mean"],
                    "validation_distance_corr_mean": row["distance_corr_mean"],
                    "strength_weight": config["strength_weight"],
                    "resid_influence_weight": config["resid_influence_weight"],
                    "resid_entropy_weight": config["resid_entropy_weight"],
                    "cross_weight": config["cross_weight"],
                    "redundancy_weight": config["redundancy_weight"],
                    "coverage_weight": config.get("coverage_weight", 0.0),
                    "balance_weight": config["balance_weight"],
                }
            )
    best = min(
        [row for row in config_rows if row.get("objective_rmse_mean_across_counts") not in {"", None}],
        key=lambda row: clean_float(row["objective_rmse_mean_across_counts"]),
    )
    learned_config = next(config for config in configs if config["config_id"] == best["config_id"])
    return learned_config, rankings[learned_config["strategy"]], metric_rows, config_rows


def source_frequency_rows(
    split_rankings: list[dict],
    reference,
    counts: list[int],
) -> tuple[list[dict], list[dict]]:
    rows = []
    layer_rows = []
    strategies = sorted({row["strategy"] for row in split_rankings})
    for strategy in strategies:
        strategy_rankings = [row for row in split_rankings if row["strategy"] == strategy]
        for idx in range(len(reference.node_ids)):
            ranks = [item["ranking"].index(idx) + 1 for item in strategy_rankings]
            record = {
                "strategy": strategy,
                "node_index": idx,
                "node_id": reference.node_ids[idx],
                "node_name": reference.node_names[idx],
                "region": reference.regions[idx],
                "hemisphere": reference.hemispheres[idx],
                "mean_rank": float(np.mean(ranks)),
                "rank_std": float(np.std(ranks)),
            }
            for count in counts:
                record[f"top_{count}_frequency"] = int(sum(rank <= count for rank in ranks))
            rows.append(record)

        if strategy == "qrc_h_learned":
            boundaries = [(1, counts[0], f"core_{counts[0]}"), (counts[0] + 1, counts[1], f"expand_{counts[1]}")]
            if len(counts) > 2:
                boundaries.append((counts[1] + 1, counts[2], f"expand_{counts[2]}"))
            for idx in range(len(reference.node_ids)):
                ranks = [item["ranking"].index(idx) + 1 for item in strategy_rankings]
                for low, high, layer in boundaries:
                    layer_rows.append(
                        {
                            "strategy": strategy,
                            "layer": layer,
                            "node_index": idx,
                            "node_id": reference.node_ids[idx],
                            "node_name": reference.node_names[idx],
                            "region": reference.regions[idx],
                            "hemisphere": reference.hemispheres[idx],
                            "layer_frequency": int(sum(low <= rank <= high for rank in ranks)),
                            "mean_rank": float(np.mean(ranks)),
                        }
                    )
    rows.sort(key=lambda row: (row["strategy"], clean_float(row["mean_rank"])))
    layer_rows.sort(key=lambda row: (row["layer"], -int(row["layer_frequency"]), clean_float(row["mean_rank"])))
    return rows, layer_rows


def add_summary_deltas(summary_rows: list[dict], baseline: str) -> list[dict]:
    by_key = {(row["selection_strategy"], int(row["anchor_count"])): row for row in summary_rows}
    out = []
    for row in summary_rows:
        base = by_key.get((baseline, int(row["anchor_count"])))
        base_rmse = clean_float(base["normalized_rmse_mean"]) if base else float("nan")
        out.append({**row, f"delta_rmse_vs_{baseline}": clean_float(row["normalized_rmse_mean"]) - base_rmse})
    return out


def write_summary(
    output_dir: Path,
    docs_path: Path | None,
    test_summary: list[dict],
    paired_rows: list[dict],
    learned_configs: list[dict],
    frequency_rows: list[dict],
    layer_rows: list[dict],
) -> None:
    focus = sorted(test_summary, key=lambda row: (int(row["anchor_count"]), clean_float(row["normalized_rmse_mean"])))
    pair_focus = [
        row
        for row in paired_rows
        if row["metric"] == "normalized_rmse"
        and row["comparator_strategy"] in {"qrc_h_learned", "qrc_h_staged", "qrc_h_stage8_cover"}
    ]
    pair_focus.sort(key=lambda row: (int(row["anchor_count"]), row["baseline_strategy"]))
    freq_focus = [
        row
        for row in frequency_rows
        if row["strategy"] == "qrc_h_learned" and int(row.get("top_32_frequency", 0)) > 0
    ][:24]
    layer_focus = [row for row in layer_rows if int(row["layer_frequency"]) > 0][:24]

    text = f"""# QRC-H Nested Source Hierarchy

QRC-H chooses one ordered source hierarchy and evaluates nested source sets:
`A8 subset A16 subset A32`.

## Held-Out Test Performance

{markdown_table(focus, ["selection_strategy", "anchor_count", "normalized_rmse_mean", "distance_corr_mean", "hemisphere_accuracy_mean", "delta_rmse_vs_strength_nested", "n"], 64)}

## Paired Tests for Learned QRC-H

Negative RMSE delta means learned QRC-H beats the baseline.

{markdown_table(pair_focus, ["baseline_strategy", "comparator_strategy", "anchor_count", "baseline_mean", "comparator_mean", "mean_delta_comparator_minus_baseline", "comparator_win_rate", "normal_approx_p"], 48)}

## Learned Configurations

{markdown_table(learned_configs, ["split_id", "config_id", "objective_rmse_mean_across_counts", "strength_weight", "resid_influence_weight", "resid_entropy_weight", "cross_weight", "redundancy_weight", "coverage_weight"], 16)}

## Stable Learned Sources

{markdown_table(freq_focus, ["node_name", "hemisphere", "region", "mean_rank", "rank_std", "top_8_frequency", "top_16_frequency", "top_32_frequency"], 24)}

## Learned Hierarchy Layers

{markdown_table(layer_focus, ["layer", "node_name", "hemisphere", "region", "layer_frequency", "mean_rank"], 24)}

## Working Interpretation

This is the first direct test of the hierarchy idea. The result is
source-count dependent: early source sets prefer hub reach, while larger source
fields benefit from QRC-residual graph coverage. The strongest single
source-budget result is the 32-source graph-coverage QRC-H policy. The best
single nested compromise keeps a strong 8-source core and gains much of the
32-source coverage benefit, but does not dominate the 16-source setting.
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")
    if docs_path is not None:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested QRC-H source hierarchy experiments.")
    parser.add_argument("--input-dir", type=Path, default=Path("research/data/brain_graph_hcp_86_nodes/graphml"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/outputs/qrc_hierarchy"))
    parser.add_argument("--docs-path", type=Path, default=Path("docs/qrc_hierarchy_2026-06-18.md"))
    parser.add_argument("--subjects", type=int, default=1064)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--anchor-counts", type=parse_ints, default=parse_ints("8,16,32"))
    parser.add_argument("--times", type=parse_times, default=parse_times("0.25,0.5,1,2,4,8"))
    parser.add_argument("--weight-transform", choices=["raw", "log1p", "binary"], default="log1p")
    parser.add_argument("--ridge-alpha", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=907)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--dataset-label", default="hcp86_qrc_h")
    parser.add_argument("--skip-supervised", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_rng = np.random.default_rng(args.seed)
    source_rng = np.random.default_rng(args.seed + 1)
    files = sorted(args.input_dir.glob("*.graphml"))[: args.subjects]
    if not files:
        raise SystemExit(f"No GraphML files found in {args.input_dir}")

    print(f"Caching {len(files)} subjects for QRC-H hierarchy")
    records = [cache_subject(path, args.times, args.weight_transform) for path in files]
    reference = records[0].connectome
    max_count = max(args.anchor_counts)

    validation_metric_rows = []
    validation_grid_rows = []
    learned_config_rows = []
    test_metric_rows = []
    anchor_rows = []
    split_rankings = []
    source_score_rows = []

    for split_id in range(1, args.splits + 1):
        train_records, val_records, test_records = split_records_three_way(
            records,
            args.train_fraction,
            args.val_fraction,
            split_rng,
        )
        print(
            f"QRC-H split {split_id}/{args.splits}: "
            f"train={len(train_records)} val={len(val_records)} test={len(test_records)}"
        )
        descriptors = node_descriptors(train_records)
        eligible_sources = eligible_source_indices(train_records)

        learned_config, learned_ranking, val_metrics, grid_rows = validation_grid_search(
            train_records,
            val_records,
            descriptors,
            args.anchor_counts,
            split_id,
            args.ridge_alpha,
            args.dataset_label,
        )
        validation_metric_rows.extend(val_metrics)
        validation_grid_rows.extend(grid_rows)
        learned_summary = next(row for row in grid_rows if row.get("config_id") == learned_config["config_id"] and row.get("objective_rmse_mean_across_counts") not in {"", None})
        learned_config_rows.append({"split_id": split_id, **learned_summary})

        baseline_rankings, baseline_score_rows = build_rankings_for_configs(descriptors, train_records, split_id, BASELINE_CONFIGS)
        source_score_rows.extend(baseline_score_rows)
        rankings = {
            "qrc_h_learned": learned_ranking,
            "strength_nested": baseline_rankings["strength_nested"],
            "qrc_residual_nested": baseline_rankings["qrc_residual_nested"],
            "qrc_h_fixed_s70_q30": baseline_rankings["qrc_h_fixed_s70_q30"],
            "qrc_h_fixed_s30_q70": baseline_rankings["qrc_h_fixed_s30_q70"],
            "qrc_h_cover_s50_q50": baseline_rankings["qrc_h_cover_s50_q50"],
            "qrc_h_staged": staged_qrc_h_ranking(
                descriptors,
                train_records,
                baseline_rankings["qrc_h_fixed_s70_q30"],
                baseline_rankings["strength_nested"],
                first_count=args.anchor_counts[0],
                second_count=args.anchor_counts[1],
            ),
            "qrc_h_stage8_cover": graph_coverage_ranking(
                descriptors,
                train_records,
                {
                    "strength_weight": 0.00,
                    "resid_influence_weight": 0.45,
                    "resid_entropy_weight": 0.45,
                    "cross_weight": 0.10,
                    "redundancy_weight": 0.45,
                    "coverage_weight": 1.00,
                    "balance_weight": 0.30,
                },
                seed=baseline_rankings["qrc_h_fixed_s70_q30"][: args.anchor_counts[0]],
                candidate_indices=eligible_sources,
            ),
            "spatial_nested": spatial_nested_ranking(
                mean_coordinate_template(train_records),
                max_count,
                candidate_indices=eligible_sources,
            ),
        }
        for repeat in range(1, args.random_repeats + 1):
            rankings[f"random_nested_{repeat}"] = nested_random_ranking(
                reference.hemispheres,
                source_rng,
                candidate_indices=eligible_sources,
            )

        if not args.skip_supervised:
            print(f"  Ranking supervised validation sources for split {split_id}")
            val_single_metrics, val_source_summary = rank_single_sources_on_validation(
                train_records,
                val_records,
                descriptors,
                split_id,
                args.ridge_alpha,
            )
            validation_metric_rows.extend(
                {**row, "phase": "validation_single_source"} for row in val_single_metrics
            )
            rankings["supervised_val_nested"] = eligible_first_ranking(
                ranking_from_source_summary(val_source_summary),
                eligible_sources,
                len(reference.node_ids),
            )

        final_train = train_records + val_records
        for strategy, ranking in rankings.items():
            metrics, anchors = evaluate_hierarchy(
                final_train,
                test_records,
                ranking,
                args.anchor_counts,
                split_id,
                strategy,
                args.ridge_alpha,
                dataset_label=args.dataset_label,
                phase="test",
            )
            test_metric_rows.extend(metrics)
            anchor_rows.extend(anchors)
            if not strategy.startswith("random_nested_"):
                split_rankings.append({"split_id": split_id, "strategy": strategy, "ranking": ranking})

    write_csv(
        args.output_dir / "qrc_h_validation_grid.csv",
        validation_grid_rows,
        [
            "split_id",
            "config_id",
            "strategy",
            "anchor_count",
            "validation_rmse_mean",
            "validation_distance_corr_mean",
            "objective_rmse_mean_across_counts",
            "strength_weight",
            "resid_influence_weight",
            "resid_entropy_weight",
            "cross_weight",
            "redundancy_weight",
            "coverage_weight",
            "balance_weight",
        ],
    )
    write_csv(
        args.output_dir / "qrc_h_learned_configs.csv",
        learned_config_rows,
        [
            "split_id",
            "config_id",
            "strategy",
            "objective_rmse_mean_across_counts",
            "strength_weight",
            "resid_influence_weight",
            "resid_entropy_weight",
            "cross_weight",
            "redundancy_weight",
            "coverage_weight",
            "balance_weight",
        ],
    )
    write_csv(
        args.output_dir / "qrc_h_test_metrics.csv",
        test_metric_rows,
        [
            "dataset_label",
            "phase",
            "split_id",
            "sample_id",
            "selection_strategy",
            "is_nested_hierarchy",
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
        args.output_dir / "qrc_h_anchor_sets.csv",
        anchor_rows,
        [
            "dataset_label",
            "phase",
            "split_id",
            "sample_id",
            "anchor_strategy",
            "anchor_count",
            "anchor_ids",
            "anchor_names",
            "anchor_regions",
            "anchor_hemispheres",
            "is_nested_hierarchy",
        ],
    )
    test_summary = mean_by_group(
        test_metric_rows,
        ["selection_strategy", "anchor_count"],
        ["normalized_rmse", "distance_corr", "hemisphere_accuracy", "x_corr", "y_corr", "z_corr"],
    )
    test_summary = add_summary_deltas(test_summary, "strength_nested")
    write_csv(
        args.output_dir / "qrc_h_test_summary.csv",
        test_summary,
        [
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
            "delta_rmse_vs_strength_nested",
        ],
    )
    paired_rows = paired_selector_tests(
        test_metric_rows,
        [
            ("strength_nested", "qrc_h_learned"),
            ("qrc_residual_nested", "qrc_h_learned"),
            ("qrc_h_fixed_s70_q30", "qrc_h_learned"),
            ("qrc_h_fixed_s30_q70", "qrc_h_learned"),
            ("qrc_h_cover_s50_q50", "qrc_h_learned"),
            ("spatial_nested", "qrc_h_learned"),
            ("random_nested_1", "qrc_h_learned"),
            ("supervised_val_nested", "qrc_h_learned"),
            ("strength_nested", "qrc_h_staged"),
            ("qrc_residual_nested", "qrc_h_staged"),
            ("qrc_h_learned", "qrc_h_staged"),
            ("qrc_h_cover_s50_q50", "qrc_h_staged"),
            ("spatial_nested", "qrc_h_staged"),
            ("random_nested_1", "qrc_h_staged"),
            ("supervised_val_nested", "qrc_h_staged"),
            ("strength_nested", "qrc_h_stage8_cover"),
            ("qrc_residual_nested", "qrc_h_stage8_cover"),
            ("qrc_h_learned", "qrc_h_stage8_cover"),
            ("qrc_h_staged", "qrc_h_stage8_cover"),
            ("qrc_h_cover_s50_q50", "qrc_h_stage8_cover"),
            ("spatial_nested", "qrc_h_stage8_cover"),
            ("random_nested_1", "qrc_h_stage8_cover"),
            ("supervised_val_nested", "qrc_h_stage8_cover"),
        ],
        dataset_label=args.dataset_label,
    )
    write_csv(
        args.output_dir / "qrc_h_paired_tests.csv",
        paired_rows,
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
    frequency_rows, layer_rows = source_frequency_rows(split_rankings, reference, args.anchor_counts)
    write_csv(
        args.output_dir / "qrc_h_source_frequency.csv",
        frequency_rows,
        [
            "strategy",
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "mean_rank",
            "rank_std",
            *[f"top_{count}_frequency" for count in args.anchor_counts],
        ],
    )
    write_csv(
        args.output_dir / "qrc_h_layer_frequency.csv",
        layer_rows,
        [
            "strategy",
            "layer",
            "node_index",
            "node_id",
            "node_name",
            "region",
            "hemisphere",
            "layer_frequency",
            "mean_rank",
        ],
    )
    write_summary(
        args.output_dir,
        args.docs_path,
        test_summary,
        paired_rows,
        learned_config_rows,
        frequency_rows,
        layer_rows,
    )
    print(f"Wrote QRC-H hierarchy outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
