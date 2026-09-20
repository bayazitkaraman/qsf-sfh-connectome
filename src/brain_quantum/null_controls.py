"""Null-control experiments for QRC-v2.

Controls:
- real_qrc: QRC-v2 on the real graph and real coordinates
- real_classical: matched heat-diffusion baseline on the real graph
- coord_shuffle_qrc: real graph but independently shuffled coordinate targets
- degree_rewire_qrc: degree-preserving randomized graph with real coordinates
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain_quantum.analysis_suite import (  # noqa: E402
    all_pairs_shortest_path,
    anchor_metrics,
    compute_walk_stack,
    mean_by_group,
    position_extent,
)
from brain_quantum.qrc_connectome import (  # noqa: E402
    interregional_weights,
    normalized_laplacian,
    parse_times,
    write_csv,
)
from brain_quantum.source_coordinate_v2 import (  # noqa: E402
    SubjectRecord,
    anchor_features,
    cache_subject,
    choose_balanced_anchors,
    fit_global_ridge,
    laplacian_eigenmap,
    predict_ridge,
)


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def select_split_indices(n: int, train_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n)
    rng.shuffle(indices)
    train_n = max(1, int(n * train_fraction))
    return indices[:train_n], indices[train_n:]


def take(records: list[SubjectRecord], indices: np.ndarray) -> list[SubjectRecord]:
    return [records[int(i)] for i in indices]


def record_from_weights(record: SubjectRecord, weights: np.ndarray, times: list[float]) -> SubjectRecord:
    weights = interregional_weights(weights)
    laplacian = normalized_laplacian(weights)
    walk = compute_walk_stack(laplacian, times)
    shortest_paths = all_pairs_shortest_path(weights).astype(np.float32)
    return SubjectRecord(
        connectome=record.connectome,
        weights=weights.astype(np.float32),
        shortest_paths=shortest_paths,
        spectral_embedding=laplacian_eigenmap(walk["evecs"]).astype(np.float32),
        q_real=[arr.astype(np.float32) for arr in walk["q_real"]],
        q_imag=[arr.astype(np.float32) for arr in walk["q_imag"]],
        q_prob=[arr.astype(np.float32) for arr in walk["q_prob"]],
        c_heat=[arr.astype(np.float32) for arr in walk["c_heat"]],
        extent=record.extent,
        c_heat_raw=[arr.astype(np.float32) for arr in walk["c_heat_raw"]],
    )


def coordinate_shuffle_records(records: list[SubjectRecord], rng: np.random.Generator) -> list[SubjectRecord]:
    shuffled = []
    for record in records:
        perm = rng.permutation(record.connectome.positions.shape[0])
        positions = record.connectome.positions[perm].copy()
        connectome = replace(record.connectome, positions=positions)
        shuffled.append(replace(record, connectome=connectome, extent=position_extent(positions)))
    return shuffled


def degree_preserving_rewire(weights: np.ndarray, rng: np.random.Generator, swaps_per_edge: int = 5) -> np.ndarray:
    weights = interregional_weights(weights)
    n = weights.shape[0]
    upper = np.transpose(np.nonzero(np.triu(weights > 0, 1)))
    edges = [tuple(map(int, edge)) for edge in upper]
    if len(edges) < 2:
        return weights.copy()
    edge_set = set(edges)
    original_values = np.array([weights[i, j] for i, j in edges], dtype=float)

    target_swaps = swaps_per_edge * len(edges)
    attempts = 0
    swaps = 0
    max_attempts = target_swaps * 20
    while swaps < target_swaps and attempts < max_attempts:
        attempts += 1
        e1, e2 = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[int(e1)]
        c, d = edges[int(e2)]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            new_edges = [(min(a, d), max(a, d)), (min(c, b), max(c, b))]
        else:
            new_edges = [(min(a, c), max(a, c)), (min(b, d), max(b, d))]
        if new_edges[0][0] == new_edges[0][1] or new_edges[1][0] == new_edges[1][1]:
            continue
        old_edges = {edges[int(e1)], edges[int(e2)]}
        if new_edges[0] == new_edges[1]:
            continue
        if any(edge in edge_set and edge not in old_edges for edge in new_edges):
            continue
        edge_set.remove(edges[int(e1)])
        edge_set.remove(edges[int(e2)])
        edge_set.add(new_edges[0])
        edge_set.add(new_edges[1])
        edges[int(e1)] = new_edges[0]
        edges[int(e2)] = new_edges[1]
        swaps += 1

    shuffled_values = rng.permutation(original_values)
    rewired = np.zeros_like(weights, dtype=float)
    for (i, j), value in zip(edges, shuffled_values):
        rewired[i, j] = value
        rewired[j, i] = value
    return rewired


def degree_rewired_records(records: list[SubjectRecord], times: list[float], rng: np.random.Generator) -> list[SubjectRecord]:
    out = []
    for idx, record in enumerate(records, start=1):
        rewired = degree_preserving_rewire(record.weights, rng)
        out.append(record_from_weights(record, rewired, times))
        if idx % 200 == 0 or idx == len(records):
            print(f"Built degree-rewired null record {idx}/{len(records)}")
    return out


def evaluate_records(
    train_records: list[SubjectRecord],
    test_records: list[SubjectRecord],
    anchors: np.ndarray,
    method: str,
    alpha: float,
) -> list[dict]:
    model = fit_global_ridge(train_records, anchors, method, alpha=alpha, qrc_variant="full")
    rows = []
    for record in test_records:
        features, _ = anchor_features(record, anchors, method, qrc_variant="full")
        pred = predict_ridge(model, features)
        rows.append(
            anchor_metrics(
                record.connectome,
                method,
                pred,
                anchors,
                "random_balanced",
                len(anchors),
                "global_ridge",
                record.extent,
            )
        )
    return rows


def paired_control_tests(rows: list[dict]) -> list[dict]:
    real = {}
    for row in rows:
        if row["control"] == "real_qrc":
            key = (row["split_id"], row["sample_id"], row["subject_id"], row["anchor_count"])
            real[key] = row

    metrics = {"normalized_rmse": "lower", "distance_corr": "higher", "hemisphere_accuracy": "higher"}
    controls = sorted({row["control"] for row in rows if row["control"] != "real_qrc"})
    out = []
    for control in controls:
        for count in sorted({row["anchor_count"] for row in rows}, key=int):
            for metric, direction in metrics.items():
                diffs = []
                wins = 0
                total = 0
                for row in rows:
                    if row["control"] != control or row["anchor_count"] != count:
                        continue
                    key = (row["split_id"], row["sample_id"], row["subject_id"], row["anchor_count"])
                    if key not in real:
                        continue
                    q_val = float(real[key][metric])
                    c_val = float(row[metric])
                    if not (np.isfinite(q_val) and np.isfinite(c_val)):
                        continue
                    diffs.append(q_val - c_val)
                    wins += int(q_val < c_val) if direction == "lower" else int(q_val > c_val)
                    total += 1
                if total < 3:
                    continue
                arr = np.array(diffs, dtype=float)
                std = float(arr.std(ddof=1))
                t_value = float(arr.mean() / (std / np.sqrt(total))) if std > 0 else 0.0
                p_value = float(math.erfc(abs(t_value) / math.sqrt(2.0)))
                out.append(
                    {
                        "control": control,
                        "anchor_count": count,
                        "metric": metric,
                        "direction": direction,
                        "n": total,
                        "mean_delta_real_qrc_minus_control": float(arr.mean()),
                        "paired_t": t_value,
                        "normal_approx_p": p_value,
                        "real_qrc_win_rate": wins / total,
                    }
                )
    return out


def write_summary(output_dir: Path, summary_rows: list[dict], pair_rows: list[dict]) -> None:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def table(rows: list[dict], fields: list[str], limit: int = 24) -> str:
        header = "| " + " | ".join(fields) + " |"
        sep = "| " + " | ".join(["---"] * len(fields)) + " |"
        body = ["| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |" for row in rows[:limit]]
        return "\n".join([header, sep, *body])

    ranked = sorted(summary_rows, key=lambda row: (int(row["anchor_count"]), float(row["normalized_rmse_mean"])))
    control_tests = [
        row
        for row in pair_rows
        if row["metric"] in {"normalized_rmse", "distance_corr"} and row["control"] != "real_classical"
    ]
    control_tests = sorted(control_tests, key=lambda row: (int(row["anchor_count"]), row["control"], row["metric"]))
    text = f"""# Null-Control Summary

## Mean Performance

{table(ranked, ["anchor_count", "control", "method", "normalized_rmse_mean", "distance_corr_mean", "hemisphere_accuracy_mean"])}

## Real QRC vs Null Controls

For RMSE, negative delta means real QRC is lower/better. For distance
correlation, positive delta means real QRC is higher/better.

{table(control_tests, ["anchor_count", "control", "metric", "mean_delta_real_qrc_minus_control", "real_qrc_win_rate", "normal_approx_p"], 24)}
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QRC-v2 null controls.")
    parser.add_argument("--input-dir", type=Path, default=Path("research/data/brain_graph_hcp_86_nodes/graphml"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/outputs/qrc_v2_null_controls"))
    parser.add_argument("--subjects", type=int, default=1064)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--anchor-counts", type=parse_ints, default=parse_ints("8,16,32"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--times", type=parse_times, default=parse_times("0.25,0.5,1,2,4,8"))
    parser.add_argument("--weight-transform", choices=["raw", "log1p", "binary"], default="log1p")
    parser.add_argument("--ridge-alpha", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    files = sorted(args.input_dir.glob("*.graphml"))[: args.subjects]
    if not files:
        raise SystemExit(f"No GraphML files found in {args.input_dir}")

    print(f"Caching real records for {len(files)} subjects")
    real_records = [cache_subject(path, args.times, args.weight_transform) for path in files]
    print("Building coordinate-shuffle null")
    coord_records = coordinate_shuffle_records(real_records, np.random.default_rng(args.seed + 1))
    print("Building degree-preserving rewired null")
    rewired_records = degree_rewired_records(real_records, args.times, np.random.default_rng(args.seed + 2))

    controls = {
        "real_qrc": (real_records, "qrc"),
        "real_classical": (real_records, "classical"),
        "coord_shuffle_qrc": (coord_records, "qrc"),
        "degree_rewire_qrc": (rewired_records, "qrc"),
    }

    rows = []
    sample_counter = 0
    for split_id in range(1, args.splits + 1):
        split_rng = np.random.default_rng(args.seed + 1000 + split_id)
        train_idx, test_idx = select_split_indices(len(real_records), args.train_fraction, split_rng)
        reference = real_records[int(train_idx[0])]
        print(f"Split {split_id}/{args.splits}: train={len(train_idx)} test={len(test_idx)}")
        for count in args.anchor_counts:
            for repeat in range(args.repeats):
                sample_counter += 1
                anchors = choose_balanced_anchors(reference.connectome.hemispheres, count, split_rng)
                for control_name, (control_records, method) in controls.items():
                    train_records = take(control_records, train_idx)
                    test_records = take(control_records, test_idx)
                    metrics = evaluate_records(train_records, test_records, anchors, method, alpha=args.ridge_alpha)
                    for metric in metrics:
                        metric.update(
                            {
                                "split_id": split_id,
                                "sample_id": sample_counter,
                                "repeat": repeat,
                                "control": control_name,
                            }
                        )
                        rows.append(metric)

    fields = [
        "split_id",
        "sample_id",
        "repeat",
        "subject_id",
        "control",
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
    write_csv(args.output_dir / "null_control_metrics.csv", rows, fields)
    summary = mean_by_group(
        rows,
        ["anchor_count", "control", "method"],
        ["normalized_rmse", "distance_corr", "hemisphere_accuracy"],
    )
    summary_fields = [
        "anchor_count",
        "control",
        "method",
        "normalized_rmse_mean",
        "normalized_rmse_std",
        "distance_corr_mean",
        "distance_corr_std",
        "hemisphere_accuracy_mean",
        "hemisphere_accuracy_std",
        "n",
    ]
    write_csv(args.output_dir / "null_control_summary.csv", summary, summary_fields)
    pair_rows = paired_control_tests(rows)
    write_csv(
        args.output_dir / "null_control_pairwise_tests.csv",
        pair_rows,
        [
            "control",
            "anchor_count",
            "metric",
            "direction",
            "n",
            "mean_delta_real_qrc_minus_control",
            "paired_t",
            "normal_approx_p",
            "real_qrc_win_rate",
        ],
    )
    write_summary(args.output_dir, summary, pair_rows)
    print(f"Wrote null-control outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
