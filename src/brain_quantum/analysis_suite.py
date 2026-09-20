"""
Extended analyses for Quantum-like Relational Coordinates (QRC).

Analyses included:
- embedding baseline comparison
- anchor/source coordinate recovery
- QRC influence versus weighted-strength hub overlap

The script intentionally stays NumPy-only so it runs in the lightweight local
venv created for this project.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain_quantum.graph_propagation import component_support, supported_spectral_matrix
from brain_quantum.qrc_connectome import (  # noqa: E402
    Connectome,
    best_axis_corr,
    hemisphere_accuracy,
    node_influence,
    normalized_laplacian,
    pairwise_distance_corr,
    parse_graphml,
    parse_times,
    pca_scores,
    pearson_abs,
    pearson_corr,
    row_normalize,
    transform_weights,
    write_csv,
)


def compute_walk_stack(laplacian: np.ndarray, times: list[float]) -> dict:
    evals, evecs = np.linalg.eigh(laplacian)
    support = component_support(laplacian)
    q_real = []
    q_imag = []
    q_prob = []
    c_heat = []
    c_heat_raw = []

    for t in times:
        q_phase = np.exp(-1j * evals * t)
        q_u = supported_spectral_matrix(evecs, q_phase, support)
        q_p = row_normalize((np.abs(q_u) ** 2).real)

        c_decay = np.exp(-evals * t)
        c_h_raw = np.maximum(supported_spectral_matrix(evecs, c_decay, support).real, 0.0)
        c_h = row_normalize(c_h_raw)

        q_real.append(q_u.real)
        q_imag.append(q_u.imag)
        q_prob.append(q_p)
        c_heat.append(c_h)
        c_heat_raw.append(c_h_raw)

    return {
        "evals": evals,
        "evecs": evecs,
        "q_real": q_real,
        "q_imag": q_imag,
        "q_prob": q_prob,
        "c_heat": c_heat,
        "c_heat_raw": c_heat_raw,
        "q_avg": np.mean(q_prob, axis=0),
        "c_avg": np.mean(c_heat, axis=0),
    }


def all_pairs_shortest_path(weights: np.ndarray) -> np.ndarray:
    n = weights.shape[0]
    dist = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist, 0.0)
    mask = weights > 0
    # Stronger edges become shorter paths.
    dist[mask] = 1.0 / (weights[mask] + 1e-9)

    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])

    finite = np.isfinite(dist)
    if finite.any():
        fill_value = float(np.max(dist[finite]) * 1.25)
        dist[~finite] = fill_value
    return dist


def position_extent(positions: np.ndarray) -> float:
    valid = np.all(np.isfinite(positions), axis=1)
    pts = positions[valid]
    if len(pts) < 2:
        return 1.0
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    return max(float(np.max(dists)), 1.0)


def finite_position_mask(positions: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(positions), axis=1)


def hemi_accuracy_from_x(values: np.ndarray, hemispheres: list[str], mask: np.ndarray) -> float:
    labels = np.array([1 if hemi.startswith("left") else 0 for hemi in hemispheres], dtype=int)
    selected = np.where(mask)[0]
    if len(selected) < 3:
        return float("nan")
    v = values[selected]
    threshold = float(np.median(v))
    pred = (v >= threshold).astype(int)
    truth = labels[selected]
    acc = float((pred == truth).mean())
    return max(acc, 1.0 - acc)


def source_informed_hemi_accuracy(
    values: np.ndarray,
    hemispheres: list[str],
    mask: np.ndarray,
    source_positions: np.ndarray,
    sources: np.ndarray,
) -> float:
    """Classify targets in a left-right frame calibrated by known sources."""
    source_positions = np.asarray(source_positions, dtype=float)
    sources = np.asarray(sources, dtype=int)
    finite_sources = sources[np.all(np.isfinite(source_positions[sources]), axis=1)]
    left_x = np.array(
        [source_positions[index, 0] for index in finite_sources if hemispheres[index].startswith("left")],
        dtype=float,
    )
    right_x = np.array(
        [source_positions[index, 0] for index in finite_sources if hemispheres[index].startswith("right")],
        dtype=float,
    )
    if len(left_x) == 0 or len(right_x) == 0:
        return float("nan")

    left_center = float(np.median(left_x))
    right_center = float(np.median(right_x))
    threshold = (left_center + right_center) / 2.0
    selected = np.array(
        [
            index
            for index in np.where(mask)[0]
            if hemispheres[index].startswith("left") or hemispheres[index].startswith("right")
        ],
        dtype=int,
    )
    if len(selected) == 0:
        return float("nan")
    if left_center > right_center:
        predicted_left = values[selected] > threshold
    else:
        predicted_left = values[selected] < threshold
    true_left = np.array([hemispheres[index].startswith("left") for index in selected], dtype=bool)
    return float(np.mean(predicted_left == true_left))


def pairwise_distance_corr_mask(predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    idx = np.where(mask & finite_position_mask(predicted) & finite_position_mask(truth))[0]
    if len(idx) < 3:
        return float("nan")
    pred = predicted[idx]
    true = truth[idx]
    pred_d = np.sqrt(np.sum((pred[:, None, :] - pred[None, :, :]) ** 2, axis=2))
    true_d = np.sqrt(np.sum((true[:, None, :] - true[None, :, :]) ** 2, axis=2))
    upper = np.triu_indices(len(idx), k=1)
    return pearson_corr(pred_d[upper], true_d[upper])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Good enough for continuous scores; ties are rare after weighted transforms.
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def farthest_spread_anchors(positions: np.ndarray, k: int) -> np.ndarray:
    valid_mask = finite_position_mask(positions)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return np.arange(min(k, positions.shape[0]), dtype=int)

    valid_positions = positions[valid_indices]
    first = int(valid_indices[np.argmin(valid_positions[:, 0])])
    selected = [first]
    selectable = set(int(idx) for idx in valid_indices)
    while len(selected) < k and set(selected) != selectable:
        selected_arr = np.array(selected, dtype=int)
        diffs = positions[valid_indices, None, :] - positions[selected_arr][None, :, :]
        nearest = np.sqrt(np.sum(diffs * diffs, axis=2)).min(axis=1)
        for chosen in selected:
            nearest[np.where(valid_indices == chosen)[0]] = -1.0
        selected.append(int(valid_indices[np.argmax(nearest)]))
    if len(selected) < k:
        selected_set = set(selected)
        selected.extend(idx for idx in range(positions.shape[0]) if idx not in selected_set)
    return np.array(selected[:k], dtype=int)


def hub_anchors(strength: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-strength)[:k]


def normalize_response(response: np.ndarray) -> np.ndarray:
    response = np.maximum(np.asarray(response, dtype=float), 0.0)
    row_sums = response.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 0
    if np.any(zero_rows):
        response[zero_rows, :] = 1.0
        row_sums = response.sum(axis=1, keepdims=True)
    return response / row_sums


def response_coordinate_moments(
    response_to_anchors: np.ndarray,
    anchor_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return response-weighted coordinate means and standard deviations.

    Sources without finite coordinates remain valid graph-response sources, but
    they are excluded and the remaining response weights are renormalized for
    this coordinate-only feature block. Propagation callers must enforce exact
    connected-component support before normalization; a magnitude cutoff here
    would also discard legitimate weak within-component responses.
    """
    response = np.asarray(response_to_anchors, dtype=float)
    positions = np.asarray(anchor_positions, dtype=float)
    if response.ndim != 2 or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Responses must be 2D and anchor positions must have shape (sources, 3).")
    if response.shape[1] != positions.shape[0]:
        raise ValueError("Response columns must match the number of anchor positions.")

    valid = np.all(np.isfinite(positions), axis=1)
    if not np.any(valid):
        zeros = np.zeros((response.shape[0], 3), dtype=float)
        return zeros, zeros.copy()

    valid_response = np.maximum(response[:, valid], 0.0)
    row_sums = valid_response.sum(axis=1, keepdims=True)
    weights = np.divide(
        valid_response,
        row_sums,
        out=np.zeros_like(valid_response),
        where=row_sums > 0,
    )
    finite_positions = positions[valid]
    mean = weights @ finite_positions
    second = weights @ (finite_positions * finite_positions)
    spread = np.sqrt(np.maximum(second - mean * mean, 0.0))
    return mean, spread


def barycentric_predict(response_to_anchors: np.ndarray, anchor_positions: np.ndarray) -> np.ndarray:
    mean, _ = response_coordinate_moments(response_to_anchors, anchor_positions)
    return mean


def standardize_train_test(features: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mean = features[train_mask].mean(axis=0, keepdims=True)
    std = features[train_mask].std(axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    return (features - mean) / std


def ridge_predict(features: np.ndarray, positions: np.ndarray, train_idx: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    train_mask = np.zeros(features.shape[0], dtype=bool)
    train_mask[train_idx] = True
    x = standardize_train_test(features, train_mask)
    x_train = x[train_idx]
    y_train = positions[train_idx]

    # Dual ridge is much faster here because anchor_count << feature_count.
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_centered = y_train - y_mean
    kernel = x_train @ x_train.T
    coef = np.linalg.pinv(kernel + alpha * np.eye(kernel.shape[0])) @ y_centered
    return (x @ x_train.T) @ coef + y_mean


def anchor_feature_sets(
    weights: np.ndarray,
    walk: dict,
    shortest_paths: np.ndarray,
    anchors: np.ndarray,
) -> dict:
    q_columns = []
    c_columns = []
    for real, imag, prob, heat in zip(walk["q_real"], walk["q_imag"], walk["q_prob"], walk["c_heat"]):
        q_columns.extend([real[:, anchors], imag[:, anchors], prob[:, anchors]])
        c_columns.append(heat[:, anchors])

    sp = shortest_paths[:, anchors]
    sp_scale = np.median(sp[np.isfinite(sp)])
    sp_response = np.exp(-sp / max(sp_scale, 1e-9))

    direct = weights[:, anchors]

    return {
        "qrc": {
            "features": np.concatenate(q_columns, axis=1),
            "response": np.mean([prob[:, anchors] for prob in walk["q_prob"]], axis=0),
        },
        "classical": {
            "features": np.concatenate(c_columns, axis=1),
            "response": np.mean([heat[:, anchors] for heat in walk["c_heat"]], axis=0),
        },
        "shortest_path": {
            "features": sp,
            "response": sp_response,
        },
        "direct_weight": {
            "features": direct,
            "response": direct,
        },
    }


def anchor_metrics(
    connectome: Connectome,
    method_name: str,
    predicted: np.ndarray,
    anchors: np.ndarray,
    strategy: str,
    count: int,
    mode: str,
    extent: float,
) -> dict:
    mask = np.ones(len(connectome.node_ids), dtype=bool)
    mask[anchors] = False
    mask &= finite_position_mask(connectome.positions)
    if int(mask.sum()) == 0:
        return {
            "subject_id": connectome.subject_id,
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
    errors = np.linalg.norm(predicted[mask] - connectome.positions[mask], axis=1)
    return {
        "subject_id": connectome.subject_id,
        "anchor_strategy": strategy,
        "anchor_count": count,
        "method": method_name,
        "mode": mode,
        "normalized_rmse": float(np.sqrt(np.mean(errors**2)) / extent),
        "mean_error": float(np.mean(errors)),
        "x_corr": pearson_abs(predicted[mask, 0], connectome.positions[mask, 0]),
        "y_corr": pearson_abs(predicted[mask, 1], connectome.positions[mask, 1]),
        "z_corr": pearson_abs(predicted[mask, 2], connectome.positions[mask, 2]),
        "distance_corr": pairwise_distance_corr_mask(predicted, connectome.positions, mask),
        "hemisphere_accuracy": source_informed_hemi_accuracy(
            predicted[:, 0],
            connectome.hemispheres,
            mask,
            connectome.positions,
            anchors,
        ),
    }


def embedding_baselines(connectome: Connectome, weights: np.ndarray, walk: dict, shortest_paths: np.ndarray) -> list[dict]:
    q_features = np.concatenate(
        [block for triples in zip(walk["q_real"], walk["q_imag"], walk["q_prob"]) for block in triples],
        axis=1,
    )
    c_features = np.concatenate(walk["c_heat"], axis=1)
    spectral = walk["evecs"][:, 1:4]
    adjacency = weights
    path_embedding = -shortest_paths

    methods = {
        "qrc_full": pca_scores(q_features, dims=3),
        "classical_full": pca_scores(c_features, dims=3),
        "spectral_laplacian": spectral,
        "adjacency_pca": pca_scores(adjacency, dims=3),
        "shortest_path_pca": pca_scores(path_embedding, dims=3),
    }

    rows = []
    for name, embedding in methods.items():
        hemi_acc, hemi_dim = hemisphere_accuracy(embedding, connectome.hemispheres)
        rows.append(
            {
                "subject_id": connectome.subject_id,
                "method": name,
                "hemisphere_accuracy": hemi_acc,
                "hemisphere_component": hemi_dim,
                "best_x_corr": best_axis_corr(embedding, connectome.positions[:, 0]),
                "best_y_corr": best_axis_corr(embedding, connectome.positions[:, 1]),
                "best_z_corr": best_axis_corr(embedding, connectome.positions[:, 2]),
                "position_distance_corr": pairwise_distance_corr(embedding, connectome.positions),
            }
        )
    return rows


def hub_overlap(connectome: Connectome, weights: np.ndarray, walk: dict, top_k: int) -> tuple[dict, list[dict]]:
    influence = node_influence(walk["q_avg"], connectome.hemispheres)
    strength = weights.sum(axis=1)
    top_influence = np.argsort(-influence)[:top_k]
    top_strength = set(np.argsort(-strength)[:top_k])
    overlap = len(set(top_influence) & top_strength) / top_k
    corr = spearman_corr(influence, strength)
    rows = []
    for rank, idx in enumerate(top_influence, start=1):
        rows.append(
            {
                "subject_id": connectome.subject_id,
                "rank": rank,
                "node_id": connectome.node_ids[idx],
                "node_name": connectome.node_names[idx],
                "hemisphere": connectome.hemispheres[idx],
                "region": connectome.regions[idx],
                "quantum_influence": float(influence[idx]),
                "weighted_strength": float(strength[idx]),
                "is_top_strength_hub": int(idx in top_strength),
            }
        )
    return (
        {
            "subject_id": connectome.subject_id,
            "top_k": top_k,
            "influence_strength_spearman": corr,
            "top_k_overlap": overlap,
        },
        rows,
    )


def mean_by_group(rows: list[dict], group_fields: list[str], metric_fields: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups[key].append(row)

    out = []
    for key, group_rows in sorted(groups.items()):
        record = {field: value for field, value in zip(group_fields, key)}
        for metric in metric_fields:
            vals = [float(row[metric]) for row in group_rows if row.get(metric) not in {"", None}]
            vals = [val for val in vals if np.isfinite(val)]
            record[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            record[f"{metric}_std"] = float(np.std(vals)) if vals else float("nan")
        record["n"] = len(group_rows)
        out.append(record)
    return out


def write_markdown_summary(
    output_dir: Path,
    baseline_summary: list[dict],
    anchor_summary: list[dict],
    hub_summary: list[dict],
    frequency_rows: list[dict],
    subject_count: int,
) -> None:
    def top_table(rows: list[dict], fields: list[str], limit: int = 12) -> str:
        header = "| " + " | ".join(fields) + " |"
        sep = "| " + " | ".join(["---"] * len(fields)) + " |"
        body = []
        for row in rows[:limit]:
            vals = []
            for field in fields:
                val = row.get(field, "")
                if isinstance(val, float):
                    vals.append(f"{val:.3f}")
                else:
                    vals.append(str(val))
            body.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep, *body])

    baseline_ranked = sorted(baseline_summary, key=lambda r: -float(r["hemisphere_accuracy_mean"]))
    anchor_ranked = sorted(
        anchor_summary,
        key=lambda r: (
            int(r["anchor_count"]),
            str(r["anchor_strategy"]),
            float(r["normalized_rmse_mean"]),
        ),
    )

    text = f"""# Extended Analysis Summary

Dataset: BrainGraph.org HCP 86-node structural connectomes

Subjects analyzed: {subject_count}

## Embedding Baselines

{top_table(baseline_ranked, ["method", "hemisphere_accuracy_mean", "position_distance_corr_mean", "best_x_corr_mean"])}

## Anchor/Source Coordinate Recovery

Lower normalized RMSE is better. Higher hemisphere accuracy and distance
correlation are better.

{top_table(anchor_ranked, ["anchor_strategy", "anchor_count", "method", "mode", "normalized_rmse_mean", "hemisphere_accuracy_mean", "distance_corr_mean"], limit=24)}

## QRC Influence vs Weighted-Strength Hubs

{top_table(hub_summary, ["top_k", "influence_strength_spearman_mean", "top_k_overlap_mean"])}

## Most Frequent Top QRC-Influence Regions

{top_table(frequency_rows, ["node_name", "hemisphere", "region", "top_influence_count"], limit=20)}

## Interpretation

The baseline analysis asks whether whole-network relational embeddings recover
known anatomy. The anchor analysis asks the more original question: when only a
small set of source nodes is treated as known, can the network infer the
relative coordinates of the rest?

The next research step is to keep the anchor/source setup, then add stronger
decoders and statistical testing.
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def analyze_subject(
    path: Path,
    times: list[float],
    weight_transform: str,
    anchor_counts: list[int],
    top_k: int,
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    connectome = parse_graphml(path)
    weights = transform_weights(connectome.adjacency, weight_transform)
    laplacian = normalized_laplacian(weights)
    walk = compute_walk_stack(laplacian, times)
    shortest_paths = all_pairs_shortest_path(weights)
    extent = position_extent(connectome.positions)

    baseline_rows = embedding_baselines(connectome, weights, walk, shortest_paths)

    anchor_rows = []
    strategies = {
        "spatial_spread": lambda k: farthest_spread_anchors(connectome.positions, k),
        "strength_hubs": lambda k: hub_anchors(weights.sum(axis=1), k),
    }
    for strategy_name, selector in strategies.items():
        for count in anchor_counts:
            anchors = selector(min(count, len(connectome.node_ids) - 1))
            feature_sets = anchor_feature_sets(weights, walk, shortest_paths, anchors)
            for method_name, method_data in feature_sets.items():
                bary_pred = barycentric_predict(method_data["response"], connectome.positions[anchors])
                anchor_rows.append(
                    anchor_metrics(
                        connectome,
                        method_name,
                        bary_pred,
                        anchors,
                        strategy_name,
                        count,
                        "barycentric",
                        extent,
                    )
                )
                ridge_pred = ridge_predict(method_data["features"], connectome.positions, anchors, alpha=1.0)
                anchor_rows.append(
                    anchor_metrics(
                        connectome,
                        method_name,
                        ridge_pred,
                        anchors,
                        strategy_name,
                        count,
                        "ridge",
                        extent,
                    )
                )

    hub_row, influence_rows = hub_overlap(connectome, weights, walk, top_k=top_k)
    return baseline_rows, anchor_rows, hub_row, influence_rows


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the extended Brain Quantum analysis suite.")
    parser.add_argument("--input-dir", type=Path, default=Path("research/data/brain_graph_hcp_86_nodes/graphml"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/outputs/analysis_suite"))
    parser.add_argument("--subjects", type=int, default=1064)
    parser.add_argument("--times", type=parse_times, default=parse_times("0.25,0.5,1,2,4,8"))
    parser.add_argument("--anchor-counts", type=parse_ints, default=parse_ints("4,8,16,32"))
    parser.add_argument("--weight-transform", choices=["raw", "log1p", "binary"], default="log1p")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.graphml"))[: args.subjects]
    if not files:
        raise SystemExit(f"No GraphML files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows: list[dict] = []
    anchor_rows: list[dict] = []
    hub_rows: list[dict] = []
    influence_rows: list[dict] = []

    for idx, path in enumerate(files, start=1):
        subject_baseline, subject_anchor, subject_hub, subject_influence = analyze_subject(
            path,
            times=args.times,
            weight_transform=args.weight_transform,
            anchor_counts=args.anchor_counts,
            top_k=args.top_k,
        )
        baseline_rows.extend(subject_baseline)
        anchor_rows.extend(subject_anchor)
        hub_rows.append(subject_hub)
        influence_rows.extend(subject_influence)
        if idx % 100 == 0 or idx == len(files):
            print(f"Analyzed {idx}/{len(files)} subjects")

    baseline_fields = [
        "subject_id",
        "method",
        "hemisphere_accuracy",
        "hemisphere_component",
        "best_x_corr",
        "best_y_corr",
        "best_z_corr",
        "position_distance_corr",
    ]
    anchor_fields = [
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
    hub_fields = ["subject_id", "top_k", "influence_strength_spearman", "top_k_overlap"]
    influence_fields = [
        "subject_id",
        "rank",
        "node_id",
        "node_name",
        "hemisphere",
        "region",
        "quantum_influence",
        "weighted_strength",
        "is_top_strength_hub",
    ]

    write_csv(args.output_dir / "baseline_comparison.csv", baseline_rows, baseline_fields)
    write_csv(args.output_dir / "anchor_recovery.csv", anchor_rows, anchor_fields)
    write_csv(args.output_dir / "hub_overlap.csv", hub_rows, hub_fields)
    write_csv(args.output_dir / "top_influence_regions_by_subject.csv", influence_rows, influence_fields)

    baseline_summary = mean_by_group(
        baseline_rows,
        ["method"],
        ["hemisphere_accuracy", "best_x_corr", "position_distance_corr"],
    )
    anchor_summary = mean_by_group(
        anchor_rows,
        ["anchor_strategy", "anchor_count", "method", "mode"],
        ["normalized_rmse", "hemisphere_accuracy", "distance_corr", "x_corr"],
    )
    hub_summary = mean_by_group(
        hub_rows,
        ["top_k"],
        ["influence_strength_spearman", "top_k_overlap"],
    )

    frequency = Counter(
        (row["node_name"], row["hemisphere"], row["region"]) for row in influence_rows if int(row["rank"]) <= args.top_k
    )
    frequency_rows = [
        {
            "node_name": node_name,
            "hemisphere": hemisphere,
            "region": region,
            "top_influence_count": count,
        }
        for (node_name, hemisphere, region), count in frequency.most_common()
    ]

    write_csv(
        args.output_dir / "baseline_summary.csv",
        baseline_summary,
        [
            "method",
            "hemisphere_accuracy_mean",
            "hemisphere_accuracy_std",
            "best_x_corr_mean",
            "best_x_corr_std",
            "position_distance_corr_mean",
            "position_distance_corr_std",
            "n",
        ],
    )
    write_csv(
        args.output_dir / "anchor_summary.csv",
        anchor_summary,
        [
            "anchor_strategy",
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
            "n",
        ],
    )
    write_csv(
        args.output_dir / "hub_summary.csv",
        hub_summary,
        [
            "top_k",
            "influence_strength_spearman_mean",
            "influence_strength_spearman_std",
            "top_k_overlap_mean",
            "top_k_overlap_std",
            "n",
        ],
    )
    write_csv(
        args.output_dir / "top_influence_frequency.csv",
        frequency_rows,
        ["node_name", "hemisphere", "region", "top_influence_count"],
    )

    write_markdown_summary(
        args.output_dir,
        baseline_summary,
        anchor_summary,
        hub_summary,
        frequency_rows,
        subject_count=len(files),
    )

    print(f"Wrote analysis suite outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
