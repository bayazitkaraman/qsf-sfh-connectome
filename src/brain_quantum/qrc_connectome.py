"""
Quantum-like Relational Coordinates (QRC) prototype for brain connectomes.

This is an intentionally small first implementation:
- parse BrainGraph.org HCP GraphML files without external graph libraries
- build a weighted connectome matrix
- compare classical diffusion features with quantum-like phase/walk features
- test whether the features recover known hemisphere and physical coordinates

The quantum-like step is a continuous-time walk on the graph Laplacian:
    U(t) = exp(-i L t)

Rows of U(t), Im(U(t)), and |U(t)|^2 are treated as relational responses:
each node is described by how it responds to signals launched from all source
nodes across several time scales.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

if __package__ in {None, ""}:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain_quantum.graph_propagation import component_support, supported_spectral_matrix


GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


@dataclass
class Connectome:
    path: Path
    subject_id: str
    adjacency: np.ndarray
    node_ids: list[str]
    node_names: list[str]
    hemispheres: list[str]
    regions: list[str]
    positions: np.ndarray


def parse_graphml(path: Path, weight_name: str = "number_of_fibers") -> Connectome:
    """Read an undirected weighted export, retaining raw loops once for auditing.

    Analysis uses ``transform_weights``, which removes self-connections before
    applying the requested transform. Invalid required weights are never imputed.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    key_names = {}
    for key in root.findall("g:key", GRAPHML_NS):
        key_id = key.attrib["id"]
        key_names[key_id] = key.attrib.get("attr.name", key_id)

    graph = root.find("g:graph", GRAPHML_NS)
    if graph is None:
        raise ValueError(f"No graph element found in {path}")
    if graph.attrib.get("edgedefault") != "undirected":
        raise ValueError(f"Expected an undirected graph in {path}")

    node_records = []
    for node in graph.findall("g:node", GRAPHML_NS):
        raw_id = node.attrib["id"]
        attrs = {}
        for data in node.findall("g:data", GRAPHML_NS):
            attrs[key_names.get(data.attrib["key"], data.attrib["key"])] = data.text or ""
        node_records.append((raw_id, attrs))

    node_records.sort(key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0]))
    index_by_id = {raw_id: idx for idx, (raw_id, _) in enumerate(node_records)}
    n = len(node_records)
    if n == 0 or len(index_by_id) != n:
        raise ValueError(f"Empty graph or duplicate node identifiers in {path}")
    adjacency = np.zeros((n, n), dtype=float)

    seen_edges = set()
    for edge in graph.findall("g:edge", GRAPHML_NS):
        source = edge.attrib["source"]
        target = edge.attrib["target"]
        if edge.attrib.get("directed", "false").lower() not in {"false", "0"}:
            raise ValueError(f"Directed edge {source}-{target} in {path}")
        if source not in index_by_id or target not in index_by_id:
            raise ValueError(f"Unknown endpoint on edge {source}-{target} in {path}")
        edge_key = tuple(sorted((source, target)))
        if edge_key in seen_edges:
            raise ValueError(f"Parallel edge {source}-{target} in {path}")
        seen_edges.add(edge_key)
        attrs = {}
        for data in edge.findall("g:data", GRAPHML_NS):
            attrs[key_names.get(data.attrib["key"], data.attrib["key"])] = data.text or ""
        raw_weight = attrs.get(weight_name)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid {weight_name} on edge {source}-{target} in {path}") from exc
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"Nonfinite or negative {weight_name} on edge {source}-{target} in {path}")
        i = index_by_id[source]
        j = index_by_id[target]
        adjacency[i, j] += weight
        if i != j:
            adjacency[j, i] += weight

    node_ids = [raw_id for raw_id, _ in node_records]
    node_names = [attrs.get("dn_name", attrs.get("dn_fsname", raw_id)) for raw_id, attrs in node_records]
    hemispheres = [attrs.get("dn_hemisphere", "unknown").lower() for _, attrs in node_records]
    regions = [attrs.get("dn_region", "unknown").lower() for _, attrs in node_records]
    positions = np.array(
        [
            [
                float(attrs.get("dn_position_x", "nan")),
                float(attrs.get("dn_position_y", "nan")),
                float(attrs.get("dn_position_z", "nan")),
            ]
            for _, attrs in node_records
        ],
        dtype=float,
    )

    subject_id = path.stem.split("_")[0]
    return Connectome(
        path=path,
        subject_id=subject_id,
        adjacency=adjacency,
        node_ids=node_ids,
        node_names=node_names,
        hemispheres=hemispheres,
        regions=regions,
        positions=positions,
    )


def interregional_weights(adjacency: np.ndarray) -> np.ndarray:
    """Return validated, zero-diagonal weights without modifying the input.

    Preserve the dtype so applying the policy to an already hollow graph does
    not change accumulation precision in existing null-control calculations.
    """
    weights = np.array(adjacency, copy=True)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("Expected a square adjacency matrix")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Adjacency weights must be finite and nonnegative")
    if not np.allclose(weights, weights.T, rtol=0, atol=1e-12):
        raise ValueError("Expected symmetric undirected weights")
    np.fill_diagonal(weights, 0)
    return weights


def transform_weights(adjacency: np.ndarray, mode: str) -> np.ndarray:
    adjacency = interregional_weights(adjacency)
    if mode == "raw":
        return adjacency.astype(float)
    if mode == "log1p":
        return np.log1p(adjacency)
    if mode == "binary":
        return (adjacency > 0).astype(float)
    raise ValueError(f"Unknown weight transform: {mode}")


def normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    weights = interregional_weights(adjacency).astype(float)
    weights = (weights + weights.T) / 2.0
    degree = weights.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    sym_norm = inv_sqrt[:, None] * weights * inv_sqrt[None, :]
    # The symmetric normalized Laplacian is zero on isolated vertices. Using
    # an identity diagonal there would create artificial time-dependent phase
    # and heat decay even though no edge can carry a response.
    return np.diag(mask.astype(float)) - sym_norm


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    return matrix / safe


def pca_scores(features: np.ndarray, dims: int = 3) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :dims] * s[:dims]


def walk_features(laplacian: np.ndarray, times: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(laplacian)
    support = component_support(laplacian)

    quantum_blocks = []
    classical_blocks = []
    q_avg = np.zeros_like(laplacian, dtype=float)
    c_avg = np.zeros_like(laplacian, dtype=float)

    for t in times:
        quantum_phase = np.exp(-1j * evals * t)
        quantum_u = supported_spectral_matrix(evecs, quantum_phase, support)
        quantum_prob = np.abs(quantum_u) ** 2
        quantum_prob = row_normalize(quantum_prob.real)

        classical_decay = np.exp(-evals * t)
        classical_heat = supported_spectral_matrix(evecs, classical_decay, support)
        classical_heat = np.maximum(classical_heat.real, 0.0)
        classical_heat = row_normalize(classical_heat)

        # Amplitude phase plus probability carries the "quantum-like" relational signal.
        quantum_blocks.extend([quantum_u.real, quantum_u.imag, quantum_prob])
        classical_blocks.append(classical_heat)
        q_avg += quantum_prob
        c_avg += classical_heat

    q_avg /= len(times)
    c_avg /= len(times)
    return np.concatenate(quantum_blocks, axis=1), np.concatenate(classical_blocks, axis=1), q_avg, c_avg


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def pearson_abs(x: np.ndarray, y: np.ndarray) -> float:
    return abs(pearson_corr(x, y))


def best_axis_corr(embedding: np.ndarray, target: np.ndarray) -> float:
    return max(pearson_abs(embedding[:, dim], target) for dim in range(embedding.shape[1]))


def hemisphere_accuracy(embedding: np.ndarray, hemispheres: list[str]) -> tuple[float, int]:
    labels = np.array([1 if hemi.startswith("left") else 0 for hemi in hemispheres], dtype=int)
    best = 0.0
    best_dim = 0
    for dim in range(embedding.shape[1]):
        values = embedding[:, dim]
        threshold = float(np.median(values))
        pred = (values >= threshold).astype(int)
        acc = float((pred == labels).mean())
        acc = max(acc, 1.0 - acc)
        if acc > best:
            best = acc
            best_dim = dim + 1
    return best, best_dim


def pairwise_distance_corr(embedding: np.ndarray, positions: np.ndarray) -> float:
    n = embedding.shape[0]
    emb_dist = []
    pos_dist = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.all(np.isfinite(positions[i])) and np.all(np.isfinite(positions[j])):
                emb_dist.append(float(np.linalg.norm(embedding[i] - embedding[j])))
                pos_dist.append(float(np.linalg.norm(positions[i] - positions[j])))
    return pearson_corr(np.array(emb_dist), np.array(pos_dist))


def node_influence(q_avg: np.ndarray, hemispheres: list[str]) -> np.ndarray:
    n = q_avg.shape[0]
    scores = np.zeros(n, dtype=float)
    for source in range(n):
        p = q_avg[:, source].copy()
        p = p / max(p.sum(), 1e-12)
        positive = p > 0
        entropy = -float(np.sum(p[positive] * np.log(p[positive]))) / math.log(n)
        opposite = sum(p[i] for i in range(n) if hemispheres[i] != hemispheres[source])
        scores[source] = entropy * (1.0 + opposite)
    return scores


def analyze_connectome(connectome: Connectome, times: list[float], weight_transform: str) -> dict:
    weights = transform_weights(connectome.adjacency, weight_transform)
    laplacian = normalized_laplacian(weights)
    q_features, c_features, q_avg, c_avg = walk_features(laplacian, times)
    q_embedding = pca_scores(q_features, dims=3)
    c_embedding = pca_scores(c_features, dims=3)

    q_acc, q_dim = hemisphere_accuracy(q_embedding, connectome.hemispheres)
    c_acc, c_dim = hemisphere_accuracy(c_embedding, connectome.hemispheres)

    q_dist_corr = pairwise_distance_corr(q_embedding, connectome.positions)
    c_dist_corr = pairwise_distance_corr(c_embedding, connectome.positions)

    gain = q_avg - c_avg
    interference_gain = float(np.linalg.norm(gain) / max(np.linalg.norm(c_avg), 1e-12))

    influence = node_influence(q_avg, connectome.hemispheres)
    strength = weights.sum(axis=1)

    return {
        "subject_id": connectome.subject_id,
        "nodes": weights.shape[0],
        "edges": int(np.count_nonzero(np.triu(weights, 1))),
        "quantum_hemi_accuracy": q_acc,
        "quantum_hemi_component": q_dim,
        "classical_hemi_accuracy": c_acc,
        "classical_hemi_component": c_dim,
        "quantum_best_x_corr": best_axis_corr(q_embedding, connectome.positions[:, 0]),
        "classical_best_x_corr": best_axis_corr(c_embedding, connectome.positions[:, 0]),
        "quantum_best_y_corr": best_axis_corr(q_embedding, connectome.positions[:, 1]),
        "classical_best_y_corr": best_axis_corr(c_embedding, connectome.positions[:, 1]),
        "quantum_best_z_corr": best_axis_corr(q_embedding, connectome.positions[:, 2]),
        "classical_best_z_corr": best_axis_corr(c_embedding, connectome.positions[:, 2]),
        "quantum_position_distance_corr": q_dist_corr,
        "classical_position_distance_corr": c_dist_corr,
        "interference_gain": interference_gain,
        "quantum_embedding": q_embedding,
        "classical_embedding": c_embedding,
        "quantum_influence": influence,
        "strength": strength,
        "q_avg": q_avg,
        "c_avg": c_avg,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_embedding_csv(path: Path, connectome: Connectome, result: dict) -> None:
    fields = [
        "node_id",
        "node_name",
        "hemisphere",
        "region",
        "x",
        "y",
        "z",
        "qrc_1",
        "qrc_2",
        "qrc_3",
        "classical_1",
        "classical_2",
        "classical_3",
        "quantum_influence",
        "weighted_strength",
    ]
    rows = []
    for i, node_id in enumerate(connectome.node_ids):
        rows.append(
            {
                "node_id": node_id,
                "node_name": connectome.node_names[i],
                "hemisphere": connectome.hemispheres[i],
                "region": connectome.regions[i],
                "x": connectome.positions[i, 0],
                "y": connectome.positions[i, 1],
                "z": connectome.positions[i, 2],
                "qrc_1": result["quantum_embedding"][i, 0],
                "qrc_2": result["quantum_embedding"][i, 1],
                "qrc_3": result["quantum_embedding"][i, 2],
                "classical_1": result["classical_embedding"][i, 0],
                "classical_2": result["classical_embedding"][i, 1],
                "classical_3": result["classical_embedding"][i, 2],
                "quantum_influence": result["quantum_influence"][i],
                "weighted_strength": result["strength"][i],
            }
        )
    write_csv(path, rows, fields)


def write_node_scores(path: Path, connectome: Connectome, result: dict, limit: int = 20) -> None:
    order = np.argsort(-result["quantum_influence"])[:limit]
    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "node_id": connectome.node_ids[idx],
                "node_name": connectome.node_names[idx],
                "hemisphere": connectome.hemispheres[idx],
                "region": connectome.regions[idx],
                "quantum_influence": result["quantum_influence"][idx],
                "weighted_strength": result["strength"][idx],
            }
        )
    write_csv(
        path,
        rows,
        ["rank", "node_id", "node_name", "hemisphere", "region", "quantum_influence", "weighted_strength"],
    )


def svg_scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        return np.full_like(values, (low + high) / 2.0)
    return low + (values - vmin) * (high - low) / (vmax - vmin)


def write_svg(path: Path, connectome: Connectome, result: dict) -> None:
    emb = result["quantum_embedding"]
    xs = svg_scale(emb[:, 0], 80, 720)
    ys = svg_scale(-emb[:, 1], 80, 520)
    influence = result["quantum_influence"]
    radii = svg_scale(influence, 4, 13)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">',
        '<rect width="800" height="600" fill="#f8fafc"/>',
        '<text x="40" y="38" font-family="Arial" font-size="20" fill="#111827">Quantum-like relational coordinates</text>',
        '<text x="40" y="62" font-family="Arial" font-size="12" fill="#475569">Blue=left, red=right, circle size=quantum influence</text>',
    ]
    for i, (x, y) in enumerate(zip(xs, ys)):
        color = "#2563eb" if connectome.hemispheres[i].startswith("left") else "#dc2626"
        name = connectome.node_names[i].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radii[i]:.2f}" fill="{color}" fill-opacity="0.75">'
            f"<title>{connectome.node_ids[i]} {name} {connectome.hemispheres[i]}</title></circle>"
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, results: list[dict], first_connectome: Connectome, first_result: dict, times: list[float]) -> None:
    metric_names = [
        "quantum_hemi_accuracy",
        "classical_hemi_accuracy",
        "quantum_best_x_corr",
        "classical_best_x_corr",
        "quantum_position_distance_corr",
        "classical_position_distance_corr",
        "interference_gain",
    ]
    means = {name: float(np.mean([row[name] for row in results])) for name in metric_names}
    top_idx = np.argsort(-first_result["quantum_influence"])[:8]
    top_lines = [
        f"- {rank}. {first_connectome.node_names[idx]} ({first_connectome.hemispheres[idx]}), "
        f"score={first_result['quantum_influence'][idx]:.4f}"
        for rank, idx in enumerate(top_idx, start=1)
    ]

    text = f"""# QRC Prototype Summary

Dataset: BrainGraph.org HCP 86-node connectomes

Subjects analyzed: {len(results)}

Times used: {", ".join(str(t) for t in times)}

## Core idea

Each brain region is treated as both a source and receiver. A quantum-like
continuous-time walk sends phase signals through the connectome. For every node,
we collect its response to all sources across several time scales, then compress
those relational responses into three coordinates.

This tests the hypothesis that global brain organization can emerge from local
relationships and wave-like propagation.

## Mean metrics

- Quantum-like hemisphere recovery: {means["quantum_hemi_accuracy"]:.3f}
- Classical diffusion hemisphere recovery: {means["classical_hemi_accuracy"]:.3f}
- Quantum-like best x-coordinate correlation: {means["quantum_best_x_corr"]:.3f}
- Classical best x-coordinate correlation: {means["classical_best_x_corr"]:.3f}
- Quantum-like 3D distance correlation: {means["quantum_position_distance_corr"]:.3f}
- Classical 3D distance correlation: {means["classical_position_distance_corr"]:.3f}
- Quantum/classical interference gain: {means["interference_gain"]:.3f}

## First subject

Subject: {first_connectome.subject_id}

Top quantum-influence regions:

{chr(10).join(top_lines)}

## Outputs

- `subject_metrics.csv`: per-subject metrics
- `embedding_first_subject.csv`: node labels, physical coordinates, QRC coordinates
- `node_scores_first_subject.csv`: highest quantum-influence nodes
- `qrc_first_subject.svg`: quick visual sketch of QRC coordinates
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_times(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QRC connectome prototype.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("research/data/brain_graph_hcp_86_nodes/graphml"),
        help="Directory containing BrainGraph GraphML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/outputs/qrc_demo"),
        help="Directory for generated metrics and reports.",
    )
    parser.add_argument("--subjects", type=int, default=30, help="Number of subjects to analyze.")
    parser.add_argument("--times", type=parse_times, default=parse_times("0.25,0.5,1,2,4,8"))
    parser.add_argument("--weight-transform", choices=["raw", "log1p", "binary"], default="log1p")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.graphml"))
    if not files:
        raise SystemExit(f"No GraphML files found in {args.input_dir}")
    selected = files[: args.subjects]

    results = []
    first_connectome = None
    first_result = None
    for path in selected:
        connectome = parse_graphml(path)
        result = analyze_connectome(connectome, times=args.times, weight_transform=args.weight_transform)
        results.append(result)
        if first_connectome is None:
            first_connectome = connectome
            first_result = result

    fields = [
        "subject_id",
        "nodes",
        "edges",
        "quantum_hemi_accuracy",
        "quantum_hemi_component",
        "classical_hemi_accuracy",
        "classical_hemi_component",
        "quantum_best_x_corr",
        "classical_best_x_corr",
        "quantum_best_y_corr",
        "classical_best_y_corr",
        "quantum_best_z_corr",
        "classical_best_z_corr",
        "quantum_position_distance_corr",
        "classical_position_distance_corr",
        "interference_gain",
    ]
    metric_rows = [{field: row[field] for field in fields} for row in results]
    write_csv(args.output_dir / "subject_metrics.csv", metric_rows, fields)
    write_embedding_csv(args.output_dir / "embedding_first_subject.csv", first_connectome, first_result)
    write_node_scores(args.output_dir / "node_scores_first_subject.csv", first_connectome, first_result)
    write_svg(args.output_dir / "qrc_first_subject.svg", first_connectome, first_result)
    write_summary(args.output_dir / "summary.md", results, first_connectome, first_result, args.times)

    print(f"Analyzed {len(results)} connectomes")
    print(f"Wrote outputs to {args.output_dir}")
    print(f"Mean quantum hemisphere accuracy: {np.mean([r['quantum_hemi_accuracy'] for r in results]):.3f}")
    print(f"Mean classical hemisphere accuracy: {np.mean([r['classical_hemi_accuracy'] for r in results]):.3f}")
    print(f"Mean interference gain: {np.mean([r['interference_gain'] for r in results]):.3f}")


if __name__ == "__main__":
    main()
