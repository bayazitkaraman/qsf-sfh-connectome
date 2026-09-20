"""Post hoc, training-only preprocessing sensitivities for source-field decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_quantum.analysis_suite import finite_position_mask
from brain_quantum.dense_heat_capacity_control import DenseHeatRecord, dense_heat_features, dense_heat_kernels
from brain_quantum.graph_propagation import component_support
from brain_quantum.methodology_controls import (
    METRIC_FIELDS, clustered_pair_summary, evaluate_prediction, inner_subject_split,
    mean_summary, read_csv, select_decoder_alpha,
)
from brain_quantum.qrc_connectome import write_csv
from brain_quantum.source_coordinate_v2 import (
    anchor_features, cache_subject, choose_balanced_anchors, eligible_source_indices, non_anchor_mask,
)

MODES = ("standard", "asinh", "tanh3")
METHODS = ("qsf_full", "heat_full", "heat_raw_full", "qsf_dynamic", "heat_dense", "heat_raw_dense")
ALPHAS = (0., .001, .01, .1, 1., 10., 25., 100., 1000.)


def transform(z: np.ndarray, mode: str) -> np.ndarray:
    if mode == "standard":
        return z
    if mode == "asinh":
        return np.arcsinh(z)
    if mode == "tanh3":
        return 3 * np.tanh(z / 3)
    raise ValueError(f"Unknown preprocessing: {mode}")


@dataclass
class Decoder:
    mean: np.ndarray
    scale: np.ndarray
    transformed_mean: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray
    mode: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(x)):
            raise ValueError("Decoder prediction requires finite features.")
        z = transform((np.asarray(x, dtype=float) - self.mean) / self.scale, self.mode)
        return (z - self.transformed_mean) @ self.coef + self.y_mean


class RidgePath:
    """One training-design SVD supports every penalty without test-data access."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mode: str):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or not len(x) or len(x) != len(y):
            raise ValueError("Expected nonempty, row-aligned feature and target arrays.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Training arrays must be finite.")
        self.mode = mode
        self.mean = x.mean(axis=0, keepdims=True)
        self.scale = x.std(axis=0, keepdims=True)
        self.scale[self.scale < 1e-8] = 1.
        z = transform((x - self.mean) / self.scale, mode)
        self.transformed_mean = z.mean(axis=0, keepdims=True)
        z -= self.transformed_mean
        self.y_mean = y.mean(axis=0, keepdims=True)
        u, self.singular, self.vt = np.linalg.svd(z, full_matrices=False)
        self.projected_y = u.T @ (y - self.y_mean)
        self.cutoff = np.finfo(float).eps * max(z.shape) * self.singular[0]

    def model(self, alpha: float) -> Decoder:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Ridge penalty must be finite and nonnegative.")
        if alpha == 0:
            weights = np.divide(1., self.singular, out=np.zeros_like(self.singular),
                                where=self.singular > self.cutoff)
        else:
            weights = self.singular / (self.singular ** 2 + alpha)
        coef = (self.vt.T * weights) @ self.projected_y
        return Decoder(self.mean, self.scale, self.transformed_mean, self.y_mean, coef, self.mode)


def features(record, dense, anchors, method):
    if method in {"heat_dense", "heat_raw_dense"}:
        return dense_heat_features(DenseHeatRecord(record, dense), anchors, method == "heat_dense")
    name = {"qsf_full": "qrc", "qsf_dynamic": "qrc", "heat_full": "classical", "heat_raw_full": "classical_raw"}[method]
    mode = "dynamic" if method == "qsf_dynamic" else "full"
    return anchor_features(record, anchors, name, "full", mode)[0]


def distribution_summary(rows):
    output = mean_summary(rows, ["preprocessing", "method"])
    for summary in output:
        selected = [r for r in rows if all(r[k] == summary[k] for k in ("preprocessing", "method"))]
        values = np.array([r["normalized_rmse"] for r in selected])
        by_subject = defaultdict(list)
        for row in selected:
            by_subject[row["subject_id"]].append(row["normalized_rmse"])
        summary.update({"median_nrmse": float(np.median(values)), "p95_nrmse": float(np.quantile(values, .95)),
                        "p99_nrmse": float(np.quantile(values, .99)), "max_nrmse": float(values.max()),
                        "rms_of_subject_split_nrmse": float(np.sqrt(np.mean(values ** 2))),
                        "equal_subject_mean_nrmse": float(np.mean([np.mean(v) for v in by_subject.values()])),
                        "equal_subject_root_mean_normalized_mse": float(np.sqrt(np.mean([np.mean(np.square(v)) for v in by_subject.values()]))),
                        "rows_over_one": int(np.count_nonzero(values > 1)), "unique_subjects": len(by_subject)})
    return output


def choose_preprocessing(validation_rows):
    """Select within training validation only, with a fixed deterministic tie order."""
    candidates = []
    for priority, mode in enumerate(MODES):
        selected = [r for r in validation_rows if r["preprocessing"] == mode]
        alpha = select_decoder_alpha([(float(r["validation_nrmse"]), float(r["validation_se"]), float(r["alpha"]))
                                      for r in selected], "one_se")
        score = next(float(r["validation_nrmse"]) for r in selected if float(r["alpha"]) == alpha)
        candidates.append((score, priority, mode, alpha))
    _, _, mode, alpha = min(candidates)
    return mode, alpha


def write_summaries(output_dir, metrics, grids):
    selected_rows, selections = [], []
    keys = sorted({(r["split_id"], r["sample_id"], r["method"]) for r in grids})
    for split, sample, method in keys:
        grid = [r for r in grids if (r["split_id"], r["sample_id"], r["method"]) == (split, sample, method)]
        mode, alpha = choose_preprocessing(grid)
        selections.append({"split_id": split, "sample_id": sample, "method": method, "selected_preprocessing": mode, "selected_alpha": alpha})
        chosen = [r for r in metrics if (r["split_id"], r["sample_id"], r["method"], r["preprocessing"]) == (split, sample, method, mode)]
        assert chosen and all(float(r["selected_alpha"]) == alpha for r in chosen)
        selected_rows.extend({**r, "preprocessing": "validation_selected", "selected_preprocessing": mode} for r in chosen)
    write_csv(output_dir / "preprocessing_selections.csv", selections, list(selections[0]))
    write_csv(output_dir / "validation_selected_metrics.csv", selected_rows, list(selected_rows[0]))
    all_rows = metrics + selected_rows
    summary = distribution_summary(all_rows)
    write_csv(output_dir / "robustness_summary.csv", summary, list(summary[0]))
    comparisons = (("qsf_full", "heat_full"), ("qsf_full", "heat_raw_full"), ("qsf_dynamic", "heat_dense"), ("qsf_dynamic", "heat_raw_dense"))
    pairs = []
    for a, b in comparisons:
        pairs.extend(clustered_pair_summary(all_rows, ["anchor_count", "preprocessing"], a, b, 5000,
                                            np.random.default_rng(6801 + len(pairs))))
    write_csv(output_dir / "robustness_clustered_pairs.csv", pairs, list(pairs[0]))
    squared = [{**r, "normalized_rmse": r["normalized_rmse"] ** 2} for r in all_rows]
    squared_pairs = []
    for a, b in comparisons:
        selected = clustered_pair_summary(squared, ["anchor_count", "preprocessing"], a, b, 5000,
                                          np.random.default_rng(6901 + len(squared_pairs)))
        for row in selected:
            if row["metric"] == "normalized_rmse":
                squared_pairs.append({**row, "metric": "normalized_mse"})
    write_csv(output_dir / "robustness_squared_error_pairs.csv", squared_pairs, list(squared_pairs[0]))


def external_source_sets(records, output_dir):
    rng = np.random.default_rng(8021)
    source_rows, test_membership = [], []
    reference = records[0].connectome
    for split in range(1, 6):
        order = rng.permutation(len(records))
        train_count = int(round(.8 * len(records)))
        train = [records[i] for i in order[:train_count]]
        eligible = eligible_source_indices(train)
        relative = choose_balanced_anchors([reference.hemispheres[i] for i in eligible], 32, rng)
        anchors = eligible[relative]
        if len(anchors) != 32:
            raise ValueError("External source sampling did not produce exactly 32 sources.")
        source_rows.append({"split_id": split, "sample_id": f"external-{split}-32", "repeat": 0, "anchor_count": 32,
                            "anchor_names": " | ".join(reference.node_names[i] for i in anchors), "training_eligible_labels": len(eligible)})
        test_membership.extend({"split_id": split, "subject_id": records[i].connectome.subject_id, "method": "qrc", "mode": "global_ridge"}
                               for i in order[train_count:])
    write_csv(output_dir / "external_source_sets.csv", source_rows, list(source_rows[0]))
    write_csv(output_dir / "external_test_membership.csv", test_membership, list(test_membership[0]))
    return source_rows, test_membership


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("research/data/brain_graph_hcp_86_nodes/graphml"))
    parser.add_argument("--validation-dir", type=Path, default=Path("research/outputs/qrc_v2_validation"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=1064)
    parser.add_argument("--seed", type=int, default=4221)
    parser.add_argument("--external-random-sources", action="store_true",
                        help="Generate five outer splits and training-eligible sources with frozen outer seed 8021.")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        metrics = read_csv(args.output_dir / "robustness_metrics.csv")
        grids = read_csv(args.output_dir / "robustness_validation_grid.csv")
        for row in metrics:
            for key in ("split_id", "anchor_count", "feature_count"):
                row[key] = int(row[key])
            for key in (*METRIC_FIELDS, "selected_alpha"):
                row[key] = float(row[key])
        for row in grids:
            row["split_id"] = int(row["split_id"])
        write_summaries(args.output_dir, metrics, grids)
        (args.output_dir / "POSTPROCESS.json").write_text(json.dumps({"command": sys.argv, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "selection_uses": "validation scores only"}, indent=2) + "\n")
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    environment = {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                   "numpy_config": np.show_config(mode="dicts"), "command": sys.argv,
                   "thread_settings": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                   "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                   "scope": "Post hoc sensitivity; all preprocessing modes reported; penalties chosen using inner validation only"}
    (args.output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    files = sorted(args.input_dir.glob("*.graphml"))[:args.subjects]
    if len(files) != args.subjects:
        raise ValueError("Requested subject count is unavailable.")
    records, kernels = [], []
    for index, path in enumerate(files, 1):
        record = cache_subject(path, [.25, .5, 1, 2, 4, 8], "log1p")
        records.append(record)
        kernels.append(dense_heat_kernels(record, np.geomspace(.25, 8., 18).tolist()))
        if index % 100 == 0:
            print(f"Cached {index}/{len(files)}", flush=True)
    lookup = {name: i for i, name in enumerate(records[0].connectome.node_names)}
    assert all(r.connectome.node_names == records[0].connectome.node_names for r in records)
    if args.external_random_sources:
        source_rows, saved = external_source_sets(records, args.output_dir)
    else:
        saved = read_csv(args.validation_dir / "validation_metrics.csv")
        source_rows = [r for r in read_csv(args.validation_dir / "validation_anchor_sets.csv")
                       if r["anchor_count"] == "32" and r["repeat"] == "0"]
    metrics, grids, strata, memberships = [], [], [], []
    for source in source_rows:
        split = int(source["split_id"])
        test_ids = {r["subject_id"] for r in saved if int(r["split_id"]) == split and r["method"] == "qrc" and r["mode"] == "global_ridge"}
        train_indices = [i for i, r in enumerate(records) if r.connectome.subject_id not in test_ids]
        test_indices = [i for i, r in enumerate(records) if r.connectome.subject_id in test_ids]
        inner_train, inner_val = inner_subject_split(train_indices, .2, np.random.default_rng(args.seed + split))
        for partition, indices in (("inner_train", inner_train), ("inner_validation", inner_val), ("outer_test", test_indices)):
            memberships.extend({"split_id": split, "partition": partition, "subject_id": records[i].connectome.subject_id} for i in indices)
        anchors = np.array([lookup[n] for n in source["anchor_names"].split(" | ")])
        mask = non_anchor_mask(len(lookup), anchors)
        masks = [mask & finite_position_mask(r.connectome.positions) for r in records]
        y_inner = np.vstack([records[i].connectome.positions[masks[i]] for i in inner_train])
        y_train = np.vstack([records[i].connectome.positions[masks[i]] for i in train_indices])
        topology = {}
        for i in test_indices:
            support = component_support(records[i].weights)
            positioned_sources = anchors[finite_position_mask(records[i].connectome.positions)[anchors]]
            reach = support[:, positioned_sources].any(axis=1)
            size = support.sum(axis=1)
            topology[i] = (reach, size)
        for method in METHODS:
            all_features = [features(r, k, anchors, method) for r, k in zip(records, kernels)]
            x_inner = np.vstack([all_features[i][masks[i]] for i in inner_train])
            x_train = np.vstack([all_features[i][masks[i]] for i in train_indices])
            for mode in MODES:
                path = RidgePath(x_inner, y_inner, mode)
                scores = []
                for alpha in ALPHAS:
                    model = path.model(alpha)
                    values = []
                    for i in inner_val:
                        error = np.linalg.norm(model.predict(all_features[i])[masks[i]] - records[i].connectome.positions[masks[i]], axis=1)
                        values.append(float(np.sqrt(np.mean(error ** 2)) / records[i].extent))
                    mean, se = float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))
                    scores.append((mean, se, alpha))
                    grids.append({"split_id": split, "sample_id": source["sample_id"], "method": method, "preprocessing": mode,
                                  "alpha": alpha, "validation_nrmse": mean, "validation_se": se})
                alpha = select_decoder_alpha(scores, "one_se")
                del path
                model = RidgePath(x_train, y_train, mode).model(alpha)
                for i in test_indices:
                    record = records[i]
                    predicted = model.predict(all_features[i])
                    row = {"split_id": split, "sample_id": source["sample_id"], "subject_id": record.connectome.subject_id,
                           "anchor_count": len(anchors), "method": method, "preprocessing": mode, "selected_alpha": alpha,
                           "feature_count": all_features[i].shape[1]}
                    row.update(evaluate_prediction(record, predicted, mask, anchors))
                    metrics.append(row)
                    errors = np.linalg.norm(predicted - record.connectome.positions, axis=1) / record.extent
                    reach, size = topology[i]
                    for reachable in (False, True):
                        for label, selection in (("isolated", size == 1), ("2_to_5", (size >= 2) & (size <= 5)), ("6_or_more", size >= 6)):
                            allowed = masks[i] & (reach == reachable) & selection
                            if allowed.any():
                                strata.append({"split_id": split, "subject_id": record.connectome.subject_id, "method": method,
                                               "preprocessing": mode, "reachable_positioned_source": reachable, "component_size": label,
                                               "target_count": int(allowed.sum()), "normalized_rmse": float(np.sqrt(np.mean(errors[allowed] ** 2))),
                                               "maximum_target_normalized_error": float(errors[allowed].max())})
                print(f"Split {split}, {method}, {mode}: alpha={alpha}", flush=True)
                for name, rows in (("robustness_metrics", metrics), ("robustness_validation_grid", grids), ("component_strata", strata), ("split_membership", memberships)):
                    write_csv(args.output_dir / f"{name}.csv", rows, list(rows[0]))
    write_summaries(args.output_dir, metrics, grids)
    (args.output_dir / "COMPLETE.json").write_text(json.dumps({"metric_rows": len(metrics), "validation_rows": len(grids), "modes": MODES, "methods": METHODS}, indent=2) + "\n")


if __name__ == "__main__":
    main()
