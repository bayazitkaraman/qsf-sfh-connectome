"""Compare QSF with heat diffusion using the same dynamic feature budget."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain_quantum.analysis_suite import finite_position_mask  # noqa: E402
from brain_quantum.graph_propagation import component_support, supported_spectral_matrix
from brain_quantum.methodology_controls import (  # noqa: E402
    METRIC_FIELDS,
    clustered_pair_summary,
    evaluate_prediction,
    fit_ridge_arrays,
    inner_subject_split,
    mean_summary,
    select_decoder_alpha,
)
from brain_quantum.qrc_connectome import normalized_laplacian, parse_times, write_csv  # noqa: E402
from brain_quantum.source_coordinate_v2 import (  # noqa: E402
    RidgeModel,
    SubjectRecord,
    anchor_features,
    cache_subject,
    non_anchor_mask,
    predict_ridge,
)


@dataclass
class DenseHeatRecord:
    subject: SubjectRecord
    raw_heat: np.ndarray


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dense_heat_kernels(record: SubjectRecord, times: list[float]) -> np.ndarray:
    """Return canonical heat kernels without storing a second normalized stack."""
    laplacian = normalized_laplacian(record.weights)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    support = component_support(laplacian)
    kernels = []
    for time in times:
        decay = np.exp(-eigenvalues * time)
        kernel = supported_spectral_matrix(eigenvectors, decay, support).real
        kernels.append(np.maximum(kernel, 0.0).astype(np.float32))
    return np.stack(kernels, axis=0)


def dense_heat_features(
    dense_record: DenseHeatRecord,
    anchors: np.ndarray,
    normalized: bool,
) -> np.ndarray:
    kernels = dense_record.raw_heat
    selected = kernels[:, :, anchors]
    if normalized:
        row_sums = kernels.sum(axis=2, keepdims=True)
        selected = selected / np.where(row_sums > 0, row_sums, 1.0)
    return selected.transpose(1, 0, 2).reshape(kernels.shape[1], -1).astype(np.float32)


def method_features(
    dense_record: DenseHeatRecord,
    anchors: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "qsf_full":
        features, _ = anchor_features(
            dense_record.subject,
            anchors,
            "qrc",
            qrc_variant="full",
            feature_mode="dynamic",
        )
        return features
    if method == "heat_dense":
        return dense_heat_features(dense_record, anchors, normalized=True)
    if method == "heat_raw_dense":
        return dense_heat_features(dense_record, anchors, normalized=False)
    raise ValueError(f"Unknown matched-capacity method: {method}")


def fit_method(
    records: list[DenseHeatRecord],
    anchors: np.ndarray,
    method: str,
    alpha: float,
) -> RidgeModel:
    x_parts = []
    y_parts = []
    for dense_record in records:
        record = dense_record.subject
        features = method_features(dense_record, anchors, method)
        mask = non_anchor_mask(features.shape[0], anchors)
        mask &= finite_position_mask(record.connectome.positions)
        x_parts.append(features[mask])
        y_parts.append(record.connectome.positions[mask])
    return fit_ridge_arrays(np.vstack(x_parts), np.vstack(y_parts), alpha)


def validation_summary(
    records: list[DenseHeatRecord],
    anchors: np.ndarray,
    method: str,
    model: RidgeModel,
) -> tuple[float, float]:
    mask = non_anchor_mask(len(records[0].subject.connectome.node_ids), anchors)
    values = []
    for dense_record in records:
        predicted = predict_ridge(model, method_features(dense_record, anchors, method))
        values.append(evaluate_prediction(dense_record.subject, predicted, mask, anchors)["normalized_rmse"])
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(array):
        return float("inf"), float("inf")
    se = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return float(np.mean(array)), se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=Path("research/outputs/dense_heat_capacity_hcp83_final"),
    )
    parser.add_argument("--subjects", type=int, default=1064)
    parser.add_argument("--anchor-count", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--qsf-times", type=parse_times, default=parse_times("0.25,0.5,1,2,4,8"))
    parser.add_argument(
        "--dense-heat-times",
        type=parse_times,
        default=np.geomspace(0.25, 8.0, 18).tolist(),
    )
    parser.add_argument("--weight-transform", default="log1p")
    parser.add_argument("--decoder-alpha-grid", default="0,0.001,0.01,0.1,1,10,25,100,1000")
    parser.add_argument("--inner-validation-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=4221)
    args = parser.parse_args()

    if len(args.dense_heat_times) != 3 * len(args.qsf_times):
        raise SystemExit("Dense heat must have three time scales per QSF time scale.")
    alphas = [float(value) for value in args.decoder_alpha_grid.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.graphml"))[: args.subjects]
    if not files:
        raise SystemExit(f"No GraphML files found in {args.input_dir}")

    print(f"Caching {len(files)} subjects and 18-scale heat kernels", flush=True)
    dense_records = []
    for index, path in enumerate(files, start=1):
        subject = cache_subject(path, args.qsf_times, args.weight_transform)
        dense_records.append(DenseHeatRecord(subject, dense_heat_kernels(subject, args.dense_heat_times)))
        if index % 100 == 0:
            print(f"  cached {index}/{len(files)}", flush=True)
    by_subject = {record.subject.connectome.subject_id: record for record in dense_records}
    reference = dense_records[0].subject.connectome
    name_to_index = {name: index for index, name in enumerate(reference.node_names)}

    metrics = read_csv(args.validation_dir / "validation_metrics.csv")
    anchor_rows = read_csv(args.validation_dir / "validation_anchor_sets.csv")
    test_subjects: dict[int, set[str]] = {}
    for row in metrics:
        if row["method"] == "qrc" and row["qrc_variant"] == "full" and row["mode"] == "global_ridge":
            test_subjects.setdefault(int(row["split_id"]), set()).add(row["subject_id"])
    selected_sets = [
        row
        for row in anchor_rows
        if int(row["anchor_count"]) == args.anchor_count and int(row["repeat"]) == args.repeat
    ]

    grid_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    methods = ["qsf_full", "heat_dense", "heat_raw_dense"]
    for source_set in selected_sets:
        split_id = int(source_set["split_id"])
        test_ids = test_subjects[split_id]
        train = [record for record in dense_records if record.subject.connectome.subject_id not in test_ids]
        test = [by_subject[subject_id] for subject_id in sorted(test_ids)]
        inner_train, inner_validation = inner_subject_split(
            train,
            args.inner_validation_fraction,
            np.random.default_rng(args.seed + split_id),
        )
        anchors = np.array(
            [name_to_index[name] for name in source_set["anchor_names"].split(" | ")],
            dtype=int,
        )
        non_source = non_anchor_mask(len(reference.node_ids), anchors)
        print(f"Matched capacity split {split_id}: sources={len(anchors)}", flush=True)
        for method in methods:
            scores = []
            for alpha in alphas:
                model = fit_method(inner_train, anchors, method, alpha)
                mean_rmse, se_rmse = validation_summary(
                    inner_validation,
                    anchors,
                    method,
                    model,
                )
                scores.append((mean_rmse, se_rmse, alpha))
                grid_rows.append(
                    {
                        "split_id": split_id,
                        "sample_id": source_set["sample_id"],
                        "method": method,
                        "alpha": alpha,
                        "inner_validation_normalized_rmse": mean_rmse,
                        "inner_validation_normalized_rmse_se": se_rmse,
                    }
                )
            selected_alpha = select_decoder_alpha(scores, "one_se")
            final_model = fit_method(train, anchors, method, selected_alpha)
            for dense_record in test:
                record = dense_record.subject
                features = method_features(dense_record, anchors, method)
                predicted = predict_ridge(final_model, features)
                row: dict[str, object] = {
                    "split_id": split_id,
                    "sample_id": source_set["sample_id"],
                    "subject_id": record.connectome.subject_id,
                    "anchor_count": len(anchors),
                    "method": method,
                    "feature_count": features.shape[1],
                    "selected_alpha": selected_alpha,
                }
                row.update(evaluate_prediction(record, predicted, non_source, anchors))
                test_rows.append(row)

    write_csv(args.output_dir / "dense_heat_validation_grid.csv", grid_rows, list(grid_rows[0]))
    write_csv(args.output_dir / "dense_heat_capacity_metrics.csv", test_rows, list(test_rows[0]))
    summary = mean_summary(test_rows, ["anchor_count", "method", "feature_count"])
    write_csv(args.output_dir / "dense_heat_capacity_summary.csv", summary, list(summary[0]))
    paired = []
    for heat_method in ["heat_dense", "heat_raw_dense"]:
        paired.extend(
            clustered_pair_summary(
                test_rows,
                ["anchor_count"],
                "qsf_full",
                heat_method,
                args.bootstrap_samples,
                np.random.default_rng(args.seed + len(paired)),
            )
        )
    write_csv(args.output_dir / "dense_heat_capacity_clustered_pairs.csv", paired, list(paired[0]))
    print(f"Wrote matched-capacity controls to {args.output_dir}")


if __name__ == "__main__":
    main()
