"""Training-only geometry, protected source policies, and decoder selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_quantum.analysis_suite import finite_position_mask
from brain_quantum.decoder_robustness import ALPHAS, MODES, RidgePath, transform
from brain_quantum.graph_propagation import component_support
from brain_quantum.methodology_controls import select_decoder_alpha
from brain_quantum.qrc_connectome import normalized_laplacian
from brain_quantum.source_coordinate_v2 import anchor_features, mean_coordinate_template

ENCODERS = ("qsf_full", "heat_raw_full", "heat_full")
FAMILIES = ("ordinary", "projection", "residual", "bounded_residual")
POLICIES = ("random", "graph_coverage", "spatial", "sfh_fixed", "spectral", "qsf_discrimination", "heat_discrimination")
POSITIVE = tuple(a for a in ALPHAS if a > 0)
RADII = (1., 2., 4.)
BUDGETS = (8, 16, 32)
NAMES = {"qsf_full": "qrc", "heat_full": "classical", "heat_raw_full": "classical_raw"}


def geometry_parameters(records, sources, fit):
    values, radii = [], []
    for i in fit:
        p = records[i].connectome.positions[sources]
        p = p[finite_position_mask(p)]
        if len(p):
            values.extend(p)
        if len(p) >= 2:
            r = float(np.sqrt(np.mean(np.sum((p-p.mean(axis=0))**2, axis=1))))
            if r > 1e-8:
                radii.append(r)
    center = np.mean(values, axis=0) if values else np.zeros(3)
    scale = float(np.median(radii)) if radii else 1.
    return center, scale


def source_frame(response, source_positions, fallback_center, fallback_scale):
    p = np.asarray(source_positions, dtype=float)
    valid = finite_position_mask(p)
    finite = p[valid]
    centroid = finite.mean(axis=0) if len(finite) else np.asarray(fallback_center)
    w = np.maximum(np.asarray(response, dtype=float)[:, valid], 0.)
    denominator = w.sum(axis=1)
    center = np.tile(centroid, (len(response), 1))
    supported = denominator > 0
    if len(finite):
        center[supported] = (w[supported] @ finite)/denominator[supported, None]
    radius = float(np.sqrt(np.mean(np.sum((finite-centroid)**2, axis=1)))) if len(finite) >= 2 else 0.
    fallback = radius <= 1e-8
    radius = fallback_scale if fallback else radius
    if radius <= 0 or not np.isfinite(center).all():
        raise ValueError("Source frame must have a finite center and positive scale.")
    return center, radius, int(not len(finite)), int(fallback), int((~supported).sum())


def project(prediction, centers, scales, radius):
    delta = prediction-centers
    distance = np.linalg.norm(delta, axis=-1)
    limit = np.broadcast_to(np.asarray(scales)*radius, distance.shape)
    clipped = distance > limit
    factor = np.minimum(1., np.divide(limit, distance, out=np.ones_like(distance), where=distance > 0))
    return centers+delta*factor[..., None], clipped


@dataclass
class Bundle:
    x: np.ndarray
    y: np.ndarray
    finite: np.ndarray
    extent: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    unsupported: np.ndarray
    fallback: np.ndarray

    def fit_arrays(self, indices):
        valid = self.finite[indices]
        x = self.x[indices][valid]
        y = self.y[indices]
        residual = (y-self.center[indices])/self.scale[indices, None, None]
        return x, np.concatenate([y, residual], axis=2)[valid]


def bundle(records, sources, targets, encoder, fit):
    prior, scale_prior = geometry_parameters(records, sources, fit)
    x, y, centers, scales, unsupported, fallback = [], [], [], [], [], []
    for r in records:
        f, response = anchor_features(r, sources, NAMES[encoder])
        b, s, no_sources, scale_fallback, no_response = source_frame(response[targets], r.connectome.positions[sources], prior, scale_prior)
        x.append(f[targets])
        y.append(r.connectome.positions[targets])
        centers.append(b)
        scales.append(s)
        reach = component_support(r.weights)[np.ix_(targets, sources[finite_position_mask(r.connectome.positions[sources])])]
        unsupported.append(float(np.mean(~reach.any(axis=1))))
        fallback.append([no_sources, scale_fallback, no_response])
    y = np.stack(y)
    finite = np.isfinite(y).all(axis=2)
    assert finite.any(axis=1).all()
    return Bundle(np.stack(x), y, finite, np.array([r.extent for r in records]), np.stack(centers),
                  np.array(scales), np.array(unsupported), np.array(fallback))


def reconstruct(raw, b, indices, family, radius):
    center, scale = b.center[indices], b.scale[indices, None]
    prediction = raw[..., :3] if family in ("ordinary", "projection") else center+scale[..., None]*raw[..., 3:]
    if family in ("projection", "bounded_residual"):
        return project(prediction, center, scale, radius)
    return prediction, np.zeros(prediction.shape[:2], dtype=bool)


def losses(prediction, b, indices):
    squared = np.sum((prediction-b.y[indices])**2, axis=2)
    squared = np.where(b.finite[indices], squared, 0.)
    mse = squared.sum(axis=1)/b.finite[indices].sum(axis=1)
    nmse = mse/b.extent[indices]**2
    return np.sqrt(nmse), nmse


def choose_families(grid):
    chosen = {}
    for family in FAMILIES:
        candidates = []
        for priority, mode in enumerate(MODES):
            radii = RADII if family in ("projection", "bounded_residual") else (0.,)
            for radius in radii:
                subset = [r for r in grid if r["family"] == family and r["preprocessing"] == mode and r["radius"] == radius]
                alpha = select_decoder_alpha([(r["validation_nrmse"], r["validation_se"], r["alpha"]) for r in subset], "one_se")
                row = next(r for r in subset if r["alpha"] == alpha)
                candidates.append((row["validation_nrmse"], priority, -radius, row))
        chosen[family] = min(candidates, key=lambda v: v[:3])[3]
    return chosen


def choose_track(candidates, baseline):
    """Use validation outcomes only; ordinary/random is the default, not test-best."""
    qualified = [r for r in candidates if r["validation_nrmse"] <= .99*baseline["validation_nrmse"]
                 and r["validation_nmse"] <= baseline["validation_nmse"]]
    if not qualified:
        return baseline
    best = min(r["validation_nrmse"] for r in qualified)
    near = [r for r in qualified if r["validation_nrmse"] <= 1.01*best]
    return min(near, key=lambda r: (FAMILIES.index(r["family"]), POLICIES.index(r.get("policy", "random")), r["validation_nrmse"]))


def fit_families(inner, final, fit, val, train):
    xi, yi = inner.fit_arrays(fit)
    xo, yo = final.fit_arrays(train)
    grid = []
    for mode in MODES:
        path = RidgePath(xi, yi, mode)
        xv = inner.x[val]
        for alpha in POSITIVE:
            model = path.model(alpha)
            raw = model.predict(xv.reshape(-1, xv.shape[-1])).reshape(len(val), xv.shape[1], 6)
            for family in FAMILIES:
                for radius in RADII if family in ("projection", "bounded_residual") else (0.,):
                    prediction, _ = reconstruct(raw, inner, val, family, radius)
                    nr, nm = losses(prediction, inner, val)
                    grid.append({"family": family, "preprocessing": mode, "alpha": alpha, "radius": radius,
                        "validation_nrmse": float(nr.mean()), "validation_nmse": float(nm.mean()),
                        "validation_se": float(nr.std(ddof=1)/np.sqrt(len(nr))), "validation_max": float(nr.max()),
                        "inner_rows": len(xi), "alpha_over_inner_N": alpha/len(xi)})
    selected = choose_families(grid)
    models, diagnostics, gate = {}, [], None
    for mode in sorted({r["preprocessing"] for r in selected.values()}):
        path = RidgePath(xo, yo, mode)
        for family, selection in selected.items():
            if selection["preprocessing"] != mode:
                continue
            alpha = selection["alpha"]
            model = path.model(alpha)
            models[family] = model
            columns = slice(0, 3) if family in ("ordinary", "projection") else slice(3, 6)
            diagnostics.append({**selection, "outer_rows": len(xo), "alpha_over_N": alpha/len(xo),
                "coefficient_frobenius": float(np.linalg.norm(model.coef[:, columns])),
                "smallest_singular": float(path.singular[-1]), "largest_singular": float(path.singular[0])})
            if family == "ordinary":
                factors = path.singular/(path.singular**2+alpha)
                gate = (model, path.vt.T*factors, 1./len(xo))
    return grid, selected, models, diagnostics, gate


def gate_scores(b, gate, train):
    model, basis, intercept = gate
    sensitivities, magnitudes = [], []
    for start in range(0, len(b.x), 64):
        x = b.x[start:start+64]
        z = transform((x-model.mean)/model.scale, model.mode)-model.transformed_mean
        flat = z.reshape(-1, z.shape[-1])
        leverage = np.sum((flat@basis)**2, axis=1).reshape(x.shape[:2])+intercept
        magnitude = np.mean(z**2, axis=2)
        sensitivities.extend(leverage.max(axis=1))
        magnitudes.extend(magnitude.max(axis=1))
    raw = np.column_stack([sensitivities, magnitudes])
    median = np.median(raw[train], axis=0)
    median[median <= 0] = 1.
    return raw/median, median


def farthest_sequence(distances, candidates, first):
    candidates = np.asarray(candidates, dtype=int)
    result = [int(first)]
    while len(result) < len(candidates):
        scores = distances[np.ix_(candidates, result)].min(axis=1).astype(float)
        scores[np.isin(candidates, result)] = -np.inf
        result.append(int(candidates[np.argmax(scores)]))
    return result


def discrimination_contributions(records, dense, targets, candidates, kind):
    """Fixed training-only squared signature differences for each candidate."""
    stacks = []
    for r, h in zip(records, dense):
        if kind == "qsf":
            a = np.stack([v for triplet in zip(r.q_real, r.q_imag, r.q_prob) for v in triplet], axis=-1)
        else:
            a = h.transpose(1, 2, 0)
        stacks.append(a[np.ix_(targets, candidates)].astype(float))
    a = np.stack(stacks)
    scales = np.sqrt(np.mean(a*a, axis=(0, 1, 2)))
    scales[scales < 1e-8] = 1.
    a /= scales
    u, v = np.triu_indices(len(targets), 1)
    contribution = np.zeros((len(candidates), len(u)))
    for sample in a:
        contribution += np.sum((sample[u]-sample[v])**2, axis=2).T
    contribution /= len(a)
    mean = contribution.mean()
    if mean > 0:
        contribution /= mean
    return contribution, scales


def discrimination_sequence(contribution, candidates):
    current = np.zeros(contribution.shape[1])
    remaining = list(range(len(candidates)))
    result, objectives = [], []
    while remaining:
        values = np.log1p(current[None, :]+contribution[remaining]).sum(axis=1)
        best = remaining[int(np.argmax(values))]
        current += contribution[best]
        result.append(int(candidates[best]))
        objectives.append(float(np.log1p(current).sum()))
        remaining.remove(best)
    return result, objectives


def spectral_sequence(weights, candidates, width=16, precision=.001):
    _, basis = np.linalg.eigh(normalized_laplacian(weights))
    basis = basis[:, :min(width, len(weights))]
    inverse = np.eye(basis.shape[1])/precision
    remaining = list(map(int, candidates))
    sequence = []
    while remaining:
        rows = basis[remaining]
        gains = np.einsum("ij,ij->i", rows@inverse, rows)
        increments = np.log1p(gains)
        tied = np.flatnonzero(np.isclose(increments, increments.max(), rtol=1e-10, atol=1e-12))
        node = min(remaining[i] for i in tied)
        v = inverse@basis[node]
        inverse -= np.outer(v, v)/(1.+basis[node]@v)
        sequence.append(node)
        remaining.remove(node)
    return sequence
