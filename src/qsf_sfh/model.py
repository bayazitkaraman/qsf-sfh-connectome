"""Restricted, feature-matched consolidation using existing numerical helpers."""
from __future__ import annotations

import numpy as np

from brain_quantum.analysis_suite import response_coordinate_moments
from brain_quantum.decoder_robustness import Decoder, RidgePath
from brain_quantum.methodology_controls import select_decoder_alpha
from .core import Bundle, FAMILIES, POSITIVE, RADII, losses, reconstruct, source_frame

ENCODERS = ('qsf', 'heat_raw18', 'heat_norm18')
POLICIES = ('random', 'graph_coverage', 'sfh_fixed', 'spectral',
            'qsf_discrimination', 'heat_discrimination')


def features(stack, source_positions, sources, targets, encoder):
    """Propagation is normalized over full rows before selecting sources."""
    if encoder == 'heat_norm18':
        sums = stack.sum(axis=2, keepdims=True)
        stack = stack / np.where(sums > 0, sums, 1.)
    selected = stack[:, targets][:, :, sources]
    response = selected[2::3].mean(axis=0) if encoder == 'qsf' else selected.mean(axis=0)
    base = selected.transpose(1, 0, 2).reshape(len(targets), -1)
    mean, spread = response_coordinate_moments(response, source_positions)
    result = np.concatenate([base, mean, spread], axis=1).astype(np.float32)
    assert result.shape == (len(targets), 18*len(sources)+6)
    return result, response


def geometry_prior(positions, sources, fit):
    values, radii = [], []
    for i in fit:
        p = positions[i, sources]
        p = p[np.isfinite(p).all(axis=1)]
        if len(p):
            values.extend(p)
        if len(p) >= 2:
            r = np.sqrt(np.mean(np.sum((p-p.mean(axis=0))**2, axis=1)))
            if r > 1e-8:
                radii.append(r)
    return (np.mean(values, axis=0) if values else np.zeros(3),
            float(np.median(radii)) if radii else 1.)


def make_bundle(data, sources, targets, encoder, fit, template=None):
    positions = data['positions']
    prior, radius = geometry_prior(positions, sources, fit)
    xx, bb, ss, fallbacks = [], [], [], []
    stack_name = 'qsf' if encoder == 'qsf' else 'heat'
    for i in range(len(positions)):
        stack = data[stack_name][i] if template is None else template[stack_name]
        x, response = features(stack, positions[i, sources], sources, targets, encoder)
        b, s, no_sources, small_radius, no_response = source_frame(
            response, positions[i, sources], prior, radius)
        xx.append(x)
        bb.append(b)
        ss.append(s)
        fallbacks.append([no_sources, small_radius, no_response])
    y = positions[:, targets]
    finite = np.isfinite(y).all(axis=2)
    if not finite.any(axis=1).all():
        raise ValueError('Every participant must have a finite protected target.')
    return Bundle(np.stack(xx), y, finite, data['extent'], np.stack(bb), np.array(ss),
                  np.zeros(len(y)), np.array(fallbacks))


def grouped_scale(x, source_count, epsilon=.1):
    std = x.std(axis=0)
    if x.shape[1] != 18*source_count+6:
        raise ValueError('Feature grouping does not match the encoder layout.')
    groups = [np.arange(t*source_count, (t+1)*source_count) for t in range(18)]
    groups += [np.arange(18*source_count, 18*source_count+3),
               np.arange(18*source_count+3, 18*source_count+6)]
    scale = np.empty_like(std)
    for indices in groups:
        group = np.sqrt(np.mean(std[indices]**2))
        if group == 0:
            group = np.sqrt(np.mean(x[:, indices]**2))
        if group == 0:
            group = 1.
        scale[indices] = np.sqrt(std[indices]**2+(epsilon*group)**2)
    return scale[None, :]


class FloorPath(RidgePath):
    """Same direct-design SVD, with a fixed training-only grouped scale floor."""
    def __init__(self, x, y, source_count):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite fitting arrays.')
        self.mode = 'standard'
        self.mean = x.mean(axis=0, keepdims=True)
        self.scale = grouped_scale(x, source_count)
        z = (x-self.mean)/self.scale
        self.transformed_mean = z.mean(axis=0, keepdims=True)
        z -= self.transformed_mean
        self.y_mean = y.mean(axis=0, keepdims=True)
        u, self.singular, self.vt = np.linalg.svd(z, full_matrices=False)
        self.projected_y = u.T @ (y-self.y_mean)
        self.cutoff = np.finfo(float).eps*max(z.shape)*self.singular[0]


def path(x, y, scaling, count):
    return RidgePath(x, y, 'tanh3') if scaling == 'tanh3' else FloorPath(x, y, count)


def choose_families(grid):
    result = {}
    for family in FAMILIES:
        candidates = []
        for radius in RADII if family in ('projection', 'bounded_residual') else (0.,):
            rows = [r for r in grid if r['family'] == family and r['radius'] == radius]
            alpha = select_decoder_alpha([(r['validation_nrmse'], r['validation_se'], r['alpha'])
                                          for r in rows], 'one_se')
            candidates.append(next(r for r in rows if r['alpha'] == alpha))
        result[family] = min(candidates, key=lambda r: (r['validation_nrmse'], -r['radius']))
    return result


def robust_choice(families):
    p, r = families['projection'], families['bounded_residual']
    if (p['validation_nrmse'] <= 1.01*r['validation_nrmse'] and
            p['validation_nmse'] <= 1.01*r['validation_nmse']):
        return p
    return r if r['validation_nrmse'] < p['validation_nrmse'] else p


def policy_choice(rows):
    by_policy = {r['policy']: r for r in rows}
    baseline = by_policy['random']
    qualified = [r for r in rows if r['validation_nrmse'] <= .99*baseline['validation_nrmse']
                 and r['validation_nmse'] <= baseline['validation_nmse']]
    if not qualified:
        return baseline
    best = min(r['validation_nrmse'] for r in qualified)
    return min((r for r in qualified if r['validation_nrmse'] <= 1.01*best),
               key=lambda r: POLICIES.index(r['policy']))


def fit(inner, final, fit_ids, val, train, count, scaling):
    x, y = inner.fit_arrays(fit_ids)
    p = path(x, y, scaling, count)
    xv = inner.x[val]
    grid = []
    for alpha in POSITIVE:
        raw = p.model(alpha).predict(xv.reshape(-1, xv.shape[-1])).reshape(len(val), xv.shape[1], 6)
        for family in FAMILIES:
            for radius in RADII if family in ('projection', 'bounded_residual') else (0.,):
                prediction, _ = reconstruct(raw, inner, val, family, radius)
                nr, nm = losses(prediction, inner, val)
                grid.append(dict(family=family, alpha=alpha, radius=radius,
                                 validation_nrmse=float(nr.mean()), validation_nmse=float(nm.mean()),
                                 validation_se=float(nr.std(ddof=1)/np.sqrt(len(nr))),
                                 alpha_over_inner_N=alpha/len(x)))
    chosen = choose_families(grid)
    xo, yo = final.fit_arrays(train)
    final_path = path(xo, yo, scaling, count)
    models = {family: final_path.model(row['alpha']) for family, row in chosen.items()}
    diagnostics = dict(feature_count=xo.shape[1], fit_rows=len(x), refit_rows=len(xo),
                       smallest_singular=float(final_path.singular[-1]),
                       largest_singular=float(final_path.singular[0]),
                       alpha_over_N={f: r['alpha']/len(xo) for f, r in chosen.items()})
    return grid, chosen, models, diagnostics


def projection_diagnostics(raw, prediction, truth, centers, scales, radius, finite, extent):
    """Evaluate the identical fitted prediction before and after projection."""
    distance = np.linalg.norm(truth-centers, axis=2)
    excess = np.maximum(0., distance-radius*scales[:, None])/extent[:, None]
    before = np.linalg.norm(raw-truth, axis=2)/extent[:, None]
    after = np.linalg.norm(prediction-truth, axis=2)/extent[:, None]
    inside = distance <= radius*scales[:, None]
    if np.any((after > before+1e-10) & inside & finite):
        raise AssertionError('Euclidean projection worsened an inside-ball target.')
    if np.any((after+1e-10 < excess) & finite):
        raise AssertionError('Bounded prediction violates the unavoidable error floor.')
    return before, after, excess, inside
