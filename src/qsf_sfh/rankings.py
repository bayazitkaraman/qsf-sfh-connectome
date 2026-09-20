"""Recompute fitting-only source rankings using the original scoring functions."""
import hashlib
import numpy as np
from brain_quantum.analysis_suite import finite_position_mask
from brain_quantum.mechanism_suite import node_descriptors
from brain_quantum.qrc_hierarchy_suite import graph_coverage_ranking, nested_random_ranking
from brain_quantum.source_coordinate_v2 import mean_coordinate_template
from .core import POLICIES, discrimination_contributions, discrimination_sequence, farthest_sequence, spectral_sequence

def rankings(records, dense, fit, targets, candidates, split, mask):
    rr = [records[i] for i in fit]
    hh = [dense[i] for i in fit]
    reference = records[0].connectome
    mean_weights = np.mean([r.weights for r in rr], axis=0)
    distances = np.mean([r.shortest_paths for r in rr], axis=0)
    first = int(candidates[np.argmax(mean_weights.sum(axis=1)[candidates])])
    template = mean_coordinate_template(rr)
    positions = template[candidates]
    assert finite_position_mask(positions).all()
    spatial_distances = np.linalg.norm(template[:,None,:]-template[None,:,:], axis=2)
    order = nested_random_ranking(reference.hemispheres, np.random.default_rng(18071+100*split+mask), candidates)
    result = {"random": [i for i in order if i in set(candidates)],
              "graph_coverage": farthest_sequence(distances, candidates, first),
              "spatial": farthest_sequence(spatial_distances, candidates, int(candidates[np.argmin(positions[:,0])])),
              "spectral": spectral_sequence(mean_weights, candidates)}
    config = {"strength_weight": .5, "resid_influence_weight": .25, "resid_entropy_weight": .25,
              "cross_weight": 0., "redundancy_weight": .45, "coverage_weight": 1., "balance_weight": .30}
    order = graph_coverage_ranking(node_descriptors(rr), rr, config, candidate_indices=candidates)
    result["sfh_fixed"] = [i for i in order if i in set(candidates)]
    positioned = np.any([finite_position_mask(r.connectome.positions) for r in rr], axis=0)
    active = targets[positioned[targets]]
    diagnostics = {"active_target_labels": active.tolist(), "fit_people": [r.connectome.subject_id for r in rr],
                   "contribution_details": {}}
    for kind in ("qsf", "heat"):
        contribution, scales = discrimination_contributions(rr, hh, active, candidates, kind)
        order, objective = discrimination_sequence(contribution, candidates)
        result[kind+"_discrimination"] = order
        diagnostics["contribution_details"][kind] = {"scales": scales.tolist(), "objective": objective,
            "contribution_sha256": hashlib.sha256(contribution.tobytes()).hexdigest(),
            "constant_pairs": int(np.count_nonzero(contribution.sum(axis=0) == 0))}
    for policy in POLICIES:
        assert set(result[policy]) == set(candidates) and len(result[policy]) == len(candidates)
        assert not set(result[policy]) & set(targets)
    return result, diagnostics
