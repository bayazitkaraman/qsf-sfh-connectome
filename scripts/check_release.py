"""Verify code/configuration and aggregate outputs without downloading data."""
from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from qsf_sfh.data import ROOT, check_release, digest, load_design


def main():
    check_release()
    design = load_design()
    for cohort, n in [('hcp83',1062),('oasis',695)]:
        config = design['cohorts'][cohort]
        assert len(config['graph_sha256']) == n
        for split, parts in config['partitions'].items():
            assert sorted(sum([parts[k] for k in ('fit','validation','calibration','test')],[])) == list(range(n))
            assert parts['train'] == sorted(parts['fit']+parts['validation'])
            for mask, targets in enumerate(design['target_masks'][cohort],1):
                ranks = config['rankings'][f's{split}_m{mask}']
                candidates = set(ranks['random'])
                assert not candidates.intersection(targets)
                assert len(candidates) >= 32
                assert all(len(v)==len(candidates) and set(v)==candidates for v in ranks.values())
    expected = json.loads((ROOT/'results/manifest.json').read_text())
    for name, sha in expected['sha256'].items():
        assert digest(ROOT/'results'/name) == sha, name
        with (ROOT/'results'/name).open(newline='') as handle:
            headers = next(csv.reader(handle))
        assert not {'subject_id','participant_id','patient_id'}.intersection(headers), name
    for name, sha in expected.get('verification_sha256', {}).items():
        assert digest(ROOT/'results'/name) == sha, name
    print('PASS: release hashes, 1757 input-hash entries, partition/mask separation, frozen source rankings, aggregate outputs, and verification records.')


if __name__ == '__main__':
    main()
