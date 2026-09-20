"""Portable, hash-checked inputs for the frozen consolidation design."""
from pathlib import Path
import hashlib
import json
import os

from brain_quantum.source_coordinate_v2 import cache_subject

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'configs/protocol.txt'
WORK = Path(os.environ.get('QSF_WORKDIR', ROOT/'work')).resolve()


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_design():
    return json.loads((ROOT/'configs/design.json').read_text(encoding='utf-8'))


def check_release():
    manifest = json.loads((ROOT/'configs/release_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Release file changed: '+name)
    if digest(PROTOCOL) != manifest['protocol_sha256']:
        raise ValueError('Protocol digest mismatch.')
    return manifest


def locate_graphs(cohort, directory=None):
    expected = load_design()['cohorts'][cohort]['graph_sha256']
    if len(expected) != len(set(expected)):
        raise ValueError('The recorded input list contains duplicate graph content.')
    directory = Path(directory or os.environ.get('QSF_'+cohort.upper()+'_DATA', ROOT/'data'/cohort))
    if not directory.is_dir():
        raise FileNotFoundError(f'Place the original GraphML files in {directory} or set QSF_{cohort.upper()}_DATA.')
    needed = set(expected)
    matched = {}
    for path in sorted(directory.rglob('*.graphml')):
        value = digest(path)
        if value in needed:
            matched.setdefault(value, path)
    missing = needed-set(matched)
    if missing:
        raise ValueError(f'{cohort}: {len(missing)} required graph hashes are missing. No reduced-cohort run will be substituted.')
    return [matched[value] for value in expected]


def load_records(cohort):
    records = []
    for index, path in enumerate(locate_graphs(cohort)):
        record = cache_subject(path, [.25,.5,1.,2.,4.,8.], 'log1p')
        # Ordered row keys preserve the original sort order without publishing IDs.
        record.connectome.subject_id = f'row{index:04d}'
        records.append(record)
        if (index+1) % 250 == 0:
            print(f'{cohort}: prepared {index+1} graphs', flush=True)
    reference = records[0].connectome
    if any(r.connectome.node_ids != reference.node_ids or
           r.connectome.node_names != reference.node_names or
           r.connectome.hemispheres != reference.hemispheres for r in records):
        raise ValueError('Atlas correspondence differs across input graphs.')
    return records
