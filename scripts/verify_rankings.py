"""Optionally recompute every fitting-only ranking from verified local graphs."""
import argparse
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from brain_quantum.dense_heat_capacity_control import dense_heat_kernels
from brain_quantum.source_coordinate_v2 import eligible_source_indices
from qsf_sfh.data import check_release, load_design, load_records
from qsf_sfh.rankings import rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, choices=('hcp83','oasis'))
    args = parser.parse_args()
    check_release()
    design = load_design()
    cohort = design['cohorts'][args.cohort]
    records = load_records(args.cohort)
    dense = [dense_heat_kernels(r,np.geomspace(.25,8.,18).tolist()) for r in records]
    for split in range(1,6):
        fit = cohort['partitions'][str(split)]['fit']
        eligible = eligible_source_indices([records[i] for i in fit])
        for mask, targets in enumerate(design['target_masks'][args.cohort],1):
            targets = np.array(targets)
            candidates = eligible[~np.isin(eligible,targets)]
            actual, _ = rankings(records,dense,fit,targets,candidates,split,mask)
            if actual != cohort['rankings'][f's{split}_m{mask}']:
                raise ValueError(f'Ranking mismatch at split {split}, mask {mask}; recorded rankings were not overwritten.')
            print(args.cohort,split,mask,'rankings match',flush=True)


if __name__ == '__main__':
    main()
