"""Run the fixed additional controls after preparing the original cohort responses."""
import argparse
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from qsf_sfh import controls
from qsf_sfh.data import check_release, load_design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('check','prepare','fit','summarize','all'))
    parser.add_argument('--cohort',choices=('hcp83','oasis'))
    parser.add_argument('--workers',type=int,default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.cohort and args.stage in ('summarize','all'):
        parser.error('Complete reporting requires both cohorts; omit --cohort.')
    check_release()
    if args.stage == 'check':
        print(f'Verified release and {len(controls.plans(load_design()))} planned controls. No data required.')
        return
    for cohort in ([args.cohort] if args.cohort else ['hcp83','oasis']):
        if args.stage in ('prepare','all'):
            controls.prepare(cohort)
        if args.stage in ('fit','all'):
            controls.run(cohort,args.workers)
    if args.stage in ('summarize','all'):
        from qsf_sfh.control_summary import main as summarize
        summarize()


if __name__ == '__main__':
    main()
