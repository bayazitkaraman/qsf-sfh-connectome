"""Run the released consolidation from separately obtained GraphML inputs."""
import argparse
from pathlib import Path
import os
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from qsf_sfh import benchmark
from qsf_sfh.data import check_release, locate_graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('check','inputs','prepare','main','template','sensitivity','summarize','audit','all'))
    parser.add_argument('--cohort', choices=('hcp83','oasis'))
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.cohort and args.stage in ('summarize','audit','all'):
        parser.error('Reporting stages require both cohorts; omit --cohort or run individual stages.')
    check_release()
    if args.stage == 'check':
        print('Release source/configuration hashes verified. No data required.')
        return
    cohorts = [args.cohort] if args.cohort else ['hcp83','oasis']
    for cohort in cohorts:
        if args.stage == 'inputs':
            print(cohort, len(locate_graphs(cohort)), 'matching graphs')
        if args.stage in ('prepare','all'):
            benchmark.prepare(cohort)
        for phase in ('main','template','sensitivity'):
            if args.stage in (phase,'all'):
                benchmark.run(cohort, phase, args.workers)
    if args.stage in ('summarize','all'):
        from qsf_sfh.summarize import main as summarize
        summarize()
    if args.stage in ('audit','all'):
        from qsf_sfh.projection_audit import main as projection_audit
        from qsf_sfh.verify_models import main as verify_models
        projection_audit()
        verify_models()


if __name__ == '__main__':
    main()
