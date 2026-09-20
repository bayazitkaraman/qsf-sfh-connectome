"""Regenerate numerical panels; no manuscript or schematic artwork is included."""
import argparse
from pathlib import Path
import hashlib
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('QSF_WORKDIR', ROOT/'work'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=WORK/'consolidation')
parser.add_argument('--output', type=Path, default=WORK/'figures')
args = parser.parse_args()
DATA, OUT = args.input, args.output
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = pd.read_csv(DATA/'track_summary.csv')
PAIRS = pd.read_csv(DATA/'paired_comparisons.csv')
LEDGER = []
COHORTS = {'hcp83': 'HCP-83', 'oasis': 'OASIS'}
ENCODERS = {'qsf': 'QSF', 'heat_raw18': 'Raw heat', 'heat_norm18': 'Normalized heat'}
POLICIES = {'random': 'Random', 'graph_coverage': 'Graph coverage', 'sfh_fixed': 'Fixed SFH',
            'spectral': 'Spectral', 'qsf_discrimination': 'QSF discrimination',
            'heat_discrimination': 'Heat discrimination', 'spatial': 'Spatial spread'}
COLORS = ['#176b87', '#a45119', '#716078', '#267349']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.labelpad': 6, 'savefig.dpi': 600})

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stat(cohort, k, encoder='qsf', track='random_robust'):
    rows = SUMMARY[(SUMMARY.cohort == cohort) & (SUMMARY.source_count == k) &
                   (SUMMARY.encoder == encoder) & (SUMMARY.track == track)]
    assert len(rows) == 1, (cohort, k, encoder, track)
    return rows.iloc[0]

def pair(cohort, k, contrast, metric='normalized_rmse', encoder='qsf'):
    rows = PAIRS[(PAIRS.cohort == cohort) & (PAIRS.source_count == k) &
                 (PAIRS.encoder == encoder) & (PAIRS.contrast == contrast) & (PAIRS.metric == metric)]
    assert len(rows) == 1, (cohort, k, contrast)
    return rows.iloc[0]

def record(place, source, filters, column, value):
    LEDGER.append(dict(location=place, source=source, filters=json.dumps(filters, sort_keys=True),
                       column=column, value=float(value), source_sha256=sha(DATA/source)))
    return float(value)

def mean(cohort, k, encoder, track, place):
    r = stat(cohort, k, encoder, track)
    return record(place, 'track_summary.csv', dict(cohort=cohort, source_count=k,
                  encoder=encoder, track=track), 'equal_participant_nrmse', r.equal_participant_nrmse)

def save(fig, name):
    for extension in ('pdf', 'png'):
        fig.savefig(OUT/f'{name}.{extension}', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)

def axes(ylabel='Normalized RMSE'):
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    fig.subplots_adjust(left=.20, right=.98, bottom=.19, top=.98)
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 1, 2], ['8 sources', '16 sources', '32 sources'])
    ax.set_xlim(-.13, 2.13)
    ax.grid(axis='y', color='#e5e5e5', lw=.6)
    ax.set_axisbelow(True)
    return fig, ax

def lines(cohort, name, conditions, ylim=None):
    fig, ax = axes()
    for j, (encoder, track, label) in enumerate(conditions):
        y = [mean(cohort, k, encoder, track, name) for k in (8,16,32)]
        ax.plot(range(3), y, ['o-', 's--', '^:', 'd-.'][j], color=COLORS[j], lw=1.3,
                ms=4, label=label)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc='upper right', frameon=True, framealpha=.95, edgecolor='none', borderpad=.25)
    save(fig, name)

def main():
    # Figure 1 is author-supplied artwork; only numerical panels are generated.
    for c, letter in [('hcp83','a'),('oasis','b')]:
        lines(c,'Fig2'+letter,[(e,'random_robust',n) for e,n in ENCODERS.items()],(.035,.19))
        lines(c,'Fig3'+letter,[('qsf','random_robust','Individual QSF'),
              ('qsf','template_random_robust','Template QSF'),
              ('qsf','atlas_random_aligned','Aligned atlas')],(.012,.18))
        lines(c,'Fig4'+letter,[('qsf','random_robust','Random'),
              ('qsf','selected_robust','Selected nonspatial'),('qsf','spatial_robust','Spatial spread')],(.035,.18))
        fig, ax = axes('Paired change in normalized RMSE')
        contrasts = [('random_robust minus random_ordinary','Decoder, random sources'),
                     ('ordinary_selected minus random_ordinary','Placement, ordinary decoder'),
                     ('selected_robust minus selected_sources_ordinary','Decoder, selected sources')]
        for j,(contrast,label) in enumerate(contrasts):
            r = [pair(c,k,contrast) for k in (8,16,32)]
            y = np.array([v.paired_difference for v in r])
            err = np.array([[v.paired_difference-v.ci95_low for v in r],
                            [v.ci95_high-v.paired_difference for v in r]])
            ax.errorbar(np.arange(3)+(j-1)*.035,y,yerr=err,fmt=['o-','s--','^:'][j],
                        color=COLORS[j],ms=4,lw=1.1,capsize=2,label=label)
        ax.axhline(0,color='#888888',lw=.7)
        ax.set_ylim(-.014,.005)
        ax.legend(loc='upper left',fontsize=8,frameon=True,edgecolor='none',borderpad=.2)
        save(fig,'Fig4'+('c' if c=='hcp83' else 'd'))
    proj = pd.read_csv(DATA/'projection_target_summary.csv')
    for c,letter in [('hcp83','a'),('oasis','b')]:
        fig, ax = axes('Mean change in normalized distance')
        for j,(fam,group,label) in enumerate([
            ('projection','inside','P: inside'),('projection','outside','P: outside'),
            ('bounded_residual','inside','BR: inside'),('bounded_residual','outside','BR: outside')]):
            v = proj[(proj.cohort==c)&(proj.family==fam)&(proj.truth_group==group)]
            g = v.groupby('source_count')[['error_change_sum','target_count']].sum()
            ax.plot(range(3),g.error_change_sum/g.target_count,['o-','s--','^:','d-.'][j],
                    color=COLORS[j],ms=4,lw=1.1,label=label)
        ax.axhline(0,color='#888888',lw=.7)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        lo,hi=ax.get_ylim()
        ax.set_ylim(lo,hi+(hi-lo)*.7)
        ax.legend(loc='upper left',ncol=2,fontsize=8,frameon=True,edgecolor='none',columnspacing=.6,handlelength=1.4)
        save(fig,'Fig5'+letter)
    if not (DATA/'track_metrics.csv').exists():
        print('Exported 10 panels. Fig5c/d require local record distributions from a completed rerun.')
        return
    metrics = pd.read_csv(DATA/'track_metrics.csv',usecols=['cohort','encoder','track','normalized_rmse'])
    for c,letter in [('hcp83','c'),('oasis','d')]:
        fig,ax=plt.subplots(figsize=(3.35,2.45))
        fig.subplots_adjust(left=.20,right=.98,bottom=.20,top=.98)
        for j,(track,label) in enumerate([('random_ordinary','Ordinary, random'),
                         ('random_robust','Bounded, random'),('selected_robust','Bounded, selected')]):
            x=np.sort(metrics.loc[(metrics.cohort==c)&(metrics.encoder=='qsf')&(metrics.track==track),
                                 'normalized_rmse'].to_numpy())
            ax.step(x,(len(x)-np.arange(len(x)))/len(x),where='post',color=COLORS[j],
                    ls=['-','--',':'][j],lw=1.3,label=label)
        ax.set(xlabel='Normalized RMSE',ylabel='Fraction at or above error',yscale='log',xlim=(0,.38),ylim=(1e-4,1.1))
        ax.legend(loc='upper right',frameon=True,edgecolor='none',fontsize=8)
        ax.grid(axis='y',color='#e5e5e5',lw=.6)
        save(fig,'Fig5'+letter)


main()
