"""Cache-only Fig. 1 diagnostic. No model, raw annotation or video reads."""
from pathlib import Path
import hashlib, io, json, csv, time, os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/surrogate_alignment_v1'
PUB = ROOT / 'results/surrogate_alignment_v1'
FULL = ROOT / 'outputs/simplepath_full_v2'
SIGNALS = ('temporal_confidence', 'spatial_trajectory_consistency')
SEED, REPEATS = 20261023, 5000

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha(p):
    return sha_bytes(p.read_bytes())

def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def confidence(q):
    q = np.asarray(q, dtype=np.float64)
    n = len(q)
    legal = np.triu(np.ones((n, n), dtype=bool), 1)
    assert q.shape == (n, n) and np.isfinite(q).all() and (q >= 0).all()
    assert not q[~legal].any() and abs(q.sum() - 1) < 1e-10
    p = q[q > 0]
    return float(np.clip(1 + (p * np.log(p)).sum() / np.log(legal.sum()), 0, 1))

def spatial_score(boxes):
    b = np.asarray(boxes, dtype=np.float64)
    assert b.ndim == 2 and b.shape[1] == 4 and len(b) >= 2
    assert np.isfinite(b).all() and (b[:, 2:] >= 0).all()
    lo, hi = np.clip(b[:, :2] - b[:, 2:] / 2, 0, 1), np.clip(b[:, :2] + b[:, 2:] / 2, 0, 1)
    inter = np.maximum(np.minimum(hi[1:], hi[:-1]) - np.maximum(lo[1:], lo[:-1]), 0).prod(1)
    area = np.maximum(hi - lo, 0).prod(1)
    union = area[1:] + area[:-1] - inter
    pair = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return float(pair.mean())

def bins(values):
    # Tied scores stay together. Counts may differ from exactly N/5.
    return np.minimum(4, np.floor(5 * (rankdata(values, method='average') - .5) / len(values)).astype(int))

def cluster_summary(values, group, parents, rng):
    unique, parent_index = np.unique(parents, return_inverse=True)
    sums = np.zeros((len(unique), 5)); counts = np.zeros_like(sums)
    np.add.at(sums, (parent_index, group), values)
    np.add.at(counts, (parent_index, group), 1)
    totals = counts.sum(0)
    assert (totals > 0).all()
    # Whole parent-video clusters sampled with replacement; equal-query mean.
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=REPEATS)
    den = weights @ counts
    samples = np.divide(weights @ sums, den, out=np.full_like(den, np.nan), where=den > 0)
    result = []
    for k in range(5):
        result.append(dict(quintile=k+1, nominal_percentile=[k*20,(k+1)*20], queries=int(totals[k]),
                           parents=int((counts[:, k] > 0).sum()), mean_vIoU_percent=float(sums[:, k].sum()/totals[k]),
                           ci95_percent=np.nanquantile(samples[:, k], [.025,.975]).tolist()))
    return result

def plot(ds, report):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.labelsize':11,
                         'axes.titlesize':12, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(6.8, 4.55))
    styles = [('temporal_confidence', '#4379C4', 'o', 'Temporal confidence'),
              ('spatial_trajectory_consistency', '#59A276', 's', 'Spatial consistency')]
    for signal, color, marker, label in styles:
        s = report['signals'][signal]
        y = np.array([v['mean_vIoU_percent'] for v in s['bins']])
        ci = np.array([v['ci95_percent'] for v in s['bins']])
        x = np.arange(1,6)
        ax.fill_between(x, ci[:,0], ci[:,1], color=color, alpha=.12, linewidth=0)
        ax.plot(x, y, color=color, marker=marker, markersize=6, linewidth=2,
                label=label + rf' ($\rho={s["spearman_rho"]:.3f}$)')
        ax.errorbar(x, y, yerr=np.stack([y-ci[:,0],ci[:,1]-y]), color=color, fmt='none', capsize=3, linewidth=1)
    ax.set(xlim=(.82,5.18), ylim=(0,40), xticks=np.arange(1,6), yticks=np.arange(0,41,10),
           xticklabels=['0–20','20–40','40–60','60–80','80–100'],
           xlabel='Surrogate score percentile', ylabel='Mean vIoU (%)')
    ax.set_title('Task Alignment of TTA Surrogate Signals in STVG', pad=34, fontweight='semibold')
    name = {'hc2':'HC-STVG2.0', 'vidstg':'VidSTG'}[ds]
    ax.text(.5,1.035, f'{name}  ·  Frozen Source predictions  ·  {report["queries"]:,} queries',
            ha='center', transform=ax.transAxes, fontsize=9, color='#555555')
    ax.grid(axis='y', color='#DFE3E8', linestyle='--', linewidth=.7)
    ax.set_axisbelow(True)
    for side in ('top','right'): ax.spines[side].set_visible(False)
    for side in ('bottom','left'): ax.spines[side].set_color('#C8CDD2')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    fig.text(.12,.035, '95% parent-video bootstrap CI · Each signal ranked separately', fontsize=8, color='#555555')
    fig.subplots_adjust(left=.12,right=.975,bottom=.18,top=.78)
    (OUT/'figures').mkdir(parents=True, exist_ok=True)
    for ext in ('pdf','svg','png'):
        target = PUB if ext == 'svg' else OUT/'figures'
        fig.savefig(target/f'fig1_{ds}.{ext}', dpi=300, facecolor='white')
        if ext == 'svg':
            svg = target/f'fig1_{ds}.{ext}'
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)

def main():
    assert not (OUT/'completion.json').exists(), 'Completed diagnostic must not be silently overwritten'
    torch.set_num_threads(1)
    started=time.time()
    q=np.triu(np.ones((4,4)),1);q/=q.sum()
    assert abs(confidence(q))<1e-14
    q[:]=0;q[0,3]=1;assert confidence(q)==1
    assert spatial_score(np.array([[.5,.5,.2,.2]]*3))==1
    assert spatial_score(np.array([[.1,.1,.1,.1],[.9,.9,.1,.1]]))==0
    assert len(set(bins(np.ones(10))))==1
    protocol=dict(schema='surrogate_alignment_v1',purpose='Sample-level cache-only diagnostic, not a TTA optimization experiment',
      datasets=['hc2','vidstg'],scope='All sealed official queries, domain-specific figures; no cross-domain pooling or subset selection',
      downstream='Original frozen Source vIoU from previously completed official metrics; no new GT/annotation access',
      temporal_confidence='1 - H(q0)/log(T*(T-1)/2); original native two-offset physical-hull interval prior, natural logs, all legal intervals',
      spatial_trajectory_consistency='Arithmetic mean of adjacent Source box IoUs over the entire sampled trajectory; normalized cxcywh boxes clipped to image [0,1] before IoU; degenerate union gives zero',
      spatial_boundary='Trajectory smoothness proxy, NOT augmentation-view consistency; real object/camera motion and sampling affect it',
      binning='Five percentile bins independently for each score using average ranks; equal scores never split, unequal bin sizes possible',
      correlation='Query-level Spearman score versus Source vIoU, with average-rank ties; not correlation of five plotted means',
      uncertainty='Pointwise 95% percentile parent-video cluster bootstrap; fixed full-data bin assignment, equal-query mean, 5000 resamples',
      seed=SEED,bootstrap_repeats=REPEATS,primary_title='Task Alignment of TTA Surrogate Signals in STVG',
      selection='Definitions fixed before extracting scores; datasets and downstream results already exposed; no proxy/scale/bin-count sweep',
      no_new_model_or_expert_or_GPU_or_predictions_or_GT_score=True,
      source_sha256=sha(Path(__file__)),production_plan_sha256=sha(FULL/'plan.json'))
    write(PUB/'protocol.json',protocol)
    all_reports={}
    for ds in protocol['datasets']:
        dest=FULL/ds;seal_bytes=(dest/'sealed.json').read_bytes();seal=json.loads(seal_bytes)
        metric_sha=sha(dest/'metrics.json');scores=[]
        for i,(qid,entry) in enumerate(sorted(seal['files'].items())):
            raw=(dest/entry['capture_path']).read_bytes()
            assert sha_bytes(raw)==entry['capture_sha256']
            cap=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
            assert cap['GT_read'] is False
            scores.append(dict(id=qid,temporal_confidence=confidence(cap['native_prior']),
                               spatial_trajectory_consistency=spatial_score(cap['Source'])))
            if (i+1)%1000==0: print(ds,i+1,'surrogates extracted; CPU-only',flush=True)
        write(OUT/ds/'scores.json',scores)
        write(OUT/ds/'score_seal.json',dict(queries=len(scores),scores_sha256=sha(OUT/ds/'scores.json'),
             source_seal_sha256=sha_bytes(seal_bytes),no_quality_in_score_computation=True))
        # Only join saved quality after all scores for the domain have been sealed.
        metrics=json.loads((dest/'metrics.json').read_text());rows={r['id']:r for r in metrics['private_rows']}
        assert len(scores)==seal['queries']==len(rows) and set(rows)=={r['id']for r in scores}
        y=np.array([100*rows[r['id']]['values']['Source']['v']for r in scores]);parents=np.array([rows[r['id']]['parent']for r in scores])
        report=dict(dataset=ds,queries=len(scores),parents=len(set(parents)),Source_mean_vIoU_percent=float(y.mean()),signals={})
        for si,signal in enumerate(SIGNALS):
            x=np.array([r[signal]for r in scores]);g=bins(x)
            report['signals'][signal]=dict(spearman_rho=float(spearmanr(x,y).statistic),
              score_min=float(x.min()),score_max=float(x.max()),unique_scores=len(np.unique(x)),
              bins=cluster_summary(y,g,parents,np.random.default_rng(SEED)))
        assert abs(y.mean()-100*metrics['means']['Source']['m_vIoU'])<1e-10
        write(PUB/ds/'report.json',report)
        with (PUB/ds/'bins.csv').open('w')as f:
            w=csv.writer(f,lineterminator="\n");w.writerow(['signal','quintile','queries','parents','mean_vIoU_percent','CI95_low','CI95_high'])
            for signal in SIGNALS:
                for b in report['signals'][signal]['bins']:
                    w.writerow([signal,b['quintile'],b['queries'],b['parents'],b['mean_vIoU_percent'],*b['ci95_percent']])
        # Anonymized scores preserve reproducibility without captions/GT coordinates.
        pm={p:f'P{i+1:04d}'for i,p in enumerate(sorted(set(parents)))}
        with (PUB/ds/'scores.tsv').open('w')as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','parent',*SIGNALS,'Source_vIoU_percent'])
            for i,r in enumerate(scores):w.writerow([f'Q{i+1:05d}',pm[parents[i]],*[r[s]for s in SIGNALS],y[i]])
        plot(ds,report)
        assert sha(dest/'metrics.json')==metric_sha and sha(dest/'sealed.json')==sha_bytes(seal_bytes)
        all_reports[ds]=report
        print(ds,json.dumps(report),flush=True)
    write(PUB/'verification.json',dict(status='complete',queries=sum(v['queries']for v in all_reports.values()),
      all_capture_SHA_verified=True,scores_sealed_before_quality_join=True,existing_Source_summary_reproduced=True,
      production_metrics_and_seals_unchanged=True,math_invariants=5,model_calls=0,expert_calls=0,
      GPU_calls=0,new_predictions=0,new_GT_scoring=0,seconds=time.time()-started))
    write(OUT/'completion.json',dict(status='complete',files={str(p.relative_to(PUB)):sha(p)for p in PUB.rglob('*')if p.is_file()}))

if __name__=='__main__':main()
