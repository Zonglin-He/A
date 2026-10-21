"""Offline attribution of the fixed paired-hull temporal reference.

Import and pure helpers read no assets. ``evaluate`` first calls the supplied
cohort validator, then opens the fixed labels. Original posteriors, old source,
predictions and old results are read-only. Candidate oracles are restricted to
the twelve precomputed single-layer/single-offset MBRs plus old Global6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
TRIAL=ROOT/'outputs/interval_pool/development'
PUBLIC=ROOT/'results/interval_pool/development'
SEED=20261021
DRAWS=10000
EPS=1e-12
THRESHOLDS=(.5,.7)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def interval_iou(intervals,truth):
    a=np.asarray(intervals,dtype=np.float64)
    assert a.ndim>=1 and a.shape[-1]==2 and np.isfinite(a).all()and(a[...,1]>a[...,0]).all()
    g=np.asarray(truth,dtype=np.float64);assert g.shape==(2,)and g[1]>g[0]
    intersection=np.maximum(0,np.minimum(a[...,1],g[1])-np.maximum(a[...,0],g[0]))
    union=(a[...,1]-a[...,0])+(g[1]-g[0])-intersection
    return intersection/union


def interval_error(interval,truth,fps):
    """Signed errors use predicted minus GT physical boundaries, not indices."""
    assert isinstance(fps,(int,float))and not isinstance(fps,bool)and np.isfinite(fps)and fps>0
    a,b=map(float,interval);s,e=map(float,truth)
    assert np.isfinite([a,b,s,e]).all()and b>a and e>s
    length=b-a;glength=e-s;overlap=max(0.,min(b,e)-max(a,s))
    outside=length-overlap;missing=glength-overlap
    if a==s and b==e:shape='exact'
    elif overlap==0:shape='disjoint'
    elif a<=s and b>=e:shape='contains_GT'
    elif a>=s and b<=e:shape='inside_GT'
    else:shape='partial_intersection'
    t=float(interval_iou(np.asarray([a,b]),[s,e]))
    return dict(start_signed_seconds=(a-s)/fps,end_signed_seconds=(b-e)/fps,
        start_signed_GT_normalized=(a-s)/glength,end_signed_GT_normalized=(b-e)/glength,
        boundary_MAE_seconds=(abs(a-s)+abs(b-e))/(2*fps),
        overcoverage_seconds=outside/fps,undercoverage_seconds=missing/fps,intersection_seconds=overlap/fps,
        overcoverage_GT_normalized=outside/glength,undercoverage_GT_normalized=missing/glength,
        intersection_GT_normalized=overlap/glength,predicted_length_seconds=length/fps,GT_length_seconds=glength/fps,
        predicted_to_GT_length_ratio=length/glength,length_signed_seconds=(length-glength)/fps,
        early_start_seconds=max(0.,s-a)/fps,late_start_seconds=max(0.,a-s)/fps,
        early_end_seconds=max(0.,e-b)/fps,late_end_seconds=max(0.,b-e)/fps,
        starts_early=a<s,ends_late=b>e,has_undercoverage=missing>0,has_overcoverage=outside>0,
        nearly_disjoint=t<.1,shape=shape,tIoU=t)


def duration_group(seconds):
    assert np.isfinite(seconds)and seconds>0
    return 'short_<5s'if seconds<5 else'medium_5-10s'if seconds<10 else'long_>=10s'


def disagreement_group(seconds):
    assert np.isfinite(seconds)and seconds>=-EPS
    return 'low_<.25s'if seconds<.25 else'medium_.25-1s'if seconds<1 else'high_>=1s'


def distribution(intervals,probabilities):
    """Validate already-normalized mass; never silently change its support."""
    intervals=np.asarray(intervals,dtype=np.float64);prob=np.asarray(probabilities,dtype=np.float64)
    assert intervals.ndim==2 and intervals.shape[1]==2 and prob.shape==(len(intervals),)
    assert len(prob)>0 and np.isfinite(intervals).all()and np.isfinite(prob).all()
    assert(intervals[:,1]>intervals[:,0]).all()and(prob>=0).all()
    assert abs(float(prob.sum())-1)<1e-10,'Distribution must be normalized before diagnosis'
    return intervals,prob


def _expect_abs(first,p,second,q):
    """Independent E|X-Y| via a sorted weighted CDF; O(n log n), no pair grid."""
    order=np.argsort(second,kind='stable');y=second[order];mass=q[order]
    cm=np.concatenate(([0.],np.cumsum(mass)));cy=np.concatenate(([0.],np.cumsum(mass*y)))
    indices=np.searchsorted(y,first,side='right');left=first*cm[indices]-cy[indices]
    right=(cy[-1]-cy[indices])-first*(cm[-1]-cm[indices])
    return float(np.dot(p,left+right))


def _ordered_gap(ends,p,starts,q):
    """P(end <= start) and E[(start-end)+] for independent intervals."""
    order=np.argsort(ends,kind='stable');ends=ends[order];p=p[order]
    mass=np.concatenate(([0.],np.cumsum(p)));moment=np.concatenate(([0.],np.cumsum(p*ends)))
    ix=np.searchsorted(ends,starts,side='right')
    return float(q@mass[ix]),float(q@(starts*mass[ix]-moment[ix]))


def distribution_summary(intervals,probabilities,fps,truth=None):
    intervals,p=distribution(intervals,probabilities);assert fps>0
    means=(p[:,None]*intervals).sum(0)
    variance=(p[:,None]*(intervals-means)**2).sum(0)
    result=dict(total_mass=float(p.sum()),support_size=int((p>0).sum()),
        expected_length_seconds=float(p@(intervals[:,1]-intervals[:,0]))/fps,
        start_std_seconds=float(np.sqrt(variance[0]))/fps,end_std_seconds=float(np.sqrt(variance[1]))/fps)
    if truth is not None:
        iou=interval_iou(intervals,truth)
        result.update({f'mass_tIoU_ge_{int(100*a):02d}':float(p[iou>=a].sum())for a in THRESHOLDS})
    return result


def layer_attribution(views,hull,fps,truth=None):
    """Per-layer hull identity and original-vs-hull effective event mass."""
    assert len(views)==2
    v=[distribution(*item)for item in views];h=distribution(*hull)
    summaries=[distribution_summary(*item,fps,truth)for item in v]
    hs=distribution_summary(*h,fps,truth)
    start=_expect_abs(v[0][0][:,0],v[0][1],v[1][0][:,0],v[1][1])/fps
    end=_expect_abs(v[0][0][:,1],v[0][1],v[1][0][:,1],v[1][1])/fps
    original_mean=.5*sum(x['expected_length_seconds']for x in summaries)
    predicted=original_mean+.5*(start+end)
    error=abs(hs['expected_length_seconds']-predicted)
    assert error<1e-8,'Paired hull expected-length identity failed'
    disjoint01,gap01=_ordered_gap(v[0][0][:,1],v[0][1],v[1][0][:,0],v[1][1])
    disjoint10,gap10=_ordered_gap(v[1][0][:,1],v[1][1],v[0][0][:,0],v[0][1])
    result=dict(views=summaries,hull=hs,original_view_mean_expected_length_seconds=original_mean,
        independent_start_absolute_disagreement_seconds=start,independent_end_absolute_disagreement_seconds=end,
        hull_expected_length_increase_seconds=hs['expected_length_seconds']-original_mean,
        endpoint_disagreement_seconds=.5*(start+end),expected_length_identity_absolute_error_seconds=error,
        independent_pair_disjoint_probability=disjoint01+disjoint10,
        independent_expected_between_interval_gap_seconds=(gap01+gap10)/fps,
        interpretation='Independent endpoint disagreement includes within-view uncertainty; it is not a pure offset-alignment error.')
    if truth is not None:
        for a in THRESHOLDS:
            field=f'mass_tIoU_ge_{int(100*a):02d}'
            original=.5*sum(x[field]for x in summaries)
            result[f'original_view_mean_{field}']=original
            result[f'hull_minus_views_{field}']=hs[field]-original
    return result


def mean_ci(values,scale=1.):
    a=scale*np.asarray(values,dtype=np.float64);assert a.ndim==1 and np.isfinite(a).all()
    if not len(a):return dict(queries=0,mean=None,CI95=None)
    ix=np.random.default_rng(SEED).integers(0,len(a),(DRAWS,len(a)))
    return dict(queries=len(a),mean=float(a.mean()),CI95=np.quantile(a[ix].mean(1),[.025,.975]).tolist(),
        positive=int((a>EPS).sum()),negative=int((a < -EPS).sum()),unchanged=int((abs(a)<=EPS).sum()))


def _ranks(x):
    x=np.asarray(x,float);order=np.argsort(x,kind='stable');rank=np.empty(len(x),float);i=0
    while i<len(x):
        j=i+1
        while j<len(x)and x[order[j]]==x[order[i]]:j+=1
        rank[order[i:j]]=(i+j-1)/2;i=j
    return rank


def association(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);assert x.shape==y.shape and np.isfinite(x).all()and np.isfinite(y).all()
    def corr(a,b):return None if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0 else float(np.corrcoef(a,b)[0,1])
    return dict(queries=len(x),Pearson=corr(x,y),Spearman=corr(_ranks(x),_ranks(y)),
        interpretation='Offline association only; neither selection criterion nor causal attribution.')


def candidate_oracle(intervals,names,truth,baseline='Global6'):
    """The thirteen fixed decisions only; no all-legal-interval oracle."""
    a=np.asarray(intervals,dtype=np.float64);assert a.shape==(13,2)and len(names)==len(set(names))==13
    assert baseline in names and names.count(baseline)==1
    iou=interval_iou(a,truth);index=int(iou.argmax());base=float(iou[names.index(baseline)])
    return dict(candidates=13,tIoU_by_candidate={name:float(value)for name,value in zip(names,iou)},
        best_candidate=names[index],best_tIoU=float(iou[index]),Global6_tIoU=base,
        oracle_minus_Global6_tIoU_pp=100*(float(iou[index])-base),
        oracle_only=True,scope='Twelve frozen single-layer/single-offset MBRs plus old Global6. Not deployed selection or a model upper bound.')


def _mean_records(rows,field):return float(np.mean([r[field]for r in rows]))if rows else None


def error_aggregate(rows):
    if not rows:return dict(queries=0)
    errors=[r['error']for r in rows]
    fields=[name for name,value in errors[0].items()if isinstance(value,(int,float))and not isinstance(value,bool)]
    return dict(queries=len(rows),means={name:_mean_records(errors,name)for name in fields},
        counts={name:sum(bool(x[name])for x in errors)for name in('starts_early','ends_late','has_undercoverage','has_overcoverage','nearly_disjoint')},
        shape_counts={name:sum(x['shape']==name for x in errors)for name in('exact','contains_GT','inside_GT','partial_intersection','disjoint')},
        fixed_space_vIoU=mean_ci([r['fixed_spatial_vIoU']for r in rows],100),full_vIoU=mean_ci([r['full_vIoU']for r in rows],100),
        hull_minus_views_mass50_pp=mean_ci([r['hull_minus_views_mass50']for r in rows],100),
        hull_minus_views_mass70_pp=mean_ci([r['hull_minus_views_mass70']for r in rows],100),
        candidate_oracle_tIoU_gap_pp=mean_ci([r['candidate_oracle']['oracle_minus_Global6_tIoU_pp']for r in rows]))


def aggregate(rows):
    assert len(rows)==len({r['query']for r in rows})==64
    return dict(overall=error_aggregate(rows),
        event_duration_groups={name:error_aggregate([r for r in rows if r['duration_group']==name])for name in('short_<5s','medium_5-10s','long_>=10s')},
        unlabeled_endpoint_disagreement_groups={name:error_aggregate([r for r in rows if r['disagreement_group']==name])for name in('low_<.25s','medium_.25-1s','high_>=1s')},
        associations=dict(disagreement_vs_actual_overcoverage=association([r['endpoint_disagreement_seconds']for r in rows],[r['error']['overcoverage_seconds']for r in rows]),
            effective_mass50_loss_vs_actual_overcoverage=association([-r['hull_minus_views_mass50']for r in rows],[r['error']['overcoverage_seconds']for r in rows]),
            effective_mass50_loss_vs_candidate_headroom=association([-r['hull_minus_views_mass50']for r in rows],[r['candidate_oracle']['oracle_minus_Global6_tIoU_pp']for r in rows])),
        expected_length_identity_max_error_seconds=max(layer['expected_length_identity_absolute_error_seconds']for r in rows for layer in r['layers']),
        interpretation='All64 fixed parents retained. Duration groups use GT only for offline description; disagreement groups use posteriors without labels. Expected-length inflation alone is not evidence of MBR harm.')


def mechanism_conclusion(summary):
    """A provisional descriptive statement, not a gate or proof of root cause."""
    overall=summary['overall'];m50=overall['hull_minus_views_mass50_pp'];m70=overall['hull_minus_views_mass70_pp']
    damaged=(m50['mean'] < -1e-10)or(m70['mean'] < -1e-10)
    strong=(m50['CI95'][1] < -1e-10)and(m70['CI95'][1] < -1e-10)
    correlation=summary['associations']['effective_mass50_loss_vs_actual_overcoverage']['Spearman']
    linked=correlation is not None and correlation>0
    if strong and linked:label='支持（暂定）'
    elif damaged:label='部分支持（暂定）'
    else:label='未得到支持（本轮归因）'
    return dict(automatic_descriptive_hint=label,final_mechanism_verdict='Requires joint interpretation of quality, actual over/undercoverage and both positive/negative cases; this helper does not decide.',
        effective_mass50_decreases=m50['mean'] < -1e-10,effective_mass70_decreases=m70['mean'] < -1e-10,
        both_mass_loss_CIs_below_zero=strong,mass_loss_overcoverage_association_positive=linked,
        rule='Automatic descriptive hint only, fixed before labels: both effective-mass CIs below zero plus positive loss/overcoverage association; otherwise partial when either mean falls. These are not validated mechanism thresholds, a formal verdict, an adaptation gate or a reason to alter the two fixed rounds.',
        limitation='This attribution cannot establish that hull fusion is the main failure cause or that mixture improves MBR. The fixed second-round intervention is required; no tuning follows these labels.')


def grid_distribution(grid):
    ids=np.asarray(grid['ids'],dtype=np.int64);ij=np.asarray(grid['ij'],dtype=np.int64)
    assert ids.ndim==1 and(np.diff(ids)>0).all()and ij.ndim==2 and ij.shape[0]==2
    assert(ij[0]<ij[1]).all()and ij.min()>=0 and ij.max()<len(ids)
    intervals=np.stack((ids[ij[0]],ids[ij[1]]+1),axis=1)
    return distribution(intervals,grid['p'])


def table_distribution(ids,q):
    ids=np.asarray(ids,dtype=np.int64);q=np.asarray(q,dtype=np.float64);n=len(ids)
    assert ids.ndim==1 and(np.diff(ids)>0).all()and q.shape==(n,n)
    legal=np.triu(np.ones((n,n),bool),1);assert np.isfinite(q).all()and(q>=0).all()and not q[~legal].any()
    ij=np.nonzero(legal)
    return distribution(np.stack((ids[ij[0]],ids[ij[1]]+1),axis=1),q[ij])


def pack_attribution(pack,fps,truth=None):
    assert pack['GT_read']is False and len(pack['views'])==len(pack['hull_layers'])==len(pack['statistics'])==6
    layers=[];ids=pack['ids'];statistics=pack['statistics']
    assert np.asarray(pack['hull_layers']).shape==(6,len(ids),len(ids))
    assert np.max(np.abs(np.asarray(pack['hull_layers']).mean(0)-np.asarray(pack['hull'])))<2e-15
    table_distribution(ids,pack['hull'])
    for l in range(6):
        views=[grid_distribution(g)for g in pack['views'][l]]
        layer=layer_attribution(views,table_distribution(ids,pack['hull_layers'][l]),fps,truth)
        old=statistics[l]
        for computed,stored in((layer['original_view_mean_expected_length_seconds']*fps,old['mean_view_length_frames']),
            (layer['hull']['expected_length_seconds']*fps,old['hull_expected_length_frames']),
            (layer['endpoint_disagreement_seconds']*fps,old['endpoint_disagreement_frames'])):
            assert abs(float(computed)-float(stored))<1e-7
        assert old['identity_abs_error_frames']<1e-7
        for new,previous in zip(layer['views'],old['views']):
            for a,b in(('expected_length_seconds','expected_length_frames'),('start_std_seconds','start_std_frames'),('end_std_seconds','end_std_frames')):
                assert abs(new[a]*fps-previous[b])<1e-7
        layer['layer']=l+1;layers.append(layer)
    return layers


def _case_summary(row):
    return dict(query=row['query'],error=row['error'],endpoint_disagreement_seconds=row['endpoint_disagreement_seconds'],
        hull_minus_views_mass50_pp=100*row['hull_minus_views_mass50'],hull_minus_views_mass70_pp=100*row['hull_minus_views_mass70'],
        Global6_minus_mean12_decision_tIoU_pp=row['Global6_minus_mean12_decision_tIoU_pp'],
        independent_pair_disjoint_probability=row['independent_pair_disjoint_probability'],
        candidate_oracle_headroom_tIoU_pp=row['candidate_oracle']['oracle_minus_Global6_tIoU_pp'])


def select_cases(rows):
    """Declared mechanical cases complement the complete anonymous table."""
    result={'selection_rule':'Up to three largest signed gains and losses under each stated diagnostic, ties by anonymous manifest order. All64 rows are published; these are not selected training/confirmation cases.'}
    for name,field in(('effective_mass50','hull_minus_views_mass50'),('Global6_vs_mean12_MBR','Global6_minus_mean12_decision_tIoU_pp')):
        negative=sorted((r for r in rows if r[field] < -EPS),key=lambda r:(r[field],r['query']))[:3]
        positive=sorted((r for r in rows if r[field]>EPS),key=lambda r:(-r[field],r['query']))[:3]
        result[name]=dict(benefiting=[_case_summary(r)for r in positive],harmed=[_case_summary(r)for r in negative])
    return result


def _render(public):
    summary=public['summary'];overall=summary['overall'];e=overall['means']
    lines=['# HC64 first-round temporal error attribution','',
        'All64 predictions, the twelve single-layer/single-offset MBRs and old Global6 were frozen before GT. '
        'This is attribution on exposed development data, not a newly deployed method or independent confirmation. '
        'MBR, physical frame endpoints and original BoxFixed remain unchanged.','',
        '| Global6 error | Mean |','|---|---:|']
    for label,field in(('Signed start error s (prediction − GT)','start_signed_seconds'),('Signed end error s','end_signed_seconds'),
        ('Signed start / GT duration','start_signed_GT_normalized'),('Signed end / GT duration','end_signed_GT_normalized'),
        ('Overcoverage s','overcoverage_seconds'),('Undercoverage s','undercoverage_seconds'),('Intersection s','intersection_seconds'),
        ('Predicted / GT duration','predicted_to_GT_length_ratio'),('tIoU','tIoU')):
        lines.append(f'| {label} | {e[field]:.6f} |')
    lines+=['',f'Early starts: {overall["counts"]["starts_early"]}/64; late ends: {overall["counts"]["ends_late"]}/64; '
        f'any undercoverage: {overall["counts"]["has_undercoverage"]}/64; nearly disjoint (tIoU < .1): {overall["counts"]["nearly_disjoint"]}/64.','',
        '| Diagnostic | Query mean [parent bootstrap 95% CI] |','|---|---:|']
    for label,key in(('Hull − original-view effective mass at tIoU ≥ .5, pp','hull_minus_views_mass50_pp'),
        ('Hull − original-view effective mass at tIoU ≥ .7, pp','hull_minus_views_mass70_pp'),
        ('13-candidate tIoU oracle − Global6, pp','candidate_oracle_tIoU_gap_pp')):
        v=overall[key];lines.append(f'| {label} | {v["mean"]:+.6f} [{v["CI95"][0]:.6f}, {v["CI95"][1]:.6f}] |')
    lines+=['',f'The independently recomputed per-layer expected-length identity has maximum error '
        f'{summary["expected_length_identity_max_error_seconds"]:.3g} seconds. This verifies the hull operator; '
        'it does not by itself show that the actual MBR decision is too long or incorrect. All twelve original view distributions '
        'and six hull distributions have separate mass, length and endpoint-spread summaries in JSON.','',
        '## Fixed descriptive groups','',
        'Event duration: <5 s, 5–10 s, ≥10 s. Unlabeled endpoint disagreement: <.25 s, .25–1 s, ≥1 s. '
        'The disagreement is the six-layer mean of (E|start₀−start₁| + E|end₀−end₁|)/(2 fps); '
        'it includes uncertainty within each view. No group changes online inference or selects confirmation queries.','',
        '| Group | Queries | Overcoverage s | Undercoverage s | tIoU | Effective-mass .5 delta pp |','|---|---:|---:|---:|---:|---:|']
    for group in('event_duration_groups','unlabeled_endpoint_disagreement_groups'):
        for name,row in summary[group].items():
            if not row['queries']:lines.append(f'| {name} | 0 | — | — | — | — |');continue
            x=row['means'];lines.append(f'| {name} | {row["queries"]} | {x["overcoverage_seconds"]:.5f} | {x["undercoverage_seconds"]:.5f} | {x["tIoU"]:.5f} | {row["hull_minus_views_mass50_pp"]["mean"]:+.5f} |')
    lines+=['','## Evidence and limits','',
        'The oracle uses exactly the twelve saved single-view MBR decisions plus saved Global6. It never searches all legal intervals '
        'with GT, generates no additional DP, and is not a deployable result or the full model upper bound. The fixed-space vIoU '
        'oracle is separately labeled; it cannot be substituted for full-method accuracy.','',
        'Benefit and harm cases are selected by the declared symmetric rule and accompany the complete signed anonymous table. '
        'Mass changes and MBR outcomes are distinct: a view may gain effective mass through hull coverage, and MBR can select an '
        'interval different from expected length. Correlations are descriptive and do not isolate a causal mechanism.','',
        public['mechanism']['automatic_descriptive_hint']+'：自动描述提示，非正式机制裁决。需结合全部质量、过/欠覆盖和正反案例作人工结论；第二轮固定干预才检验替换是否有效。','',
        '448 posterior attribution: '+public['448_attribution']['status']+'. '+public['448_attribution']['reason'],'',
        'The two fixed rounds proceed without temperature, layer-weight, duration-prior or spatial-feedback search. '
        'Historical full panels in both domains have prior exposure; parent-isolated frozen follow-up must be called development '
        'recheck unless independent nonexposure can actually be established.']
    return'\n'.join(lines)+'\n'


def evaluate(trial=TRIAL,validate_fn=None):
    """The only real-data/GT entry. Never invoked implicitly at import."""
    trial=Path(trial).resolve();assert trial==TRIAL.resolve()
    if validate_fn is None:
        from scripts.interval_pool.development import validate as validate_fn
    receipts=validate_fn(trial);assert isinstance(receipts,dict)and len(receipts)==64
    lock=read(trial/'lock.json');seal=read(trial/'sealed.json');locksha=sha(trial/'lock.json');sealsha=sha(trial/'sealed.json')
    assert seal['GT_read']is False and seal['queries']==64 and set(receipts)==set(lock['ids'])==set(seal['files'])
    assert not torch.cuda.is_initialized()
    manifest=ROOT/'data/evaluation/development.jsonl';assert sha(manifest)==lock['manifest_sha256']
    records=[json.loads(line)for line in manifest.read_text().splitlines()if line.strip()]
    records=[r for r in records if r['dataset']=='hc2'and r['id']in lock['ids']]
    assert [r['id']for r in records]==lock['ids']
    expected_names=[f'L{l}O{o}'for l in range(1,7)for o in(0,1)]+['Global6']
    # All source/prediction/receipt checks precede this label-file access.
    labels=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(labels)
    previous=read(ROOT/'results/tube_fusion_hc64/report.json')
    assert labelsha==previous['proof']['labels_sha256']
    truth={g['id']:g for g in [json.loads(line)for line in labels.read_text().splitlines()if line.strip()]if g['dataset']=='hc2'and g['id']in lock['ids']}
    assert set(truth)==set(lock['ids'])
    from scripts.state_opd.evaluate import score_boxes
    rows=[];private=[];maxerror=baselineerror=0.;checks=0
    for i,record in enumerate(records):
        key=record['id'];g=truth[key];fps=record['metadata']['fps'];r=receipts[key]
        assert r['GT_read']is False
        posterior_path=(trial/r['posterior_path']).resolve();assert posterior_path.is_relative_to(trial)and sha(posterior_path)==r['posterior_sha256']
        pack=torch.load(posterior_path,map_location='cpu',weights_only=False)
        assert pack['GT_read']is False and list(pack['candidate_names'])==expected_names
        gtboxes={int(k):v for k,v in g['boxes'].items()}
        with np.load(trial/seal['files'][key]['path'],allow_pickle=False)as p:
            assert p['frame_ids'].tolist()==list(pack['ids'])==record['metadata']['frame_ids']
            assert np.array_equal(np.asarray(pack['candidate_intervals'])[-1],p['Global6_interval'])
            layers=pack_attribution(pack,fps,g['interval']);err=interval_error(p['Global6_interval'],g['interval'],fps)
            values={}
            for method,boxes,interval in(('Global6',p['Global6'],p['Global6_interval']),('FixedSpatial',p['FixedSpatial'],p['Global6_interval']),('Source',p['Source'],p['Source_interval'])):
                v,e,_=score_boxes(boxes,record,gtboxes,interval.tolist(),g['interval'],p['positions8'].tolist());checks+=1;maxerror=max(maxerror,e);values[method]=v
            for new,oldname in(('Global6','Global6'),('Source','Source_Native224')):
                old=previous['queries_anonymous'][i]['values'][oldname]
                for field in('v','t','s','observed_actual_sIoU','unobserved_actual_sIoU'):
                    assert(values[new][field]is None)==(old[field]is None)
                    if old[field]is not None:baselineerror=max(baselineerror,abs(values[new][field]-old[field]))
            assert abs(err['tIoU']-values['Global6']['t'])<2e-10
            oracle=candidate_oracle(pack['candidate_intervals'],expected_names,g['interval'])
            fixed_v={};candidate_scores={}
            for name,interval in zip(expected_names,pack['candidate_intervals']):
                v,e,_=score_boxes(p['FixedSpatial'],record,gtboxes,np.asarray(interval).tolist(),g['interval'],p['positions8'].tolist());checks+=1;maxerror=max(maxerror,e)
                fixed_v[name]=v['v'];candidate_scores[name]=v
                assert abs(v['t']-oracle['tIoU_by_candidate'][name])<2e-10
            oracle.update(fixed_spatial_vIoU_by_candidate=fixed_v,best_fixed_spatial_candidate=max(fixed_v,key=fixed_v.get),
                best_fixed_spatial_vIoU=max(fixed_v.values()),Global6_fixed_spatial_vIoU=fixed_v['Global6'],
                oracle_minus_Global6_fixed_spatial_vIoU_pp=100*(max(fixed_v.values())-fixed_v['Global6']),
                fixed_spatial_oracle_independently_selects_vIoU=True,no_new_DP=True)
            row=dict(query=f'H{i+1:03d}',error=err,full_vIoU=values['Global6']['v'],fixed_spatial_vIoU=values['FixedSpatial']['v'],
                Source_metrics=values['Source'],Global6_metrics=values['Global6'],layers=layers,
                endpoint_disagreement_seconds=float(np.mean([x['endpoint_disagreement_seconds']for x in layers])),
                hull_minus_views_mass50=float(np.mean([x['hull_minus_views_mass_tIoU_ge_50']for x in layers])),
                hull_minus_views_mass70=float(np.mean([x['hull_minus_views_mass_tIoU_ge_70']for x in layers])),candidate_oracle=oracle,
                independent_pair_disjoint_probability=float(np.mean([x['independent_pair_disjoint_probability']for x in layers])),
                independent_expected_between_interval_gap_seconds=float(np.mean([x['independent_expected_between_interval_gap_seconds']for x in layers])),
                Global6_minus_mean12_decision_tIoU_pp=100*(values['Global6']['t']-np.mean([oracle['tIoU_by_candidate'][n]for n in expected_names[:-1]])))
            row.update(duration_group=duration_group(err['GT_length_seconds']),disagreement_group=disagreement_group(row['endpoint_disagreement_seconds']))
            rows.append(row);private.append(dict(id=key,parent=g['parent'],interval=p['Global6_interval'].tolist(),GT_interval=g['interval'],candidate_metrics=candidate_scores))
    assert len(rows)==len({x['parent']for x in private})==64 and maxerror<2e-10 and baselineerror<2e-10
    assert sha(labels)==labelsha and sha(trial/'lock.json')==locksha and sha(trial/'sealed.json')==sealsha
    validate_fn(trial);assert not torch.cuda.is_initialized()
    summary=aggregate(rows)
    proof=dict(fingerprint=lock['fingerprint'],lock_sha256=locksha,seal_sha256=sealsha,labels_sha256=labelsha,
        all64_inputs_sources_receipts_predictions_before_GT=True,all_hashes_rechecked_after_scoring=True,
        original_Global6_and_Source_metric_max_error=baselineerror,official_dense_checks=checks,official_dense_max_absolute_error=maxerror,
        candidate_count_per_query=13,candidates_saved_without_GT=True,candidate_oracle_not_all_legal_intervals=True,
        new_RGB_encoder_decoder_expert_DP_or_updates=0,CUDA_initialized=False)
    public=dict(status='completed_round1_HC64_exposed_development_attribution',queries=64,parents=64,proof=proof,summary=summary,
        mechanism=mechanism_conclusion(summary),cases=select_cases(rows),queries_anonymous=rows,
        reference_448='Historical Strong precision/cost reference only.',
        **{'448_attribution':dict(status='not measured',reason='No complete448 interval-posterior artifact is supplied to this attribution; no new448 computation is authorized or performed.')})
    assert not(PUBLIC/'diagnostics.json').exists(),'Do not overwrite an earlier diagnosis'
    atomic(trial/'diagnostics.json',dict(public,private_rows=private));atomic(PUBLIC/'diagnostics.json',public)
    (PUBLIC/'error_attribution.md').write_text(_render(public))
    flat=[]
    for row in rows:
        flat.append(dict(query=row['query'],**row['error'],duration_group=row['duration_group'],disagreement_group=row['disagreement_group'],
            full_vIoU=row['full_vIoU'],fixed_spatial_vIoU=row['fixed_spatial_vIoU'],endpoint_disagreement_seconds=row['endpoint_disagreement_seconds'],
            hull_minus_views_mass50_pp=100*row['hull_minus_views_mass50'],hull_minus_views_mass70_pp=100*row['hull_minus_views_mass70'],
            candidate_oracle_headroom_tIoU_pp=row['candidate_oracle']['oracle_minus_Global6_tIoU_pp']))
    fields=list(flat[0]);(PUBLIC/'signed_errors.tsv').write_text('\n'.join(['\t'.join(fields)]+['\t'.join(str(row[f])for f in fields)for row in flat])+'\n')
    atomic(trial/'diagnosis_receipt.json',dict(status='diagnosed',proof=proof,public_json_sha256=sha(PUBLIC/'diagnostics.json'),
        public_md_sha256=sha(PUBLIC/'error_attribution.md'),signed_errors_sha256=sha(PUBLIC/'signed_errors.tsv')))
    return public


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-after-seal',action='store_true');args=parser.parse_args()
    if not args.run_after_seal:parser.error('Explicit --run-after-seal and complete64 seal required')
    result=evaluate();print(json.dumps(dict(status=result['status'],summary=result['summary'],mechanism=result['mechanism']),indent=2))
