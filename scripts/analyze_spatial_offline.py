"""CPU-only post-seal spatial analysis. Never infer, tune, or inspect partial jobs.

Usage: CUDA_VISIBLE_DEVICES='' python -B scripts/analyze_spatial_offline.py
Public output contains aggregate numbers only; query records stay under outputs/.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.metrics import DenseTube, box_iou
from stvg.metrics_base import xyxy


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def lines(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def clean(v):
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def avg(a):
    return float(np.mean(a)) if len(a) else None


def cxwh(b):
    return np.concatenate(((b[..., :2]+b[..., 2:])/2, b[..., 2:]-b[..., :2]), -1)


def chart(b):
    return np.concatenate((b[..., :2], np.log(b[..., 2:])), -1)


def local_cost(b, target):
    # Exact same 5*L1(cxcywh)+2*(1-GIoU) definition as stvg.geometry.
    a = xyxy(b, 1, 1); t = xyxy(target, 1, 1)
    inter = np.maximum(np.minimum(a[..., 2:], t[..., 2:])-np.maximum(a[..., :2], t[..., :2]), 0).prod(-1)
    union = b[..., 2:].prod(-1)+target[..., 2:].prod(-1)-inter
    enclosure = (np.maximum(a[..., 2:], t[..., 2:])-np.minimum(a[..., :2], t[..., :2])).prod(-1)
    giou = inter/union-(enclosure-union)/enclosure
    return 5*np.abs(b-target).sum(-1)+2*(1-giou)


def frame_metrics(a, b, gt, mask, covered):
    """a=initial E_Ref, b=Full. All arrays share a fixed GT support."""
    a, b, gt, covered = a[mask], b[mask], gt[mask], covered[mask]
    ia, ib = box_iou(a, gt), box_iou(b, gt)
    ia[~covered] = 0; ib[~covered] = 0
    bad, good = ia < .3, ia >= .5
    out = dict(frames=len(a), initial_s=avg(ia), final_s=avg(ib), delta_s=avg(ib-ia),
               recovery_den=int(bad.sum()), recovery_num=int((bad & (ib >= .5)).sum()),
               damage_den=int(good.sum()), damage_num=int((good & (ib < .3)).sum()),
               missing_prediction_frames=int((~covered).sum()))
    out['recovery'] = out['recovery_num']/out['recovery_den'] if out['recovery_den'] else None
    out['damage'] = out['damage_num']/out['damage_den'] if out['damage_den'] else None
    aa, bb, gg = cxwh(a), cxwh(b), cxwh(gt)
    valid = covered & (aa[..., 2:] > 0).all(-1) & (bb[..., 2:] > 0).all(-1) & (gg[..., 2:] > 0).all(-1)
    out['geometry_valid_frames'] = int(valid.sum())
    aa, bb, gg = aa[valid], bb[valid], gg[valid]
    if len(aa):
        D = np.concatenate((aa[:, 2:], np.ones_like(aa[:, 2:])), -1)
        update = (chart(bb)-chart(aa))/D
        residual = (chart(gg)-chart(aa))/D
        un, rn = np.linalg.norm(update, axis=-1), np.linalg.norm(residual, axis=-1)
        both = (un > 1e-12) & (rn > 1e-12)
        dot = (update*residual).sum(-1)
        energy = (update*update).sum(-1)-2*dot
        direct = (((chart(bb)-chart(gg))/D)**2).sum(-1)-(residual**2).sum(-1)
        assert np.allclose(energy,direct,rtol=1e-9,atol=1e-8)
        alignment = dot[both]/(un[both]*rn[both])
        out.update(alignment=avg(alignment), alignment_valid_frames=int(both.sum()),
                   zero_update_frames=int((un <= 1e-12).sum()), zero_gt_residual_frames=int((rn <= 1e-12).sum()),
                   magnitude_ratio=avg(un[both]/rn[both]), aligned_fraction=avg((dot[both] > 0).astype(float)),
                   direction_wrong_fraction=avg((dot[both] < 0).astype(float)),
                   overshoot_fraction=avg(((dot[both] > 0) & (energy[both] > 1e-12)).astype(float)),
                   squared_error_improve_fraction=avg((energy < -1e-12).astype(float)),
                   squared_error_delta=avg(energy),
                   initial_center_error=avg(np.linalg.norm((aa[:, :2]-gg[:, :2])/gg[:, 2:], axis=-1)),
                   final_center_error=avg(np.linalg.norm((bb[:, :2]-gg[:, :2])/gg[:, 2:], axis=-1)),
                   initial_size_error=avg(np.abs(np.log(aa[:, 2:]/gg[:, 2:])).mean(-1)),
                   final_size_error=avg(np.abs(np.log(bb[:, 2:]/gg[:, 2:])).mean(-1)))
        out['delta_center_error'] = out['final_center_error']-out['initial_center_error']
        out['delta_size_error'] = out['final_size_error']-out['initial_size_error']
    return out


def estimate(rows, getter, scale=1):
    """Equal-query estimate, paired parent-cluster CI (not independent frames)."""
    usable = [(r['parent'], getter(r)) for r in rows]
    usable = [(p, float(v)*scale) for p, v in usable if v is not None and np.isfinite(v)]
    if not usable: return dict(mean=None, CI95=None, queries=0, parents=0)
    parents = list(dict.fromkeys(p for p, _ in usable)); ix = {p:i for i,p in enumerate(parents)}
    sums = np.zeros(len(parents)); counts = np.zeros(len(parents))
    for p, v in usable: sums[ix[p]] += v; counts[ix[p]] += 1
    rng = np.random.default_rng(20261021); samples = rng.integers(0, len(parents), (10000, len(parents)))
    boot = sums[samples].sum(1)/counts[samples].sum(1)
    return dict(mean=sum(v for _,v in usable)/len(usable), CI95=np.quantile(boot, [.025,.975]).tolist(),
                queries=len(usable), parents=len(parents))


def summarize(rows):
    out = dict(queries=len(rows), parents=len({r['parent'] for r in rows}),
               delta_v_pp=estimate(rows, lambda r:r['delta_v'], 100),
               delta_overlap_s_pp=estimate(rows, lambda r:r['delta_overlap_s'], 100), groups={})
    for name in ['dense', 'actual', 'observed', 'unobserved', 'unobserved_dense']:
        fields=['initial_s','final_s','delta_s','recovery','damage','alignment','aligned_fraction',
                'direction_wrong_fraction','overshoot_fraction','magnitude_ratio','squared_error_delta',
                'squared_error_improve_fraction','delta_center_error','delta_size_error']
        out['groups'][name] = {k:estimate(rows,lambda r,k=k,n=name:r['groups'][n].get(k),
                                    100 if k in ['initial_s','final_s','delta_s','recovery','damage',
                                                 'aligned_fraction','direction_wrong_fraction','overshoot_fraction',
                                                 'squared_error_improve_fraction'] else 1) for k in fields}
        g=out['groups'][name]
        for k in ['frames','recovery_den','recovery_num','damage_den','damage_num','geometry_valid_frames',
                  'alignment_valid_frames','zero_update_frames','zero_gt_residual_frames','missing_prediction_frames']:
            g[k] = sum(r['groups'][name].get(k,0) for r in rows)
        g['pooled_frame_recovery_percent'] = 100*g['recovery_num']/g['recovery_den'] if g['recovery_den'] else None
        g['pooled_frame_damage_percent'] = 100*g['damage_num']/g['damage_den'] if g['damage_den'] else None
    out['teacher_cost_initial'] = estimate(rows, lambda r:r.get('teacher_cost_initial'))
    out['teacher_cost_final'] = estimate(rows, lambda r:r.get('teacher_cost_final'))
    out['teacher_cost_delta'] = estimate(rows, lambda r:r.get('teacher_cost_delta'))
    out['teacher_quality'] = estimate(rows, lambda r:r.get('teacher_quality'),100)
    out['tail']={}
    for field in ['delta_v','delta_dense_s','delta_actual_s','delta_unobserved_s']:
        d=np.array([r[field]*100 for r in rows if r.get(field) is not None]); n=len(d)
        out['tail'][field]=dict(queries=n, positive=int((d>1e-8).sum()), negative=int((d < -1e-8).sum()),
            unchanged=int((np.abs(d)<=1e-8).sum()), below_minus5=int((d < -5).sum()),
            positive_contribution_pp=float(np.maximum(d,0).sum()/n) if n else None,
            negative_contribution_pp=float(np.minimum(d,0).sum()/n) if n else None,
            quantiles_pp=np.quantile(d,[0,.05,.1,.5,.9,.95,1]).tolist() if n else [])
    return out


def main(dataset):
    root=ROOT/'outputs/formal'/ 'tastvg'/dataset
    state=read(ROOT/'outputs/formal/state.json'); jobstate=state['jobs']['tastvg-'+dataset]
    if jobstate['status']!='scored': raise ValueError('Only fully sealed and scored jobs are eligible; no partial scoring')
    plan=read(ROOT/'outputs/formal/plan.json');job=next(j for j in plan['jobs'] if j['id']=='tastvg-'+dataset)
    from tools.run_formal import validate_worker
    validate_worker(ROOT,job,require_seal=True)
    seal=read(root/'sealed.json')
    assert sha(root/'sealed.json')==jobstate['seal_sha256']
    assert sha(root/'metrics.json')==jobstate['metrics_sha256']
    assert sha(ROOT/job['labels'])==job['labels_sha256']==jobstate['scoring_labels_sha256']
    records=lines(ROOT/job['manifest']); metrics=read(root/'metrics.json'); old={r['id']:r for r in metrics['rows']}
    assert len(records)==len(seal['files'])==len(old)==job['queries']
    # validate_worker has hashed ALL sealed predictions before any GT is opened.
    truth={r['id']:r for r in lines(ROOT/job['labels'])}
    cache_index={}
    for p in (ROOT/'outputs/formal/teacher_cache').glob('*.json'):
        m=read(p);c=m.get('context',{})
        if c.get('dataset')!=dataset or c.get('fingerprint')!=seal['runtime_fingerprint']: continue
        key=(c['caption'],tuple(c['frame_ids']),tuple(c['positions']))
        cache_index.setdefault(key,[]).append((p,m))
    rows=[];teacher_proofs=[];max_metric_err=0.;max_factor_err=0.;coverage_gaps=0
    for idx,rec in enumerate(records):
        key=rec['id'];meta=rec['metadata'];g=truth[key];span=g['interval'];gt={int(k):v for k,v in g['boxes'].items()}
        assert g['dataset']==dataset
        with np.load(root/seal['files'][key]['path'],allow_pickle=False) as z:
            ids=z['frame_ids'].copy();pos=z['positions8'].copy();base=z['E_Ref'].copy();full=z['Live'].copy();iv=z['Strong_interval'].copy()
        row=dict(frame_ids=ids,input=meta); initial=DenseTube(base,row,gt,span,clip=dataset=='hc2'); final=DenseTube(full,row,gt,span,clip=dataset=='hc2')
        for arm,dense in [('WithoutOPD',initial),('Full',final)]:
            vals=dense.score(iv)
            for k in ['v','t','s']: max_metric_err=max(max_metric_err,abs(vals[k]-old[key]['values'][arm][k]))
        a=xyxy(base,meta['width'],meta['height']);b=xyxy(full,meta['width'],meta['height'])
        if dataset=='hc2': a=np.maximum(a,0);b=np.maximum(b,0)
        fids=np.array(sorted(gt));target=np.array([gt[int(f)] for f in fids]);observed=set(ids[pos].tolist())
        da=np.column_stack([np.interp(fids,ids,a[:,k]) for k in range(4)])
        db=np.column_stack([np.interp(fids,ids,b[:,k]) for k in range(4)])
        covered=(fids>=ids[0]) & (fids<=ids[-1])
        actual=np.isin(fids,ids);obs=np.isin(fids,list(observed));unobs=actual & ~obs
        masks=dict(dense=np.ones(len(fids),bool), actual=actual,observed=obs,unobserved=unobs,unobserved_dense=~obs)
        rr=dict(id=key,parent=g['parent'],ordinal=idx+1,groups={n:frame_metrics(da,db,target,m,covered)for n,m in masks.items()})
        for n,field in [('dense','s'),('actual','fixed_actual_sIoU'),('observed','observed_actual_sIoU'),('unobserved','unobserved_actual_sIoU')]:
            for tag,arm in [('initial_s','WithoutOPD'),('final_s','Full')]:
                saved=old[key]['values'][arm][field];value=rr['groups'][n][tag]
                assert (saved is None)==(value is None)
                if saved is not None:max_metric_err=max(max_metric_err,abs(saved-value))
        values=initial.score(iv);rr.update(tiou=values['t'],delta_v=final.score(iv)['v']-values['v'],
            delta_dense_s=rr['groups']['dense']['delta_s'],delta_actual_s=rr['groups']['actual']['delta_s'],delta_unobserved_s=rr['groups']['unobserved']['delta_s'])
        overlap=max(0,min(iv[1],span[1])-max(iv[0],span[0]));mask=(fids>=max(iv[0],span[0]))&(fids<min(iv[1],span[1]))
        if overlap:
            ds=float((final.iou[mask]-initial.iou[mask]).sum()/overlap)
            max_factor_err=max(max_factor_err,abs(rr['delta_v']-rr['tiou']*ds));coverage_gaps+=int(mask.sum()!=overlap);rr['delta_overlap_s']=ds
        else:rr['delta_overlap_s']=None;assert abs(rr['delta_v'])<1e-12
        cachekey=(meta['caption'],tuple(ids.tolist()),tuple(pos.tolist())); candidates=cache_index.get(cachekey,[])
        assert candidates, 'No exact-context teacher cache; do not invent references'
        hashes={m['sha256']for _,m in candidates};assert len(hashes)==1,'Ambiguous teacher evidence'
        p,m=candidates[0];payload=p.with_suffix('.pt');assert sha(payload)==m['sha256'];teacher_proofs.append((p.name,m['sha256']))
        import torch
        teacher=np.asarray(torch.load(payload,map_location='cpu',weights_only=False)['boxes'],dtype=float)
        assert teacher.shape==(len(pos),4)
        from stvg.geometry import local_cost as original_cost
        c0=local_cost(base[pos],teacher);c1=local_cost(full[pos],teacher)
        assert np.max(np.abs(c0-original_cost(torch.tensor(base[pos]),torch.tensor(teacher)).numpy()))<1e-10
        rr.update(teacher_cost_initial=float(c0.mean()),teacher_cost_final=float(c1.mean()),teacher_cost_delta=float((c1-c0).mean()),
                  teacher_cost_reduced=float(c1.mean()<c0.mean()-1e-12),
                  d0=float(np.linalg.norm((chart(base[pos])-chart(teacher))/np.concatenate((base[pos,2:],np.ones_like(base[pos,2:])),axis=-1),axis=-1).mean()))
        teacherpix=xyxy(teacher,meta['width'],meta['height'])
        if dataset=='hc2':teacherpix=np.maximum(teacherpix,0)
        anchors=[i for i,p0 in enumerate(pos)if int(ids[p0]) in gt]
        rr['teacher_quality']=avg([float(box_iou(teacherpix[i],gt[int(ids[pos[i]])]))for i in anchors]);rr['teacher_gt_anchors']=len(anchors)
        rr['distance_bins']={}
        dist=np.min(np.abs(fids[:,None]-ids[pos][None,:]),axis=-1)/float(meta['fps'])
        for n,m0 in [('0',dist==0),('(0,0.5]',(dist>0)&(dist<=.5)),('(0.5,1]',(dist>.5)&(dist<=1)),('(1,2]',(dist>1)&(dist<=2)),('>2',dist>2)]:
            rr['distance_bins'][n]=dict(frames=int(m0.sum()),delta_s=avg(final.iou[m0]-initial.iou[m0]))
        rows.append(rr)
        if (idx+1)%400==0:print('processed',idx+1,flush=True)
    assert max_metric_err<2e-10 and max_factor_err<2e-10
    result=dict(status='completed_offline_analysis_of_sealed_predictions',dataset=dataset,
        created_utc=datetime.now(timezone.utc).isoformat(),comparison='Full - WithoutOPD, identical Strong intervals',
        provenance=dict(plan_sha256=state['plan_sha256'],manifest_sha256=job['manifest_sha256'],labels_sha256=job['labels_sha256'],
            metrics_sha256=jobstate['metrics_sha256'],seal_sha256=jobstate['seal_sha256'],runtime_fingerprint=seal['runtime_fingerprint'],
            script_sha256=sha(__file__),teacher_evidence_hashes_sha256=hashlib.sha256(json.dumps(sorted(teacher_proofs)).encode()).hexdigest()),
        thresholds=dict(recovery='initial IoU <0.3 and final IoU >=0.5',damage='initial IoU >=0.5 and final IoU <0.3',
                        vector_nonzero=1e-12,serious_drop_pp=5),
        aggregation='equal query means; 10000 parent-cluster paired bootstrap, seed20261021; recovery/damage macro over eligible queries, pooled frame counts separately',
        geometry='cxcylogwh after official HC lower-coordinate clipping; center divided by initial width/height, log sizes unit scale; invalid sizes and frames outside sampled support excluded only from geometry; outside-support IoU zero as official, retained in IoU/recovery denominators',
        quality_scope='GT available planned anchors only; teacher discrepancies use all eight observed positions',
        checks=dict(max_saved_metric_absolute_error=max_metric_err,max_temporal_factorization_error=max_factor_err,overlap_GT_coverage_gap_queries=coverage_gaps,
                    no_model_inference=True,no_expert_calls=True,no_parameter_changes=True,only_complete_jobs_scored=True),
        overall=summarize(rows),strata={},distance={})
    for name,choose in [('tIoU=0',lambda r:r['tiou']==0),('0<tIoU<0.3',lambda r:0<r['tiou']<.3),
                        ('0.3<=tIoU<0.5',lambda r:.3<=r['tiou']<.5),('tIoU>=0.5',lambda r:r['tiou']>=.5)]:
        rr=[r for r in rows if choose(r)];result['strata'][name]=summarize(rr)
    for field in ['d0','teacher_quality']:
        edges=np.quantile([r[field]for r in rows if r[field]is not None],[.25,.5,.75]);result[field+'_quartile_edges']=edges
        result['strata'][field]={}
        for i in range(4):
            subset=[r for r in rows if r[field]is not None and int(np.searchsorted(edges,r[field],side='right'))==i]
            result['strata'][field]['Q'+str(i+1)]=summarize(subset)
    for n in rows[0]['distance_bins']:
        result['distance'][n]=dict(delta_s_pp=estimate(rows,lambda r,n=n:r['distance_bins'][n]['delta_s'],100),
                                   frames=sum(r['distance_bins'][n]['frames']for r in rows))
    result['teacher_cost_reduction_query_percent']=estimate(rows,lambda r:r['teacher_cost_reduced'],100)
    result['not_available']=['Direct/Fixed/FitNets/Soft-KD matched formal predictions absent; no such comparison inferred',
        'Formal NPZ saves only endpoint arrays; policy path/rounds not serialized, so no per-step cost, candidate coverage or ESS curve']
    validate_worker(ROOT,job,require_seal=True)
    assert sha(root/'sealed.json')==jobstate['seal_sha256'] and sha(root/'metrics.json')==jobstate['metrics_sha256']
    assert sha(ROOT/job['labels'])==job['labels_sha256']
    assert not torch.cuda.is_initialized(), 'Offline analysis must never initialize CUDA'
    result['checks'].update(prediction_seal_revalidated_after_analysis=True,CUDA_initialized=False)
    public=ROOT/'results/formal'/('spatial_analysis_'+dataset+'.json');public.write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    private=ROOT/'outputs/analysis'/('spatial_'+dataset);private.mkdir(parents=True,exist_ok=True)
    (private/'queries.json').write_text(json.dumps(clean(rows),indent=2,allow_nan=False)+'\n')
    print('completed',len(rows),'queries',result['overall']['parents'],'parents','metric_error',max_metric_err,'factor_error',max_factor_err,flush=True)
    print('public',public,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hc2','vidstg'],default='hc2');a=p.parse_args();main(a.dataset)
