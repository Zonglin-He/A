"""Remove cycle-based admission/weighting, retain exact bidirectional geometry.

Only saved tensors are used. Full time and Native2D region pools stay fixed to
isolate spatial selection. Every prediction seals before independent CPU labels.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from full import read,sha,atomic,identity,check,PROVIDER,RUN
from stvg.evidence import SourceEvidence
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg._spatial_math import occupancy,soft_dice,EPS
D=ROOT/'outputs/cycle_reliability_ablation_v1'
P=ROOT/'results/cycle_reliability_ablation_v1'
ARMS=('Source','Current','NoCycle')
PAIRS=(('NoCycle','Current'),('NoCycle','Source'),('Current','Source'))

def load_evidence(item,cap):
    raw=torch.load(item['raw_path'],map_location='cpu',weights_only=False)
    mask=torch.load(item['mask_path'],map_location='cpu',weights_only=False)
    assert sha(item['raw_path'])==mask['source_cache_sha256']
    j=int(mask['selected_query']);assert np.array_equal(raw['Source'],cap['Source'])
    assert list(raw['frame_ids'])==list(cap['frame_ids']) and list(raw['planned_positions'])==list(cap['positions8'])
    assert torch.equal(mask['raw_out']['pred_boxes'][0,:,j],torch.as_tensor(cap['geometry']['native_boxes']).float())
    return SourceEvidence.from_capture(raw['spatial_inputs'],cap['frame_ids'],cap['positions8'],mask['raw_out']['pred_masks'][0,:,j].float(),mask['input_shape'][-2:],selected_query=j)

def reweight(g,ev,*,use_cycle):
    """Rebuild ALL downstream cycle-dependent fields without refitting geometry.

    Unavailable directions remain0; every geometrically available direction gets1
    for NoCycle. The original affine-fit validity checks are never relaxed.
    """
    h=SpatialGeometry.__new__(SpatialGeometry);h.__dict__.update(copy.deepcopy(g.__dict__))
    h.side_reliability=g.side_reliability.clone() if use_cycle else g.side_available.double()
    total=h.side_reliability.sum(-1)
    h.confidence=total/h.side_available.sum(-1).clamp_min(1)
    h.merged_targets=(h.side_maps*h.side_reliability[...,None,None]).sum(1)/total.clamp_min(EPS)[:,None,None]
    h.merged_targets*=ev.valid.reshape(h.frames,h.height,h.width)
    h.reliable=(h.confidence>=.1)&h.side_available.any(-1)
    h.valid=torch.zeros_like(g.valid);h.valid[:,0]=True
    h.candidate_origin=torch.full_like(g.candidate_origin,-1);h.candidate_quality=torch.zeros_like(g.candidate_quality)
    for t in range(h.frames):
        for side in (0,1):
            if h.side_available[t,side] and (h.reliable[t] or h.observed[t]):
                h.valid[t,side+1]=True;h.candidate_origin[t,side+1]=-2 if h.observed[t] else side
                h.candidate_quality[t,side+1]=h.side_reliability[t,side]
        if h.valid[t,1] and h.valid[t,2] and torch.equal(h.boxes[t,1],h.boxes[t,2]):h.valid[t,2]=False
    side_score=.5*h.extent_scores+.5*h.association_scores
    h.projection=(side_score*h.side_reliability[:,None]).sum(-1)/total.clamp_min(EPS)[:,None];h.u=h.projection
    occ,_,_=occupancy(h.boxes,h.height,h.width);occ*=ev.valid.reshape(h.frames,1,h.height,h.width)
    masked=(occ*h.merged_targets[:,None]).flatten(-2)
    previous=ev.transport_adjacent(masked[:-1].transpose(0,1)).transpose(0,1)
    h.link=soft_dice(previous[:,:,None],masked[1:,None])
    if not use_cycle:h.metadata={**h.metadata,'confidence':'1 for a geometrically available side, else0; no return-cycle screening or weighting'}
    return h

def tensor_equal(a,b,fields=None):
    for f in fields or [k for k,v in a.__dict__.items() if torch.is_tensor(v)]:
        assert torch.equal(getattr(a,f),getattr(b,f)),f

def verify():
    p=read(D/'plan.json');assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    for f,h in p['code'].items():assert sha(ROOT/f)==h,f
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert not torch.cuda.is_initialized();return p

def prepare():
    assert not(D/'plan.json').exists();prod=check();pre=read(D/'source_only_preflight.json');assert pre['status']=='passed' and pre['queries']==67 and pre['all_cached_geometry_tensor_fields_exact']
    code={f:sha(ROOT/f)for f in prod['code']}
    for f in ('studies/cycle_reliability_ablation/run.py','studies/cycle_reliability_ablation/preflight.py'):code[f]=sha(ROOT/f)
    inputs={str(f):sha(f)for f in [D/'source_only_preflight.json',RUN/'plan.json',RUN/'hc2/sealed.json',RUN/'hc2/score_receipt.json']}
    for i in pre['items']:
        for k in ('raw','mask','capture','prediction'):
            assert sha(i[k+'_path'])==i[k+'_sha256'];inputs[i[k+'_path']]=i[k+'_sha256']
    p=dict(schema='cycle_reliability_ablation_v1',authorization='User requests cached removal of return-cycle reliability screening and weighting, preserving bidirectional geometric propagation; compare unobserved-frame sIoU, complete vIoU and severe harm without interrupting current full.',
        primary='NoCycle-Current',arms=ARMS,pairs=PAIRS,
        intervention='Available propagation directions use reliability1 instead of return-cycleDice; unavailable directions0. Remove cycle gate and cycle weights in side support fusion, candidate projection, unobserved unary and adjacent pair terms. Recompute merged target and link using exact cached Source forward adjacent transport. Never reuse old link after changing merged support.',
        preserved='Exact bidirectional affinities/affine-fitted complete boxes, fit mass/ESS/variance/scale validity, native eight observations, seeds, candidates/features/appearance association/extent/guard, E, all DP/tie order and score coefficients. No extra candidates fabricated for invalid geometry.',
        fixed_time='Current frozen native-prior kappa64/beta1/ONEround Full interval for BOTH Current and NoCycle. Native2D features/auxiliary trajectory/posterior held fixed. This isolates spatial reliability; not an end-to-end rerun with a changed temporal-feedback ROI.',
        metrics='Fixed actual GT-supported sampled nodes; identical observed/unobserved split, per-query average then equal-query mean. Exclude empty metric support and report effective N. Complete official vIoU at shared Full interval. Separate severe>5pp complete-v, fixed-unobserved-s, and union (one query once); explicit Current and Source baselines.',
        scope='67 exact-official HC queries with fully reconstructible saved spatial evidence from historical128HC development inputs; selection solely input/cache availability, before labels. Other compact caches omit source adjacency needed for faithful link recomputation, no new capture. Historically exposed, not full1902 or fresh confirmation; no Vid claim.',
        budget=dict(queries=67,new_model=0,new_RGB=0,new_expert=0,new_training=0,GPU=0),production_unchanged=True,no_selection=True,GT_in_prediction=False,code=code,inputs=inputs,items=pre['items'],job=prod['jobs']['hc2'],production_fingerprint=prod['fingerprint'],seed=20260920)
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(D/'plan.json',p)
    for f in code:
        dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    atomic(P/'protocol.json',{k:v for k,v in p.items()if k not in ('items','inputs','job')})
    atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print('prepared',p['fingerprint'],flush=True)

def predict():
    p=verify();assert not(D/'sealed.json').exists();torch.set_num_threads(1)
    atomic(D/'worker.json',identity());atomic(D/'state.json',dict(status='predicting',**identity()))
    files={};start=time.perf_counter();counts=dict(restored_geometry_tensor_exact=0,current_final_boxes_exact=0,geometry_candidates_features_exact=0,uniform_idempotence=0,unavailable_not_admitted=0,reenabled_nodes=0,reenabled_candidate_slots=0,changed_actions=0,changed_unobserved_actions=0)
    keep=('boxes','candidates','side_reference_boxes','side_available','source_anchors','native_boxes','features','E_features','extent_scores','association_scores','guard','frame_ids','observed','seed_features','seed_valid')
    for n,i in enumerate(p['items'],1):
        cap=torch.load(i['capture_path'],map_location='cpu',weights_only=False);ev=load_evidence(i,cap);g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry'])
        restored=reweight(g,ev,use_cycle=True);tensor_equal(g,restored);counts['restored_geometry_tensor_exact']+=1
        h=reweight(g,ev,use_cycle=False);tensor_equal(g,h,keep);counts['geometry_candidates_features_exact']+=1
        h2=reweight(h,ev,use_cycle=False);tensor_equal(h,h2);counts['uniform_idempotence']+=1
        assert not h.valid[~h.side_available.any(-1),1:].any() and not h.reliable[~h.side_available.any(-1)].any();counts['unavailable_not_admitted']+=1
        counts['reenabled_nodes']+=int((h.reliable&~g.reliable&~g.observed).sum());counts['reenabled_candidate_slots']+=int((h.valid&~g.valid).sum())
        with np.load(i['prediction_path'],allow_pickle=False)as a:
            span=a['Full_interval'].copy();old=evidence_dp(restored.for_interval(span));new=evidence_dp(h.for_interval(span))
            assert np.array_equal(old['final'].numpy(),a['Full']) and np.array_equal(old['actions'].numpy(),a['Full_actions']);counts['current_final_boxes_exact']+=1
            diff=old['actions']!=new['actions'];counts['changed_actions']+=int(diff.sum());counts['changed_unobserved_actions']+=int((diff&~g.observed).sum())
            arr=dict(Source=a['Source'].copy(),Source_interval=a['Source_interval'].copy(),Current=a['Full'].copy(),Current_interval=span,NoCycle=new['final'].numpy(),NoCycle_interval=span,positions8=a['positions8'].copy(),frame_ids=a['frame_ids'].copy(),Current_actions=old['actions'].numpy(),NoCycle_actions=new['actions'].numpy())
        key=f'H{n:03d}';file=D/'predictions'/(key+'.npz');file.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(file,**arr)
        trace=D/'traces'/(key+'.pt');trace.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(Current_confidence=g.confidence,NoCycle_confidence=h.confidence,Current_valid=g.valid,NoCycle_valid=h.valid,Current_link=g.link,NoCycle_link=h.link,GT_read=False),trace)
        files[i['source_key']]=dict(alias=key,path=str(file.relative_to(D)),sha256=sha(file),trace=str(trace.relative_to(D)),trace_sha256=sha(trace));atomic(D/'progress.json',dict(completed=n,total=67,GT_read=False))
        if n%16==0:print('predicted',n,flush=True)
    verify();assert all(counts[k]==67 for k in ('restored_geometry_tensor_exact','current_final_boxes_exact','geometry_candidates_features_exact','uniform_idempotence','unavailable_not_admitted'))
    atomic(D/'sealed.json',dict(status='sealed',queries=67,fingerprint=p['fingerprint'],GT_read=False,files=files,checks=counts,seconds=time.perf_counter()-start));atomic(D/'state.json',dict(status='sealed_waiting_independent_CPU_score',fingerprint=p['fingerprint']));print('sealed',counts,flush=True)

def summarize(rows):
    from full_score import cluster
    fields=list(rows[0]['values']['Source']);means={a:{f:float(np.mean([x['values'][a][f]for x in rows if x['values'][a][f]is not None]))if any(x['values'][a][f]is not None for x in rows)else None for f in fields}for a in ARMS}
    pairs={a+'-'+b:cluster(rows,a,b)for a,b in PAIRS};risk={}
    for a,b in PAIRS:
        vv=[x['values'][a]['v']-x['values'][b]['v']<-.05 for x in rows];ss=[x['values'][a]['unobserved_actual_sIoU']-x['values'][b]['unobserved_actual_sIoU']<-.05 if x['values'][a]['unobserved_actual_sIoU']is not None and x['values'][b]['unobserved_actual_sIoU']is not None else False for x in rows]
        risk[a+'-'+b]=dict(complete_v_count=sum(vv),unobserved_s_count=sum(ss),union_count=sum(x or y for x,y in zip(vv,ss)),queries=len(rows),union_percent=100*sum(x or y for x,y in zip(vv,ss))/len(rows))
    return dict(queries=len(rows),parents=len({x['parent']for x in rows}),means=means,paired=pairs,severe=risk,valid_unobserved_queries={a:sum(x['values'][a]['unobserved_actual_sIoU']is not None for x in rows)for a in ARMS})

def score():
    from full_score import metric
    p=verify();s=read(D/'sealed.json');assert s['queries']==67 and not s['GT_read'] and s['fingerprint']==p['fingerprint'] and not(D/'score_receipt.json').exists();torch.set_num_threads(1)
    atomic(D/'score_worker.json',identity());atomic(D/'state.json',dict(status='scoring',**identity()))
    for e in s['files'].values():assert sha(D/e['path'])==e['sha256'] and sha(D/e['trace'])==e['trace_sha256']
    lab=PROVIDER/p['job']['labels'];assert sha(lab)==p['job']['labels_sha256'];truth={x['id']:x for x in map(json.loads,lab.read_text().splitlines())if x['dataset']=='hc2'}
    original={x['id']:x for x in read(RUN/'hc2/metrics.json')['private_rows']};parent_names=sorted({str(truth[i['source_key']]['parent'])for i in p['items']});pm={v:f'P{k+1:03d}'for k,v in enumerate(parent_names)}
    rows=[];checks=0;maxerr=0.;start=time.perf_counter();signed=['query\tparent\tduration_group\tmethod\tbaseline\tdelta_unobserved_s_pp\tdelta_complete_v_pp\tdelta_t_pp'];severe=[signed[0]]
    for i in p['items']:
        row=i['row'];gt=truth[i['source_key']];span=gt['interval'];boxes={int(k):v for k,v in gt['boxes'].items()};e=s['files'][i['source_key']];vals={}
        with np.load(D/e['path'],allow_pickle=False)as a:
            assert np.array_equal(a['Current_interval'],a['NoCycle_interval'])
            for arm in ARMS:vals[arm],err=metric(a[arm],row,boxes,span,a[arm+'_interval'].tolist(),a['positions8']);checks+=1;maxerr=max(maxerr,err)
        for a,b in [('Source','Source'),('Current','Full')]:assert vals[a]==original[i['source_key']]['values'][b]
        assert vals['Current']['t']==vals['NoCycle']['t']
        rr=dict(query=e['alias'],parent=pm[str(gt['parent'])],duration_group='short_le5s'if(span[1]-span[0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals);rows.append(rr)
        for a,b in PAIRS:
            dv=100*(vals[a]['v']-vals[b]['v']);dt=100*(vals[a]['t']-vals[b]['t']);ds=100*(vals[a]['unobserved_actual_sIoU']-vals[b]['unobserved_actual_sIoU'])if vals[a]['unobserved_actual_sIoU']is not None and vals[b]['unobserved_actual_sIoU']is not None else None
            line=f"{e['alias']}\t{rr['parent']}\t{rr['duration_group']}\t{a}\t{b}\t{ds if ds is not None else ''}\t{dv:.12g}\t{dt:.12g}";signed.append(line)
            if dv< -5 or(ds is not None and ds< -5):severe.append(line)
    groups=dict(all67=summarize(rows));groups['all67']['duration']={tag:summarize([x for x in rows if x['duration_group']==tag])for tag in ('short_le5s','long_gt5s')if any(x['duration_group']==tag for x in rows)}
    assert checks==201 and maxerr<2e-10 and len(rows)==67 and not torch.cuda.is_initialized()
    report=dict(status='complete_scored',dataset='hc2',groups=groups,rows=rows,checks=checks,max_metric_error=maxerr,seconds=time.perf_counter()-start,exposure=p['scope'],plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),proof=s['checks'],time_identical_all67=True,no_selection=True,production_unchanged=True)
    atomic(P/'report.json',report);(P/'signed.tsv').write_text('\n'.join(signed)+'\n');(P/'severe.tsv').write_text('\n'.join(severe)+'\n');verify()
    atomic(D/'score_receipt.json',dict(status='scored',report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),**identity()));atomic(D/'state.json',dict(status='complete_scored',fingerprint=p['fingerprint'],production_changed=False))
    g=groups['all67'];print(json.dumps(dict(means=g['means'],paired=g['paired']['NoCycle-Current'],severe=g['severe'])),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','predict','score'));a=ap.parse_args();globals()[a.command]()
