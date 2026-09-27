"""Foreground, bounded DeCoTA-Cal candidate. Separate dev-label selection and online API."""
import argparse
import collections
import contextlib
import fcntl
import hashlib
import itertools
import json
import math
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
OUT=ROOT/'artifacts/decota_cal_v1'
PRIOR=ROOT/'artifacts/decota_corrective_iteration_v1'
SIGNALS=['actionness','native_posterior']
CONFIGS=[dict(signal=s,high=h,low=l,lr_scale=r) for s,h,l,r in itertools.product(SIGNALS,[.8,.9],[.1,.2],[.1,.3,1.])]
WEIGHTS=[0.,.1,.3,1.,3.]


def fold_assign(rows,n):
    sources=sorted({r['source'] for r in rows},key=lambda s:hashlib.sha256(('20260912:'+s).encode()).hexdigest())
    return {s:i%n for i,s in enumerate(sources)}


def prepare():
    from vg_tta.decota_cal_spatial_v1 import parse_query
    from scripts.run_stvg_fullscale_v1 import query_parser
    if (OUT/'lock.json').exists():return plan()
    pp=read(PRIOR/'probe_v1/lock.json');parser=query_parser();cohorts={}
    pins={str(ROOT/'methods/CURRENT_METHOD.json'):sha(ROOT/'methods/CURRENT_METHOD.json')}
    for f in ('vg_tta/decota_cal_v1.py','vg_tta/decota_cal_spatial_v1.py','protocols/decota_cal_v1.md',
              'scripts/run_decota_cal_v1.py','tests/test_decota_cal_v1.py',
              'vg_tta/tg_spatial_tta_v1.py','methods/decota_refine_uniform_v1/api.py',
              'methods/decota_v1/api.py','scripts/score_stvg_fullscale_v1.py'):
        pins[str(ROOT/f)]=sha(ROOT/f)
    for c,rows in pp['cohorts'].items():
        fs=fold_assign([r for r in rows if r['split']=='source_reference'],4);new=[]
        for r in rows:
            idx=r['key'].split(':')[1]
            temporal=PRIOR/'probe_v1/temporal'/c/(idx+'.pt')
            signal=PRIOR/'matched_signals_v1/capture'/c/(idx+'.pt')
            assert sha(r['cache_path'])==r['cache_sha256']
            for p in (temporal,signal):assert sha(p)==read(p.with_suffix('.json'))['sha256']
            new.append({**r,'outer_fold':fs.get(r['source'],-1),'parsed':parse_query(parser,r['input']['caption']),
                        'temporal_path':str(temporal),'temporal_sha256':sha(temporal),
                        'signal_path':str(signal),'signal_sha256':sha(signal)})
        cohorts[c]=new
    p=dict(version='decota_cal_v1',created=time.time(),cohorts=cohorts,pins=pins,
           configs=CONFIGS,spatial_weights=WEIGHTS,seed=20260912,bootstrap=1000,
           GPU_budget_seconds=7200,production_changed=False,target_labeled_calibration=True,
           original_test_historically_exposed=True,GT_online=False)
    write(OUT/'lock.json',p);return p


def plan():
    p=read(OUT/'lock.json')
    for f,h in p['pins'].items():assert sha(f)==h,('pin changed',f)
    return p


@contextlib.contextmanager
def gpu_lease(stage):
    p=plan();lock=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);begin=time.time();failure=None
    used=sum(read(f)['seconds'] for f in (OUT/'leases').glob('*.json'))
    deadline=begin+max(0,p['GPU_budget_seconds']-used)
    def guard():
        if time.time()>deadline:raise TimeoutError('Bounded total GPU lease exhausted')
    try:guard();yield guard
    except Exception:failure=traceback.format_exc();raise
    finally:
        write(OUT/'leases'/f'{time.time_ns()}.json',dict(stage=stage,seconds=time.time()-begin,failure=failure))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def time_weights(row):
    ids=np.asarray(row['input']['frame_ids'],float)
    edges=np.r_[row['input']['start_frame'],(ids[1:]+ids[:-1])/2,row['input']['end_frame']]
    w=np.diff(edges);assert (w>0).all();return w/w.sum()


def fit_calibrator(rows, signal_arrays, labels, kind):
    """Offline-only source-equal NLL. Never called by the online fit function."""
    from scipy.special import expit,logit
    from scipy.optimize import minimize
    sources=collections.Counter(r['source'] for r in rows);xs=[];ys=[];ws=[]
    for r in rows:
        key=r['key'];p=np.asarray(signal_arrays[key]['full'][kind],dtype=np.float64)
        xs.append(logit(np.clip(p,1e-6,1-1e-6)))
        ids=np.asarray(r['input']['frame_ids']);g,h=labels[key]['interval']
        ys.append(((ids>=g)&(ids<h)).astype(float))
        ws.append(time_weights(r)/sources[r['source']]/len(sources))
    x,y,w=[np.concatenate(v).astype(np.float64) for v in (xs,ys,ws)]
    def fg(v):
        z=v[0]*x+v[1];p=expit(z);d=w*(p-y)
        return float(np.sum(w*(np.logaddexp(0,z)-y*z))),np.array([np.sum(d*x),np.sum(d)])
    opt=minimize(fg,[1.,0.],method='L-BFGS-B',jac=True,bounds=[(1e-4,100.),(-30.,30.)],
                 options=dict(maxiter=300,ftol=1e-12,gtol=1e-9))
    if not opt.success:raise RuntimeError('Calibration did not converge: '+str(opt.message))
    a,b=map(float,opt.x);p=expit(a*x+b)
    return dict(alpha=a,beta=b,train_sources=sorted(sources),train_keys=[r['key'] for r in rows],
        sources=len(sources),queries=len(rows),NLL_before=fg([1.,0.])[0],NLL_after=fg([a,b])[0],
        Brier_after=float(np.sum(w*(p-y)**2)),foreground_rate=float(w@y),
        bound_hit=a<1.01e-4 or a>99.999 or abs(b)>29.999,optimizer_success=bool(opt.success))


def temporal():
    import torch
    from scripts.run_stvg_fullscale_v1 import plan as source_plan,label_payload
    from scripts.score_stvg_fullscale_v1 import score
    from scripts.run_decota_temporal_optimizer_probe_v1 import make_head
    from scripts.run_decota_refine_v1 import configure
    from vg_tta.decota_cal_v1 import calibrated,fit_episode
    p=plan();labels=label_payload(source_plan());configure();torch.set_num_threads(4)
    if (OUT/'temporal/barrier.json').exists():return
    receipts=[]
    with gpu_lease('temporal') as guard:
        for c,rows in p['cohorts'].items():
            data={};signals={}
            for r in rows:
                assert sha(r['temporal_path'])==r['temporal_sha256'] and sha(r['signal_path'])==r['signal_sha256']
                x=load(r['temporal_path']);sx=load(r['signal_path'])
                assert not x['GT_used'] and not sx['GT_used'] and x['frame_ids']==sx['frame_ids']
                data[r['key']]=x;signals[r['key']]={k:{s:np.asarray(v['signals'][s]) for s in SIGNALS} for k,v in sx['variants'].items()}
            head=make_head(load(PRIOR/'probe_v1/temporal'/c/'source_head.pt'))
            refs=[r for r in rows if r['split']=='source_reference']
            def run_one(row,cal,cfg,variant='full',keep=False,lr_override=None,steps=5):
                guard();x=data[row['key']]
                assert row['source'] not in cal['train_sources']
                raw=signals[row['key']][variant][cfg['signal']] if variant not in {'constant','uncalibrated'} else (
                    np.full(len(x['frame_ids']),.5) if variant=='constant' else signals[row['key']]['full'][cfg['signal']])
                prob=raw if variant=='uncalibrated' else calibrated(raw,cal['alpha'],cal['beta'])
                positions=[[x['frame_ids'].index(f) for f in v['frame_ids']] for v in x['records']]
                z=fit_episode(head,x['head_inputs'],positions,x['frame_ids'],x['native_indices'],
                    x['native_logits'],prob,cfg['high'],cfg['low'],
                    x['config']['lr']*cfg['lr_scale'] if lr_override is None else lr_override,
                    steps=steps,keep_state=keep)
                z['config']=cfg;z['calibration']=cal;z['probability']=prob.tolist();return z
            for outer in [-1,0,1,2,3]:
                eval_rows=[r for r in rows if r['outer_fold']==outer]
                train=[r for r in refs if outer<0 or r['outer_fold']!=outer]
                assert {r['source'] for r in eval_rows}.isdisjoint(r['source'] for r in train)
                selpath=OUT/'temporal'/c/f'selection_{outer}.json'
                if selpath.exists():selection=read(selpath)
                else:
                    inner=fold_assign(train,4 if outer<0 else 3);trials=[];calibrators={}
                    for innerfold in sorted(set(inner.values())):
                        it=[r for r in train if inner[r['source']]!=innerfold]
                        iv=[r for r in train if inner[r['source']]==innerfold]
                        cal={s:fit_calibrator(it,signals,labels,s) for s in SIGNALS}
                        calibrators[str(innerfold)]=cal
                        for r in iv:
                            x=data[r['key']];baseline=score(x['boxes'],labels[r['key']],x['frame_ids'],x['outputs']['loss_min']['indices'])[0]['vIoU_corrected']
                            for ci,cfg in enumerate(CONFIGS):
                                z=run_one(r,cal[cfg['signal']],cfg)
                                m=score(x['boxes'],labels[r['key']],x['frame_ids'],z['indices'])[0]
                                trials.append(dict(key=r['key'],source=r['source'],config_index=ci,inner_fold=innerfold,
                                    delta=m['vIoU_corrected']-baseline,metrics=m,indices=z['indices'],
                                    reason=z['audit']['reason'],best_step=z['best_step'],parameter_l2=z['parameter_l2'],
                                    losses=[v['loss'] for v in z['curve']],state_sha256=z.get('state_sha256')))
                    utilities=[]
                    for ci,cfg in enumerate(CONFIGS):
                        d=np.array([t['delta'] for t in trials if t['config_index']==ci]);assert len(d)==len(train)
                        utilities.append(float(np.mean(d)-np.mean(np.maximum(-d,0))))
                    best=max(range(len(CONFIGS)),key=lambda i:(utilities[i],-i))
                    finalcal={s:fit_calibrator(train,signals,labels,s) for s in SIGNALS}
                    selection=dict(outer_fold=outer,selected_config_index=best,config=CONFIGS[best],utilities=utilities,
                        trials=trials,inner_calibrators=calibrators,calibrators=finalcal,
                        train_sources=[r['source'] for r in train],eval_sources=[r['source'] for r in eval_rows],
                        historical_development_exposure=True)
                    write(selpath,selection)
                print('SELECTED',c,outer,selection['config'],flush=True)
                for r in eval_rows:
                    key=r['key'];idx=key.split(':')[1];path=OUT/'temporal'/c/(idx+'.pt')
                    if path.exists():receipts.append(read(path.with_suffix('.json')));continue
                    cfg=selection['config'];cal=selection['calibrators'][cfg['signal']]
                    z=run_one(r,cal,cfg,keep=True);x=data[key]
                    ctr={v:run_one(r,cal,cfg,v) for v in ['constant','uncalibrated','wrong_20260912','wrong_20260913','wrong_20260914']}
                    noop=run_one(r,cal,cfg,lr_override=0.,steps=5)
                    zero=run_one(r,cal,cfg,steps=0)
                    assert noop['indices']==zero['indices']==z['no_parameter_indices']
                    assert noop['parameter_l2']==zero['parameter_l2']==0.
                    assert noop.get('state_sha256')==zero.get('state_sha256')
                    out=dict(key=key,cohort=c,source=r['source'],split=r['split'],outer_fold=outer,
                        frame_ids=x['frame_ids'],B=x['outputs']['loss_min'],C=z,controls=ctr,
                        F_indices=z['no_parameter_indices'],selection_sha256=sha(selpath),
                        no_op_exact=True,GT_online=False,GT_only_calibration_and_inner_selection=True,
                        frozen_boxes=x['boxes'],native_indices=x['native_indices'])
                    save(path,out);receipt=dict(key=key,path=str(path),sha256=sha(path));write(path.with_suffix('.json'),receipt);receipts.append(receipt)
                    status(OUT/'progress.json',dict(stage='temporal',done=len(receipts),total=42,last=key))
                    print('TEMPORAL',len(receipts),key,z['audit']['reason'],'best',z['best_step'],flush=True)
            del head,data,signals;torch.cuda.empty_cache()
    assert len(receipts)==42;plan()
    write(OUT/'temporal/barrier.json',dict(receipts=receipts,queries=42,created=time.time(),lock_sha256=sha(OUT/'lock.json')))


def spatial_capture():
    import torch
    from scripts.run_decota_refine_v1 import configure
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from vg_tta.decota_cal_spatial_v1 import observe
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from methods.decota_refine_uniform_v1.api import uniform_positions
    from vg_tta.foreground_runtime import state_digest
    p=plan();bar=read(OUT/'temporal/barrier.json');assert len(bar['receipts'])==42
    if (OUT/'spatial/barrier.json').exists():return
    lookup={r['key']:r for rr in p['cohorts'].values() for r in rr};receipts=[];configure()
    oldlock=read(ROOT/'artifacts/decota_refine_v1/lock.json')
    with gpu_lease('spatial') as guard:
        expert=SpatialExpert(oldlock['expert_snapshot']);digest=state_digest(expert.model)
        for rr in bar['receipts']:
            guard();r=lookup[rr['key']];c,idx=r['key'].split(':');path=OUT/'spatial'/c/(idx+'.pt')
            if path.exists():receipts.append(read(path.with_suffix('.json')));continue
            assert sha(rr['path'])==rr['sha256'];t=load(rr['path']);old=load(r['cache_path']);ids=t['frame_ids']
            spans=dict(B=t['B']['indices'],C=t['C']['indices'],F=t['F_indices'])
            frames={k:uniform_positions(ids,v,8) for k,v in spans.items()}
            raw,newids=decode(r['input']);assert newids==ids
            old_phrase=old.get('visual_query')
            if old_phrase is None:
                # Production cache has its actual phrase under expert_phrase.
                old_phrase=old.get('phrase')
            current={};repaired={};calls=0;reuse=0
            # Existing expert records carry exact original prompt / scores.
            cached={z['position']:z for z in old['expert']}
            parsed=r['parsed']
            # Original parser output is stored explicitly by the full runner.
            original_parse=old.get('visual',old.get('query'))
            if not isinstance(original_parse,dict) or 'phrase' not in original_parse:
                from scripts.run_stvg_fullscale_v1 import query_parser
                from vg_tta.tg_spatial_tta_v1 import visual_query
                if not hasattr(spatial_capture,'parser'):spatial_capture.parser=query_parser()
                original_parse=visual_query(spatial_capture.parser,r['input']['caption'])
            begin=time.perf_counter()
            for pos in sorted(set(frames['B']+frames['C'])):
                guard()
                if not original_parse['phrase']:continue
                if pos in cached:current[pos]=cached[pos];reuse+=1
                else:
                    z=expert(raw[pos],original_parse['phrase'],original_parse['entity'])
                    current[pos]={**z,'position':pos,'frame_id':ids[pos]};calls+=1
            for pos in sorted(set(frames['B']+frames['C']+frames['F'])):
                guard()
                if not parsed['phrase']:continue
                z=observe(expert,raw[pos],parsed);repaired[pos]={**z,'position':pos,'frame_id':ids[pos]};calls+=1
            assert state_digest(expert.model)==digest
            save(path,dict(key=r['key'],cohort=c,source=r['source'],split=r['split'],outer_fold=r['outer_fold'],
                frame_ids=ids,spans=spans,positions=frames,current=current,repaired=repaired,
                parsed=parsed,original_parse=original_parse,GT_online=False,expert_unchanged=True,
                total_new_calls=calls,reused_calls=reuse,seconds=time.perf_counter()-begin,
                per_arm_calls={k:len(v) if parsed['phrase'] else 0 for k,v in frames.items()}))
            receipt=dict(key=r['key'],path=str(path),sha256=sha(path),calls=calls);write(path.with_suffix('.json'),receipt);receipts.append(receipt)
            status(OUT/'progress.json',dict(stage='spatial',done=len(receipts),total=42,last=r['key'],new_calls=calls))
            print('SPATIAL',len(receipts),r['key'],calls,parsed['phrase'],flush=True)
            del raw,t,old
    plan();write(OUT/'spatial/barrier.json',dict(receipts=receipts,queries=42,created=time.time()))


def spatial_predict_and_score():
    import torch
    from vg_tta.decota_cal_spatial_v1 import associate,reconstruct_path
    from methods.decota_refine_uniform_v1.api import reconstruct
    from scripts.score_stvg_fullscale_v1 import score
    from scripts.run_stvg_fullscale_v1 import plan as source_plan,label_payload
    from vg_tta.corrective_evidence_v1 import source_mean_ci
    from vg_tta.box_stability_diagnostics_v1 import overlap
    p=plan();labels=label_payload(source_plan());bar=read(OUT/'spatial/barrier.json');outputs={};rows=[]
    lookup={r['key']:r for rr in p['cohorts'].values() for r in rr}
    # First construct every GT-free spatial path BEFORE any weight selection.
    for receipt in bar['receipts']:
        assert sha(receipt['path'])==receipt['sha256'];x=load(receipt['path']);r=lookup[x['key']]
        c,idx=x['key'].split(':');t=load(OUT/'temporal'/c/(idx+'.pt'));old=load(r['cache_path']);base=t['frozen_boxes'];ids=x['frame_ids']
        pred={'A':old['predictions']['mymethod'],'Frozen':old['predictions']['frozen'],
              'B_time':dict(boxes=base,indices=x['spans']['B']),
              'C_time':dict(boxes=base,indices=x['spans']['C']),
              'F_time':dict(boxes=base,indices=x['spans']['F'])};audits={}
        for arm in ['B','C']:
            pseudo=[dict(position=j,frame_id=ids[j],box=x['current'][j]['box'],score=x['current'][j]['score'],margin=x['current'][j]['margin'])
                    for j in x['positions'][arm] if j in x['current'] and x['current'][j]['accepted']]
            boxes,audit=reconstruct(base,pseudo,ids,'absolute');pred[arm]=dict(boxes=boxes,indices=x['spans'][arm]);audits[arm]=audit
        if x['spans']['B']==old['predictions']['temporal_only']['indices']:
            assert torch.equal(pred['B']['boxes'],pred['A']['boxes']),x['key']
        for arm in ['B','C','F']:
            probes=[x['repaired'][j] for j in x['positions'][arm] if j in x['repaired']]
            independent=[]
            for i,z in enumerate(probes):
                u=np.asarray(z['reference']);order=np.argsort(-u,kind='stable')
                if len(u) and u[order[0]]>=.35 and u[order[0]]-(u[order[1]] if len(u)>1 else 0)>=.05:
                    independent.append((i,int(order[0])))
            boxes,a=reconstruct_path(base,ids,probes,independent)
            pred[arm+'_repaired_independent']=dict(boxes=boxes,indices=x['spans'][arm]);audits[arm+'_repaired_independent']=a
            for weight in WEIGHTS:
                for control in ['reference','unreferenced','wrong_roi']:
                    path,a=associate(probes,weight,require_reference=control!='unreferenced',shuffle=control=='wrong_roi')
                    boxes,rec=reconstruct_path(base,ids,probes,path);name=f'{arm}_{control}_{weight:g}'
                    pred[name]=dict(boxes=boxes,indices=x['spans'][arm]);audits[name]={**a,**rec}
        for name,z in t['controls'].items():pred['time_'+name]=dict(boxes=base,indices=z['indices'])
        outpath=OUT/'predictions'/c/(idx+'.pt')
        obj=dict(key=x['key'],predictions=pred,audits=audits,GT_online=False,
                 spatial_source_path=receipt['path'],spatial_sha256=receipt['sha256'])
        if not outpath.exists():save(outpath,obj)
        outputs[x['key']]=(x,t,obj,outpath)
    write(OUT/'prediction_barrier.json',dict(queries=42,created=time.time(),GT_online=False,
        receipts=[dict(key=k,path=str(o[3]),sha256=sha(o[3])) for k,o in outputs.items()]))
    # All labels below are offline. Spatial tuning uses B only and other sources.
    allmetrics={}
    for key,(x,t,o,_) in outputs.items():
        allmetrics[key]={name:score(z['boxes'],labels[key],x['frame_ids'],z['indices'])[0] for name,z in o['predictions'].items()}
    selections={}
    for c,rr in p['cohorts'].items():
        for outer in [-1,0,1,2,3]:
            train=[r for r in rr if r['split']=='source_reference' and (outer<0 or r['outer_fold']!=outer)]
            values=[]
            for w in WEIGHTS:
                d=np.array([allmetrics[r['key']][f'B_reference_{w:g}']['vIoU_corrected']-allmetrics[r['key']]['B']['vIoU_corrected'] for r in train])
                values.append(float(d.mean()-np.maximum(-d,0).mean()))
            selected=WEIGHTS[max(range(len(WEIGHTS)),key=lambda i:(values[i],-i))]
            selections[c+':'+str(outer)]=dict(weight=selected,utility=values,train_sources=[r['source'] for r in train])
    write(OUT/'spatial_selection.json',selections)
    for key,(x,t,o,path) in outputs.items():
        sel=selections[x['cohort']+':'+str(x['outer_fold'])];assert x['source'] not in sel['train_sources'];w=sel['weight']
        metrics=allmetrics[key];chosen={'D':f'B_reference_{w:g}','E':f'C_reference_{w:g}','F':f'F_reference_{w:g}'}
        metrics.update({k:metrics[v] for k,v in chosen.items()})
        # Frozen event calibration, pseudo-label and reference quality: offline only.
        gt=labels[key];ids=np.asarray(x['frame_ids']);g,h=gt['interval'];event=(ids>=g)&(ids<h)
        prob=np.clip(t['C']['probability'],1e-12,1-1e-12);weights=time_weights(lookup[key])
        cfg=t['C']['config'];pos=prob>=cfg['high'];neg=prob<=cfg['low']
        diagnostic=dict(NLL=float(np.sum(weights*np.where(event,-np.log(prob),-np.log1p(-prob)))),
            Brier=float(np.sum(weights*(prob-event)**2)),positive_precision=float(event[pos].mean()) if pos.any() else None,
            negative_contamination=float(event[neg].mean()) if neg.any() else None,
            positive_coverage=float(pos.mean()),negative_coverage=float(neg.mean()),
            reason=t['C']['audit']['reason'],parameter_l2=t['C']['parameter_l2'],best_step=t['C']['best_step'])
        space={}
        for arm in ['B','C','F']:
            probes=[x['repaired'][j] for j in x['positions'][arm] if j in x['repaired']]
            quality=[];recall=[]
            for z in probes:
                j=z['position']
                if gt['valid'][j] and len(z['boxes']):
                    q=overlap(np.asarray(z['boxes']),np.repeat(np.array(gt['boxes'][j])[None],len(z['boxes']),axis=0))
                    recall.append(float(q.max()>=.5));quality.append(q)
                else:quality.append(None)
            a=o['audits'][f'{arm}_reference_{w:g}'];ref=a.get('reference_node');qref=None
            if ref is not None and quality[ref[0]] is not None:qref=float(quality[ref[0]][ref[1]])
            space[arm]=dict(calls=len(probes),top3_recall=float(np.mean(recall)) if recall else None,
                recall_observations=len(recall),reference_IoU=qref,reference_wrong=(qref<.5 if qref is not None else None),
                reason=a['reason'],accepted=len(a['pseudo']),modified_fraction=float(np.mean(a['spatial_modified'])),
                low_score_selected=sum(z['score']<.35 for z in a['pseudo']))
        rows.append(dict(key=key,cohort=x['cohort'],source=x['source'],split=x['split'],outer_fold=x['outer_fold'],
            metrics=metrics,temporal=diagnostic,spatial=space,spatial_weight=w,temporal_config=cfg,
            chosen=chosen,parsed=x['parsed'],predictions_path=str(path),predictions_sha256=sha(path),
            indices={k:list(o['predictions'][v]['indices']) for k,v in {**{n:n for n in ['A','B','C','Frozen']},**chosen}.items()},
            event_fraction=(h-g)/(lookup[key]['input']['end_frame']-lookup[key]['input']['start_frame'])))
    summary=dict(run='completed',measurement='pending_independent_audit',groups={},original_test_historically_exposed=True,
        GT_online=False,target_labeled_calibration=True,production_changed=False,
        GPU_seconds=sum(read(f)['seconds'] for f in (OUT/'leases').glob('*.json')))
    for c in p['cohorts']:
        for split in ['source_reference','reviewed_cases']:
            rr=[r for r in rows if r['cohort']==c and r['split']==split];sources=[r['source'] for r in rr];methods={}
            for name in rr[0]['metrics']:
                vals={m:source_mean_ci([r['metrics'][name][m] for r in rr],sources,seed=p['seed'],bootstrap=1000)
                      for m in ['vIoU_corrected','sIoU','tIoU','temporal_recall','temporal_precision']}
                d=np.array([r['metrics'][name]['vIoU_corrected']-r['metrics']['A']['vIoU_corrected'] for r in rr])
                methods[name]=dict(metrics=vals,delta_v_vs_A=source_mean_ci(d,sources,seed=p['seed'],bootstrap=1000),
                    improved=int((d>.001).sum()),harmed=int((d<-.001).sum()),neutral=int((abs(d)<=.001).sum()),
                    mean_harm=float(np.maximum(-d,0).mean()),median_delta=float(np.median(d)))
            comparisons={}
            for a,b in [('B','A'),('C','B'),('D','B'),('E','A'),('E','F'),('C_time','F_time')]:
                comparisons[a+'-'+b]=source_mean_ci([r['metrics'][a]['vIoU_corrected']-r['metrics'][b]['vIoU_corrected'] for r in rr],sources,seed=p['seed'],bootstrap=1000)
            summary['groups'][c+'/'+split]=dict(queries=len(rr),sources=len(set(sources)),methods=methods,comparisons=comparisons,
                temporal_reasons=dict(collections.Counter(r['temporal']['reason'] for r in rr)),
                actual_updated=sum(r['temporal']['parameter_l2']>0 for r in rr),
                reference_wrong=sum(r['spatial']['B']['reference_wrong'] is True for r in rr),
                reference_labeled=sum(r['spatial']['B']['reference_wrong'] is not None for r in rr),
                spatial_abstain=sum(r['spatial']['B']['accepted']==0 for r in rr))
    write(OUT/'rows.json',rows);write(OUT/'summary.json',summary)
    write(OUT/'complete.json',dict(created=time.time(),rows_sha256=sha(OUT/'rows.json'),summary_sha256=sha(OUT/'summary.json')))
    print('SCORED_ALL',len(rows),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','temporal','spatial','score']);args=ap.parse_args()
    try:
        if args.stage=='prepare':print('PREPARED',sum(len(v) for v in prepare()['cohorts'].values()))
        elif args.stage=='temporal':temporal()
        elif args.stage=='spatial':spatial_capture()
        else:spatial_predict_and_score()
    except Exception:
        write(OUT/'failures'/f'{time.time_ns()}.json',dict(stage=args.stage,error=traceback.format_exc(),created=time.time()))
        raise
