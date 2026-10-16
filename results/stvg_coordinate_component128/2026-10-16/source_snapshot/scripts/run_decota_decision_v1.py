"""Bounded R0--R5 experiment; explicit capture/selection/score barriers."""
import argparse,collections,contextlib,fcntl,gc,hashlib,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,load,save,sha,status
OUT=ROOT/'artifacts/decota_decision_v1'
PRIOR=ROOT/'artifacts/decota_corrective_iteration_v1'
CAL=ROOT/'artifacts/decota_cal_v1'
CS=[-2,-1,0,1,2];WS=[0.,.1,.3,1.,3.,10.,30.,100.];SEEDS=[20260912,20260913,20260914]
GROUPS={'hcstvg1_test':'vid_to_hc','vidstg_test':'hc_to_vid'}

def rank(k):return hashlib.sha256(('decota_decision_v1:'+k).encode()).hexdigest()
def file(stage,key):
    c,i=key.split(':');return OUT/stage/c/(i+'.pt')
def commit(path,obj):
    save(path,obj);r=dict(key=obj['key'],path=str(path),sha256=sha(path));write(path.with_suffix('.json'),r);return r
def checked(path):
    r=read(Path(path).with_suffix('.json'));assert sha(path)==r['sha256'];return load(path)
def allrows(p):return [r for rr in p['cohorts'].values() for r in rr]

def prepare():
    from scripts.run_stvg_fullscale_v1 import query_parser
    from scripts.run_decota_cal_v1 import fold_assign
    from vg_tta.decota_decision_v1 import repair_query,transition_prompt
    if (OUT/'lock.json').exists():return plan()
    old=read(CAL/'lock.json');full=read(ROOT/'artifacts/stvg_fullscale_diagnostics_v1/lock.json');parser=query_parser()
    cohorts={};audit={};pins=dict(old['pins'])
    for f in ['vg_tta/decota_decision_v1.py','scripts/run_decota_decision_v1.py','protocols/decota_decision_v1.md','tests/test_decota_decision_v1.py',
              'vg_tta/temporal_optimizer_probe_v1.py','vg_tta/matched_signals_v1.py','vg_tta/posterior_mass_coverage_v1.py']:
        pins[str(ROOT/f)]=sha(ROOT/f)
    for c,rr in old['cohorts'].items():
        blocked={r['source'] for r in rr};hashes={r['input']['video_sha256'] for r in rr};eligible=collections.defaultdict(list)
        for r in full['rows'][c]:
            q=r['input']
            if r['input_unavailable'] or q['source'] in blocked or q['video_sha256'] in hashes:continue
            eligible[q['source']].append(r)
        selected=sorted(eligible,key=rank)[:32];assert len(selected)==32
        extension=[]
        for src in selected:
            r=min(eligible[src],key=lambda r:rank(r['key']));idx=r['key'].split(':')[1]
            cache=ROOT/'artifacts/stvg_fullscale_diagnostics_v1/method/tastvg'/c/(idx+'.pt')
            assert cache.exists()
            extension.append({**r,'source':src,'split':'extension','outer_fold':-2,'cache_path':str(cache),'cache_sha256':sha(cache)})
        combined=[dict(r) for r in rr]+extension
        for r in combined:
            r['parsed'],r['parser_audit']=repair_query(parser,r['input']['caption'])
            r['transition_parse']=transition_prompt(r['parsed'],r['input']['caption'])
            r['subject']=parser(r['input']['caption'])['subject']
            r['wrong']={}
            for seed in SEEDS:
                candidates=[v for v in rr if v['source']!=r['source']]
                donor=min(candidates,key=lambda v:(v['query_type']!=r['query_type'],rank(str(seed)+r['key']+v['key'])))
                r['wrong'][str(seed)]={'key':donor['key'],'source':donor['source'],'caption':donor['input']['caption'],
                    'subject':parser(donor['input']['caption'])['subject']}
        cohorts[c]=combined;audit[c]=dict(original_queries=21,extension_queries=32,extension_sources=selected,
            source_overlap=0,media_hash_overlap=0,historically_untouched=False,total_queries=len(combined))
    p=dict(version='decota_decision_v1',created=time.time(),cohorts=cohorts,pins=pins,selection=CS,spatial_weights=WS,
        GPU_budget_seconds=7200,bootstrap=1000,seed=20260912,production_changed=False,online_GT=False,split_audit=audit)
    write(OUT/'lock.json',p);return p

def plan():
    p=read(OUT/'lock.json')
    for f,h in p['pins'].items():assert sha(f)==h,('protected dependency changed',f)
    return p

@contextlib.contextmanager
def lease(stage):
    p=plan();lock=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();used=sum(read(f)['seconds'] for f in (OUT/'leases').glob('*.json'));failure=None
    def guard():
        if time.time()-start+used>p['GPU_budget_seconds']:raise TimeoutError('GPU budget exhausted')
    try:guard();yield guard
    except Exception:failure=traceback.format_exc();raise
    finally:
        write(OUT/'leases'/f'{time.time_ns()}.json',dict(stage=stage,seconds=time.time()-start,failure=failure))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def temporal_capture():
    import torch
    from scripts.run_decota_refine_v1 import configure,student
    from scripts.run_decota_temporal_optimizer_probe_v1 import make_head,decode_output
    from vg_tta.temporal_optimizer_probe_v1 import snapshot,cpu_state
    from vg_tta.decota_decision_v1 import replay_path,candidate_path
    from vg_tta.decota_tastvg_episode_v1 import make_batch
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.matched_signals_v1 import capture
    from vg_tta.foreground_runtime import state_digest
    p=plan();configure();receipts=[];oldlock=read(ROOT/'artifacts/decota_refine_v1/lock.json')
    if (OUT/'temporal_barrier.json').exists():return
    configs=read(ROOT/'methods/decota_refine_uniform_v1/configs.json')['tastvg']
    with lease('temporal_capture') as guard:
        for c,rows in p['cohorts'].items():
            m=None;source=load(PRIOR/'probe_v1/temporal'/c/'source_head.pt');head=make_head(source)
            for row in rows:
                guard();key=row['key'];idx=key.split(':')[1];dest=file('temporal',key)
                if dest.exists():checked(dest);receipts.append(read(dest.with_suffix('.json')));continue
                t=time.time();original=load(row['cache_path']);assert sha(row['cache_path'])==row['cache_sha256']
                if row['split']!='extension':
                    x=checked(PRIOR/'probe_v1/temporal'/c/(idx+'.pt'))
                    trajectory=checked(PRIOR/'temporal_optimizer_v1/capture'/c/(idx+'.pt'))['path']
                    sx=checked(PRIOR/'matched_signals_v1/capture'/c/(idx+'.pt'))
                    signals={k:{s:np.asarray(v['signals'][s]) for s in ['actionness','native_posterior']} for k,v in sx['variants'].items() if k=='full' or k.startswith('wrong_')}
                    provenance=dict(reused=True,temporal_path=str(PRIOR/'probe_v1/temporal'/c/(idx+'.pt')),
                        trajectory_path=str(PRIOR/'temporal_optimizer_v1/capture'/c/(idx+'.pt')),signal_path=str(PRIOR/'matched_signals_v1/capture'/c/(idx+'.pt')))
                else:
                    if m is None:
                        m=student(oldlock,'tastvg',GROUPS[c]);state0=state_digest(m)
                        assert all(torch.equal(v,source[k]) for k,v in cpu_state(m.temp_embed).items())
                    raw,ids=decode(row['input']);assert ids==original['frame_ids'];signals={};hs=[]
                    inputspecs={'full':dict(caption=row['input']['caption'],subject=row['subject'])}
                    inputspecs.update({'wrong_'+k:v for k,v in row['wrong'].items()})
                    for variant,spec in inputspecs.items():
                        guard();batch=make_batch(raw,ids,{**row['input'],'caption':spec['caption']},spec['subject'],m)
                        handles=[]
                        if variant=='full':handles=[m.temp_embed.register_forward_pre_hook(lambda m,a:hs.append(a[0].detach().cpu()))]
                        try:v=capture(m,batch)
                        finally:
                            for h in handles:h.remove()
                        signals[variant]={s:v['signals'][s] for s in ['actionness','native_posterior']}
                        if variant=='full':
                            base=v['base'];assert len(hs)==2
                            assert torch.equal(base['raw_boxes'].float(),original['predictions']['frozen']['boxes'])
                            assert list(base['predicted_indices'])==list(original['predictions']['frozen']['indices'])
                            x=dict(head_inputs=hs,boxes=base['raw_boxes'].float().cpu(),native_indices=list(base['predicted_indices']),
                                native_logits=base['temporal_logits'],records=[{'frame_ids':z['frame_ids']} for z in v['views']],
                                frame_ids=ids,config=configs[GROUPS[c]])
                        del v,batch
                    trajectory=replay_path(head,hs,x['config']['lr'])
                    for snap in trajectory:snap['prediction']=decode_output(snap,x)
                    assert state_digest(m)==state0
                    provenance=dict(reused=False,source_state_digest=state0,pixel_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),full_and_three_wrong=True)
                    del raw
                candidates,audit=candidate_path(trajectory)
                assert trajectory[-1]['prediction']['indices']==list(original['predictions']['temporal_only']['indices'])
                assert [s['loss'] for s in trajectory[1:]]==[s['loss_after'] for s in original['temporal_audit']['steps']]
                # Real restore and forward every eligible state; not just index swapping.
                for cand in candidates:
                    snap=trajectory[cand['step']];head.load_state_dict(snap['head_state']);z=snapshot(head,x['head_inputs'])
                    assert z['loss']==snap['loss'] and all(torch.equal(a,b) for a,b in zip(z['logits'],snap['logits']))
                    assert decode_output(z,x)['indices']==cand['indices']
                head.load_state_dict(source)
                for s in trajectory:s.pop('process',None)
                out=dict(key=key,cohort=c,source=row['source'],split=row['split'],frame_ids=x['frame_ids'],
                    boxes=x['boxes'],native_indices=x['native_indices'],native_logits=x['native_logits'],head_inputs=x['head_inputs'],
                    records=x['records'],trajectory=trajectory,candidates=candidates,audit=audit,signals=signals,
                    source_unchanged=True,restore_exact=True,reference5_exact=True,GT_online=False,provenance=provenance,seconds=time.time()-t)
                receipts.append(commit(dest,out));status(OUT/'progress.json',dict(stage='temporal',done=len(receipts),total=106,last=key))
                print('TRAJECTORY',len(receipts),key,'candidates',len(candidates),'sec',round(time.time()-t,2),flush=True)
                del original,x,out,trajectory;gc.collect();torch.cuda.empty_cache()
            del head
            if m is not None:del m;gc.collect();torch.cuda.empty_cache()
    assert len(receipts)==106;write(OUT/'temporal_barrier.json',dict(receipts=receipts,created=time.time(),queries=106))

def pools(x):
    from vg_tta.native_coverage_calibration_v1 import native_at_length
    from vg_tta.posterior_mass_coverage_v1 import select as pm
    z=x['native_logits'];ids=x['frame_ids'];native=dict(indices=x['native_indices'],step=0)
    out={'tta':x['candidates']}
    for name,grid in [('prior',[.2,.4,.6,.8,1.]),('pm',[.5,.7,.8,.9,.95])]:
        cs=[native]
        for v in grid:
            ij=list(native_at_length(z,ids,fraction=v)['indices'] if name=='prior' else pm(z,ids,tau=v)['indices'])
            if ij not in [v['indices'] for v in cs]:cs.append(dict(indices=ij,step=None,setting=v))
        out[name]=cs
    return out

def selections():
    from scripts.run_stvg_fullscale_v1 import plan as fp,label_payload
    from scripts.score_stvg_fullscale_v1 import score
    from vg_tta.decota_decision_v1 import select,weights
    p=plan();read(OUT/'temporal_barrier.json');labels=label_payload(fp());trials={};configs={}
    if (OUT/'selection_barrier.json').exists():return
    for c,rows in p['cohorts'].items():
        refs=[r for r in rows if r['split']=='source_reference']
        for r in refs:
            x=checked(file('temporal',r['key']));w=weights(x['frame_ids'],r['input']['start_frame'],r['input']['end_frame']);ps=pools(x)
            base=score(x['boxes'],labels[r['key']],x['frame_ids'],x['candidates'][0]['indices'])[0]['vIoU_corrected'];trials[r['key']]={}
            for pool,cs in ps.items():
                for signal in ['actionness','native_posterior']:
                    trials[r['key']][pool+':'+signal]=[]
                    for cc in CS:
                        result=select(cs,x['signals']['full'][signal],w,cc);metric=score(x['boxes'],labels[r['key']],x['frame_ids'],result['indices'])[0]
                        trials[r['key']][pool+':'+signal].append(dict(c=cc,metrics=metric,delta=metric['vIoU_corrected']-base))
        for fold in [-1,0,1,2,3]:
            train=[r for r in refs if fold<0 or r['outer_fold']!=fold]
            choices={}
            for spec in ['tta:actionness','tta:native_posterior','prior:actionness','prior:native_posterior','pm:actionness','pm:native_posterior']:
                utilities=[]
                for ci in range(len(CS)):
                    ds=np.array([trials[r['key']][spec][ci]['delta'] for r in train]);utilities.append(float(ds.mean()-np.maximum(-ds,0).mean()))
                k=max(range(len(CS)),key=lambda i:(utilities[i],-abs(CS[i]),-CS[i]));choices[spec]=dict(c=CS[k],utilities=utilities)
            configs[c+':'+str(fold)]=dict(choices=choices,train_sources=[r['source'] for r in train],train_keys=[r['key'] for r in train])
    # Freeze configurations before any expanded-cohort labels are evaluated.
    write(OUT/'temporal_selection.json',dict(configurations=configs,trials=trials,extension_labels_used=False,created=time.time()))
    receipts=[]
    for r in allrows(p):
        x=checked(file('temporal',r['key']));spec=configs[r['key'].split(':')[0]+':'+str(max(-1,r['outer_fold']))]
        assert r['source'] not in spec['train_sources'];w=weights(x['frame_ids'],r['input']['start_frame'],r['input']['end_frame']);ps=pools(x);out={}
        for pool,cs in ps.items():
            for signal in ['actionness','native_posterior']:
                cc=spec['choices'][pool+':'+signal]['c'];name=pool+':'+signal
                out[name]=select(cs,x['signals']['full'][signal],w,cc)
                out[name+':c0']=select(cs,x['signals']['full'][signal],w,0.)
                for variant in ['constant']+['wrong_'+str(s) for s in SEEDS]:
                    v=np.full(len(w),.5) if variant=='constant' else x['signals'][variant][signal]
                    out[name+':'+variant]=select(cs,v,w,cc)
        for cc in CS:out['tta:actionness:c'+str(cc)]=select(ps['tta'],x['signals']['full']['actionness'],w,cc)
        result=dict(key=r['key'],choices=out,pools=ps,selection=spec,GT_online=False,
            temporal_sha256=sha(file('temporal',r['key'])),selection_sha256=sha(OUT/'temporal_selection.json'))
        receipts.append(commit(file('decisions',r['key']),result))
    write(OUT/'selection_barrier.json',dict(receipts=receipts,created=time.time(),queries=106))

def spatial_capture():
    import torch
    from scripts.run_decota_refine_v1 import configure
    from vg_tta.decota_decision_v1 import observe
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from methods.decota_refine_uniform_v1.api import uniform_positions
    p=plan();read(OUT/'selection_barrier.json');configure();receipts=[]
    if (OUT/'spatial_barrier.json').exists():return
    with lease('spatial_capture') as guard:
        expert=SpatialExpert(read(ROOT/'artifacts/decota_refine_v1/lock.json')['expert_snapshot']);digest=state_digest(expert.model)
        for r in allrows(p):
            guard();key=r['key'];dest=file('spatial',key)
            if dest.exists():checked(dest);receipts.append(read(dest.with_suffix('.json')));continue
            x=checked(file('temporal',key));d=checked(file('decisions',key));old=load(r['cache_path']);ids=x['frame_ids']
            spans={'B':x['candidates'][0]['indices'],'T':d['choices']['tta:actionness']['indices']}
            positions={k:uniform_positions(ids,ij,8) for k,ij in spans.items()};cache={};probes={};old_probes={};calls=0;raw=None
            # Exactly current old detections may be reused for the old interface.
            old_cache={(v['position'],v['text']):v for v in old['expert']}
            if r['split']!='extension':
                cs=checked(CAL/'spatial'/r['key'].split(':')[0]/(r['key'].split(':')[1]+'.pt'))
                old_cache.update({(j,v['text']):v for j,v in cs['current'].items()})
            for arm in ['B','T','transition']:
                ij=positions['B'] if arm=='transition' else positions[arm];seq=[];found=False
                for j in ij:
                    pp=r['transition_parse'] if arm=='transition' and found else r['parsed']
                    if not pp['phrase']:continue
                    tag=(j,pp['phrase']+'.')
                    if tag not in cache:
                        guard()
                        if raw is None:raw,newids=decode(r['input']);assert newids==ids
                        z=observe(expert,raw[j],pp);cache[tag]={**z,'position':j,'frame_id':ids[j]};calls+=1
                        if tag in old_cache:
                            dz=z['detection'];oz=old_cache[tag]
                            assert torch.equal(dz['all_boxes'],oz['all_boxes']) and torch.equal(dz['all_phrase_scores'],oz['all_phrase_scores']),(key,tag,'old expert mismatch')
                            assert dz['accepted']==oz['accepted']
                        old_cache[tag]=z['detection']|dict(position=j,frame_id=ids[j])
                    z=cache[tag];seq.append(z);found=found or z['accepted']
                probes[arm]=seq
            original_parse=r['parser_audit']['original']
            for arm in ['B']:
                seq=[]
                for j in positions[arm]:
                    if not original_parse['phrase']:continue
                    tag=(j,original_parse['phrase']+'.')
                    if tag not in old_cache:
                        guard()
                        if raw is None:raw,newids=decode(r['input']);assert newids==ids
                        z=expert(raw[j],original_parse['phrase'],original_parse['entity']);old_cache[tag]=z|dict(position=j,frame_id=ids[j]);calls+=1
                    seq.append(old_cache[tag])
                old_probes[arm]=seq
            assert state_digest(expert.model)==digest
            obj=dict(key=key,spans=spans,positions=positions,probes=probes,old_probes=old_probes,new_calls=calls,
                per_arm_budget=8,expert_unchanged=True,GT_online=False,decision_sha256=sha(file('decisions',key)))
            receipts.append(commit(dest,obj));status(OUT/'progress.json',dict(stage='spatial',done=len(receipts),total=106,last=key,new_calls=calls))
            print('SPATIAL_FIXED',len(receipts),key,'calls',calls,'support',[sum(z['accepted'] for z in probes[a]) for a in ['B','T']],flush=True)
            del raw,x,old,obj;gc.collect()
    write(OUT/'spatial_barrier.json',dict(receipts=receipts,queries=106,created=time.time()))

def score_results():
    import torch
    from scripts.run_stvg_fullscale_v1 import plan as fp,label_payload
    from scripts.score_stvg_fullscale_v1 import score
    from vg_tta.decota_decision_v1 import fixed_associate
    from vg_tta.decota_cal_spatial_v1 import reconstruct_path,associate
    from methods.decota_refine_uniform_v1.api import reconstruct
    from vg_tta.corrective_evidence_v1 import source_mean_ci
    p=plan();read(OUT/'spatial_barrier.json');labels=label_payload(fp());objects={};metrics={};rows=[]
    def evaluate(row):
        key=row['key'];x=checked(file('temporal',key));s=checked(file('spatial',key));d=checked(file('decisions',key));old=load(row['cache_path']);ids=x['frame_ids'];base=x['boxes']
        pred={'Frozen':old['predictions']['frozen'],'R0':old['predictions']['mymethod']};audits={}
        ps=[dict(position=z['position'],frame_id=z['frame_id'],box=z['box'],score=z['score'],margin=z['margin']) for z in s['old_probes']['B'] if z['accepted']]
        bb,_=reconstruct(base,ps,ids,'absolute');pred['R1']=dict(boxes=bb,indices=s['spans']['B'])
        for arm in ['B','T','transition']:
            probes=s['probes'][arm];ij=s['spans']['B'] if arm=='transition' else s['spans'][arm]
            independent=[(i,0) for i,z in enumerate(probes) if z['accepted']];b,a=reconstruct_path(base,ids,probes,independent)
            name={'B':'R2','T':'R4','transition':'transition_independent'}[arm];pred[name]=dict(boxes=b,indices=ij);audits[name]=a
            for w in WS:
                controls=['reference','unreferenced']+['wrong_'+str(seed) for seed in SEEDS]
                if arm=='transition':controls=['reference']
                for control in controls:
                    path,a=fixed_associate(probes,w,reference=control!='unreferenced',wrong_seed=int(control[6:]) if control.startswith('wrong_') else None)
                    b,rec=reconstruct_path(base,ids,probes,path);name=f'{arm}:{control}:{w:g}'
                    pred[name]=dict(boxes=b,indices=ij);audits[name]={**a,**rec,'path':path}
                    assert a['support']==[i for i,j in independent]
                    if w==0:assert torch.equal(b,pred[{'B':'R2','T':'R4','transition':'transition_independent'}[arm]]['boxes'])
            # Old variable-count objective, SAME admitted pool. For attribution only.
            admitted=[z for z in probes if z['accepted']]
            for w in [10.,30.,100.]:
                path,a=associate(admitted,w);b,rec=reconstruct_path(base,ids,admitted,path)
                name=f'{arm}:variable:{w:g}';pred[name]=dict(boxes=b,indices=ij);audits[name]={**a,**rec}
        for name,choice in d['choices'].items():pred['time:'+name]=dict(boxes=base,indices=choice['indices'])
        pred['time:B']=dict(boxes=base,indices=s['spans']['B']);pred['time:A']=old['predictions']['temporal_only']
        mm={k:score(v['boxes'],labels[key],ids,v['indices'])[0] for k,v in pred.items()}
        # Candidate oracle uses fixed native boxes. No labels returned online.
        oracle={pool:max(score(base,labels[key],ids,v['indices'])[0]['vIoU_corrected'] for v in poolrows) for pool,poolrows in d['pools'].items()}
        recall={}
        from vg_tta.box_stability_diagnostics_v1 import overlap
        gt=labels[key]
        for arm in ['B','T']:
            aa=[]
            for i,z in enumerate(s['probes'][arm]):
                j=z['position']
                if not z['accepted'] or not bool(gt['valid'][j]):continue
                ious=[float(overlap(b,gt['boxes'][j])) for b in z['boxes']]
                aa.append(dict(probe=i,position=j,IoUs=ious,top3_hit=max(ious)>=.5,top1_hit=ious[0]>=.5))
            recall[arm]=aa
        obj=dict(key=key,predictions=pred,audits=audits,GT_online=False,source_spatial_sha256=sha(file('spatial',key)))
        dest=file('predictions',key);commit(dest,obj);objects[key]=dest;metrics[key]=mm
        return dict(key=key,cohort=key.split(':')[0],source=row['source'],split=row['split'],outer_fold=row['outer_fold'],query_type=row['query_type'],
            metrics=mm,oracle=oracle,candidate_counts={k:len(v) for k,v in d['pools'].items()},candidate_recall=recall,
            selected_step=d['choices']['tta:actionness']['step'],best_step=x['audit']['best_step'],temporal_selection=d['selection'],
            caption=row['input']['caption'],parser_repaired=row['parser_audit']['repaired'],
            original_phrase=row['parser_audit']['original']['phrase'],phrase=row['parsed']['phrase'],transition_phrase=row['transition_parse']['phrase'],
            predictions_path=str(dest),predictions_sha256=sha(dest))
    if (OUT/'complete.json').exists():return
    # Score old panel first; freeze spatial weights before loading extension metrics.
    panel=[r for r in allrows(p) if r['split']!='extension']
    for r in panel:rows.append(evaluate(r))
    selections={}
    for c in p['cohorts']:
        for fold in [-1,0,1,2,3]:
            train=[r for r in rows if r['cohort']==c and r['split']=='source_reference' and (fold<0 or r['outer_fold']!=fold)]
            choices={}
            for control in ['reference','unreferenced']:
                uu=[]
                for w in WS:
                    ds=np.array([r['metrics'][f'B:{control}:{w:g}']['vIoU_corrected']-r['metrics']['R2']['vIoU_corrected'] for r in train]);uu.append(float(ds.mean()-np.maximum(-ds,0).mean()))
                choices[control]=dict(weight=WS[int(np.argmax(uu))],utilities=uu)
            selections[c+':'+str(fold)]=dict(choices=choices,train_sources=[r['source'] for r in train])
    write(OUT/'spatial_selection.json',dict(configurations=selections,extension_labels_used=False,created=time.time()))
    for r in allrows(p):
        if r['split']=='extension':rows.append(evaluate(r))
    for row in rows:
        key=row['key'];selection=selections[row['cohort']+':'+str(max(-1,row['outer_fold']))];assert row['source'] not in selection['train_sources']
        w=selection['choices']['reference']['weight'];u=selection['choices']['unreferenced']['weight'];m=row['metrics']
        chosen={'R3':f'B:reference:{w:g}','R5':f'T:reference:{w:g}',
            'transition_R3':f'transition:reference:{w:g}','B:unreferenced:selected':f'B:unreferenced:{u:g}'}
        chosen.update({f'R3_wrong_{seed}':f'B:wrong_{seed}:{w:g}' for seed in SEEDS})
        m.update({k:m[v] for k,v in chosen.items()});row.update(chosen=chosen,weight=w,spatial_selection=selection)
    summary={}
    comparisons=[('R1','R0'),('R2','R1'),('R3','R2'),('R4','R2'),('R5','R1'),('R5','R4'),
        ('time:tta:actionness','time:B'),('time:tta:native_posterior','time:B'),('time:prior:actionness','time:B'),('time:pm:actionness','time:B')]
    for c in p['cohorts']:
        for split in ['source_reference','reviewed_cases','extension']:
            rr=[r for r in rows if r['cohort']==c and r['split']==split];ss=[r['source'] for r in rr];methods={}
            for name in rr[0]['metrics']:
                methods[name]={k:source_mean_ci([r['metrics'][name][k] for r in rr],ss,seed=20260912,bootstrap=1000) for k in ['vIoU_corrected','sIoU','tIoU']}
            diffs={}
            for a,b in comparisons:
                d=[r['metrics'][a]['vIoU_corrected']-r['metrics'][b]['vIoU_corrected'] for r in rr]
                diffs[a+' minus '+b]={**source_mean_ci(d,ss,seed=20260912,bootstrap=1000),
                    'improved':sum(v>.001 for v in d),'harmed':sum(v<-.001 for v in d),'neutral':sum(abs(v)<=.001 for v in d)}
            summary[c+'/'+split]=dict(queries=len(rr),sources=len(set(ss)),methods=methods,differences=diffs,
                candidate_oracles={pool:source_mean_ci([r['oracle'][pool] for r in rr],ss,seed=20260912,bootstrap=1000) for pool in ['tta','prior','pm']})
    write(OUT/'rows.json',rows);write(OUT/'summary.json',summary)
    write(OUT/'complete.json',dict(run='completed',measurement='pending_independent_audit',queries=len(rows),production_changed=False,
        summary_sha256=sha(OUT/'summary.json'),rows_sha256=sha(OUT/'rows.json'),created=time.time()))
    print('SCORED_R0_R5',len(rows),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','temporal','select','spatial','score']);args=parser.parse_args()
    try:{'prepare':prepare,'temporal':temporal_capture,'select':selections,'spatial':spatial_capture,'score':score_results}[args.action]()
    except Exception:
        write(OUT/'failures'/f'{time.time_ns()}.json',dict(action=args.action,error=traceback.format_exc(),created=time.time()));raise
