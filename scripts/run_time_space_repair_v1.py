"""F25 finite runners. Selection scoring is a separate, label-bearing process."""
import argparse
import collections
import copy
import fcntl
import gc
import hashlib
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
OUT=ROOT/'artifacts/decota_time_space_repair_v1'
F24=ROOT/'artifacts/decota_design_claims_v1'
F22=ROOT/'artifacts/decota_s2_system_v1'
GROUPS={'hcstvg1_test':'vid_to_hc','vidstg_test':'hc_to_vid'}


def hashkey(x):return hashlib.sha256(x.encode()).hexdigest()


def prepare():
    full=read(F22/'system_lock.json');old=read(F24/'lock.json');rows={};counts={}
    watched={r['key'] for rr in old['rows'].values() for r in rr if r['role']=='watched'}
    watched|={r['key'] for r in read(ROOT/'artifacts/decota_spatial_extension_v1/lock.json')['rows'] if r['role']=='watched'}
    watched|={f'hcstvg1_test:{k:06d}' for k in [1036]}|{f'vidstg_test:{k:06d}' for k in [9356,621,8335,4989,2421,7783]}
    for c,rr in full['rows'].items():
        bykey={r['key']:r for r in rr};sel=[r for r in old['rows'][c] if r['role']=='selection']
        ws=[bykey[k] for k in sorted(watched) if k in bykey]
        excluded={r['source'] for r in sel}|{r['input']['source'] for r in ws}
        pool=collections.defaultdict(list)
        for r in rr:
            if r['input']['source'] not in excluded:pool[r['input']['source']].append(r)
        sources=sorted(pool,key=lambda s:hashkey('F25/source/'+c+s))[:32]
        selected=[(r['key'],'selection') for r in sel]
        selected += [(min(pool[s],key=lambda r:hashkey('F25/query/'+r['key']))['key'],'expansion') for s in sources]
        selected += [(r['key'],'watched') for r in ws]
        out=[]
        for key,role in selected:
            r=bykey[key];path=F22/'system'/c/(key.split(':')[1]+'.pt');receipt=read(path.with_suffix('.json'))
            assert sha(path)==receipt['sha256']
            out.append(dict(key=key,role=role,source=r['input']['source'],input=r['input'],subject=r['parses'].get('subject',''),
                            query_type=r['query_type'],F22_path=str(path),F22_sha256=receipt['sha256'],
                            best_step=r.get('best_step'),input_unavailable=r.get('input_unavailable')))
        rows[c]=out;counts[c]=dict(available_remaining_sources=len(pool),selection=8,expansion=len(sources),watched=len(ws),
                                  excluded_sources=sorted(excluded))
        for i,r in enumerate(out):
            donor=next(s for s in out[i+1:]+out[:i] if s['source']!=r['source'] and s['input']['caption']!=r['input']['caption'])
            r['wrong_query']=dict(key=donor['key'],source=donor['source'],caption=donor['input']['caption'],subject=donor['subject'])
        print(c,counts[c],flush=True)
    files=['artifacts/decota_time_space_repair_v1/PROTOCOL.md','vg_tta/time_space_repair_v1.py','scripts/run_time_space_repair_v1.py',
           'tests/test_time_space_repair_v1.py','methods/CURRENT_METHOD.json','vg_tta/design_claims_v1.py','vg_tta/decota_tastvg_episode_v1.py',
           'vg_tta/decota_s2_system_v1.py','external/TA-STVG/models/pipeline.py','external/TA-STVG/models/grounding_model/query_decoder.py']
    context=[r['key'] for rr in rows.values() for role in ['selection','expansion'] for r in [s for s in rr if s['role']==role][:4]]
    write(OUT/'LOCK.json',dict(rows=rows,counts=counts,context_keys=context,configs=full['configs'],created=time.time(),
        temporal_grid=[dict(eta=e,lr=l) for e in [.25,1.,4.] for l in [.0001,.001,.01]],spatial_lr={'hcstvg1_test':.01,'vidstg_test':.1},
        new_dino_max=2000,walltime_seconds=14400,bytes_max=20*2**30,GT_online=False,
        pins={f:sha(ROOT/f) for f in files},parent_lock_sha256=sha(F24/'lock.json'),F22_lock_sha256=sha(F22/'system_lock.json'),
        exposure='historically exposed; source-isolated development expansion, not untouched confirmation'))


def plan():
    p=read(OUT/'LOCK.json');pins=dict(p['pins'])
    for f in sorted(OUT.glob('IMPLEMENTATION_AMENDMENT_*.json')):pins.update(read(f)['new_pins'])
    for f,h in pins.items():assert sha(ROOT/f)==h,('changed pin',f)
    return p


def dest(stage,r):return OUT/stage/r['key'].split(':')[0]/(r['key'].split(':')[1]+'.pt')


def existing(path):
    if not path.with_suffix('.json').exists():
        assert not path.exists(),('unreceipted output preserved',path);return False
    assert sha(path)==read(path.with_suffix('.json'))['sha256'];return True


def receipt(path,x):
    save(path,{**x,'lock_sha256':sha(OUT/'LOCK.json'),'GT_online':False})
    write(path.with_suffix('.json'),dict(key=x['key'],sha256=sha(path),path=str(path),seconds=x.get('seconds',0)))


def capture(model,batch):
    from scripts.run_design_claims_v1 import capture as oldcapture
    aa=[];h=model.register_forward_hook(lambda m,a,o:aa.append(o['pred_actioness'].detach().float().cpu().reshape(-1)))
    try:base,inputs,records,caches=oldcapture(model,batch)
    finally:h.remove()
    assert len(aa)==2 and all(len(a)==len(r['frame_ids']) for a,r in zip(aa,records))
    n=len(batch['targets'][0]['frame_ids']);import torch
    action=torch.stack([aa[i%2][i//2] for i in range(n)]).numpy()
    return base,inputs,records,caches,action


def spatial_arms(model,caches,ids,f22,lr):
    from vg_tta.time_space_repair_v1 import SpatialInterface,spatial_fit
    from vg_tta.design_claims_v1 import regate
    from vg_tta.decota_s2_system_v1 import independent
    b=f22['predictions']['F0']['boxes'];f4=f22['predictions']['F4'];out={}
    for scope in ['query','ln','query_ln']:
        it=SpatialInterface(model,caches,len(ids),scope)
        out[scope]=spatial_fit(it,b,f4['anchors'],.1 if scope=='query' else lr);del it;gc.collect()
    score=independent(b,ids,regate(f4['probes'],.35,0)) if f4['context_active'] else f4
    it=SpatialInterface(model,caches,len(ids),'query_ln')
    if score['anchors']==f4['anchors']:out['score_only']=out['query_ln']
    else:out['score_only']=spatial_fit(it,b,score['anchors'],lr)
    out['score_only_absolute']=dict(boxes=score['boxes'],anchors=score['anchors'],context_fallback=not f4['context_active'])
    f5=f22['predictions']['F5']
    if f5['anchors']==f4['anchors']:out['F5_query_ln']=out['query_ln']
    else:out['F5_query_ln']=spatial_fit(it,b,f5['anchors'],lr)
    del it
    return out


def run(cohort,stage,limit=0):
    import numpy as np
    import torch
    from scripts.run_decota_refine_v1 import configure,student
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.exact_frame_decode_audit_v2 import decode as frames_decode
    from vg_tta.decota_tastvg_episode_v1 import make_batch
    from vg_tta.time_space_repair_v1 import boundary_evidence,boundary_fit,paired_readout,SpatialInterface,spatial_fit
    from methods.decota_v1.api import fit
    p=plan();configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.time();done=0;failure=None
    used=sum(read(f)['seconds'] for f in (OUT/'leases').glob('*.json'))
    try:
        model=student(read(ROOT/'artifacts/decota_refine_v1/lock.json'),'tastvg',GROUPS[cohort]);digest=state_digest(model)
        rows=[r for r in p['rows'][cohort] if (r['role']=='selection')==(stage=='selection')] if stage in ['selection','expanded'] else p['rows'][cohort]
        if stage=='budget20':
            selected=read(OUT/'TEMPORAL_SELECTION.json')[cohort]
            rows=[r for r in p['rows'][cohort] if r['role']=='selection'] if selected['twenty_step_trigger'] else []
        for r in rows:
            path=dest(stage,r)
            if existing(path):continue
            assert time.time()-began+used<p['walltime_seconds'];tick=time.time()
            assert sum(f.stat().st_size for f in OUT.rglob('*.pt'))<p['bytes_max']
            if r['input_unavailable']:
                receipt(path,dict(key=r['key'],status='input_unavailable',reason=r['input_unavailable']));continue
            assert sha(r['input']['video_path'])==r['input']['video_sha256']
            frames,ids=frames_decode(r['input']);assert ids==r['input']['frame_ids']
            batch=make_batch(frames,ids,r['input'],r['subject'],model);base,inputs,records,caches,action=capture(model,batch)
            f22=load(r['F22_path']);b=f22['predictions']['F0']['boxes']
            assert torch.equal(base['raw_boxes'].float().cpu(),b) and list(base['predicted_indices'])==f22['predictions']['F0']['indices']
            if stage in ['selection','expanded']:
                e=boundary_evidence(action,ids,r['input']['fps']);arms={}
                grid=p['temporal_grid'] if stage=='selection' else [read(OUT/'TEMPORAL_SELECTION.json')[cohort]['config']]
                for conf in grid:
                    name=f"eta{conf['eta']}_lr{conf['lr']}";arms[name]=boundary_fit(model.temp_embed,inputs,records,ids,base,e,**conf)
                bh,bz,_=fit(model.temp_embed,inputs,backbone='tastvg',lr=p['configs'][GROUPS[cohort]]['lr'],steps=f22['best_step'])
                old=paired_readout(bz,records,ids,base);assert old['reposition']==f22['predictions']['F2']['indices'];del bh,bz
                spatial=spatial_arms(model,caches,ids,f22,p['spatial_lr'][cohort])
                result=dict(temporal=arms,evidence=e,old_B=old,spatial=spatial,native_logits=base['temporal_logits'],native_indices=list(base['predicted_indices']))
                # Real no-op tests on the first selection query of each direction.
                if r==rows[0] and stage=='selection':
                    noop={}
                    for name,kw in [('lr0',dict(lr=0.,steps=5)),('steps0',dict(lr=.001,steps=0))]:
                        z=boundary_fit(model.temp_embed,inputs,records,ids,base,e,eta=1.,**kw)
                        assert z['direct']==list(base['predicted_indices']) and z['path'][-1]['state_delta_norm']==0
                        noop[name]=z
                    result['noops']=noop
            elif stage=='budget20':
                conf=read(OUT/'TEMPORAL_SELECTION.json')[cohort]['config'];e=boundary_evidence(action,ids,r['input']['fps'])
                result=dict(temporal=boundary_fit(model.temp_embed,inputs,records,ids,base,e,steps=20,**conf),
                            diagnostic_only=True,main_budget_unchanged=5)
            elif stage=='controls':
                conf=read(OUT/'TEMPORAL_SELECTION.json')[cohort]['config'];arms={};evidences={}
                donor=r['wrong_query'];wrong={**r['input'],'caption':donor['caption']}
                wb=make_batch(frames,ids,wrong,donor['subject'],model);_,_,_,wc,wa=capture(model,wb);del wb,wc
                rng=np.random.default_rng(17+int(hashkey(r['key'])[:8],16))
                for name,a,w in [('wrong_query',wa,2),('constant',np.full_like(action,action.mean()),2),('shuffle',rng.permutation(action),2),('h4',action,4)]:
                    e=boundary_evidence(a,ids,r['input']['fps'],w);evidences[name]=e
                    arms[name]=boundary_fit(model.temp_embed,inputs,records,ids,base,e,**conf)
                assert arms['constant']['direct']==list(base['predicted_indices']) and arms['constant']['path'][-1]['state_delta_norm']==0
                result=dict(temporal=arms,evidences=evidences,wrong_query=donor)
            elif stage=='joint':
                ex=load(dest('expert',r));out={};it=SpatialInterface(model,caches,len(ids),'query_ln')
                for name,v in ex['arms'].items():
                    out[name]=spatial_fit(it,b,v['anchors'],p['spatial_lr'][cohort])
                del it;result=dict(spatial=out)
            else:raise ValueError(stage)
            assert state_digest(model)==digest
            receipt(path,dict(key=r['key'],status='completed',result=result,frame_ids=ids,native_boxes=b,
                              source_unchanged=True,seconds=time.time()-tick))
            done+=1;status(OUT/'progress.json',dict(stage=stage,cohort=cohort,last=r['key'],done=done,total=len(rows),seconds=time.time()-began))
            print(stage,r['key'],'seconds',round(time.time()-tick,2),flush=True)
            del frames,batch,base,inputs,records,caches,f22,result;gc.collect();torch.cuda.empty_cache()
            if limit and done>=limit:break
        plan()
    except Exception:failure=traceback.format_exc();raise
    finally:
        write(OUT/'leases'/f'{time.time_ns()}.json',dict(stage=stage,cohort=cohort,seconds=time.time()-began,done=done,failure=failure))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


def expert_run(limit=0):
    import torch
    from scripts.run_decota_refine_v1 import configure
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from scripts.run_decota_spatial_extension_v1 import ContextView
    from scripts.run_design_claims_v1 import prompt_specs
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.decota_s2_system_v1 import probe_from_detection,independent,old_reconstruct
    from methods.decota_refine_uniform_v1.api import uniform_positions
    p=plan();configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.time();done=0;failure=None;new=0
    calls=sum(load(f).get('new_calls',0) for f in (OUT/'expert').glob('*/*.pt'))
    if (OUT/'expert_call_counter.json').exists():calls=max(calls,read(OUT/'expert_call_counter.json')['total'])
    used=sum(read(f)['seconds'] for f in (OUT/'leases').glob('*.json'))
    try:
        exp=SpatialExpert(read(ROOT/'artifacts/decota_refine_v1/lock.json')['expert_snapshot']);digest=state_digest(exp.model)
        for c,rr in p['rows'].items():
            conf=read(OUT/'TEMPORAL_SELECTION.json')[c]['name']
            for r in rr:
                path=dest('expert',r)
                if existing(path):continue
                tick=time.time();f22=load(r['F22_path']);b=f22['predictions']['F0']['boxes'];ids=f22['frame_ids']
                t=load(dest('selection' if r['role']=='selection' else 'expanded',r))['result']['temporal'][conf]
                spec=f22['parses']['context'];f4=f22['predictions']['F4'];tasks={'T4':(t['direct'],spec),'T5':(t['analytic']['direct'],spec)}
                if r['key'] in p['context_keys']:
                    specs,reason=prompt_specs(f22,r['wrong_query']['caption'])
                    tasks['modifier']=(f4['indices'],specs.get('appearance',spec))
                cache={};arms={};raw=None;newrow=0;reused=0
                def ckey(pos,sp):return (ids[pos],sp['context'],tuple(sp['span']))
                for ob in f22['observations']:
                    d=ob['detection']
                    if 'target_span' in d:cache[(ids[ob['cache_key'][1]],d['text'],tuple(d['target_span']))]=d
                tp=F24/'text_context'/c/(r['key'].split(':')[1]+'.pt')
                if tp.exists():
                    for arm in load(tp)['arms'].values():
                        for ob in arm['observations']:cache[ckey(ob['position'],arm['spec'])]=ob['detection']
                observations=[]
                for name,(iv,sp) in tasks.items():
                    positions=uniform_positions(ids,iv,8);probes=[]
                    if not sp.get('eligible',True):
                        # Preserve S2's explicitly unresolved query policy; old short expert still at own frames.
                        old=f22['parses']['old']
                        for pos in positions:
                            found=next((ob['detection'] for ob in f22['observations'] if ob['cache_key'][0]=='short' and ob['cache_key'][1]==pos),None)
                            if found is None and old['phrase']:
                                if raw is None:raw,actual=decode(r['input']);assert actual==ids
                                assert calls+new<p['new_dino_max'];found=exp(raw[pos],old['phrase'],old['entity']);new+=1;newrow+=1
                            if found is None:continue
                            probes.append({**found,'position':pos,'frame_id':ids[pos]})
                        z=old_reconstruct(b,ids,probes);fallback=True
                    else:
                        assert len(exp.processor.tokenizer(sp['context'])['input_ids'])<=256
                        for pos in positions:
                            assert time.time()-began+used<p['walltime_seconds'];ck=ckey(pos,sp);found=cache.get(ck)
                            if found is None:
                                if raw is None:raw,actual=decode(r['input']);assert actual==ids
                                assert calls+new<p['new_dino_max'];found=ContextView(exp,sp)(raw[pos],None,None);cache[ck]=found;new+=1;newrow+=1
                                # Persist budget immediately even if this row later fails.
                                status(OUT/'expert_call_counter.json',dict(total=calls+new,key=r['key']))
                            else:reused+=1
                            observations.append(dict(position=pos,spec=sp,detection=found))
                            probes.append(probe_from_detection(found,pos,ids[pos]))
                        z=independent(b,ids,probes);fallback=False
                    arms[name]=dict(**z,indices=iv,positions=positions,probes=probes,spec=sp,context_fallback=fallback)
                assert state_digest(exp.model)==digest
                receipt(path,dict(key=r['key'],arms=arms,observations=observations,new_calls=newrow,reused=reused,seconds=time.time()-tick))
                done+=1;print('expert',r['key'],'new',newrow,'total',calls+new,flush=True)
                if limit and done>=limit:return
        plan()
    except Exception:failure=traceback.format_exc();raise
    finally:
        write(OUT/'leases'/f'{time.time_ns()}.json',dict(stage='expert',seconds=time.time()-began,done=done,new_calls=new,failure=failure))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','selection','expanded','controls','expert','joint','budget20']);a.add_argument('--cohort',choices=list(GROUPS));a.add_argument('--limit',type=int,default=0);q=a.parse_args()
    if q.stage=='prepare':prepare()
    elif q.stage=='expert':expert_run(q.limit)
    else:run(q.cohort,q.stage,q.limit)
