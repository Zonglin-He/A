"""F24 finite execution; offline targets are isolated from label-free outputs."""
import argparse
import collections
import fcntl
import gc
import hashlib
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
OUT=ROOT/'artifacts/decota_design_claims_v1'
PARENT=ROOT/'artifacts/decota_temporal_capacity_v1'
F22=ROOT/'artifacts/decota_s2_system_v1'
GROUPS={'hcstvg1_test':'vid_to_hc','vidstg_test':'hc_to_vid'}


def prepare():
    p=read(PARENT/'lock.json')
    files=['protocols/decota_design_claims_v1.md','vg_tta/design_claims_v1.py','scripts/run_design_claims_v1.py',
        'methods/CURRENT_METHOD.json','methods/decota_s2_work_v1/predictor.py','vg_tta/decota_tastvg_episode_v1.py',
        'methods/decota_s_v1/api.py','vg_tta/tg_spatial_tta_v1.py','methods/decota_refine_v1/api.py',
        'methods/decota_refine_uniform_v1/api.py','external/TA-STVG/models/grounding_model/query_decoder.py']
    lock=dict(rows=p['rows'],configs=p['configs'],pins={f:sha(ROOT/f) for f in files},created=time.time(),
        GPU_seconds=7200,output_max_bytes=20*2**30,parent_sha=sha(PARENT/'lock.json'),
        F22_sha=sha(F22/'system_lock.json'),spatial_lrs=[.001,.01,.1,1.],spatial_budgets=[3,10],seeds=[17,41,83],
        exposure='historically exposed development, eight selection/eight source holdout per direction, watched excluded')
    write(OUT/'lock.json',lock);print('LOCKED',sum(len(v) for v in lock['rows'].values()),flush=True)


def plan():
    p=read(OUT/'lock.json')
    pins=dict(p['pins'])
    if (OUT/'IMPLEMENTATION_AMENDMENT_1.json').exists():
        a=read(OUT/'IMPLEMENTATION_AMENDMENT_1.json');assert a['parent_sha']==sha(OUT/'lock.json');pins.update(a['new_pins'])
    for f,h in pins.items():assert sha(ROOT/f)==h,('pin changed',f)
    return p


def oracle_targets():
    from scripts.run_stvg_fullscale_v1 import plan as fullplan,label_payload
    import numpy as np
    p=plan();labels=label_payload(fullplan());out={}
    for c,rr in p['rows'].items():
        for r in rr:
            x=load(r['F22_path']);gt=labels[r['key']];pp=x['predictions']['F4']['anchors']
            common=[a for a in pp if gt['valid'][a['position']]]
            out[r['key']]=dict(expert_common=common,GT_common=[{**a,'box':gt['boxes'][a['position']]} for a in common],
                event_mask=gt['event_mask'],known=gt['valid'],interval=gt['interval'])
    write(OUT/'ORACLE_ONLY_targets.json',dict(rows=out,GT_used=True,label_hash=fullplan()['labels_sha256'],
        purpose='matched supervision and temporal key diagnostic; not online method input'))


def receipt(path,x):
    x.update(lock_sha256=sha(OUT/'lock.json'));save(path,x)
    write(path.with_suffix('.json'),dict(path=str(path),sha256=sha(path),key=x['key'],seconds=x.get('seconds',0.)))


def existing(path):
    if not path.with_suffix('.json').exists():
        assert not path.exists(),('unreceipted file preserved',path);return False
    assert sha(path)==read(path.with_suffix('.json'))['sha256'];return True


def capture(model,batch):
    from vg_tta.tg_spatial_tta_v1 import detached_tree
    from vg_tta.decota_tastvg_episode_v1 import forward
    caches=[];h=model.ground_decoder.decoder.register_forward_pre_hook(lambda m,a,kw:caches.append(detached_tree(kw)),with_kwargs=True)
    try:base,inputs,records=forward(model,batch)
    finally:h.remove()
    assert len(caches)==4
    return base,inputs,records,[caches[1],caches[3]]


def run(cohort,stage,limit=0):
    import numpy as np
    import torch
    from scripts.run_decota_refine_v1 import configure,student
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.decota_tastvg_episode_v1 import make_batch
    from vg_tta.design_claims_v1 import temporal_probe,SpatialInterface,spatial_fit
    p=plan();configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.time();used=sum(read(f)['seconds'] for f in (OUT/'leases').glob('*.json'));failure=None;done=0
    try:
        model=student(read(ROOT/'artifacts/decota_refine_v1/lock.json'),'tastvg',GROUPS[cohort]);digest=state_digest(model)
        for r in p['rows'][cohort]:
            key=r['key'];path=OUT/stage/cohort/(key.split(':')[1]+'.pt')
            if existing(path):continue
            assert time.time()-began+used<p['GPU_seconds'],'F24 GPU budget reached'
            assert sum(f.stat().st_size for f in OUT.rglob('*.pt'))<p['output_max_bytes']
            tick=time.time();frames,ids=decode(r['input']);batch=make_batch(frames,ids,r['input'],r['subject'],model)
            base,inputs,records,caches=capture(model,batch);f22=load(r['F22_path']);b=f22['predictions']['F0']['boxes'];B=f22['predictions']['F2']['indices']
            assert torch.equal(base['raw_boxes'].float().cpu(),b) and list(base['predicted_indices'])==f22['predictions']['F0']['indices']
            if stage=='temporal':
                z=temporal_probe(model.temp_embed,inputs,records,ids,base,p['configs'][GROUPS[cohort]]['lr'])
                assert [s['loss'] for s in z['path']]==r['original_losses'],('F22 loss path mismatch',key)
                assert z['path'][z['best_step']]['final']==B
                z.update(native_boxes=b,native_indices=f22['predictions']['F0']['indices'],frame_ids=ids,records=records)
                receipt(path,dict(key=key,GT_used=False,result=z,seconds=time.time()-tick,source_unchanged=state_digest(model)==digest))
            elif stage=='spatial':
                # Legal expert-only output is persisted BEFORE opening oracle payload.
                legal={};anchors=f22['predictions']['F4']['anchors']
                for scope in ['query','query_ln']:
                    it=SpatialInterface(model,caches,len(ids),scope)
                    with torch.no_grad():zero,_=it.values()
                    assert torch.equal(zero.cpu(),b)
                    legal[scope]=[spatial_fit(it,b,anchors,lr,tuple(p['spatial_budgets'])) for lr in p['spatial_lrs']]
                    del it;gc.collect()
                receipt(path,dict(key=key,GT_used=False,arms=legal,anchors=anchors,frame_ids=ids,B=B,seconds=time.time()-tick))
            elif stage=='spatial_oracle':
                tar=read(OUT/'ORACLE_ONLY_targets.json')['rows'][key];arms={}
                for supervision in ['expert_common','GT_common']:
                    arms[supervision]={}
                    for scope in ['query','query_ln']:
                        it=SpatialInterface(model,caches,len(ids),scope)
                        arms[supervision][scope]=[spatial_fit(it,b,tar[supervision],lr,tuple(p['spatial_budgets'])) for lr in p['spatial_lrs']]
                        del it;gc.collect()
                receipt(path,dict(key=key,GT_used=True,matched_positions=True,arms=arms,B=B,frame_ids=ids,seconds=time.time()-tick))
            elif stage=='video_context':
                tar=read(OUT/'ORACLE_ONLY_targets.json')['rows'][key];it=SpatialInterface(model,caches,len(ids),'query')
                full=[np.ones(len(c['query_tgt']),bool) for c in caches];bm=np.zeros(len(ids),bool);bm[B[0]:B[1]+1]=True
                native=f22['predictions']['F0']['indices'];nm=np.zeros(len(ids),bool);nm[native[0]:native[1]+1]=True
                masks={'full':full,'predicted_B':[bm[o::2] for o in [0,1]],'predicted_native':[nm[o::2] for o in [0,1]],
                    'GT':[np.asarray(tar['event_mask'],bool)[o::2] for o in [0,1]]}
                for ref in ['predicted_B','predicted_native','GT']:
                    for seed in p['seeds']:
                        rng=np.random.default_rng(seed+int(hashlib.sha256(key.encode()).hexdigest()[:8],16));mm=[]
                        for m in masks[ref]:
                            out=np.zeros(len(m),bool);out[rng.choice(len(m),int(m.sum()),replace=False)]=True;mm.append(out)
                        masks[ref+'_random'+str(seed)]=mm
                results={}
                for name,mm in masks.items():
                    with torch.no_grad():bb,audit=it.values(mm)
                    results[name]=dict(boxes=bb.cpu(),audit=audit,masks=[m.tolist() for m in mm])
                assert torch.equal(results['full']['boxes'],b)
                receipt(path,dict(key=key,GT_used=True,oracle_diagnostic=True,arms=results,frame_ids=ids,B=B,seconds=time.time()-tick));del it
            else:raise ValueError(stage)
            assert state_digest(model)==digest;done+=1
            status(OUT/'progress.json',dict(stage=stage,cohort=cohort,last=key,new_completed=done,total=len(p['rows'][cohort]),seconds=time.time()-began))
            print(stage,key,'seconds',round(time.time()-tick,2),flush=True)
            del frames,batch,base,inputs,records,caches,f22;gc.collect();torch.cuda.empty_cache()
            if limit and done>=limit:break
        plan()
    except Exception:failure=traceback.format_exc();raise
    finally:
        write(OUT/'leases'/str(time.time_ns()).__add__('.json'),dict(stage=stage,cohort=cohort,seconds=time.time()-began,done=done,failure=failure))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


def prompt_specs(f22,donor):
    import re
    c=f22['parses']['context'];old=f22['parses']['old']
    if not c['eligible']:return {},'unresolved_target_keep_S2'
    target=c['context'][c['span'][0]:c['span'][1]];noun=target+'.'
    attr=old['phrase'].lower().strip()
    # Keep exactly the current referent text. Lemmatization is not a new noun.
    match=re.search(r'\b'+re.escape(old.get('entity',''))+r'\b',attr) if old.get('entity') else None
    if match:attr=attr[:match.start()]+target+attr[match.end():];sp=[match.start(),match.start()+len(target)]
    else:attr=target;sp=[0,len(target)]
    words=re.findall(r'\b[\w-]+\b',c['context']);pool=re.findall(r'\b[\w-]+\b',donor.lower())
    pool=[w for w in pool if w not in target.split()] or ['near','something']
    n=max(len(words)-len(target.split()),1);unrelated=target+' '+' '.join((pool*((n+len(pool)-1)//len(pool)))[:n])+'.'
    return dict(noun=dict(context=noun,span=[0,len(target)],entity=target),
        appearance=dict(context=attr+'.',span=sp,entity=target),full=c,
        unrelated=dict(context=unrelated,span=[0,len(target)],entity=target)),None


def text_context():
    import torch
    from scripts.run_decota_refine_v1 import configure
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from scripts.run_decota_spatial_extension_v1 import ContextView
    from vg_tta.decota_s2_system_v1 import probe_from_detection
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.foreground_runtime import state_digest
    p=plan();configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.time();used=sum(read(f)['seconds'] for f in (OUT/'leases').glob('*.json'));done=calls=0;failure=None
    try:
        old=read(ROOT/'artifacts/decota_refine_v1/lock.json');expert=SpatialExpert(old['expert_snapshot']);digest=state_digest(expert.model)
        for cohort,rr in p['rows'].items():
            for i,r in enumerate(rr):
                path=OUT/'text_context'/cohort/(r['key'].split(':')[1]+'.pt')
                if existing(path):continue
                tick=time.time();f22=load(r['F22_path']);f4=f22['predictions']['F4'];donor=rr[(i+1)%len(rr)];assert donor['source']!=r['source']
                specs,reason=prompt_specs(f22,donor['input']['caption']);arms={};raw=None;ids=r['input']['frame_ids'];cached=0;new=0
                for name,spec in specs.items():
                    probes=[];observations=[]
                    if len(expert.processor.tokenizer(spec['context'])['input_ids'])>256:raise ValueError('Unexpected bounded context truncation')
                    for pos in f4['positions']:
                        assert time.time()-began+used<p['GPU_seconds']
                        # Existing observations have a physical position/prompt key.
                        found=None
                        for ob in f22['observations']:
                            key=ob['cache_key'];d=ob['detection']
                            if d.get('text')==spec['context'] and list(d.get('target_span',[]))==list(spec['span']):
                                if key[1]==pos:found=d;break
                        if found is None:
                            if raw is None:raw,actual=decode(r['input']);assert actual==ids
                            found=ContextView(expert,spec)(raw[pos],None,None);calls+=1;new+=1
                        else:cached+=1
                        observations.append(dict(position=pos,detection=found));probes.append(probe_from_detection(found,pos,ids[pos]))
                    arms[name]=dict(probes=probes,observations=observations,spec=spec)
                assert state_digest(expert.model)==digest
                receipt(path,dict(key=r['key'],arms=arms,fallback=reason,GT_used=False,new_calls=new,cached=cached,
                    donor=donor['key'],donor_source=donor['source'],seconds=time.time()-tick))
                done+=1;status(OUT/'progress.json',dict(stage='text_context',new_completed=done,calls=calls,last=r['key']))
                print('TEXT',r['key'],'new',new,'reused',cached,'sec',round(time.time()-tick,2),flush=True)
        plan()
    except Exception:failure=traceback.format_exc();raise
    finally:
        write(OUT/'leases'/str(time.time_ns()).__add__('.json'),dict(stage='text_context',seconds=time.time()-began,done=done,calls=calls,failure=failure))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','oracle_targets','temporal','spatial','spatial_oracle','video_context','text_context']);ap.add_argument('--cohort',choices=list(GROUPS));ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.stage in ['prepare','oracle_targets','text_context']:globals()[a.stage]()
    else:run(a.cohort,a.stage,a.limit)
