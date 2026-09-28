"""Finite native-DINO/PTD bridge: GT-free acquisition; separate offline scoring."""
import argparse,collections,gc,hashlib,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.ptd_teacher_qualification_v1 import rows,file,checked,put,read,write,status,sha,load,geometry,support,stats,OLD
from scripts.ptd_spatial_adapter_ab_v1 import frames_for
from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256
from methods.decota_final_simplified_v1.observations import SpatialExpert,ContextView,probe_from_detection,tensor_hash
OUT=ROOT/'artifacts/ptd_dino_bridge_v1'
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def dest(r):return OUT/'references'/(digest(r['key'])+'.pt')
def point(r,j):return OUT/'points'/digest(r['key'])/f'{j}.pt'
def used():return sum(read(f)['seconds'] for f in (OUT/'worker_receipts').glob('*.json'))
def verify():
    reg=read(OUT/'REGISTRATION.json');pins=dict(reg['pins'])
    for f in sorted((OUT/'amendments').glob('*.json')):pins.update(read(f).get('pins',{}))
    for f,h in {**pins,**reg['parents'],**reg['protected']}.items():assert sha(f)==h,f
    return reg
def old_observations(parent):
    ex=parent['expert']
    if 'observations' not in ex and 'reference' in ex:
        ref=ex['reference'];assert sha(ref['path'])==ref['sha256'];ex=load(ref['path'])['expert']
    return {o['receipt']['frame_id']:o for (view,pos),o in ex.get('observations',{}).items() if view=='original'}
def register():
    parents={};specs=[]
    for r in rows('development'):
        z=checked(file(r,'native'));assert z['format_ok']
        assert sha(r['parent_file'])==r['parent_sha'];p=load(r['parent_file']);assert p['input']['caption']==r['input']['caption']
        for f in [file(r,'native'),Path(r['parent_file'])]:parents[str(f)]=sha(f)
        specs.append(dict(key=r['key'],domain=r['domain'],source=r['source'],positions=z['positions'],frame_ids=z['frame_ids'],parses=p['parses'],native_sha=sha(file(r,'native')),engineering=r['engineering']))
    assert len(specs)==64 and sum(len(x['positions']) for x in specs)==1014
    for f in [OLD/'INPUTS.json',OLD/'LABELS_SCORER_ONLY.json']:
        parents[str(f)]=sha(f)
    weights=ROOT/EXPERT_SNAPSHOT/'model.safetensors';assert sha(weights)==EXPERT_SHA256
    parents[str(weights)]=EXPERT_SHA256
    protected={str(ROOT/f):sha(ROOT/f) for f in ['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json','methods/C1_FINAL_RESEARCH_CONFIG.json','artifacts/c1_fresh_confirmation_v1/vid_train_audit/PROSPECTIVE_SOURCE_RESERVATION.json']}
    pins=[Path(__file__),ROOT/'protocols/ptd_dino_bridge_v1.md',ROOT/'methods/decota_final_simplified_v1/observations.py',ROOT/'scripts/score_ptd_spatial_adapter_ab_v1.py',ROOT/'scripts/ptd_teacher_qualification_v1.py',ROOT/'vg_tta/ptd_spatial_adapter_ab_v1.py']
    write(OUT/'REGISTRATION.json',dict(time=time.time(),request='./private_authorization_notes/authorization.txt',pins={str(f):sha(f) for f in pins},parents=parents,protected=protected,
        sources=64,queries=64,positions=1014,GT_worker=False,historically_exposed=True,max_GPU_seconds=7200,stage1_GPU_seconds=1800,max_output_bytes=3*2**30,
        stage2_entry='full64 direct-reference delta vIoU source bootstrap95 lower bound > 0',stage2_seeds=[20260925,20260926,20260927],stage2_LR=.002,stage2_steps=6,beta=1,M=64))
    write(OUT/'INPUTS.json',specs);status(OUT/'STATUS.json',dict(state='registered',measurement='unscored',time=time.time()))
def check_budget(start,past):
    assert time.monotonic()-start+past<1800,'STAGE1_GPU_TIME_LIMIT'
    assert shutil.disk_usage(ROOT).free>25*2**30,'DISK_FLOOR'
def acquire(stage):
    from scripts.run_final_simplification_v1 import lease
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();guard=lease();start=time.monotonic();past=used();count=0;controls=[];expert=None
    rr=rows('development')
    if stage=='engineering':rr=[r for dom in ['HC','Vid'] for r in [x for x in rr if x['engineering'] and x['domain']==dom][:2]]
    try:
        torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);before=state_hash(expert.model.state_dict())
        for r in rr:
            check_budget(start,past)
            if dest(r).exists():checked(dest(r));continue
            source_start=time.monotonic();z=checked(file(r,'native'));parent=load(r['parent_file']);parses=parent['parses'];historical=old_observations(parent)
            frames,ids=frames_for(r,'clean');rgbhash=hashlib.sha256(frames.tobytes()).hexdigest();assert rgbhash==z['preprocess']['pixel_sha'];assert ids==r['input']['frame_ids']
            context,s1=parses['context'],parses['old'];eligible=context['eligible'] and len(expert.processor.tokenizer(context['context'])['input_ids'])<=256
            points=[]
            for j,pos in enumerate(z['positions']):
                check_budget(start,past);f=point(r,j)
                if f.exists():points.append(checked(f));continue
                base=dict(key=r['key'],local_position=j,position=pos,frame_id=ids[pos],RGB_sha=hashlib.sha256(frames[pos].tobytes()).hexdigest(),GT_used=False)
                if not eligible and not s1['phrase']:
                    out=dict(**base,accepted=False,reason='query_parser_abstention',probe=None,detection=None,inputs={},context_active=False,seconds=0.)
                else:
                    inp={}
                    def hook(m,args,kw):
                        for k in ['pixel_values','pixel_mask','input_ids','attention_mask']:
                            if k in kw:inp[k]=dict(sha256=tensor_hash(kw[k]),shape=list(kw[k].shape),dtype=str(kw[k].dtype))
                    handle=expert.model.register_forward_pre_hook(hook,with_kwargs=True);tick=time.monotonic()
                    try:
                        with torch.no_grad():d=ContextView(expert,context)(frames[pos],context['context'],context['entity']) if eligible else expert(frames[pos],s1['phrase'],s1['entity'])
                    finally:handle.remove()
                    if eligible:probe=probe_from_detection(d,pos,ids[pos])
                    else:probe=dict(position=pos,frame_id=ids[pos],accepted=d['accepted'],reason=d['reason'],margin=d['margin'],boxes=torch.tensor([d['box']]) if d['box'] else torch.empty(0,4),target_scores=[d['score']] if d['box'] else [],candidate_ids=[0] if d['box'] else [])
                    out=dict(**base,accepted=probe['accepted'],reason=probe['reason'],probe=probe,detection=d,inputs=inp,context_active=bool(eligible),seconds=time.monotonic()-tick)
                    count+=1
                    if ids[pos] in historical:
                        old=historical[ids[pos]];assert old['receipt']['rgb_sha256']==base['RGB_sha'];assert old['receipt']['inputs']==inp
                        op=old['probe'];assert bool(op['accepted'])==bool(probe['accepted']) and op['reason']==probe['reason']
                        boxerr=float((torch.as_tensor(op['boxes'])-torch.as_tensor(probe['boxes'])).abs().max()) if len(op['boxes']) else 0.
                        serr=float((torch.as_tensor(op['target_scores'])-torch.as_tensor(probe['target_scores'])).abs().max()) if len(op['target_scores']) else 0.
                        assert max(boxerr,serr)<1e-5,(boxerr,serr)
                        out['historical_control']=dict(passed=True,box_error=boxerr,score_error=serr,RGB_and_processor_exact=True);controls.append(out['historical_control'])
                put(f,out);points.append(out)
            xy=z['base_tokens'].double()/1000;target=xy.clone();mask=[]
            for j,v in enumerate(points):
                ok=v['accepted'];mask.append(ok)
                if ok:
                    pr=v['probe'];i=int(np.argmax(pr['target_scores']));b=torch.as_tensor(pr['boxes'][i]).double();target[j]=torch.cat([b[:2]-b[2:]/2,b[:2]+b[2:]/2]).clamp(0,1)
                    assert (target[j]>=0).all() and (target[j]<=1).all() and (target[j,2:]>target[j,:2]).all()
            result=dict(key=r['key'],positions=z['positions'],frame_ids=ids,parses=parses,mask=mask,xyxy=target,source_seconds=time.monotonic()-source_start,
                        point_files={str(point(r,j)):sha(point(r,j)) for j in range(len(points))},RGB_sha=rgbhash,GT_used=False)
            assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<3*2**30,'OUTPUT_BUDGET'
            put(dest(r),result);status(OUT/'STATUS.json',dict(state='acquiring',stage=stage,last_source=r['key'],completed=sum(dest(q).exists() for q in rows('development')),calls=count,time=time.time()))
            print('DINO',r['key'],len(points),sum(mask),round(result['source_seconds'],2),flush=True)
            del frames,points,result;gc.collect();torch.cuda.empty_cache()
        assert state_hash(expert.model.state_dict())==before
        files={str(dest(r)):sha(dest(r)) for r in rr};write(OUT/f'REFERENCE_BARRIER_{stage}.json',dict(time=time.time(),files=files,sources=len(rr),GT_used=False,expert_state_unchanged=True))
    except BaseException as e:
        failure=dict(time=time.time(),stage=stage,error=repr(e),traceback=traceback.format_exc());write(OUT/'failures'/f'{time.time_ns()}.json',failure);status(OUT/'STATUS.json',dict(state='failed',**failure));raise
    finally:
        write(OUT/'worker_receipts'/f'{time.time_ns()}.json',dict(stage=stage,seconds=time.monotonic()-start,calls=count,controls=controls,peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0))
        guard.close()
def forecast():
    verify();done=[checked(dest(r)) for r in rows('development') if dest(r).exists()];positions=sum(len(x['positions']) for x in done);remain=1014-positions
    estimate=sum(x['source_seconds'] for x in done)/max(1,positions)*remain*1.5+60
    result=dict(time=time.time(),observed_sources=len(done),observed_positions=positions,remaining_positions=remain,used_GPU_seconds=used(),estimate_with_margin=estimate,allowed=estimate+used()<1800,GT_used=False)
    write(OUT/'STAGE1_FORECAST.json',result);print(result);assert result['allowed'],'MEASURED_STAGE1_BUDGET_STOP'
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','acquire','forecast']);p.add_argument('--stage',choices=['engineering','development'],default='development');a=p.parse_args()
    if a.action=='register':register()
    elif a.action=='forecast':forecast()
    else:acquire(a.stage)
if __name__=='__main__':main()
