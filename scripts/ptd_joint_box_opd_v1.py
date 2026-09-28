"""Resumable finite 256-source runner. No label access in any inference stage."""
import argparse,gc,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts.ptd_opd_information_v1 import Budget as BaseBudget
from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for,infer
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter
from vg_tta.ptd_joint_box_opd_v1 import Episode,SEEDS,sampled,sampling_seed
from scripts.prepare_ptd_joint_box_opd_v1 import OUT,digest

def rows(stage='all'):
    rr=read(OUT/'INPUTS.json')
    return [r for r in rr if stage=='all' or (r['engineering'] if stage=='engineering' else r['split']==stage)]
def file(r,folder,arm=None,seed=None):return OUT/folder/(digest(r['key'])+('' if arm is None else f'_{arm}_{seed}')+'.pt')
def checked(p):assert sha(p)==read(p.with_suffix('.json'))['sha'];return load(p)
def put(p,z):save(p,z);write(p.with_suffix('.json'),dict(sha=sha(p)))

class Budget(BaseBudget):
    def check(self):
        super().check()
        assert shutil.disk_usage(ROOT).free>20*2**30,'DISK_FLOOR'
        if time.monotonic()-getattr(self,'last_disk_check',0)>30:
            self.last_disk_check=time.monotonic()
            assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<24*2**30,'OUTPUT_BUDGET'

def register():
    intake=read(OUT/'INTAKE_COMPLETE.json');assert intake['count']==256 and sha(OUT/'INPUTS.json')==intake['input_sha']
    assert read(OUT/'SYNTHETIC_GRADIENT_AUDIT.json')['passed']
    pins=['protocols/ptd_joint_box_opd_v1.md','scripts/ptd_joint_box_opd_v1.py','scripts/ptd_joint_box_teacher_v1.py','vg_tta/ptd_joint_box_opd_v1.py',
          'scripts/prepare_ptd_joint_box_opd_v1.py','scripts/ptd_dense_opd_teacher_v1.py','scripts/ptd_spatial_adapter_ab_v1.py',
          'scripts/score_ptd_joint_box_opd_v1.py','vg_tta/ptd_spatial_adapter_ab_v1.py']
    protected=['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json','artifacts/c1_fresh_confirmation_v1/vid_train_audit/PROSPECTIVE_SOURCE_RESERVATION.json',
        'artifacts/ptd_dense_opd_v1/PREDICTION_BARRIER.json','artifacts/ptd_opd_information_v1/PREDICTION_BARRIER.json']
    write(OUT/'REGISTRATION.json',dict(time=time.time(),input_sha=intake['input_sha'],selection_sha=sha(OUT/'SELECTION.json'),pins={p:sha(ROOT/p) for p in pins},
        protected={p:sha(ROOT/p) for p in protected},seeds=SEEDS,LR=.002,steps=3,rank=16,M=4,primary_step=3,sources=256,development=64,confirmation=192,
        gpu_seconds=86400,engineering_seconds=3600,output_bytes=24*2**30,GT_training=False,confirmation_D1_independent_of_J2=True))

def verify():
    r=read(OUT/'REGISTRATION.json');pins=dict(r['pins'])
    for f in sorted((OUT/'amendments').glob('*.json')):pins.update(read(f).get('pins',{}))
    for p,h in {**pins,**r['protected']}.items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'INPUTS.json')==r['input_sha'];return r

def capture(budget,stage):
    pr=processor_load();model=model_load()
    for r in rows(stage):
        dest=file(r,'native')
        if dest.exists():checked(dest);continue
        start=time.monotonic();frames,_=frames_for(r,'clean');inp,pre=inputs_for(r,pr,frames)
        z=infer(model,pr,inp);z.update(key=r['key'],preprocess=pre,frame_ids=r['input']['frame_ids'],native_wall_seconds=time.monotonic()-start)
        put(dest,z);budget.calls+=1;budget.check();print('NATIVE',r['key'],z['format_ok'],len(z.get('positions',[])),flush=True)
        del z,frames,inp;gc.collect();torch.cuda.empty_cache()
    write(OUT/f'CAPTURE_{stage}_COMPLETE.json',dict(count=len(rows(stage)),time=time.time(),GT_read=False))

def marginal(t,budget,r,z):
    dest=file(r,'teacher')
    if dest.exists():return checked(dest)['logp']
    inp,text=t.inputs(r,z,'plus');p=t.prepare(inp);cache,_=t.prefill(p);points=[];start=time.monotonic()
    for j,pos in enumerate(z['positions']):
        for c,name in enumerate(['x1','y1','x2','y2']):
            point=OUT/'marginal_points'/digest(r['key'])/f'{j}_{c}.pt'
            if point.exists():v=checked(point)
            else:
                selector=f'Frame {pos+1}, {name}: ';clock=time.monotonic();calls,tokens=budget.calls,budget.tokens;v=t.distribution(p,cache,selector)
                v.update(position=pos,coordinate=name,seconds=time.monotonic()-clock,teacher_calls=budget.calls-calls,teacher_tokens=budget.tokens-tokens);put(point,v)
            audit=OUT/'numerical'/f'{digest(r["key"])}_marginal.json'
            if j==0 and c==0 and not audit.exists():
                del cache;gc.collect();torch.cuda.empty_cache();errors=[]
                for val in [0,1000]:
                    direct=t.independent(p,f'Frame {pos+1}, {name}: ',val);errors.append(abs(direct-float(v['raw'][val])))
                write(audit,dict(errors=errors,max_error=max(errors),passed=max(errors)<=.1));assert max(errors)<=.1,'MARGINAL_NUMERICAL'
                cache,_=t.prefill(p)
            points.append(v);budget.check()
    result=dict(key=r['key'],positions=z['positions'],logp=torch.stack([v['logp'] for v in points]).reshape(-1,4,1001),
        prompt=text,input_ids=inp['input_ids'],point_seconds=sum(v['seconds'] for v in points),stage_seconds=time.monotonic()-start,coordinate_paths=len(points)*1001,GT_used=False)
    put(dest,result);del cache,p,inp;gc.collect();torch.cuda.empty_cache();return result['logp']

def marginal_fits(r,z,qp):
    arms=['D1','J1'] if r['split']=='development' else ['D1']
    for seed in SEEDS:
        for arm in arms:
            dest=file(r,'fits',arm,seed)
            if dest.exists():checked(dest);continue
            ep=Episode(z,r['key'],arm,seed)
            for step in range(3):ep.update(step,qp=qp)
            put(dest,ep.finish());print('FIT',r['key'],arm,seed,flush=True)

def joint_fits(t,budget,r,z):
    if all(file(r,'fits','J2',s).exists() for s in SEEDS):
        for s in SEEDS:checked(file(r,'fits','J2',s))
        return
    inp,text=t.inputs_joint(r,z);p=t.prepare(inp);cache,_=t.prefill(p)
    for seed in SEEDS:
        dest=file(r,'fits','J2',seed)
        if dest.exists():checked(dest);continue
        ep=Episode(z,r['key'],'J2',seed)
        for step in range(3):
            acts=sampled(ep.policy(),r['key'],seed,step);scores=[]
            for j,pos in enumerate(z['positions']):
                point=OUT/'joint_points'/digest(r['key'])/f'{seed}_{step}_{j}.pt'
                selector=f'Frame {pos+1}, box: '
                if point.exists():
                    v=checked(point);assert torch.equal(v['actions'],acts[j]) and v['prompt_sha']==digest(text)
                else:
                    start=time.monotonic();calls,tokens=budget.calls,budget.tokens;v=t.score_boxes(p,cache,selector,acts[j].tolist(),text)
                    v.update(actions=acts[j],position=pos,seed=seed,step=step,prompt_sha=digest(text),seconds=time.monotonic()-start,teacher_calls=budget.calls-calls,teacher_tokens=budget.tokens-tokens)
                    put(point,v)
                audit=OUT/'numerical'/f'{digest(r["key"])}_joint.json'
                if j==0 and seed==SEEDS[0] and step==0 and not audit.exists():
                    del cache;gc.collect();torch.cuda.empty_cache();direct=t.independent_box(p,selector,acts[j,0].tolist());err=abs(direct-float(v['raw'][0]))
                    write(audit,dict(error=err,passed=err<=.1,canonical=v['canonical'][0],full=direct,trie=float(v['raw'][0])));assert err<=.1,'JOINT_NUMERICAL'
                    cache,_=t.prefill(p)
                scores.append(v['segments']);budget.check()
            ep.update(step,actions=acts,teacher_segments=torch.stack(scores))
            print('J2_STEP',r['key'],seed,step+1,ep.updates[-1]['displacement_norm'],flush=True)
        result=ep.finish();result.update(prompt=text,prompt_sha=digest(text),fresh_current_prefixes=True)
        put(dest,result)
    del inp,p,cache;gc.collect();torch.cuda.empty_cache()

def teach(budget,stage,mode):
    from scripts.ptd_joint_box_teacher_v1 import JointTeacher
    t=JointTeacher(budget)
    for r in rows(stage):
        z=checked(file(r,'native'))
        if not z['format_ok']:
            skip=OUT/'native_invalid'/f'{digest(r["key"])}.json'
            if not skip.exists():write(skip,dict(key=r['key'],reason='native_format_invalid',arms_fallback='unchanged_native',source_kept=True))
            continue
        if mode!='joint-only':
            qp=marginal(t,budget,r,z);marginal_fits(r,z,qp)
        if mode!='marginal-only' and not (OUT/'J2_UNAVAILABLE.json').exists():
            try:joint_fits(t,budget,r,z)
            except Exception as e:
                # Do not let a J2 engineering failure cancel independent D1 confirmation.
                failure=dict(key=r['key'],error=repr(e),traceback=traceback.format_exc(),time=time.time(),stage=stage)
                write(OUT/'J2_UNAVAILABLE.json',failure);print('J2_UNAVAILABLE',failure,flush=True)
                raise
        budget.check();status(OUT/'PROGRESS.json',dict(stage=stage,mode=mode,last_source=r['key'],time=time.time()))
    write(OUT/f'TEACHER_{stage}_{mode}_COMPLETE.json',dict(time=time.time(),sources=len(rows(stage)),GT_read=False))

def replay(budget,stage):
    pr=processor_load();model=model_load();arms=['D1','J1','J2'] if stage in ['engineering','development'] else ['D1']
    if stage=='confirmation' and read(OUT/'CANDIDATE_DECISION.json')['include_J2']:arms.append('J2')
    if (OUT/'J2_UNAVAILABLE.json').exists():arms=[a for a in arms if a!='J2']
    files={}
    for r in rows(stage):
        z=checked(file(r,'native'));files[str(file(r,'native'))]=sha(file(r,'native'))
        for folder in ['teacher','marginal_points','joint_points']:
            candidates=[file(r,folder)] if folder=='teacher' else list((OUT/folder/digest(r['key'])).glob('*.pt'))
            for path in candidates:
                if path.exists():checked(path);files[str(path)]=sha(path)
        frames,_=frames_for(r,'clean');inp,_=inputs_for(r,pr,frames)
        for arm in arms:
            for seed in SEEDS:
                dest=file(r,'predictions',arm,seed)
                if dest.exists():checked(dest)
                elif not z['format_ok']:put(dest,dict(**z,arm=arm,seed=seed,fallback='native_format_invalid'))
                else:
                    fitted=checked(file(r,'fits',arm,seed));adapter=Adapter(z['h'].shape[-1],seed=seed).cuda();adapter.load_state_dict(fitted['history'][-1]['state'])
                    clock=time.monotonic();result=infer(model,pr,inp,fixed=z,adapter=adapter)
                    assert torch.equal(result['h'],z['h']) and torch.equal(result['logits'],z['logits'])
                    assert torch.equal(result['adapted_logits'].argmax(-1),fitted['history'][-1]['tokens'])
                    assert result['interval']==z['interval'] and result['semantic']==z['semantic']
                    result.update(key=r['key'],arm=arm,seed=seed,full_model_reinsertion_exact=True,replay_seconds=time.monotonic()-clock);put(dest,result);budget.calls+=1
                files[str(dest)]=sha(dest)
                if file(r,'fits',arm,seed).exists():files[str(file(r,'fits',arm,seed))]=sha(file(r,'fits',arm,seed))
                budget.check()
        del inp,frames;gc.collect();torch.cuda.empty_cache();print('REPLAY',stage,r['key'],flush=True)
    write(OUT/f'PREDICTION_BARRIER_{stage}.json',dict(files=files,time=time.time(),sources=len(rows(stage)),arms=arms,seeds=SEEDS,GT_read=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['register','capture','teacher','replay']);ap.add_argument('--stage',default='all');ap.add_argument('--mode',default='both',choices=['both','marginal-only','joint-only']);a=ap.parse_args()
    torch.set_num_threads(4)
    if a.action=='register':register();return
    verify();limit=3600 if a.stage=='engineering' or (a.action=='capture' and not (OUT/'ENGINEERING_COST.json').exists()) else 86400
    b=Budget(OUT,a.action+'_'+a.stage+'_'+a.mode,limit)
    try:
        if a.action=='capture':capture(b,a.stage)
        elif a.action=='teacher':teach(b,a.stage,a.mode)
        else:replay(b,a.stage)
        status(OUT/'STATUS.json',dict(state='stage_completed',action=a.action,stage=a.stage,mode=a.mode,time=time.time()))
    except BaseException as e:
        f=dict(action=a.action,stage=a.stage,mode=a.mode,error=repr(e),traceback=traceback.format_exc(),time=time.time())
        write(OUT/'failures'/f'{time.time_ns()}.json',f);status(OUT/'STATUS.json',dict(state='failed',**f));raise
    finally:b.close()
if __name__=='__main__':main()
