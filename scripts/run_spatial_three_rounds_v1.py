"""Finite conditional spatial diagnostics, isolated from the locked paper method."""
import argparse,gc,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts.run_spatial_ssl_gpu_v1 import config,frozen_forward,flat_delta
OUT=ROOT/'artifacts/decota_spatial_three_rounds_v1'
PARENT=ROOT/'artifacts/decota_corrective_identifiability_v1'
PREVIOUS=PARENT/'spatial_gpu_v1'
OWN=['scripts/run_spatial_three_rounds_v1.py','vg_tta/spatial_three_rounds_v1.py',
     'protocols/decota_spatial_three_rounds_v1.md','scripts/score_spatial_three_rounds_v1.py',
     'tests/test_spatial_three_rounds_v1.py']
ARMS=['A_uniform','B_native','C_adapted','D_soft','E_shifted','F_GT']

def prepare():
    p=read(PARENT/'LOCK.json');old=read(PREVIOUS/'LOCK.json');pins=dict(p['protected'])
    for f in OWN+['scripts/run_spatial_ssl_gpu_v1.py','vg_tta/spatial_ssl_diagnostic_v1.py',
                  'scripts/analyze_spatial10_components_v1.py']:
        pins[f]=sha(ROOT/f)
    for d in ['methods/decota_final_simplified_v1','external/TA-STVG/models']:
        for f in sorted((ROOT/d).rglob('*.py')):pins[str(f.relative_to(ROOT))]=sha(f)
    for c in ['hcstvg1_test','vidstg_test']:
        rr=[r for r in p['rows'] if r['cohort']==c]
        assert len(rr)==len({r['group'] for r in rr})==32
    reused={}
    for r in old['rows']:
        stem=r['key'].replace(':','_');t=PREVIOUS/'teachers'/f'{stem}.pt';f=PREVIOUS/'episodes'/f'{stem}.pt'
        assert sha(t)==read(t.with_suffix('.json'))['sha256']
        assert sha(f)==read(f.with_suffix('.json'))['sha256']
        reused[r['key']]=dict(teacher=str(t),teacher_sha256=sha(t),episode=str(f),episode_sha256=sha(f))
    for r in p['rows']:assert sha(r['path'])==r['sha256']
    write(OUT/'LOCK.json',dict(rows=p['rows'],labels=p['labels'],labels_sha256=p['labels_sha256'],pins=pins,
          parent_lock_sha256=sha(PARENT/'LOCK.json'),reused=reused,historical_exposure=True,
          configs={c:config(c).to_dict() for c in ['hcstvg1_test','vidstg_test']},
          optimizer=dict(name='AdamW',eps=1e-4,betas=[.9,.999],weight_decay=0),
          parameters=1792,arms=ARMS,steps=[1,5],alphas=[0,.25,.5,.75,1],created=time.time()))

def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    assert sha(PARENT/'LOCK.json')==p['parent_lock_sha256']
    return p

def setup(model,x):
    import torch
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.spatial_three_rounds_v1 import temporal_state
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import detached
    from methods.decota_final_simplified_v1.backbone import full_prediction,query_subject
    frames,ids=decode(x['input']);batch,records,s=frozen_forward(model,frames,x)
    native=detached(s.zero);initial=s.state();actual=s.values()
    assert torch.equal(actual['boxes'],native['boxes']) and torch.equal(native['boxes'].cpu(),x['native_boxes'])
    assert all(torch.equal(z.cpu(),old) for z,old in zip(native['logits'],x['native_logits']))
    assert sum(v.numel() for _,v in s.named)==1792
    ts=temporal_state(x,config(x['cohort']).eta)
    adapted=[z.to('cuda') for z in x['predictions']['Full_DeCoTA']['logits']]
    expected=dict(boxes=native['boxes'],logits=adapted)
    with query_subject(model,batch,x['parses']['subject']):
        base=full_prediction(model,batch,ids,records,{**initial,**ts},expected)
    assert base['indices']==x['predictions']['Full_DeCoTA']['indices']
    return frames,ids,batch,records,s,native,initial,ts,adapted,base

def roi_teacher(model,frames,x,p,native):
    import torch
    from vg_tta.spatial_ssl_diagnostic_v1 import views,inverse_boxes,image_hash
    if x['key'] in p['reused']:
        r=p['reused'][x['key']];assert sha(r['teacher'])==r['teacher_sha256']
        old=load(r['teacher']);assert old['input_sha256']==image_hash(frames)
        assert not old['GT_access']
        return old['teachers']['cycle'].to('cuda'),dict(reused=r,view=old['views']['cycle'])
    v=views(frames,native['boxes'].cpu().numpy())['cycle']
    vb,vr,vs=frozen_forward(model,v['frames'],x)
    target=inverse_boxes(vs.zero['boxes'],v['transforms'],v['flip']).detach().clone()
    info=dict(raw_boxes=vs.zero['boxes'].cpu(),mapped_boxes=target.cpu(),
          image_sha256=image_hash(v['frames']),shape=list(v['frames'].shape),
          transforms=v['transforms'],pixel_crop=v.get('pixel_crop'),actual_two_offset_forward=True)
    del vb,vr,vs,v;gc.collect();torch.cuda.empty_cache()
    return target,info

def p0_episode(model,x,p):
    import numpy as np,torch
    from vg_tta.spatial_three_rounds_v1 import temporal_weights,WeightedROI,normalize
    from vg_tta.spatial_ssl_diagnostic_v1 import image_hash
    from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    started=time.time();frames,ids,batch,records,s,native,initial,ts,tl,base=setup(model,x)
    weights,confidence=temporal_weights(x,config(x['cohort']))
    target,info=roi_teacher(model,frames,x,p,native)
    path=OUT/'teachers'/f'{x["key"].replace(":","_")}.pt'
    barrier=dict(key=x['key'],target=target.cpu(),weights=weights,confidence=confidence,
          info=info,input_sha256=image_hash(frames),GT_access=False)
    if path.exists():
        assert sha(path)==read(path.with_suffix('.json'))['sha256']
        prev=load(path);assert torch.equal(prev['target'],barrier['target'])
        assert all(torch.equal(prev['weights'][k],v) for k,v in weights.items())
    else:
        save(path,barrier);write(path.with_suffix('.json'),dict(path=str(path),sha256=sha(path),GT_access=False))
    del frames
    # Label-free A-E targets above are committed before accessing diagnostic labels.
    gt=read(p['labels'])[x['key']];start,end=gt['interval']
    weights['F_GT']=normalize([(start<=i<end) for i in ids])
    losses={k:WeightedROI(target,w,k=='A_uniform') for k,w in weights.items()}
    s.restore(initial);noop=torch.optim.AdamW([v for _,v in s.named],lr=0,eps=1e-4,weight_decay=0)
    noop.zero_grad();losses['A_uniform'](s.values()['boxes']).backward();noop.step()
    assert all(torch.equal(v,initial[k]) for k,v in s.named)
    assert torch.equal(s.values()['boxes'],native['boxes']);del noop
    arms={};paths={};origins={};old=None;parity=0
    if x['key'] in p['reused']:
        rr=p['reused'][x['key']];assert sha(rr['episode'])==rr['episode_sha256'];old=load(rr['episode'])
        assert state_hash(initial)==state_hash({k:v.to('cuda') for k,v in old['initial'].items()})
    for name in ARMS:
        s.restore(initial);opt=torch.optim.AdamW([v for _,v in s.named],lr=config(x['cohort']).spatial_lr,
                          betas=(.9,.999),eps=1e-4,weight_decay=0)
        lossfn=losses[name];origins[name]=float(lossfn(s.values()['boxes']).detach());history=[]
        for step in range(1,6):
            opt.zero_grad();loss=lossfn(s.values()['boxes']);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for _,v in s.named)
            gn=float(torch.cat([v.grad.reshape(-1).double() for _,v in s.named]).norm())
            opt.step()
            with torch.no_grad():actual=detached(s.values());value=float(lossfn(actual['boxes']))
            assert torch.isfinite(actual['boxes']).all()
            assert all(torch.equal(a,b) for a,b in zip(actual['logits'],native['logits']))
            delta=flat_delta(s,initial);history.append(dict(step=step,loss=value,gradient_norm=gn,
                  delta_norm=float(delta.norm()),changed_parameters=int((delta!=0).sum())))
            if step in [1,5]:
                ss=s.state()
                if name=='A_uniform' and old is not None:
                    o=old['arms'][f'cycle_step{step}']
                    assert torch.equal(actual['boxes'].cpu(),o['prediction']['boxes']),'uniform prior boxes'
                    assert all(torch.equal(v.cpu(),o['state'][k]) for k,v in ss.items()),'uniform prior state'
                    parity+=1
                with query_subject(model,batch,x['parses']['subject']):
                    pr=full_prediction(model,batch,ids,records,{**ss,**ts},dict(boxes=actual['boxes'],logits=tl))
                assert pr['indices']==base['indices']
                arms[f'{name}_step{step}']=dict(prediction=pr,state=detached(ss,'cpu'),
                       loss=value,delta_norm=float(delta.norm()),changed_parameters=int((delta!=0).sum()))
            del actual,delta,loss
        paths[name]=history;del opt
    s.restore(initial);assert state_hash(s.state())==state_hash(initial)
    assert torch.equal(s.values()['boxes'],native['boxes'])
    assert not any(v.requires_grad or v.grad is not None for v in model.parameters())
    return dict(key=x['key'],cohort=x['cohort'],group=x['group'],source=x['input']['source'],
      input=x['input'],frame_ids=ids,baseline=base,arms=arms,weights=weights,confidence=confidence,
      initial=detached(initial,'cpu'),temporal_state=ts,paths=paths,origin_losses=origins,
      teacher_receipt=read(path.with_suffix('.json')),seconds=time.time()-started,
      peak_memory_bytes=torch.cuda.max_memory_allocated(),
      audit=dict(parameters=1792,native_exact=True,lr0_exact=True,steps0_exact=True,reset_exact=True,
            temporal_logits_invariant=True,full_reinsertions=13,uniform_prior_parity=parity,
            teacher_frozen=True,legal_weights_GT=False,GT_F_only=True,empty_F=float(weights['F_GT'].sum())==0))

def p2_episode(model,x,p):
    import torch
    from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    decision=read(OUT/'P0_DECISION.json');name=decision['P2_objective'];assert name
    started=time.time();frames,ids,batch,records,s,native,initial,ts,tl,base=setup(model,x);del frames
    f=OUT/'p0'/f'{x["key"].replace(":","_")}.pt'
    assert sha(f)==read(f.with_suffix('.json'))['sha256'];old=load(f)
    end=old['arms'][name+'_step5'];arms={}
    for alpha in p['alphas']:
        # Exact endpoints avoid floating subtraction/readdition at alpha 1.
        if alpha==0:ss=old['initial']
        elif alpha==1:ss=end['state']
        else:ss={k:v+alpha*(end['state'][k]-v) for k,v in old['initial'].items()}
        s.restore(ss)
        with torch.no_grad():actual=detached(s.values())
        with query_subject(model,batch,x['parses']['subject']):
            pr=full_prediction(model,batch,ids,records,{**ss,**ts},dict(boxes=actual['boxes'],logits=tl))
        if alpha in [0,1]:
            reference=base if alpha==0 else end['prediction']
            assert torch.equal(pr['boxes'],reference['boxes'])
        assert pr['indices']==base['indices']
        arms[str(alpha)]=dict(prediction=pr,state=detached(ss,'cpu'),delta_norm=float(flat_delta(s,initial).norm()))
    s.restore(initial);assert state_hash(s.state())==state_hash(initial)
    return dict(key=x['key'],cohort=x['cohort'],source=x['input']['source'],group=x['group'],
           arms=arms,baseline=base,objective=name,seconds=time.time()-started,
           confidence=old['confidence'],audit=dict(full_reinsertions=6,no_backward=True,
                  reset_exact=True,temporal_invariant=True,endpoints_exact=True))

def run(phase,limit=0):
    import torch,numpy as np
    from scripts.run_final_simplification_v1 import lease
    from methods.decota_final_simplified_v1._tastvg_load import load_model_on_device
    p=verify();assert sha(p['labels'])==p['labels_sha256']
    if phase=='p2':assert read(OUT/'P0_DECISION.json')['P2_run']
    gpu=lease();model=None;current=None;done=0
    torch.set_num_threads(4);torch.manual_seed(20260910);np.random.seed(20260910)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    try:
        for row in p['rows']:
            f=OUT/phase/f'{row["key"].replace(":","_")}.pt'
            if f.with_suffix('.json').exists():
                assert sha(f)==read(f.with_suffix('.json'))['sha256'];done+=1;continue
            if limit and done>=limit:break
            if row['cohort']!=current:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                c=config(row['cohort']);assert sha(ROOT/c.checkpoint)==c.checkpoint_sha256
                runtime=ROOT/'artifacts/tastvg_runtime'/('vidstg' if c.source_dataset=='vidstg' else 'hc-stvg2')
                model,_,report=load_model_on_device(c.source_dataset,runtime,checkpoint=ROOT/c.checkpoint,
                      device='cuda',source_dataset=c.source_dataset)
                model.eval().requires_grad_(False);current=row['cohort']
            assert sha(row['path'])==row['sha256'];x=load(row['path']);torch.cuda.reset_peak_memory_stats()
            status(OUT/f'{phase}_STATUS.json',dict(status='running',done=done,total=64,key=row['key'],pid=os.getpid(),updated=time.time()))
            z=(p0_episode if phase=='p0' else p2_episode)(model,x,p)
            save(f,z);write(f.with_suffix('.json'),dict(key=row['key'],sha256=sha(f),lock_sha256=sha(OUT/'LOCK.json'),seconds=z['seconds']))
            done+=1;print('COMPLETED',phase,done,64,row['key'],round(z['seconds'],2),flush=True)
            del z,x;gc.collect();torch.cuda.empty_cache()
        verify();status(OUT/f'{phase}_STATUS.json',dict(status='complete' if done==64 else 'smoke_complete',done=done,total=64,updated=time.time()))
    except BaseException as e:
        status(OUT/f'{phase}_FAILURE.json',dict(type=type(e).__name__,message=str(e),done=done,updated=time.time()));raise
    finally:gpu.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','p0','p2']);ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args();prepare() if a.stage=='prepare' else run(a.stage,a.limit)
