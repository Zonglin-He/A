"""Finite actual-forward and same-interface spatial diagnostics; independent output."""
import argparse, copy, gc, json, os, re, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
OUT=ROOT/'artifacts/decota_corrective_identifiability_v1/spatial_gpu_v1'
PARENT=ROOT/'artifacts/decota_corrective_identifiability_v1'
PROTOCOL=ROOT/'protocols/decota_spatial_ssl_gpu_v1.md'
OWN=['scripts/run_spatial_ssl_gpu_v1.py','vg_tta/spatial_ssl_diagnostic_v1.py','protocols/decota_spatial_ssl_gpu_v1.md']


def config(cohort):
    from methods.decota_final_simplified_v1.config import MethodConfig
    return MethodConfig.for_direction('vid_to_hc1' if cohort=='hcstvg1_test' else 'hc2_to_vid')


def prepare():
    p=read(PARENT/'LOCK.json');rr=[]
    for c in ['hcstvg1_test','vidstg_test']:
        rows=[r for r in p['rows'] if r['cohort']==c][:8]
        assert len(rows)==len({r['group'] for r in rows})==8
        rr.extend(rows)
    pins={**p['protected'],**{f:sha(ROOT/f) for f in OWN}}
    for r in rr: assert sha(r['path'])==r['sha256']
    write(OUT/'LOCK.json',dict(rows=rr,pins=pins,parent_lock_sha256=sha(PARENT/'LOCK.json'),
        labels=p['labels'],labels_sha256=p['labels_sha256'],historical_exposure=True,
        configs={c:config(c).to_dict() for c in ['hcstvg1_test','vidstg_test']},
        spatial_parameters=1792,steps=[1,5],primary_states='raw_steps',secondary_states='own_loss_best_prefix',
        methods=['flip','crop','cycle','smooth','ssl_joint','gt_all','gt_matched','external_matched','external_all'],
        new_expert_calls=0,created=time.time()))


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items(): assert sha(ROOT/f)==h, f
    assert sha(PARENT/'LOCK.json')==p['parent_lock_sha256']
    return p


def frozen_forward(model,frames,x):
    from methods.decota_final_simplified_v1.backbone import make_batch,query_subject,capture
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    q=dict(x['input']);q.update(height=frames.shape[1],width=frames.shape[2])
    batch=make_batch(frames,x['frame_ids'],q,model)
    with query_subject(model,batch,x['parses']['subject']):
        views,records=capture(model,batch)
        replay=SpatialReplay(model,views,len(x['frame_ids']))
    del views
    return batch,records,replay


def flat_delta(replay,initial):
    import torch
    return torch.cat([(v.detach()-initial[k]).reshape(-1).double() for k,v in replay.named])


def episode(model,x,label_path,p):
    import numpy as np, torch
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.spatial_ssl_diagnostic_v1 import views,inverse_boxes,image_hash,BoxLoss,velocity_energy,gradient_vector,cosine
    from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from methods.decota_final_simplified_v1.objectives import prediction
    started=time.time();frames,ids=decode(x['input']);batch,records,s=frozen_forward(model,frames,x)
    initial=s.state(); native=detached(s.zero);zero=s.values()
    assert torch.equal(zero['boxes'],native['boxes'])
    assert torch.equal(native['boxes'].cpu(),x['native_boxes']), 'cached native boxes mismatch'
    assert all(torch.equal(z.cpu(),v) for z,v in zip(native['logits'],x['native_logits'])), 'cached logits mismatch'
    nativepr=prediction(native['logits'],native['boxes'],records,ids)
    assert nativepr['indices']==x['predictions']['Frozen']['indices']
    with query_subject(model,batch,x['parses']['subject']):
        full_prediction(model,batch,ids,records,initial,native)
    del zero
    # Entire intrinsic target construction precedes this episode's GT access.
    teachers={};viewinfo={}
    for name,v in views(frames,native['boxes'].cpu().numpy()).items():
        vb,vr,vs=frozen_forward(model,v['frames'],x)
        raw=vs.zero['boxes']; mapped=inverse_boxes(raw,v['transforms'],v['flip']).detach()
        teachers[name]=mapped.clone()
        viewinfo[name]=dict(raw_boxes=raw.detach().cpu(),mapped_boxes=mapped.cpu(),
             logits=detached(vs.zero['logits'],'cpu'),image_sha256=image_hash(v['frames']),
             shape=list(v['frames'].shape),transforms=v['transforms'],flip=v['flip'],pixel_crop=v.get('pixel_crop'),
             max_native_difference=float((mapped-native['boxes']).abs().max()),actual_two_offset_forward=True)
        if name=='roundtrip_noop':
            assert image_hash(v['frames'])==image_hash(frames)
            assert torch.equal(mapped,native['boxes']), 'roundtrip image/model no-op mismatch'
        del vb,vr,vs;gc.collect();torch.cuda.empty_cache()
    qkey=x['key'];d=OUT/'teachers'/f'{qkey.replace(":","_")}.pt'
    label_free=dict(key=qkey,teachers=detached(teachers,'cpu'),views=viewinfo,native=nativepr,
                   input_sha256=image_hash(frames),GT_access=False)
    save(d,label_free);write(d.with_suffix('.json'),dict(path=str(d),sha256=sha(d),GT_access=False,key=qkey))
    del frames
    # Barrier above is sealed before oracle/support diagnostic labels are opened.
    gt=read(label_path)[qkey];valid=np.asarray(gt['valid'],bool);idx=np.flatnonzero(valid).tolist()
    truth=torch.as_tensor(gt['boxes'],device='cuda',dtype=torch.float32)
    anchors=x['expert']['anchors']['single4'];apos=[a['position'] for a in anchors]
    matched=[a for a in anchors if valid[a['position']]];mi=[a['position'] for a in matched]
    ext=lambda aa:torch.tensor([a['box'] for a in aa],device='cuda',dtype=torch.float32).reshape(-1,4)
    n=len(ids);times=torch.tensor(ids,device='cuda',dtype=torch.float32)/x['input']['fps']
    losses={name:BoxLoss(teachers[name],list(range(n)),n) for name in ['flip','crop','cycle']}
    losses['smooth']=lambda b:velocity_energy(b,times)
    losses['ssl_joint']=lambda b:sum(losses[k](b) for k in ['flip','crop','cycle'])/3+losses['smooth'](b)
    losses['gt_all']=BoxLoss(truth[idx],idx,len(idx))
    losses['gt_matched']=BoxLoss(truth[mi],mi,4)
    losses['external_matched']=BoxLoss(ext(matched),mi,4)
    losses['external_all']=BoxLoss(ext(anchors),apos,4)
    grads={};origin_losses={}
    for name,loss in losses.items():
        s.restore(initial);val=loss(s.values()['boxes']);origin_losses[name]=float(val.detach())
        grads[name]=gradient_vector(s,val)
        assert torch.isfinite(grads[name]).all(),name
    gradients={name:dict(norm=float(g.norm()),cos_gt_all=cosine(g,grads['gt_all']),
                 cos_gt_matched=cosine(g,grads['gt_matched'])) for name,g in grads.items()}
    # lr=0 executes backward and optimizer, not merely identity assignment.
    s.restore(initial);noop=torch.optim.AdamW([v for _,v in s.named],lr=0,eps=1e-4,weight_decay=0)
    noop.zero_grad();losses['flip'](s.values()['boxes']).backward();noop.step()
    assert all(torch.equal(v,initial[k]) for k,v in s.named)
    assert torch.equal(s.values()['boxes'],native['boxes'])
    del noop
    arms={};paths={};full_audits=0
    for name in p['methods']:
        s.restore(initial);opt=torch.optim.AdamW([v for _,v in s.named],lr=config(x['cohort']).spatial_lr,
               betas=(.9,.999),eps=1e-4,weight_decay=0)
        lossfn=losses[name];best=origin_losses[name];beststep=0;beststate=detached(initial);history=[]
        for step in range(1,6):
            opt.zero_grad();before=s.values();loss=lossfn(before['boxes']);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for _,v in s.named)
            opt.step();del before,loss
            with torch.no_grad():
                actual=detached(s.values());value=float(lossfn(actual['boxes']))
            assert torch.isfinite(actual['boxes']).all()
            assert all(torch.equal(a,b) for a,b in zip(actual['logits'],native['logits']))
            delta=flat_delta(s,initial)
            if step==1:
                gradients[name]['first_step_GT_all_descent']=float(-torch.dot(delta,grads['gt_all']))
                gradients[name]['first_step_GT_matched_descent']=float(-torch.dot(delta,grads['gt_matched']))
            if value<best:best=value;beststep=step;beststate=s.state()
            history.append(dict(step=step,loss=value,delta_norm=float(delta.norm()),best_step=beststep,
                          changed_parameters=int((delta!=0).sum())))
            if step in (1,5):
                rawstate=s.state()
                with query_subject(model,batch,x['parses']['subject']):
                    pr=full_prediction(model,batch,ids,records,rawstate,actual)
                assert pr['physical_interval']==nativepr['physical_interval'];full_audits+=1
                arms[f'{name}_step{step}']=dict(prediction=pr,state=detached(rawstate,'cpu'),state_step=step,
                      loss=value,delta_norm=float(delta.norm()))
                s.restore(beststate)
                with torch.no_grad():bv=s.values();bpr=prediction(bv['logits'],bv['boxes'],records,ids)
                arms[f'{name}_best{step}']=dict(prediction=bpr,state=detached(beststate,'cpu'),state_step=beststep,
                      loss=best,delta_norm=float(flat_delta(s,initial).norm()))
                s.restore(rawstate)
            del actual,delta
        paths[name]=history;del opt
    s.restore(initial)
    assert state_hash(s.state())==state_hash(initial)
    assert torch.equal(s.values()['boxes'],native['boxes'])
    assert not any(v.requires_grad or v.grad is not None for v in model.parameters())
    result=dict(key=qkey,cohort=x['cohort'],group=x['group'],source=x['input']['source'],input=x['input'],
         frame_ids=ids,native=nativepr,arms=arms,paths=paths,gradients=gradients,origin_losses=origin_losses,
         gradient_vectors=detached(grads,'cpu'),teacher_receipt=read(d.with_suffix('.json')),
         support=dict(gt_all=idx,accepted_external=apos,matched=mi,planned_external=4),
         initial=detached(initial,'cpu'),labels_diagnostic_only=True,GT_SSL_targets=False,
         directional_query_words=re.findall(r'\b(?:left|right|leftmost|rightmost|clockwise|counterclockwise)\b',x['input']['caption'].lower()),
         audit=dict(parameters=sum(v.numel() for _,v in s.named),parameter_names=[k for k,_ in s.named],
           native_exact=True,steps0_exact=True,lr0_exact=True,roundtrip_pixel_and_forward_exact=True,
           full_model_reinsertions=full_audits,temporal_logits_invariant=True,reset_exact=True),
         seconds=time.time()-started,peak_memory_bytes=torch.cuda.max_memory_allocated())
    return result


def run(limit=0):
    import torch,numpy as np
    from scripts.run_final_simplification_v1 import lease
    from methods.decota_final_simplified_v1._tastvg_load import load_model_on_device
    p=verify();gpu=lease();assert sha(p['labels'])==p['labels_sha256']
    torch.set_num_threads(4);torch.manual_seed(20260910);np.random.seed(20260910)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=None;current=None;done=0
    try:
        for row in p['rows']:
            f=OUT/'episodes'/f'{row["key"].replace(":","_")}.pt'
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
            status(OUT/'STATUS.json',dict(status='running',done=done,total=len(p['rows']),key=row['key'],pid=os.getpid(),updated=time.time()))
            z=episode(model,x,p['labels'],p);save(f,z);write(f.with_suffix('.json'),dict(key=row['key'],sha256=sha(f),
                 lock_sha256=sha(OUT/'LOCK.json'),seconds=z['seconds']))
            done+=1;print('COMPLETED',done,len(p['rows']),row['key'],round(z['seconds'],2),flush=True)
            del z,x;gc.collect();torch.cuda.empty_cache()
        verify();status(OUT/'STATUS.json',dict(status='inference_complete' if done==len(p['rows']) else 'bounded_smoke_complete',
                   done=done,total=len(p['rows']),updated=time.time()))
    except BaseException as e:
        status(OUT/'FAILURE.json',dict(type=type(e).__name__,message=str(e),done=done,updated=time.time()))
        raise
    finally:gpu.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args();prepare() if a.stage=='prepare' else run(a.limit)
