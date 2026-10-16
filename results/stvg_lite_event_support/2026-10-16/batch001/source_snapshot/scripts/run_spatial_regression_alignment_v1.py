"""Finite source collection and spatial fits; labels only in gradient diagnostics."""
import argparse,gc,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts.run_spatial_ssl_gpu_v1 import config,frozen_forward,flat_delta
from scripts.run_spatial_three_rounds_v1 import setup
OUT=ROOT/'artifacts/decota_spatial_regression_alignment_v1'
OLD=ROOT/'artifacts/decota_spatial_three_rounds_v1'
COHORTS=['hcstvg1_test','vidstg_test']
CODE=['scripts/run_spatial_regression_alignment_v1.py','vg_tta/spatial_regression_alignment_v1.py',
      'protocols/decota_spatial_regression_alignment_v1.md','tests/test_spatial_regression_alignment_v1.py']


def prepare():
    import torch
    torch.set_num_threads(4)
    from methods.decota_final_simplified_v1.observations import QuerySubjectParser
    if (OUT/'LOCK.json').exists():return verify()
    p=read(OLD/'LOCK.json');pins=dict(p['pins']);sources={};old={}
    parser=QuerySubjectParser(ROOT/'.cache/stanza')
    for name in CODE+['scripts/run_spatial_three_rounds_v1.py','vg_tta/spatial_three_rounds_v1.py']:
        pins[name]=sha(ROOT/name)
    for c in COHORTS:
        direction='vid_to_hc1' if c==COHORTS[0] else 'hc2_to_vid'
        path=ROOT/'artifacts/decota_paper_execution_20260917/vitta'/direction/'SOURCE_MANIFEST.json'
        m=read(path);assert sha(m['annotation'])==m['annotation_sha256'];a=read(m['annotation'])
        assert m['official_split']=='train' and m['source_count']==64
        target_sources={r['source'] for r in p['rows'] if r['cohort']==c}
        rr=[]
        for row in m['records']:
            q=row['input'];assert q['source'] not in target_sources
            ann=a[row['official_annotation_key']]
            text=ann['sentence']['description'] if m['source_dataset']=='vidstg' else ann['English']
            assert text.lower()==q['caption'].lower()
            assert sha(q['video_path'])==q['video_sha256']
            rr.append({**row,'parses':parser(q['caption'])})
        assert len(rr)==len({v['input']['source'] for v in rr})==64
        sources[c]=dict(records=rr,manifest=str(path),manifest_sha256=sha(path),
                        annotation=m['annotation'],annotation_sha256=m['annotation_sha256'],
                        official_split='train',source_dataset=m['source_dataset'],
                        overlap_target=0,GT_statistics=False)
    for row in p['rows']:
        assert sha(row['path'])==row['sha256']
        stem=row['key'].replace(':','_');f=OLD/'p0'/f'{stem}.pt';t=OLD/'teachers'/f'{stem}.pt'
        assert sha(f)==read(f.with_suffix('.json'))['sha256']
        assert sha(t)==read(t.with_suffix('.json'))['sha256']
        old[row['key']]=dict(episode=str(f),episode_sha256=sha(f),teacher=str(t),teacher_sha256=sha(t))
    write(OUT/'LOCK.json',dict(rows=p['rows'],sources=sources,reused=old,labels=p['labels'],labels_sha256=p['labels_sha256'],
          parent_lock_sha256=sha(OLD/'LOCK.json'),pins=pins,historical_dev=True,parameters=1792,
          lr={c:config(c).spatial_lr for c in COHORTS},optimizer=dict(name='AdamW',eps=1e-4,betas=[.9,.999],weight_decay=0),
          steps=[1,5],source_count=128,target_count=64,created=time.time(),auto_promote=False))
    print('PREPARED128 source+64 exposed dev',flush=True)


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    assert sha(OLD/'LOCK.json')==p['parent_lock_sha256']
    for m in p['sources'].values():assert sha(m['manifest'])==m['manifest_sha256']
    return p


def model_load(c):
    from methods.decota_final_simplified_v1._tastvg_load import load_model_on_device
    cfg=config(c);assert sha(ROOT/cfg.checkpoint)==cfg.checkpoint_sha256
    runtime=ROOT/'artifacts/tastvg_runtime'/('vidstg' if cfg.source_dataset=='vidstg' else 'hc-stvg2')
    m,_,_=load_model_on_device(cfg.source_dataset,runtime,checkpoint=ROOT/cfg.checkpoint,device='cuda',source_dataset=cfg.source_dataset)
    return m.eval().requires_grad_(False)


def commit(f,x):
    if f.exists():raise RuntimeError('Unreceipted output preserved: '+str(f))
    save(f,x);write(f.with_suffix('.json'),dict(path=str(f),sha256=sha(f),lock_sha256=sha(OUT/'LOCK.json'),completed=time.time()))


def exists(f):
    if not f.with_suffix('.json').exists():return False
    r=read(f.with_suffix('.json'))
    assert sha(f)==r['sha256'] and r['lock_sha256']==sha(OUT/'LOCK.json')
    return True


def source(model,row,c):
    import torch
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.spatial_regression_alignment_v1 import features,sensitivity
    q=row['input'];frames,ids=decode(q);x=dict(input=q,frame_ids=ids,parses=row['parses'])
    batch,records,s=frozen_forward(model,frames,x)
    with torch.no_grad():v,z=features(s)
    assert torch.equal(v['boxes'],s.zero['boxes'])
    r=sensitivity(s.decoder.decoder.bbox_embed,z)
    assert not any(p.grad is not None for p in model.parameters())
    return dict(key=row['key'],cohort=c,input=q,frame_ids=ids,z=z.cpu(),sensitivity=r.cpu(),
                boxes=v['boxes'].cpu(),parameters=1792,GT_access=False,readout_exact=True)


def seal_references(p):
    from vg_tta.spatial_regression_alignment_v1 import reference
    for c,m in p['sources'].items():
        f=OUT/'references'/f'{c}.pt'
        if exists(f):continue
        rr=[];receipts=[]
        for row in m['records']:
            q=OUT/'source'/c/f'{row["ordinal"]:03d}.pt';assert exists(q)
            r=load(q);assert not r['GT_access'];rr.append(r);receipts.append(read(q.with_suffix('.json')))
        z=reference(rr);z.update(cohort=c,source_receipts=receipts,GT_access=False,source_manifest_sha256=m['manifest_sha256'])
        commit(f,z)
    write(OUT/'REFERENCE_BARRIER.json',dict(status='sealed',sources=128,target_adaptation_started=False,
          references={c:read((OUT/'references'/f'{c}.pt').with_suffix('.json')) for c in COHORTS},created=time.time()))


def episode(model,x,p,fd=False):
    import numpy as np,torch
    from vg_tta.spatial_regression_alignment_v1 import ARMS,AlignmentLoss,features,finite_difference
    from vg_tta.spatial_ssl_diagnostic_v1 import BoxLoss,gradient_vector,cosine
    from vg_tta.spatial_three_rounds_v1 import WeightedROI
    from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    start=time.time();frames,ids,batch,records,s,native,initial,ts,tl,base=setup(model,x);del frames
    refpath=OUT/'references'/f'{x["cohort"]}.pt';assert exists(refpath);ref=load(refpath)
    losses={a:AlignmentLoss(ref,a,'cuda') for a in ARMS};rr=p['reused'][x['key']]
    assert sha(rr['teacher'])==rr['teacher_sha256'] and sha(rr['episode'])==rr['episode_sha256']
    teacher=load(rr['teacher']);assert not teacher['GT_access'];old=load(rr['episode'])
    assert torch.equal(base['boxes'],old['baseline']['boxes']) and base['indices']==old['baseline']['indices']
    assert all(torch.equal(a,b) for a,b in zip(base['logits'],old['baseline']['logits']))
    roi=WeightedROI(teacher['target'].cuda(),teacher['weights']['A_uniform'],True)
    gradients={};origins={};finite={}
    for a,fn in losses.items():
        s.restore(initial);_,z=features(s);loss=fn(z);origins[a]=float(loss.detach())
        gradients[a]=gradient_vector(s,loss).cpu();del z,loss
    s.restore(initial);loss=roi(s.values()['boxes']);gradients['roi']=gradient_vector(s,loss).cpu()
    origins['roi']=float(loss.detach());del loss
    barrier=OUT/'target_barriers'/f'{x["key"].replace(":","_")}.json'
    write(barrier,dict(key=x['key'],source_reference_sha256=sha(refpath),roi_teacher_sha256=rr['teacher_sha256'],
          GT_access=False,losses_constructed=True,initial_legal_gradients_computed=True,created=time.time()))
    # All unsupervised targets and source-only subspace choices are frozen above.
    gt=read(p['labels'])[x['key']];idx=np.flatnonzero(np.asarray(gt['valid'],bool)).tolist()
    truth=torch.tensor(gt['boxes'],device='cuda',dtype=torch.float32)
    gtfn=BoxLoss(truth[idx],idx,len(idx));s.restore(initial)
    gradients['gt_all']=gradient_vector(s,gtfn(s.values()['boxes'])).cpu()
    for g in gradients.values():assert g.numel()==1792 and torch.isfinite(g).all()
    cosines={k:cosine(v,gradients['gt_all']) for k,v in gradients.items()}
    if fd:
        finite=finite_difference(s,losses['all'],gradients['all'].cuda())
        assert finite['passed'],finite
    s.restore(initial);opt=torch.optim.AdamW([v for _,v in s.named],lr=0,eps=1e-4,weight_decay=0)
    opt.zero_grad();_,z=features(s);losses['significant64'](z).backward();opt.step();del opt,z
    assert all(torch.equal(v,initial[k]) for k,v in s.named)
    assert torch.equal(s.values()['boxes'],native['boxes'])
    arms={};paths={};first_dot={}
    for a,fn in losses.items():
        s.restore(initial);opt=torch.optim.AdamW([v for _,v in s.named],lr=config(x['cohort']).spatial_lr,
                      betas=(.9,.999),eps=1e-4,weight_decay=0);history=[]
        for step in range(1,6):
            opt.zero_grad();_,z=features(s);loss=fn(z);loss.backward()
            gv=torch.cat([v.grad.detach().flatten().double() for _,v in s.named]);assert torch.isfinite(gv).all()
            if step==1:assert torch.allclose(gv.cpu(),gradients[a],rtol=1e-5,atol=1e-8)
            opt.step();del z,loss
            with torch.no_grad():actual,z=features(s);value=float(fn(z));actual=detached(actual)
            delta=flat_delta(s,initial)
            if step==1:first_dot[a]=float(torch.dot(delta.cpu(),gradients['gt_all']))
            assert torch.isfinite(z).all() and torch.isfinite(delta).all()
            history.append(dict(step=step,loss=value,grad_norm=float(gv.norm()),delta_norm=float(delta.norm()),
                           changed=int((delta!=0).sum())))
            if step in (1,5):
                ss=s.state()
                with query_subject(model,batch,x['parses']['subject']):
                    pred=full_prediction(model,batch,ids,records,{**ss,**ts},dict(boxes=actual['boxes'],logits=tl))
                assert pred['indices']==base['indices']
                arms[f'{a}_step{step}']=dict(prediction=pred,state=detached(ss,'cpu'),loss=value,
                                delta_norm=float(delta.norm()),changed=int((delta!=0).sum()))
            del z,actual,delta,gv
        paths[a]=history;del opt
    for step in (1,5):
        arms[f'roi_step{step}']=old['arms'][f'A_uniform_step{step}']
    rd=torch.cat([(arms['roi_step1']['state'][k]-initial[k].cpu()).reshape(-1).double() for k,_ in s.named])
    first_dot['roi']=float(torch.dot(rd,gradients['gt_all']))
    paths['roi']=old['paths']['A_uniform'];s.restore(initial)
    assert state_hash(s.state())==state_hash(initial)
    assert torch.equal(s.values()['boxes'],native['boxes'])
    assert not any(v.requires_grad or v.grad is not None for v in model.parameters())
    return dict(key=x['key'],cohort=x['cohort'],source=x['input']['source'],input=x['input'],group=x['group'],
         baseline=base,frame_ids=ids,arms=arms,initial=detached(initial,'cpu'),temporal_state=ts,
         gradients=gradients,cos_gt=cosines,adam_first_dot_gt=first_dot,origin_losses=origins,paths=paths,
         finite_difference=finite,reference_sha256=sha(refpath),old=rr,seconds=time.time()-start,
         peak_memory_bytes=torch.cuda.max_memory_allocated(),audit=dict(native_exact=True,readout_exact=True,
         lr0_exact=True,steps0_exact=True,reset_exact=True,temporal_invariant=True,full_reinsertions=7,
         source_only_reference=True,GT_used_for_updates=False,parameters=1792))


def run(stage,limit=0):
    import numpy as np,torch
    from scripts.run_final_simplification_v1 import lease
    p=verify();assert sha(p['labels'])==p['labels_sha256']
    if stage=='target':
        assert read(OUT/'REFERENCE_BARRIER.json')['status']=='sealed'
        for c in COHORTS:assert exists(OUT/'references'/f'{c}.pt')
    guard=lease();m=None;current=None;done=0;total=128 if stage=='source' else 64
    torch.set_num_threads(4);torch.manual_seed(20260910);np.random.seed(20260910)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    jobs=[(c,r) for c,v in p['sources'].items() for r in v['records']] if stage=='source' else [(r['cohort'],r) for r in p['rows']]
    try:
        for c,row in jobs:
            f=(OUT/'source'/c/f'{row["ordinal"]:03d}.pt' if stage=='source' else OUT/'target'/f'{row["key"].replace(":","_")}.pt')
            if exists(f):done+=1;continue
            if limit and done>=limit:break
            if c!=current:
                if m is not None:del m;gc.collect();torch.cuda.empty_cache()
                m=model_load(c);current=c
            torch.cuda.reset_peak_memory_stats();started=time.time()
            status(OUT/f'{stage}_STATUS.json',dict(status='running',done=done,total=total,key=row['key'],pid=os.getpid(),updated=time.time()))
            if stage=='source':z=source(m,row,c)
            else:
                assert sha(row['path'])==row['sha256']
                z=episode(m,load(row['path']),p,fd=done in (0,32))
            z['seconds']=time.time()-started;commit(f,z);done+=1
            print('COMPLETED',stage,done,total,row['key'],round(z['seconds'],2),flush=True)
            del z;gc.collect();torch.cuda.empty_cache()
        if stage=='source' and done==128:seal_references(p)
        verify();status(OUT/f'{stage}_STATUS.json',dict(status='complete' if done==total else 'smoke_complete',done=done,total=total,updated=time.time()))
    except BaseException as e:
        status(OUT/f'{stage}_FAILURE.json',dict(type=type(e).__name__,message=str(e),done=done,time=time.time()));raise
    finally:guard.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','source','target']);ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args();prepare() if a.stage=='prepare' else run(a.stage,a.limit)
