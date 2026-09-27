"""Bounded route reopening. Prediction workers never open target labels."""
import argparse, collections, copy, gc, hashlib, io, itertools, math, re, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts.c1_controlled_corruption_v1 import configuration,corrupt as old_corrupt,pixelhash,digest,CONDITIONS as OLD_CONDITIONS
OUT=ROOT/'artifacts/corruption_route_retest_v1'
OLD=ROOT/'artifacts/c1_controlled_corruption_v1'
CONCEPT=ROOT/'artifacts/c1_concept_local_v1'
STATS=ROOT/'artifacts/decota_spatial_regression_alignment_v1'
NEW_CONDITIONS=['noise_strong','noise_extreme','defocus_strong','defocus_extreme','jpeg_strong','jpeg_extreme']
DOSES=OLD_CONDITIONS+NEW_CONDITIONS
SEEDS=[20260923,20260924,20260925]
QUERY='spatial.query_residual'

def corrupt(frames,ids,source,condition):
    if condition in OLD_CONDITIONS:return old_corrupt(frames,ids,source,condition)
    import cv2
    from PIL import Image
    result=np.empty_like(frames);extreme=condition.endswith('extreme');kernel=None
    if condition.startswith('defocus'):
        radius=(10 if extreme else 6)*min(frames.shape[1:3])/224
        extent=math.ceil(radius);g=np.arange(-extent,extent+1);yy,xx=np.meshgrid(g,g,indexing='ij')
        kernel=(xx*xx+yy*yy<=radius*radius).astype(np.float32);kernel/=kernel.sum()
        kernel=cv2.GaussianBlur(kernel,(3,3),.1);kernel/=kernel.sum()
    for i,(frame,fid) in enumerate(zip(frames,ids)):
        if condition.startswith('noise'):
            rng=np.random.default_rng(int(digest(f'route-retest|{source}|{condition}|{fid}')[:16],16))
            result[i]=np.clip(np.rint(frame.astype(np.float32)+rng.normal(0,(.32 if extreme else .16)*255,frame.shape)),0,255).astype(np.uint8)
        elif condition.startswith('defocus'):result[i]=cv2.filter2D(frame,-1,kernel,borderType=cv2.BORDER_REFLECT_101)
        elif condition.startswith('jpeg'):
            b=io.BytesIO();Image.fromarray(frame).save(b,format='JPEG',quality=3 if extreme else 10,subsampling=2,optimize=False,progressive=False);b.seek(0);result[i]=np.asarray(Image.open(b).convert('RGB'))
        else:raise ValueError(condition)
    return result

def prepare():
    import torch
    from vg_tta.spatial_regression_alignment_v1 import reference
    parent=read(ROOT/'artifacts/spatial_consolidation_roles_v1/LOCK.json')
    cp=read(CONCEPT/'LOCK.json');texts=read(CONCEPT/'TEXT_LOCK.json');rows=[]
    for r in cp['rows']:
        if r['cohort']!='vidstg_test':continue
        x=load(r['original']);assert sha(r['original'])==r['original_sha']
        rows.append(dict(key=r['key'],source=r['source'],cohort='vidstg_test',path=r['original'],sha256=r['original_sha'],split=r['split'],input=x['input'],frame_ids=x['frame_ids'],parses=x['parses'],text=texts['rows'][r['key']]))
    used={r['source'] for r in rows}
    dose=sorted([r for r in read(OLD/'INPUTS.json') if r['source'] not in used],key=lambda r:digest('route-dose-'+r['source']))[:8]
    used.update(r['source'] for r in dose)
    pool=sorted([r for r in parent['rows'].values() if r['cohort']=='vidstg_test' and r['source'] not in used],key=lambda r:digest('route-long-'+r['source']))[:128]
    long=[]
    for r in pool:
        x=load(r['path']);long.append(dict(**r,input=x['input'],frame_ids=x['frame_ids'],parses=x['parses']))
    for rr in [rows,dose,long]:
        assert len(rr)==len({r['source'] for r in rr})
        for r in rr:assert sha(r['input']['video_path'])==r['input']['video_sha256']
    sp=read(STATS/'LOCK.json');sourceaudit={};refs={}
    for model,c in [('vid_source','hcstvg1_test'),('hc2_source','vidstg_test')]:
        f=STATS/'references'/f'{c}.pt';assert sha(f)==read(f.with_suffix('.json'))['sha256'];ref=load(f);rr=[]
        m=sp['sources'][c];assert m['official_split']=='train' and sha(m['manifest'])==m['manifest_sha256']
        for z in ref['source_receipts']:
            assert sha(z['path'])==z['sha256'];rr.append(load(z['path']))
        rebuilt=reference(rr)
        assert all(torch.equal(rebuilt[n],ref[n]) for n in ['mu','sigma','sensitivity'])
        assert all(torch.equal(rebuilt['indices'][n],ref['indices'][n]) for n in rebuilt['indices'])
        sourceaudit[model]=dict(manifest=m['manifest'],manifest_sha=m['manifest_sha256'],official_split='train',entries=len(rr),unique_media_sha=len({r['input']['video_sha256'] for r in rr}),source_count_claim='64 training entries; media dedup reported separately',reference_recomputed_exact=True,target_clean_features_used=False,checkpoint=configuration(model).checkpoint,checkpoint_sha=configuration(model).checkpoint_sha256)
        refs[model]=dict(path=str(f),sha256=sha(f),base_lr=.005 if model=='vid_source' else .05)
    write(OUT/'INPUTS.json',dict(dose=dose,local=rows,long=long));write(OUT/'SOURCE_AUDIT.json',sourceaudit)
    pins={str(ROOT/f):sha(ROOT/f) for f in ['scripts/corruption_route_retest_v1.py','protocols/corruption_route_retest_v1.md','methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json','vg_tta/c1_concept_local_v1.py','vg_tta/spatial_ssl_diagnostic_v1.py','vg_tta/spatial_regression_alignment_v1.py','scripts/c1_controlled_corruption_v1.py']}
    for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py'):pins[str(f)]=sha(f)
    write(OUT/'LOCK.json',dict(created=time.time(),pins=pins,inputs_sha=sha(OUT/'INPUTS.json'),source_audit_sha=sha(OUT/'SOURCE_AUDIT.json'),refs=refs,labels=parent['labels'],labels_sha=parent['labels_sha256'],dose_conditions=DOSES,lr_multipliers=[.25,1.,4.],steps=[1,3,5],radii=[.05,.1,.25],query_multipliers=[1/3,1.,3.],seeds=SEEDS,old_manifest_sha=sha(OLD/'RESULT_MANIFEST.json'),reserved_sha=sha(ROOT/'artifacts/c1_fresh_confirmation_v1/vid_train_audit/PROSPECTIVE_SOURCE_RESERVATION.json'),max_seconds=21600,max_bytes=40*2**30,historical_development=True,GT_adaptation=False,GT_development_selection=True))
    status(OUT/'STATUS.json',dict(status='prepared',dose_sources=8,local_sources=32,long_reserved=128))

def verify():
    p=read(OUT/'LOCK.json')
    pins=dict(p['pins'])
    amendment=OUT/'PRELOCAL_IMPLEMENTATION_AMENDMENT.json'
    if amendment.exists():
        a=read(amendment);assert a['original_lock_sha']==sha(OUT/'LOCK.json');pins.update(a['pins'])
    transfer_amendment=OUT/'PRETRANSFER_ORDER_AMENDMENT.json'
    if transfer_amendment.exists():
        a=read(transfer_amendment);assert a['original_lock_sha']==sha(OUT/'LOCK.json');pins.update(a['pins'])
    for f,h in pins.items():assert sha(f)==h,f
    assert sha(OUT/'INPUTS.json')==p['inputs_sha']
    for r in p['refs'].values():assert sha(r['path'])==r['sha256']
    return p

def commit(f,x):
    from methods.decota_final_simplified_v1.tensors import detached
    save(f,detached(x,'cpu'));write(f.with_suffix('.json'),dict(sha256=sha(f),lock_sha=sha(OUT/'LOCK.json')))

def exists(f):
    if not f.with_suffix('.json').exists():return False
    assert sha(f)==read(f.with_suffix('.json'))['sha256'];return True

def seal(stage,start):
    ff={str(f):sha(f) for f in sorted((OUT/stage).rglob('*.pt'))}
    write(OUT/stage/'BARRIER.json',dict(files=ff,seconds=time.time()-start,GT_adaptation=False))

def launch():
    from scripts.spatial_consolidation_roles_v1 import setup
    from scripts.run_final_simplification_v1 import lease
    setup();return lease()

def dose():
    import torch
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.objectives import prediction
    p=verify();guard=launch();start=time.time();model=model_load('hcstvg1_test');expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT)
    try:
        for i,r in enumerate(read(OUT/'INPUTS.json')['dose']):
            frames,ids=decode(r['input'])
            for c in DOSES:
                f=OUT/'dose'/c/f'{i:03d}.pt'
                if exists(f):continue
                shifted=corrupt(frames,ids,r['source'],c);captured={};hooks=[]
                def capture(name,t):
                    positions=[0,len(t)//2,len(t)-1]
                    captured.setdefault(name,[]).append(dict(shape=list(t.shape),sha=pixelhash(t.detach().cpu().numpy()),positions=positions,values=t[positions].detach().cpu()))
                hooks.append(model.vis_encoder.register_forward_pre_hook(lambda m,a:capture('appearance',a[0].tensors)))
                hooks.append(model.vid.register_forward_pre_hook(lambda m,a:capture('motion',a[0])))
                try:batch,records,s=frozen_forward(model,shifted,r)
                finally:
                    for h in hooks:h.remove()
                assert all(a['sha']==b['sha'] for a,b in zip(captured['appearance'],captured['motion']))
                hook=expert.model.register_forward_pre_hook(lambda m,a,k: captured.setdefault('expert_first',dict(values=k['pixel_values'].detach().cpu(),shape=list(k['pixel_values'].shape))) and None,with_kwargs=True)
                try:ex=observations(expert,r['parses'],shifted,ids,prediction(s.zero['logits'],s.zero['boxes'],records,ids)['indices'],audit=True)
                finally:hook.remove()
                positions=[int((len(ids)-1)*v) for v in [.25,.5,.75]]
                pixels=dict(**captured,student=batch['videos'].tensors[positions].detach().cpu(),student_positions=positions,original_rgb=shifted[positions],frame_ids=[ids[j] for j in positions],query=r['input']['caption'],student_full_shape=list(batch['videos'].tensors.shape))
                commit(OUT/'dose_inputs'/c/f'{i:03d}.pt',pixels)
                commit(f,dict(key=r['key'],source=r['source'],condition=c,input=r['input'],frame_ids=ids,native=prediction(s.zero['logits'],s.zero['boxes'],records,ids),expert=ex,pixel_sha=pixelhash(shifted),GT_used=False))
                status(OUT/'STATUS.json',dict(status='dose_running',source=i,condition=c,seconds=time.time()-start));print('DOSE',i,c,round(time.time()-start,1),flush=True)
                del shifted,batch,records,s,ex,pixels,captured;gc.collect();torch.cuda.empty_cache()
        seal('dose',start)
    finally:guard.close()

def internal_fit(s,initial,fn,lr,steps=5,kind='stat',skip=False):
    import torch
    from vg_tta.spatial_regression_alignment_v1 import features
    from methods.decota_final_simplified_v1.tensors import detached
    s.restore(initial)
    if skip:
        with torch.no_grad():v=s.values();lv=float(fn(v['boxes']))
        return dict(path=[dict(step=i,loss=lv,boxes=detached(v['boxes'],'cpu'),state=detached(initial,'cpu')) for i in range(steps+1)],backwards=0,lr=lr,optimizer='AdamW_eps1e-4',initial_state=detached(initial,'cpu'),skipped='unsafe_flip_semantics')
    opt=torch.optim.AdamW([v for _,v in s.named],lr=lr,eps=1e-4,weight_decay=0);path=[]
    for step in range(steps+1):
        opt.zero_grad(set_to_none=True)
        if kind=='stat':v,z=features(s);loss=fn(z)
        else:v=s.values();loss=fn(v['boxes'])
        assert torch.isfinite(loss)
        path.append(dict(step=step,loss=float(loss.detach()),boxes=detached(v['boxes'],'cpu'),state=detached(s.state(),'cpu')))
        if step==steps:break
        loss.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for _,t in s.named)
        path[-1]['grad_norm']=float(torch.cat([t.grad.flatten() for _,t in s.named]).norm());opt.step()
    s.restore(initial);return dict(path=path,backwards=steps,lr=lr,optimizer='AdamW_eps1e-4',initial_state=detached(initial,'cpu'))

def concept_features(model,batch,s,r,before):
    import torch
    from vg_tta.c1_concept_local_v1 import expression,basis
    tr=r['text'];q=r['input']['caption'];subject=r['parses']['subject']
    original=expression(model,batch,s.initial,q,subject);assert torch.equal(original['boxes'],before['boxes'].cpu())
    groups={g:[expression(model,batch,s.initial,t,subject if g=='basis' else tr['donor_subject']) for t in tr[g]] for g in ['basis','far_basis']}
    us,ss=basis([v['z'] for v in groups['basis']],original['z']);un,sn=basis([v['z'] for v in groups['far_basis']],original['z']);rank=min(us.shape[1],un.shape[1]);random={}
    for seed in SEEDS:
        gen=torch.Generator().manual_seed(seed+int(digest(r['key'])[:8],16));random[str(seed)]=torch.linalg.qr(torch.randn(256,max(1,rank),generator=gen,dtype=torch.float64))[0][:,:rank].float()
    return dict(original=original,groups=groups,S=us[:,:rank],P=random,rank=rank,rankS=us.shape[1],rankN=un.shape[1],singular_S=ss,singular_N=sn,scale=float(original['z'].norm()),GT_used=False,basis_full_forwards=2*(1+sum(len(v) for v in groups.values())))

def local(stage,limit=0):
    import torch
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.spatial_regression_alignment_v1 import AlignmentLoss
    from vg_tta.spatial_ssl_diagnostic_v1 import BoxLoss,inverse_boxes
    from vg_tta.spatial_online_state_v1 import arrival
    from vg_tta.spatial_consolidation_v1 import consolidate
    from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.optim import fit_spatial,fit_temporal
    from methods.decota_final_simplified_v1.replay import TemporalReplay
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
    from methods.decota_refine_uniform_v1.api import reconstruct
    from vg_tta.c1_concept_local_v1 import fit as concept_fit
    p=verify();conditions=read(OUT/'CONDITIONS.json')['conditions'];selection=None if stage=='dev' else read(OUT/'SELECTION.json')
    modelname='hc2_source' if stage=='transfer' else 'vid_source';cohort='vidstg_test' if modelname=='hc2_source' else 'hcstvg1_test';cfg=configuration(modelname)
    rows=[r for r in read(OUT/'INPUTS.json')['local'] if r['split']==('dev' if stage=='dev' else 'validation')]
    order_stage='validation' if stage=='transfer' else stage
    rows=sorted(rows,key=lambda r:digest('route-'+order_stage+'-'+r['source']))
    guard=launch();start=time.time();model=model_load(cohort);mh=state_hash(model.state_dict());expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);ref=load(p['refs'][modelname]['path']);done=0
    try:
        for c in conditions:
            previous=None;previous_sha=None
            for i,r in enumerate(rows):
                f=OUT/stage/c/f'{i:03d}.pt'
                if exists(f):x=load(f);previous=x['committed'];previous_sha=sha(f);done+=1;continue
                if limit and done>=limit:return
                assert time.time()-start<p['max_seconds']
                frames,ids=decode(r['input']);shifted=corrupt(frames,ids,r['source'],c);del frames
                batch,records,s=frozen_forward(model,shifted,r);source=detached(s.initial);assert sum(v.numel() for _,v in s.named)==1792
                native=prediction(s.zero['logits'],s.zero['boxes'],records,ids);ex=observations(expert,r['parses'],shifted,ids,native['indices'],audit=True);anchors=ex['anchors']['single4']
                ep=fit_spatial(s,s.zero,anchors,cfg,trace=True);tem=TemporalReplay(s.head,s.temporal_inputs,{**s.zero,'boxes':ep['final']['boxes']});tt=fit_temporal(tem,records,cfg,trace=True);del tem
                indices=prediction(tt['shrunk']['logits'],s.zero['boxes'],records,ids)['indices'];direct,da=reconstruct(s.zero['boxes'],anchors,ids,'absolute')
                directional=re.findall(r'\b(?:left|right|leftmost|rightmost|clockwise|counterclockwise|anticlockwise|leftward|rightward)\b',r['input']['caption'].lower())
                if not directional:
                    vb,vr,vs=frozen_forward(model,shifted[:,:,::-1].copy(),r);teacher=inverse_boxes(vs.zero['boxes'],flip=True).detach();del vb,vr,vs
                else:teacher=s.zero['boxes'].detach().clone()
                fusion=(s.zero['boxes']+teacher)/2;internal={};cost=collections.Counter(DINO=ex['new_DINO'],spatial_backwards=ep['backwards'],temporal_backwards=tt['backwards'],basis_forwards=0,view_full_forwards=0 if directional else 2,full_reinsertions=0)
                for family in ['Sig64','Random64','Flip']:
                    fn=AlignmentLoss(ref,'significant64' if family=='Sig64' else 'random64','cuda') if family!='Flip' else BoxLoss(teacher,list(range(len(ids))),len(ids))
                    mults=p['lr_multipliers'] if stage=='dev' else sorted({1.,selection['families'][family]['multiplier']})
                    for mult in mults:
                        z=internal_fit(s,source,fn,p['refs'][modelname]['base_lr']*mult,kind='flip' if family=='Flip' else 'stat',skip=family=='Flip' and bool(directional));internal[f'{family}|m{mult}']=z;cost['spatial_backwards']+=z['backwards']
                # Equal-view and equal-replay-forward no-gradient control for Flip.
                control_start=time.perf_counter();control_forwards=0
                if not directional:
                    s.restore(source)
                    for mult in (p['lr_multipliers'] if stage=='dev' else sorted({1.,selection['families']['Flip']['multiplier']})):
                        for _ in range(6):
                            with torch.no_grad():control=s.values()['boxes']
                            assert torch.equal(control,s.zero['boxes']);control_forwards+=1
                cost['fusion_control_replay_forwards']=control_forwards
                fusion_control_seconds=time.perf_counter()-control_start
                initial=arrival(source,previous,'O-split');s.initial=detached(initial,'cuda');s.restore(s.initial)
                with torch.no_grad():before=detached(s.values())
                F=fit_spatial(s,before,anchors,cfg,trace=True);assert F['failure'] is None and ep['failure'] is None and tt['failure'] is None
                feat=concept_features(model,batch,s,r,before);cost['basis_forwards']=feat['basis_full_forwards'];fits={'F':F};cost['spatial_backwards']+=F['backwards']
                for family in ['R','S','P']:
                    pairs=itertools.product(p['radii'],p['query_multipliers']) if stage=='dev' else [(selection['families'][family]['rho'],selection['families'][family]['multiplier'])]
                    for rho,mult in pairs:
                        for seed in (SEEDS if family=='P' else [None]):
                            U=None if family=='R' else (feat['S'] if family=='S' else feat['P'][str(seed)]).cuda()
                            z=concept_fit(s,before,anchors,cfg,U,rho,mult,feat['scale']);key=f'{family}|r{rho}|m{mult}'+('' if seed is None else f'|s{seed}');fits[key]=z;cost['spatial_backwards']+=z['backwards']
                # Full model insertion verifies one statistical and one constrained state per arrival.
                for z in [internal['Sig64|m1.0']['path'][1],next(v for k,v in fits.items() if k.startswith('S|'))]:
                    st=z['state'];boxes=z['boxes'] if 'boxes' in z else z['final']['boxes']
                    with query_subject(model,batch,r['parses']['subject']):full_prediction(model,batch,ids,records,{**detached(st,'cuda'),**detached(tt['shrunk_state'],'cuda')},dict(boxes=detached(boxes,'cuda'),logits=detached(tt['shrunk']['logits'],'cuda')))
                    cost['full_reinsertions']+=1
                committed=consolidate(initial,F['state'],1/16);assert state_hash(model.state_dict())==mh
                payload=dict(key=r['key'],source=r['source'],model=modelname,stage=stage,condition=c,input=r['input'],frame_ids=ids,pixel_sha=pixelhash(shifted),native=native,formal_indices=indices,expert=ex,anchors=anchors,observed=ex['actual_observation_positions'],episodic=ep,temporal=tt,direct=direct,direct_audit=da,source_state=source,initial=initial,before=before['boxes'],fits=fits,features=feat,internal=internal,flip_teacher=teacher,fusion=fusion,fusion_control_seconds=fusion_control_seconds,directional=directional,committed=committed,previous_sha=previous_sha,cost=dict(cost),GT_used=False,seconds=time.time()-start)
                commit(f,payload);previous=detached(committed,'cpu');previous_sha=sha(f);done+=1
                status(OUT/'STATUS.json',dict(status=stage+'_running',condition=c,ordinal=i,done=done,key=r['key'],seconds=time.time()-start));print('LOCAL',stage,c,i,r['key'],'rank',feat['rank'],'back',cost['spatial_backwards'],round(time.time()-start,1),flush=True)
                del shifted,batch,records,s,source,ex,ep,tt,before,F,feat,fits,internal,teacher,fusion,payload;gc.collect();torch.cuda.empty_cache()
        seal(stage,start);status(OUT/'STATUS.json',dict(status=stage+'_complete_unscored',done=done,seconds=time.time()-start))
    except BaseException as e:
        write(OUT/f'FAILURE_{time.time_ns()}.json',dict(stage=stage,error=repr(e),done=done));raise
    finally:guard.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','dose','dev','validation','transfer']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    prepare() if a.stage=='prepare' else dose() if a.stage=='dose' else local(a.stage,a.limit)
