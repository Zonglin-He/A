"""Finite F34 observation and adaptation workers. NO evaluation labels."""
import argparse
import collections
import copy
import fcntl
import gc
import hashlib
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,save,sha,status
from scripts.run_closure_v1 import model_for
OUT=ROOT/'artifacts/decota_parametric_observation_v1'
OLD=ROOT/'artifacts/decota_five_round_closure_v1'
COHORTS=('hcstvg1_test','vidstg_test')


def rank(s):return hashlib.sha256(('param-observation-v1/'+s).encode()).hexdigest()
def path(stage,key):return OUT/stage/(key.replace(':','_')+'.pt')
def existing(p):
    if not p.exists():return False
    assert sha(p)==read(p.with_suffix('.json'))['sha256'];return True
def record(p,x):
    save(p,x);write(p.with_suffix('.json'),dict(sha256=sha(p),key=x['key'],stage=p.parent.name,completed=time.time()))


def prepare():
    old=read(OLD/'LOCK.json');guards={'hcstvg1_test:000116','vidstg_test:000621','vidstg_test:005972'}
    watched_groups={r['group'] for rr in old['rows'].values() for r in rr if r['key'] in guards}
    roles={g:'watched' for g in watched_groups};rows={};counts={}
    for c,rr in old['rows'].items():
        grouped=collections.defaultdict(list)
        for r in rr:
            if not r['input_unavailable']:grouped[r['group']].append(r)
        remaining=sorted([g for g in grouped if g not in roles],key=rank)
        assert len(remaining)>=8
        for i,g in enumerate(remaining[:40]):roles[g]='development' if i<8 else 'expansion'
        use=[]
        for g,rs in grouped.items():
            if g not in roles:continue
            if roles[g]=='watched':chosen=[r for r in rs if r['key'] in guards]
            else:chosen=[min(rs,key=lambda r:rank(r['key']))]
            for r in chosen:use.append({**r,'role':roles[g]})
        rows[c]=sorted(use,key=lambda r:(r['role'],rank(r['key'])))
        counts[c]={s:dict(queries=sum(r['role']==s for r in use),sources=len({r['group'] for r in use if r['role']==s})) for s in ('development','expansion','watched')}
        dev=[r for r in rows[c] if r['role']=='development']
        for r in rows[c]:
            donor=next(d for d in dev if d['group']!=r['group'])
            r['wrong_query']=dict(key=donor['key'],source=donor['source'],caption=donor['input']['caption'],subject=donor['subject'])
    assert sum(len(rr) for rr in rows.values())<=83
    for a in ('development','expansion','watched'):
        for b in ('development','expansion','watched'):
            if a>=b:continue
            ra=[r for rr in rows.values() for r in rr if r['role']==a];rb=[r for rr in rows.values() for r in rr if r['role']==b]
            assert not {r['group'] for r in ra}&{r['group'] for r in rb}
            assert not {r['input']['video_sha256'] for r in ra}&{r['input']['video_sha256'] for r in rb}
    contract=dict(protocol='DeCoTA_parametric_observation_v1',status='planned',parametric_required=True,
        final_from_updated_student=True,no_parameter_controls_only=True,automatic_promotion=False,
        old_registry_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),old_closure_completion_sha256=sha(OLD/'COMPLETION.json'),
        coverage_in_candidate=0,fixed_span_in_candidate=False,expert_frozen=True,GT_online=False,
        backbone='TA-STVG',extra_models=False,full_evaluation=False,schedules=False,historical_exposure=True,
        max_unique_inputs=83,max_fits=1500,max_backward=12000,max_new_DINO=4096,max_GPU_hours=8,
        attachments={x:sha(Path('./private_authorization_notes')/x/'pasted-text.txt') for x in
            ['private-authorization-8992e283a532','private-authorization-4c2fc5555cf3']})
    write(OUT/'GOAL_CONTRACT.json',contract)
    write(OUT/'LOCK.json',dict(rows=rows,counts=counts,created=time.time(),spatial_lr=old['spatial_lr'],
        config=dict(image_scale=.9,image_sensitivity=.95,warp=.6,warp_sensitivity=.4,tau_u=.25,kappas=[2.,4.],
            temporal_lrs=[.001,.01,.1],shared_lrs=[.0001,.001],spatial_steps=10,joint_steps=5,lambda_s=1.,gamma=1e-4),
        historical_exposure=True,GT_online=False,source_hash_disjoint=True,
        protected_pins={f:h for f,h in read(OLD/'FINAL_LOCK.json')['code_pins'].items()} if 'code_pins' in read(OLD/'FINAL_LOCK.json') else {},
        own_pins={f:sha(ROOT/f) for f in ['vg_tta/parametric_observation_v1.py','scripts/run_parametric_observation_v1.py','protocols/decota_parametric_observation_v1.md']}))
    print('LOCK',counts,flush=True)


def plan():
    p=read(OUT/'LOCK.json');g=read(OUT/'GOAL_CONTRACT.json')
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==g['old_registry_sha256']
    return p


def capture_input(model,r,middle=.5,wrong=False):
    import numpy as np
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.decota_tastvg_episode_v1 import make_batch
    from vg_tta.shared_state_v1 import capture_shared
    from vg_tta.parametric_observation_v1 import warp_grid,tensor_hash
    q=dict(r['input']);ids=q['frame_ids'];caption=r['wrong_query']['caption'] if wrong else q['caption']
    grid=warp_grid(ids,caption,middle);actual=grid['actual_ids'];unique=sorted(set(actual))
    frames,_=decode({**q,'frame_ids':unique});lookup={j:i for i,j in enumerate(unique)}
    frames=frames[[lookup[j] for j in actual]]
    batch=make_batch(frames,ids,{**q,'caption':caption},r['wrong_query']['subject'] if wrong else r['subject'],model)
    qa=dict(grid=grid,input_tensor_sha256=tensor_hash(batch['videos'].tensors),
        input_shape=list(batch['videos'].tensors.shape),rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
        caption=caption,wrong_query=wrong,actionness_placeholder_zero=not bool(batch['targets'][0]['actioness'].any()))
    assert qa['actionness_placeholder_zero']
    base,inputs,records,sc,action,views=capture_shared(model,batch)
    del batch,inputs,sc,action
    return frames,base,records,views,qa


def capture_pair(model,r,middle=.6):
    from vg_tta.parametric_observation_v1 import ObservationReplay
    import torch
    frames,base,records,views,q0=capture_input(model,r)
    f=load(r['F22_path']);assert sha(r['F22_path'])==read(Path(r['F22_path']).with_suffix('.json'))['sha256']
    assert torch.equal(base['raw_boxes'].float().cpu(),f['predictions']['F0']['boxes'])
    assert list(base['predicted_indices'])==f['predictions']['F0']['indices']
    _,_,rec1,v1,q1=capture_input(model,r,middle)
    recs=records+rec1;allviews=views+v1
    it=ObservationReplay(model,allviews,len(r['input']['frame_ids']),'spatial_head')
    with torch.no_grad():zero=it.values()
    return frames,base,recs,allviews,zero,[q0,q1],f


def expert_observations(expert,r,frames,ids,interval,include_controls=True):
    import numpy as np
    import torch
    from methods.decota_refine_uniform_v1.api import uniform_positions
    from scripts.run_decota_spatial_extension_v1 import ContextView
    from vg_tta.decota_s2_system_v1 import probe_from_detection
    from vg_tta.parametric_observation_v1 import tensor_hash,image_resample,match_observations
    context=r['parses']['context'];old=r['parses']['old']
    eligible=context['eligible'] and len(expert.processor.tokenizer(context['context'])['input_ids'])<=256
    fallback=context['reason'] if not eligible else None
    positions4=uniform_positions(ids,interval,4);positions8=uniform_positions(ids,interval,8)
    specs=[('original',p,1.) for p in sorted(set(positions4+(positions8 if include_controls else [])))]
    specs += [('weak',p,.9) for p in positions4]
    if include_controls:specs += [('duplicate',p,1.) for p in positions4]+[('sensitivity',p,.95) for p in positions4]
    observations={};calls=0;qa=[];byname={};details={}
    for name,pos,scale in specs:
        if not eligible and not old['phrase']:continue
        rgb=frames[pos] if scale==1 else image_resample(frames[pos],scale)
        inputs={}
        def hook(m,args,kw):
            for k in ('pixel_values','pixel_mask','input_ids','attention_mask'):
                if k in kw:inputs[k]=dict(sha256=tensor_hash(kw[k]),shape=list(kw[k].shape),dtype=str(kw[k].dtype))
        h=expert.model.register_forward_pre_hook(hook,with_kwargs=True)
        torch.cuda.synchronize();tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():d=ContextView(expert,context)(rgb,context['context'],context['entity']) if eligible else expert(rgb,old['phrase'],old['entity'])
        finally:h.remove()
        torch.cuda.synchronize();seconds=time.perf_counter()-tick;calls+=1
        if eligible:probe=probe_from_detection(d,pos,ids[pos])
        else:
            probe=dict(position=pos,frame_id=ids[pos],accepted=d['accepted'],reason=d['reason'],margin=d['margin'],
                boxes=torch.tensor([d['box']]) if d['box'] else torch.empty(0,4),
                target_scores=[d['score']] if d['box'] else [],candidate_ids=[0] if d['box'] else [])
        item=dict(position=pos,frame_id=ids[pos],view=name,scale=scale,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            rgb_shape=list(rgb.shape),inputs=inputs,seconds=seconds,forward_seconds=d['seconds'],peak_bytes=torch.cuda.max_memory_allocated(),
            inverse_geometry='identity: full-image down/up resampling in original canvas',text=d['text'],
            target_span=context['span'] if eligible else None,context_active=eligible,fallback=fallback,
            expert_snapshot=read(ROOT/'artifacts/decota_refine_v1/lock.json')['expert_sha256'])
        observations[(name,pos)]=dict(probe=probe,receipt=item,detection={k:v for k,v in d.items() if k!='raw_token_logits'})
        qa.append(item)
    def single(pos):
        z=observations.get(('original',pos),{}).get('probe')
        if not z or not z['accepted']:return None
        j=int(np.argmax(np.asarray(z['target_scores'])))
        return dict(position=pos,frame_id=ids[pos],box=torch.as_tensor(z['boxes'][j]).tolist(),score=float(z['target_scores'][j]),weight=1.)
    byname['single4']=[a for pos in positions4 if (a:=single(pos)) is not None]
    byname['single8']=[a for pos in positions8 if (a:=single(pos)) is not None] if include_controls else []
    if not include_controls:byname.pop('single8')
    for name in ['weak']+(['duplicate','sensitivity'] if include_controls else []):
        aa=[];matches=[]
        for pos in positions4:
            if ('original',pos) not in observations or (name,pos) not in observations:continue
            a=observations[('original',pos)];b=observations[(name,pos)]
            m=match_observations(a['probe'],b['probe']);matches.append(dict(position=pos,**m))
            if m['anchor'] is not None:aa.append(m['anchor'])
            same=a['receipt']['inputs']['pixel_values']['sha256']==b['receipt']['inputs']['pixel_values']['sha256']
            qa.append(dict(comparison=name,position=pos,same_model_tensor=same,
                boxes_exact=torch.equal(a['detection']['all_boxes'],b['detection']['all_boxes'])))
            if name=='duplicate':assert same and qa[-1]['boxes_exact']
        byname[name]=aa;details[name]=matches
    return dict(anchors=byname,observations=observations,details=details,qa=qa,new_DINO=calls,
        positions4=positions4,positions8=positions8,actual_positions=sorted({p for n,p in observations}),
        context_active=eligible,fallback=fallback,no_phrase=not eligible and not old['phrase'])


def frozen_controls(zero,records,ids,f,teacher,expert):
    import torch
    from vg_tta.parametric_observation_v1 import posterior_decode
    from vg_tta.decota_tastvg_episode_v1 import fitted_merge
    from methods.decota_refine_uniform_v1.api import reconstruct
    from vg_tta.posterior_mass_coverage_v1 import select
    native=list(fitted_merge(zero['logits'][:2],records[:2],ids));boxes=zero['boxes'].cpu()
    en=posterior_decode(teacher['q'],records,ids)
    arms=dict(NATIVE=f['predictions']['F0'],NATIVE_FP32=dict(boxes=boxes,indices=native),
        REGISTERED=f['predictions']['registered_original'],OLD_WORK=f['predictions']['F4'],OLD_FIXED_INTERP=f['predictions']['F5'],
        ENSEMBLE=dict(boxes=boxes,indices=en),
        OLD_COVERAGE=dict(boxes=boxes,indices=f['predictions']['F4']['indices']),
        FIXED_TIME=dict(boxes=boxes,indices=f['predictions']['F5']['indices']))
    z=torch.stack([zero['logits'][i%2][0,i//2] for i in range(len(ids))])
    arms['PM09']=dict(boxes=boxes,indices=list(select(z,ids,tau=.9,physical_mass=True)['indices']))
    for mode,aa in expert['anchors'].items():
        if mode not in ('single4','single8','weak','duplicate','sensitivity'):continue
        for interp in ('direct','absolute'):
            b=reconstruct(boxes,aa,ids,interp)[0]
            arms['TEACHER_'+mode+'_'+interp]=dict(boxes=b,indices=native)
            if mode=='weak':arms['SAME_INFO_'+interp]=dict(boxes=b,indices=en)
    return arms


def runtime(stage,cohort=None,limit=0):
    import torch
    from vg_tta.parametric_observation_v1 import ObservationReplay,fit_student,build_teacher,posterior_decode
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.decota_tastvg_episode_v1 import fitted_merge
    p=plan();old=read(ROOT/'artifacts/decota_refine_v1/lock.json');expert=None
    if stage in ('observe','expand'):
        assert sha(Path(old['expert_snapshot'])/'model.safetensors')==old['expert_sha256']
        expert=SpatialExpert(old['expert_snapshot']);edigest=state_digest(expert.model)
    n=0;backwards=0;newcalls=0;fits_count=0
    for c,rr in p['rows'].items():
        if cohort and cohort!=c:continue
        if limit and n>=limit:break
        model=model_for(c);mdigest=state_digest(model)
        for r in rr:
            if (r['role']=='expansion') != (stage=='expand'):continue
            dest=path(stage,r['key'])
            if existing(dest):continue
            if limit and n>=limit:break
            tick=time.perf_counter();frames,base,records,views,zero,qa,f=capture_pair(model,r)
            ids=r['input']['frame_ids'];native=list(fitted_merge(zero['logits'][:2],records[:2],ids))
            teacher=build_teacher(zero['logits'],records,ids,.6,qa[1]['grid']['active'])
            if stage in ('observe','expand'):
                ex=expert_observations(expert,r,frames,ids,native,stage=='observe');newcalls+=ex['new_DINO']
            else:
                prior=load(path('observe',r['key']));ex=prior['expert']
                assert all(torch.equal(a.cpu(),b) for a,b in zip(zero['logits'],prior['zero_logits']))
                assert torch.equal(zero['boxes'].cpu(),prior['native_boxes'])
            if stage=='observe':
                # Exact duplicate teacher uses the exact same suffix, not FP16/FP32 mismatch.
                dup=build_teacher(zero['logits'][:2]*2,records[:2]*2,ids,.5,True)
                sensitivity={};wrong=None
                for mid in (.4,):
                    _,_,rec,vv,qq=capture_input(model,r,mid)
                    it=ObservationReplay(model,vv,len(ids),'head')
                    with torch.no_grad():zz=it.values()
                    tt=build_teacher(zero['logits'][:2]+zz['logits'],records[:2]+rec,ids,mid,qq['grid']['active'])
                    sensitivity[str(mid)]=dict(logits=[z.cpu() for z in zz['logits']],qa=qq,teacher=cpu_tree(tt),indices=posterior_decode(tt['q'],records,ids))
                    del it,vv,zz
                wrong_logits=[];wrong_qa=[]
                for mid in (.5,.6):
                    _,_,rec,vv,qq=capture_input(model,r,mid,True);it=ObservationReplay(model,vv,len(ids),'head')
                    with torch.no_grad():zz=it.values()
                    wrong_logits += [z.cpu() for z in zz['logits']];wrong_qa.append(qq);del it,vv,zz
                # Wrong query is a real forward on this same video, with its own text subject.
                wrong=build_teacher(wrong_logits,records,ids,.6,wrong_qa[1]['grid']['active'])
                x=dict(key=r['key'],source=r['source'],group=r['group'],role=r['role'],frame_ids=ids,
                    teacher=cpu_tree(teacher),duplicate_teacher=cpu_tree(dup),wrong_teacher=cpu_tree(wrong),wrong_qa=wrong_qa,
                    wrong_query=r['wrong_query'],sensitivity=sensitivity,expert=ex,qa=qa,
                    zero_logits=[z.cpu() for z in zero['logits']],native_boxes=zero['boxes'].cpu(),native_indices=native,
                    arms=frozen_controls(zero,records,ids,f,teacher,ex),new_DINO=ex['new_DINO'],
                    seconds=time.perf_counter()-tick,GT_online=False)
                record(dest,x)
            else:
                selected=read(OUT/'AB_SELECTION.json')[c] if stage in ('joint','expand','fit10') else None
                config=read(OUT/'FINAL_CANDIDATE_LOCK.json')[c] if stage=='expand' else None
                arms=frozen_controls(zero,records,ids,f,teacher,ex);fits={};spatial_lr=p['spatial_lr'][c]
                spec=[];extra_qa={};mirror_views=None
                if stage=='dev':
                    for name,an,kap,k,oldmean in [('S4_OLD','single4',None,4,True),('S8_OLD','single8',None,8,True),
                        ('S4_FIXED','single4',None,4,False),('S4_DUP','duplicate',None,4,False),
                        ('S4_WEIGHT_SINGLE','single4',None,4,False),('S4_DUAL','weak',None,4,False),
                        ('S4_ROBUST2','weak',2.,4,False),('S4_ROBUST4','weak',4.,4,False),('S4_SENSITIVE','sensitivity',2.,4,False)]:
                        aa=copy.deepcopy(ex['anchors'][an])
                        if name=='S4_WEIGHT_SINGLE':
                            for a in aa:a['weight']=a['score']
                        spec.append(dict(name=name,scope='spatial',kind='spatial',anchors=aa,teacher=teacher,planned=k,kappa=kap,old_mean=oldmean,steps=10,head_lr=.01,shared_lr=.0001))
                    for lr in p['config']['temporal_lrs']:
                        spec.append(dict(name=f'T_REAL_{lr:g}',scope='head',kind='temporal',anchors=[],teacher=teacher,steps=5,head_lr=lr,shared_lr=.0001))
                    for name,tgt in [('T_DUP',prior['duplicate_teacher']),('T_WRONG',prior['wrong_teacher'])]:
                        spec.append(dict(name=name,scope='head',kind='temporal',anchors=[],teacher=tgt,steps=5,head_lr=.01,shared_lr=.0001,duplicate=name=='T_DUP'))
                    shuf=copy.deepcopy(teacher)
                    # Reverse the sorted legal-interval atom correspondence, not a label-aware shuffle.
                    shuf['targets']=[q.flip(0) for q in shuf['targets']]
                    spec.append(dict(name='T_SHUFFLE',scope='head',kind='temporal',anchors=[],teacher=shuf,steps=5,head_lr=.01,shared_lr=.0001))
                    spec.append(dict(name='T_COVERAGE_RAW',scope='head',kind='coverage',anchors=[],teacher=teacher,steps=5,head_lr=(.001199351394104754 if c=='hcstvg1_test' else .059382906423716963),shared_lr=.0001))
                    # Full backward/lr0, steps0: matched graph and final student output.
                    for name,steps,lr in [('NOOP_STEPS0',0,.01),('NOOP_LR0',1,0.)]:
                        spec.append(dict(name=name,scope='shared_spatial_head',kind='joint',anchors=ex['anchors']['weak'],teacher=teacher,steps=steps,head_lr=lr,shared_lr=lr,space_lr=lr,kappa=2.))
                elif stage=='fit10':
                    # One predeclared consequence H: ensemble better than native on Vid dev,
                    # but five-step student underfits it. Same teacher, input, LR; only steps change.
                    assert c=='vidstg_test'
                    spec.append(dict(name='T_FIT10',scope='head',kind='temporal',anchors=[],teacher=teacher,
                        steps=10,head_lr=selected['temporal']['lr'],shared_lr=.0001))
                else:
                    aa=ex['anchors']['weak'];kap=selected['spatial']['kappa'];tlr=selected['temporal']['lr']
                    spec += [dict(name='T_PRIVATE',scope='head',kind='temporal',anchors=[],teacher=teacher,steps=5,head_lr=tlr,shared_lr=.0001),
                        dict(name='S_PRIVATE5',scope='spatial',kind='spatial',anchors=aa,teacher=teacher,steps=5,head_lr=tlr,shared_lr=.0001,kappa=kap),
                        dict(name='S_PRIVATE10',scope='spatial',kind='spatial',anchors=aa,teacher=teacher,steps=10,head_lr=tlr,shared_lr=.0001,kappa=kap)]
                    slrs=[config['shared_lr']] if config else p['config']['shared_lrs']
                    for slr in slrs:
                        spec.append(dict(name=f'JOINT_{slr:g}',scope='shared_spatial_head',kind='joint',anchors=aa,teacher=teacher,steps=5,head_lr=tlr,shared_lr=slr,kappa=kap))
                    slr=config['shared_lr'] if config else .0001
                    for name,kind in [('JOINT_NO_T','spatial'),('JOINT_NO_S','temporal')]:
                        spec.append(dict(name=name,scope='shared_spatial_head',kind=kind,anchors=aa,teacher=teacher,steps=5,head_lr=tlr,shared_lr=slr,kappa=kap))
                    if stage=='joint':
                        # Controls must use the SELECTED temporal LR, not only the screening LR.
                        for name,tgt in [('T_MATCH_DUP',prior['duplicate_teacher']),('T_MATCH_WRONG',prior['wrong_teacher'])]:
                            spec.append(dict(name=name,scope='head',kind='temporal',anchors=[],teacher=tgt,steps=5,head_lr=tlr,shared_lr=.0001,duplicate=name=='T_MATCH_DUP'))
                        shuf=copy.deepcopy(teacher);shuf['targets']=[q.flip(0) for q in shuf['targets']]
                        spec.append(dict(name='T_MATCH_SHUFFLE',scope='head',kind='temporal',anchors=[],teacher=shuf,steps=5,head_lr=tlr,shared_lr=.0001))
                        _,_,_,mirror_views,mirror_qa=capture_input(model,r,.4)
                        extra_qa['mirror']=mirror_qa
                        spec.append(dict(name='T_MIRROR',scope='head',kind='temporal',anchors=[],teacher=prior['sensitivity']['0.4']['teacher'],
                            steps=5,head_lr=tlr,shared_lr=.0001,mirror=True))
                for s in spec:
                    vv=views[:2]+mirror_views if s.get('mirror') else views[:2]*2 if s.get('duplicate') else views if s['kind'] in ('temporal','joint') else views[:2]
                    it=ObservationReplay(model,vv,len(ids),s['scope']);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    z=fit_student(it,s['anchors'],s['teacher'],records,ids,kind=s['kind'],steps=s['steps'],
                        lrs=dict(spatial=s.get('space_lr',spatial_lr),head=s['head_lr'],shared=s['shared_lr']),
                        planned=s.get('planned',4),kappa=s.get('kappa'),old_mean=s.get('old_mean',False))
                    torch.cuda.synchronize();z['peak_bytes']=torch.cuda.max_memory_allocated()
                    fits[s['name']]=z;backwards+=z['backwards'];fits_count+=1
                    box=zero['boxes'].cpu() if s['scope']=='head' else z['final']['boxes']
                    iv=native if s['scope']=='spatial' else z['final']['indices']
                    arms[s['name']]=dict(boxes=box,indices=iv)
                    if stage in ('joint','expand') and s['name'].startswith('JOINT_') and s['name'] not in ('JOINT_NO_T','JOINT_NO_S'):
                        from scripts.audit_parametric_reinsertion_v1 import full_prediction
                        clock=time.perf_counter()
                        live=full_prediction(model,frames,ids,r['input'],r['subject'],z['state'],z['final'])
                        arms[s['name']]=dict(boxes=live['boxes'],indices=live['indices'])
                        z['full_forward_audit']=live['audit'];z['full_forward_seconds']=time.perf_counter()-clock
                    if s['name'].startswith('NOOP'):
                        assert z['state_delta']==0 and torch.equal(box,zero['boxes'].cpu()) and iv==native
                    print(stage,r['key'],s['name'],'step',z['best_step'],'delta',round(z['state_delta'],6),'sec',round(z['seconds'],2),flush=True)
                    del it;gc.collect()
                if stage in ('joint','expand'):
                    for nn in ('5','10'):
                        # Exact same-model private composition; replay once verifies true output dependency.
                        it=ObservationReplay(model,views,len(ids),'spatial_head');st=it.state()
                        st.update({k:v for k,v in fits['T_PRIVATE']['state'].items()});st.update(fits['S_PRIVATE'+nn]['state']);it.restore(st)
                        with torch.no_grad():v=it.values()
                        assert torch.equal(v['boxes'].cpu(),fits['S_PRIVATE'+nn]['final']['boxes'])
                        assert all(torch.equal(a.cpu(),b) for a,b in zip(v['logits'],fits['T_PRIVATE']['final']['logits']))
                        arms['PRIVATE'+nn]=dict(boxes=v['boxes'].cpu(),indices=list(fitted_merge(v['logits'][:2],records[:2],ids)))
                        if stage=='expand':
                            from scripts.audit_parametric_reinsertion_v1 import full_prediction
                            expected=dict(**arms['PRIVATE'+nn],logits=[z.cpu() for z in v['logits']])
                            live=full_prediction(model,frames,ids,r['input'],r['subject'],st,expected)
                            arms['PRIVATE'+nn]=dict(boxes=live['boxes'],indices=live['indices'],audit=live['audit'])
                        del it,v
                x=dict(key=r['key'],source=r['source'],group=r['group'],role=r['role'],frame_ids=ids,
                    arms=arms,fits=fits,teacher=cpu_tree(teacher),expert=ex,qa=qa,native_indices=native,
                    native_boxes=zero['boxes'].cpu(),seconds=time.perf_counter()-tick,new_DINO=ex['new_DINO'] if stage=='expand' else 0,
                    GT_online=False,student_output_only=True,old_state_unchanged=True,extra_qa=extra_qa,
                    executed_code_pins={f:sha(ROOT/f) for f in ['vg_tta/parametric_observation_v1.py','scripts/run_parametric_observation_v1.py','scripts/audit_parametric_reinsertion_v1.py']})
                record(dest,x)
                del mirror_views
            assert state_digest(model)==mdigest
            if expert:assert state_digest(expert.model)==edigest
            status(OUT/'STATUS.json',dict(stage=stage,cohort=c,last_key=r['key'],completed=n+1,
                invocation_backward=backwards,invocation_fits=fits_count,invocation_new_DINO=newcalls,updated=time.time()))
            n+=1
            assert backwards<=12000 and fits_count<=1500 and newcalls<=4096
            del frames,base,records,views,zero,qa,f,teacher,ex,x;gc.collect();torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()
    write(OUT/'invocations'/f'{time.time_ns()}.json',dict(stage=stage,completed=n,backward=backwards,fits=fits_count,new_DINO=newcalls,GT_online=False))


def cpu_tree(x):
    import torch
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    return x


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','observe','dev','joint','expand','fit10']);ap.add_argument('--cohort');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.stage=='prepare':return prepare()
    from scripts.run_decota_refine_v1 import configure
    configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tick=time.time();failure=None
    try:runtime(a.stage,a.cohort,a.limit)
    except BaseException as e:failure=repr(e);raise
    finally:
        write(OUT/'leases'/f'{time.time_ns()}.json',dict(stage=a.stage,seconds=time.time()-tick,failure=failure))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':main()
