"""Finite controlled natural-query audit; all outputs remain frozen diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,time,hashlib,gc,collections,shutil,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status as atomic_status
from scripts.tastvg_controlled_query_math_v1 import frozen_readout,multi_difference,norm,native_subject
from scripts.tastvg_information_atlas_math_v1 import frame_labels,candidate_labels,moments,bootstrap_summary,paired_difference
from scripts.tastvg_latent_quality_math_v1 import interval_features,geometry_features
from scripts import run_tastvg_query_swap_v1 as predecessor
BASE=ROOT/'artifacts/tastvg_controlled_query_v1';PUB=ROOT/'results/tastvg_controlled_query/2026-10-03'
OLD=predecessor.BASE;ATLAS=predecessor.ATLAS;PLAN=predecessor.PLAN;POOL=predecessor.POOL
DS=['vidstg','hc2'];ARMS=['true','event','subject','generic']
OWN=['scripts/intake_tastvg_controlled_query_v1.py','scripts/tastvg_controlled_query_math_v1.py',
     'scripts/run_tastvg_controlled_query_v1.py','scripts/test_tastvg_controlled_query_v1.py',
     'protocols/tastvg_controlled_query_v1.md']


def st(s,**x):atomic_status(BASE/'STATUS.json',dict(status=s,time=time.time(),pid=os.getpid(),**x))
def prefix(c):return predecessor.prefix(c)
def checked(p):return predecessor.checked(p)
def commit(p,z):
    save(p,z);write(p.with_suffix('.json'),dict(time=time.time(),sha256=sha(p),GT_read=False,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
def verify(payloads=False):
    lock=read(BASE/'RUNTIME_LOCK.json')
    for p,h in {**lock['code'],**lock['metadata']}.items():assert sha(ROOT/p)==h,p
    if payloads:
        for p,h in lock['payloads'].items():assert sha(ROOT/p)==h,p
    return lock


def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    assert read(OLD/'FINAL_COMPLETION.json')['commit']=='451d1c84cf74ae88cf7e6df04948e19e57526e41'
    intake=read(BASE/'INTAKE.json');matches=read(BASE/'PRIVATE_MATCHES.json');nlp=read(BASE/'PRIVATE_NLP_TEXT.json')
    co=read(OLD/'COHORT.json');cells=co['cells'];assert len(cells)==288
    coverage={};public=[];smokes={};metadata={};payloads={}
    for ds in DS:
        cc=[c for c in cells if c['dataset']==ds];rows=read(PLAN/ds/'PLAN.json')['rows'];counter=collections.Counter()
        for c in cc:
            m=matches[ds][str(c['parent'])];s=read(POOL/ds/'subjects'/f'{c["parent"]:05}.json')['parses']['subject']
            assert native_subject(nlp[norm(m['recipient']['caption'])]['words'])==s,('native subject parity',ds,c['parent'])
            for arm,p in m['pairs'].items():
                if p:counter[arm]+=1;counter[arm+'/'+p['kind']]+=1
            counter['common']+=int(all(m['pairs'].values()))
            for path in [ROOT/c['feature'],ROOT/c['state_payload'],OLD/ds/'swap_features'/f'{prefix(c)}.pt']:
                assert sha(path)==read(path.with_suffix('.json'))['sha256'];payloads[str(path.relative_to(ROOT))]=sha(path)
        coverage[ds]=dict(cells=len(cc),sources=len({c['parent'] for c in cc}),available=dict(counter),
            encoder_inputs={a:len({(c['parent'],c['condition']) for c in cc if matches[ds][str(c['parent'])]['pairs'][a]}) for a in ['event','subject']})
        for i,m in matches[ds].items():
            for arm,p in m['pairs'].items():
                d=p['donor'] if p else None
                public.append(dict(dataset=ds,source_index=int(i),arm=arm,available=p is not None,
                    tier=p['tier'] if p else None,kind=p['kind'] if p else None,eligible_donors=p['eligible_donors'] if p else 0,
                    true_caption_sha256=hashlib.sha256(m['recipient']['caption'].encode()).hexdigest(),
                    donor_caption_sha256=hashlib.sha256(d['caption'].encode()).hexdigest() if d else None,
                    donor_media_sha256=d['video_sha256'] if d else None,
                    same_exact_media=bool(d and d['video_sha256']==m['recipient']['video_sha256']),
                    same_input_segment=bool(d and d['segment']==m['recipient']['segment']),
                    same_native_subject=bool(d and d['signature']['subject']==m['recipient']['signature']['subject']),
                    exact_action_signature=bool(d and d['signature']['identity_signature']==m['recipient']['signature']['identity_signature']),
                    true_words=len(m['recipient']['caption'].split()),donor_words=len(d['caption'].split()) if d else None))
        clean=[c for c in cc if c['condition']=='clean']
        clean.sort(key=lambda c:(not all(matches[ds][str(c['parent'])]['pairs'].values()),
            hashlib.sha256(('controlled-smoke-v1|'+ds+'|'+prefix(c)).encode()).hexdigest()))
        smokes[ds]=[prefix(c) for c in clean[:2]]
        for p in [ATLAS/ds/'FROZEN_ATLAS.pt',OLD/ds/'SEALED_READOUT.pt']:
            assert sha(p)==read(p.with_suffix('.json'))['sha256'];payloads[str(p.relative_to(ROOT))]=sha(p)
        metadata[str((PLAN/ds/'PLAN.json').relative_to(ROOT))]=sha(PLAN/ds/'PLAN.json')
    write(PUB/'QUERY_MAPPING.json',public)
    write(PUB/'CONFIG.json',dict(version='tastvg_controlled_query_v1',predecessor_commit='451d1c84cf74ae88cf7e6df04948e19e57526e41',
        coverage=coverage,original_cells=288,corrupt_cells=240,clean_cells=48,target_sources={'vidstg':24,'hc2':21},
        donor_pool_queries={d:intake['summary'][d]['donor_pool_queries'] for d in DS},
        annotation_assisted_referent_matching=True,GT_time_or_metric_used_for_pairing=False,
        source_checkpoint_sha256=read(predecessor.PUBLIC/'CONFIG.json')['source_checkpoint_sha256'],
        probe_hashes={d:sha(ATLAS/d/'FROZEN_ATLAS.pt') for d in DS},source_models_unchanged=136,
        candidate_views=['Full','Contrast'],candidate_tasks=['precision','recall','tiou'],secondary='Hidden event AUROC',
        primary='HC Contrast R2(subject)-R2(event) on corrupt common_strict recipients; three correlated pointwise endpoints; weaker common is secondary',
        support=32,old8_preserved=True,probe_refits=0,new_expert_calls=0,new_candidates=0,parameter_updates=0,
        historical_exposure=True,source_bootstrap_draws=10000,seed=20261003,donors_fixed_in_bootstrap=True,
        donor_reuse_allowed=True,no_new_top1_output=True,production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),time=time.time()))
    write(BASE/'COHORT.json',dict(cells=cells,smokes=smokes))
    for p in [BASE/'COHORT.json',BASE/'PRIVATE_MATCHES.json',BASE/'PRIVATE_NLP_TEXT.json',BASE/'PRIVATE_DONOR_POOL.json',BASE/'INTAKE.json',
        PUB/'CONFIG.json',PUB/'QUERY_MAPPING.json',OLD/'FINAL_COMPLETION.json',ROOT/'methods/CURRENT_METHOD.json']:
        metadata[str(p.relative_to(ROOT))]=sha(p)
    write(BASE/'RUNTIME_LOCK.json',dict(time=time.time(),code={p:sha(ROOT/p) for p in OWN+predecessor.OWN+predecessor.DEPS},
        metadata=metadata,payloads=payloads))
    st('prepared_pending_no_GT_smoke',coverage=coverage,GT_time_read=False)
    print('CONTROLLED_QUERY_LOCKED',coverage,flush=True)


def smoke(ds):
    verify(True);tick=time.time();model,lease=predecessor.cuda_setup(ds)
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor
    from methods.decota_final_simplified_v1.tensors import state_hash
    actor=SpatialActor(model);initial=state_hash(model.state_dict());rows=read(PLAN/ds/'PLAN.json')['rows'];co=read(BASE/'COHORT.json')
    matches=read(BASE/'PRIVATE_MATCHES.json')[ds];out=[];enc=rep=0
    try:
        for c in [c for c in co['cells'] if c['dataset']==ds and prefix(c) in co['smokes'][ds]]:
            z=checked(ROOT/c['feature']);r=rows[c['parent']];frames,pixel,_=predecessor.observations(ds,r,c['condition']);assert pixel==z['pixel_sha256']
            m=matches[str(c['parent'])];sub=read(POOL/ds/'subjects'/f'{c["parent"]:05}.json')['parses']['subject']
            data=predecessor.input_data(model,actor,frames,r,r['input']['caption'],sub);enc+=1
            h,p=predecessor.replay(actor,data,c,z);rep+=1;old=load(ROOT/c['state_payload'])
            x,ctx=interval_features(h.numpy(),z['frame_ids'],z['candidate_indices'],r['input']['fps'])
            assert torch.equal(h,z['hidden']) and np.array_equal(x,z['x']) and ctx==z['context']
            assert torch.equal(p['boxes'],old['slow']['boxes']) and p['indices']==old['slow']['indices']
            value=dict(true=h.clone(),controlled={});del data,h,p;gc.collect();torch.cuda.empty_cache()
            for arm,pair in m['pairs'].items():
                if pair is None:continue
                d=pair['donor'];data=predecessor.input_data(model,actor,frames,r,d['caption'],d['signature']['subject']);enc+=1
                h,p=predecessor.replay(actor,data,c,z);rep+=1
                assert h.shape==z['hidden'].shape and torch.isfinite(h).all()
                value['controlled'][arm]=h;del data,p;gc.collect();torch.cuda.empty_cache()
            commit(BASE/ds/'smoke'/f'{prefix(c)}.pt',value)
            out.append(dict(cell=prefix(c),true_bitwise_hidden_feature_box_interval_parity=True,
                controlled_arms=list(value['controlled']),controlled_hidden_changed={a:not torch.equal(h,value['true']) for a,h in value['controlled'].items()}))
            print('CONTROLLED_SMOKE',ds,len(out),flush=True)
        actor.close();assert state_hash(model.state_dict())==initial
        write(BASE/ds/'SMOKE.json',dict(status='passed',rows=out,GT_read=False,model_restored=True,new_encoder_inputs=enc,
            new_backbone_offset_forwards=2*enc,latent_replay_calls=rep,worker_wall_seconds=time.time()-tick,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),time=time.time()))
        st('smoke_complete_pending_root_acceptance',dataset=ds,GT_time_read=False)
    finally:actor.close();lease.close()


def capture(ds):
    verify(True);assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='passed'
    tick=time.time();model,lease=predecessor.cuda_setup(ds)
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor
    from methods.decota_final_simplified_v1.tensors import state_hash
    actor=SpatialActor(model);initial=state_hash(model.state_dict());rows=read(PLAN/ds/'PLAN.json')['rows']
    matches=read(BASE/'PRIVATE_MATCHES.json')[ds];cells=[c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds]
    jobs=[(c,arm) for c in cells for arm in ['event','subject'] if matches[str(c['parent'])]['pairs'][arm]]
    jobs.sort(key=lambda z:(z[0]['parent'],z[0]['condition'],z[1],z[0]['order'],z[0]['arrival']))
    cached=None;data=frames=None;enc=done=0
    try:
        for c,arm in jobs:
            assert shutil.disk_usage(ROOT).free>8*2**30
            f=BASE/ds/'features'/arm/f'{prefix(c)}.pt';assert not f.exists()
            z=checked(ROOT/c['feature']);r=rows[c['parent']];pair=matches[str(c['parent'])]['pairs'][arm];d=pair['donor']
            marker=(c['parent'],c['condition'],arm)
            if marker!=cached:
                del data,frames;gc.collect();torch.cuda.empty_cache()
                frames,pixel,_=predecessor.observations(ds,r,c['condition']);assert pixel==z['pixel_sha256']
                data=predecessor.input_data(model,actor,frames,r,d['caption'],d['signature']['subject']);enc+=1;cached=marker
            h,_=predecessor.replay(actor,data,c,z)
            x,ctx=interval_features(h.numpy(),z['frame_ids'],z['candidate_indices'],r['input']['fps'])
            geom=geometry_features(z['frame_ids'],z['candidate_indices']);assert np.array_equal(geom,z['geometry'])
            value={k:z[k] for k in ['cell_key','dataset','split','source_id','condition','order','arrival','anchor_index',
                'frame_ids','candidate_indices','intervals','A_state_pre_sha256','A_state_post_sha256','pixel_sha256']}
            value.update(hidden=h,x=x,context=ctx,geometry=geom,arm=arm,tier=pair['tier'],kind=pair['kind'],
                donor_caption_sha256=hashlib.sha256(d['caption'].encode()).hexdigest(),true_feature_sha256=c['feature_sha256'],
                GT_read=False,candidate_support_changed=False,parameter_updates=0)
            commit(f,value);done+=1;st('controlled_capture_running',dataset=ds,done=done,total=len(jobs),encoder_inputs=enc,GT_time_read=False)
            if done%12==0:print('CONTROLLED_CAPTURE',ds,done,len(jobs),'ENCODERS',enc,flush=True)
            del h,x,value;gc.collect()
        actor.close();assert state_hash(model.state_dict())==initial
        expected=read(PUB/'CONFIG.json')['coverage'][ds];assert enc==sum(expected['encoder_inputs'].values())
        assert done==sum(expected['available'][a] for a in ['event','subject'])
        files={str(p.relative_to(BASE)):sha(p) for p in sorted((BASE/ds/'features').rglob('*.pt'))}
        write(BASE/ds/'CAPTURE_BARRIER.json',dict(status='completed',files=files,controlled_cells=done,GT_read=False,
            time=time.time(),model_restored=True,worker_wall_seconds=time.time()-tick,new_encoder_inputs=enc,
            new_backbone_offset_forwards=enc*2,latent_replay_calls=done,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated()))
        st('capture_complete_pending_next_stage',dataset=ds,controlled_cells=done,GT_time_read=False)
    finally:actor.close();lease.close()


def seal():
    verify(True);assert not torch.cuda.is_initialized();sys.addaudithook(predecessor.guard);tick=time.time();files={};controlled=0
    matches=read(BASE/'PRIVATE_MATCHES.json')
    for ds in DS:
        barrier=read(BASE/ds/'CAPTURE_BARRIER.json');models=checked(ATLAS/ds/'FROZEN_ATLAS.pt')['models']
        old={r['cell']:r for r in checked(OLD/ds/'SEALED_READOUT.pt')};rows=[]
        for c in read(BASE/'COHORT.json')['cells']:
            if c['dataset']!=ds:continue
            z=checked(ROOT/c['feature']);true=frozen_readout(z,models);generic=frozen_readout(checked(OLD/ds/'swap_features'/f'{prefix(c)}.pt'),models)
            for k in true:assert np.array_equal(true[k],old[z['cell_key']]['true'][k]) and np.array_equal(generic[k],old[z['cell_key']]['swap'][k])
            arms=dict(true=true,generic=generic)
            for arm,pair in matches[ds][str(c['parent'])]['pairs'].items():
                if pair is None:continue
                sw=checked(BASE/ds/'features'/arm/f'{prefix(c)}.pt');pred=frozen_readout(sw,models)
                for k in true:
                    if '/Geometry/' in k:assert np.array_equal(true[k],pred[k])
                arms[arm]=pred;controlled+=1
            rows.append(dict(cell=z['cell_key'],arms=arms))
        f=BASE/ds/'SEALED_READOUT.pt';commit(f,rows);files[ds]=sha(f)
    seal=dict(status='all_controlled_readouts_sealed',time=time.time(),files=files,original_cells=288,
        controlled_cells=controlled,GT_read=False,probes_retrained=0,CPU_wall_seconds=time.time()-tick)
    write(BASE/'GLOBAL_READOUT_SEAL.json',seal);write(PUB/'GLOBAL_READOUT_SEAL.json',seal);st('sealed_pending_original_GT_join',GT_time_read=False)


def group_rows(rows):
    for mode in ['corrupt','clean']:
        allrows=[r for r in rows if (r['condition']=='clean')==(mode=='clean')]
        panels={'all':allrows,'search':[r for r in allrows if r['panel']=='search'],'confirm':[r for r in allrows if r['panel']=='confirm']}
        panels.update({o:[r for r in allrows if r['order']==o] for o in sorted({r['order'] for r in allrows})})
        for panel,rr in panels.items():
            cohorts=dict(all=rr,event_available=[r for r in rr if 'event' in r['available_arms']],
                subject_available=[r for r in rr if 'subject' in r['available_arms']],
                common=[r for r in rr if 'event' in r['available_arms'] and 'subject' in r['available_arms']])
            cohorts['common_strict']=[r for r in cohorts['common'] if r['tiers']['event']<=2 and r['tiers']['subject']<=2]
            for arm in ['event','subject']:cohorts['same_video_'+arm]=[r for r in cohorts[arm+'_available'] if r['tiers'][arm]<=1]
            for cohort,zz in cohorts.items():yield f'{mode}/{panel}/{cohort}',zz


def summary(rows):
    out={}
    for label,rr in group_rows(rows):
        names=sorted(set.intersection(*(set(r['metrics']) for r in rr))) if rr else []
        metrics={n:bootstrap_summary(rr,n) for n in names};gaps={};specificity={};blocks={}
        for name in names:
            if not name.startswith('true/'):continue
            task=name.split('/')[3];fields=['auc','ap','logloss','r2','mse'] if task=='event' else ['r2','mse','mae','within_r2']
            for arm in ['event','subject','generic']:
                other=arm+'/'+name[5:]
                if other in names:gaps[arm+'/'+name[5:]]={f:paired_difference(rr,name,other,f) for f in fields}
            if 'event/'+name[5:] in names and 'subject/'+name[5:] in names:
                specificity[name[5:]]={f:paired_difference(rr,'subject/'+name[5:],'event/'+name[5:],f) for f in fields}
        if rr and all(a+'/candidate/Full/tiou/real' in names for a in ['event','subject']):
            for task in ['precision','recall','tiou']:
                terms=[(f'subject/candidate/Contrast/{task}/real',1),(f'event/candidate/Contrast/{task}/real',-1),
                    (f'subject/candidate/Full/{task}/real',-1),(f'event/candidate/Full/{task}/real',1)]
                blocks[task]=multi_difference(rr,terms)
        out[label]=dict(coverage=dict(cells=len(rr),sources=len({r['source_index'] for r in rr}),
            unique_donor_hashes={a:len({r['donor_hashes'][a] for r in rr if a in r['donor_hashes']}) for a in ['event','subject']}),
            metrics=metrics,paired_original_minus_intervention=gaps,specificity_subject_minus_event=specificity,
            Contrast_minus_Full_specificity=blocks)
    return out


def diagnose():
    verify(True);assert not torch.cuda.is_initialized();tick=time.time();torch.set_num_threads(2)
    seal=read(BASE/'GLOBAL_READOUT_SEAL.json');assert seal['original_cells']==288 and not seal['GT_read'];join=time.time()
    matches=read(BASE/'PRIVATE_MATCHES.json')
    for ds in DS:
        assert sha(BASE/ds/'SEALED_READOUT.pt')==seal['files'][ds]
        gt={s:read(POOL/ds/f'GT_LABELS_{s}.json') for s in ['search','confirm']}
        pred={r['cell']:r['arms'] for r in checked(BASE/ds/'SEALED_READOUT.pt')};rows=[]
        for c in read(BASE/'COHORT.json')['cells']:
            if c['dataset']!=ds:continue
            z=checked(ROOT/c['feature']);ids=np.asarray(z['frame_ids']);span=gt[c['split']][str(c['parent'])]['span']
            labels=dict(frame=frame_labels(ids,span),candidate=candidate_labels([[ids[i],ids[j]+1] for i,j in z['candidate_indices']],span,z['anchor_index']))
            arms=pred[z['cell_key']];metrics={}
            for arm,scores in arms.items():
                for name,p in scores.items():
                    if name.endswith('/logit'):continue
                    fam,view,task,control=name.split('/');metrics[arm+'/'+name]=moments(labels[fam][task],p,task=='event',scores.get(name+'/logit'))
            pair=matches[ds][str(c['parent'])]['pairs']
            rows.append(dict(dataset=ds,panel=c['split'],source_index=c['parent'],cell=z['cell_key'],order=c['order'],condition=c['condition'],
                candidates=32,frames=len(ids),available_arms=list(arms),tiers={a:p['tier'] for a,p in pair.items() if p},
                pair_kinds={a:p['kind'] for a,p in pair.items() if p},
                donor_hashes={a:hashlib.sha256(p['donor']['caption'].encode()).hexdigest() for a,p in pair.items() if p},metrics=metrics))
        write(PUB/ds/'ROWS.json',rows);write(PUB/ds/'SUMMARY.json',summary(rows));print('CONTROLLED_DIAGNOSE',ds,len(rows),flush=True)
    write(PUB/'LABEL_JOIN.json',dict(time=join,readout_seal_time=seal['time'],only_original_cached_GT_span=True,
        donor_GT_read=False,GT_for_pairing=False,annotation_metadata_intake_disclosed=True))
    workers={d:{s:read(BASE/d/(s+'.json')) for s in ['SMOKE','CAPTURE_BARRIER']} for d in DS}
    write(PUB/'RESOURCES.json',dict(workers=workers,CPU_readout_seconds=seal['CPU_wall_seconds'],CPU_diagnosis_seconds=time.time()-tick,
        CPU_intake_seconds=read(BASE/'INTAKE.json')['seconds'],encoder_inputs=sum(workers[d][s]['new_encoder_inputs'] for d in DS for s in workers[d]),
        backbone_offset_forwards=sum(workers[d][s]['new_backbone_offset_forwards'] for d in DS for s in workers[d]),
        expert_calls=0,probe_refits=0,backward_calls=0,parameter_updates=0,new_candidates=0,
        original_and_generic_reused_cells=288,CUDA_initialized_in_scoring=torch.cuda.is_initialized()))
    st('completed_pending_root_audit_report_publication',original_cells=288,controlled_cells=seal['controlled_cells'])


if __name__=='__main__':
    try:
        action=sys.argv[1]
        if action=='prepare':prepare()
        elif action=='smoke':smoke(sys.argv[2])
        elif action=='capture':capture(sys.argv[2])
        elif action=='seal':seal()
        elif action=='diagnose':diagnose()
        else:raise ValueError(action)
    except BaseException:
        write(BASE/'failures'/f'{time.time_ns()}.json',dict(time=time.time(),action=sys.argv[1:],traceback=traceback.format_exc()))
        st('failed',action=sys.argv[1:]);raise
