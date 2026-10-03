"""Finite same-video query intervention with frozen atlas probes and support."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,time,json,hashlib,copy,gc,traceback,collections,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status as atomic_status
from scripts.tastvg_query_swap_math_v1 import donor_map,caption_norm,readout,focused_models
from scripts.tastvg_latent_quality_math_v1 import interval_features,geometry_features
from scripts.tastvg_information_atlas_math_v1 import (
    frame_labels,candidate_labels,moments,bootstrap_summary,paired_difference)
BASE=ROOT/'artifacts/tastvg_query_swap_specificity_v1'
PUBLIC=ROOT/'results/tastvg_query_swap_specificity/2026-10-03'
ATLAS=ROOT/'artifacts/tastvg_temporal_information_atlas_v1'
LATENT=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
PLAN=ROOT/'artifacts/tastvg_current_correction_views_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
DATASETS=['vidstg','hc2']
OWN=['scripts/run_tastvg_query_swap_v1.py','scripts/tastvg_query_swap_math_v1.py',
     'scripts/test_tastvg_query_swap_v1.py','protocols/tastvg_query_swap_specificity_v1.md']
DEPS=['scripts/tastvg_information_atlas_math_v1.py','scripts/tastvg_latent_quality_math_v1.py',
      'vg_tta/tastvg_temporal_latent_quality_v1.py','scripts/run_tastvg_paper48_p5_online_v1.py',
      'scripts/run_tastvg_full_b1_experts_v1.py','methods/decota_final_simplified_v1/backbone.py',
      'vg_tta/tastvg_spatial_online_opd_s1_v1.py','vg_tta/tastvg_causal_round2_v1.py',
      'scripts/run_spatial_regression_alignment_v1.py','scripts/run_tastvg_evidence_vulnerability_v2.py',
      'vg_tta/exact_frame_decode_audit_v2.py','vg_tta/tastvg_paper48_hc2_decode_v1.py',
      'vg_tta/tastvg_deployment_corruption_v2.py','scripts/tastvg_correction_views_common_v1.py']

def st(s,**x):atomic_status(BASE/'STATUS.json',dict(status=s,time=time.time(),pid=os.getpid(),**x))
def prefix(c):return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def checked(p):
    assert sha(p)==read(p.with_suffix('.json'))['sha256'],str(p)
    return load(p)
def commit(p,x):
    save(p,x);write(p.with_suffix('.json'),dict(sha256=sha(p),time=time.time(),
        GT_read=False,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
def verify(payloads=False):
    r=read(BASE/'RUNTIME_LOCK.json');pins=dict(r['code'])
    for p in sorted((BASE/'revisions').glob('*.json')):
        z=read(p);assert not z['science_changed'];pins.update(z['pin_overrides'])
    for p,h in {**pins,**r['metadata']}.items():assert sha(ROOT/p)==h,p
    if payloads:
        for p,h in r['payloads'].items():assert sha(ROOT/p)==h,p
    return r
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        p=args[0].decode() if isinstance(args[0],bytes) else args[0]
        if any(s in p for s in ['/annos/','/annotations/','GT_LABELS','SOURCE_GT',
                '/ROWS.json','/SUMMARY.json','GT_SUBSET','GT_EXPOSURE','test_annotations.json']):
            raise PermissionError('Query-swap inference/readout forbids annotation and labelled-result access')

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    assert read(ATLAS/'FINAL_COMPLETION.json')['commit']=='fd57375d91933d1509ec914b5ff5bf6a040ad5dc'
    allcells=[c for c in read(LATENT/'COHORT.json')['cells'] if c['scheduled']]
    assert len(allcells)==288
    payloads={};metadata={};donors={};smokes={};cohort=[];maps=[]
    from scripts.tastvg_correction_views_common_v1 import oldfile
    for ds in DATASETS:
        rows=read(PLAN/ds/'PLAN.json')['rows'];assert len(rows)==48
        mapping,offset=donor_map(rows,ds);donors[ds]={}
        for i,d in sorted(mapping.items()):
            p=POOL/ds/'subjects'/f'{d:05}.json';sub=read(p)
            assert sub['caption_sha256']==hashlib.sha256(rows[d]['input']['caption'].encode()).hexdigest()
            donors[ds][str(i)]=dict(donor_index=d,caption=rows[d]['input']['caption'],
                subject=sub['parses']['subject'],source=rows[d]['source'])
            metadata[str(p.relative_to(ROOT))]=sha(p)
            maps.append(dict(dataset=ds,source_index=i,donor_index=d,cyclic_offset=offset,
                true_caption_sha256=hashlib.sha256(rows[i]['input']['caption'].encode()).hexdigest(),
                swap_caption_sha256=hashlib.sha256(rows[d]['input']['caption'].encode()).hexdigest(),
                different_source=True,different_video_hash=True,different_normalized_caption=True,
                true_word_count=len(rows[i]['input']['caption'].split()),
                swap_word_count=len(rows[d]['input']['caption'].split())))
        cells=[c for c in allcells if c['dataset']==ds];assert len(cells)==144
        smokes[ds]=[prefix(c) for c in sorted([c for c in cells if c['condition']=='clean'],
            key=lambda c:hashlib.sha256(('query-swap-smoke|'+ds+'|'+prefix(c)).encode()).hexdigest())[:2]]
        for c in cells:
            f=LATENT/ds/'target_features'/f'{prefix(c)}.pt';z=checked(f)
            assert z['source_id']==c['parent'] and z['frame_ids']==rows[c['parent']]['frame_ids']
            assert z['pixel_sha256']==c['pixel_sha256'] and len(z['candidate_indices'])==32
            af=oldfile(ds,c['split'],c['condition'],c['order'],c['arrival'])
            assert sha(af)==read(af.with_suffix('.json'))['sha256']
            for p in [f,af]:payloads[str(p.relative_to(ROOT))]=sha(p)
            sf=POOL/ds/'subjects'/f'{c["parent"]:05}.json';metadata[str(sf.relative_to(ROOT))]=sha(sf)
            cohort.append(dict(**c,feature=str(f.relative_to(ROOT)),feature_sha256=sha(f),
                state_payload=str(af.relative_to(ROOT)),state_payload_sha256=sha(af),
                query_swap_donor=mapping[c['parent']]))
        for p in [ATLAS/ds/'FROZEN_ATLAS.pt',ATLAS/ds/'SEALED_READOUT.pt']:
            assert sha(p)==read(p.with_suffix('.json'))['sha256'];payloads[str(p.relative_to(ROOT))]=sha(p)
        for p in [PLAN/ds/'PLAN.json',ATLAS/ds/'FROZEN_ATLAS.json']:
            metadata[str(p.relative_to(ROOT))]=sha(p)
    write(BASE/'PRIVATE_QUERY_DONORS.json',donors);write(BASE/'COHORT.json',dict(cells=cohort,smokes=smokes))
    write(PUBLIC/'QUERY_MAPPING.json',maps)
    checkpoints=read(ROOT/'results/tastvg_temporal_information_atlas/2026-10-03/CONFIG.json')['source_checkpoint_sha256']
    cfg=dict(version='tastvg_query_swap_specificity_v1',predecessor_commit='fd57375d91933d1509ec914b5ff5bf6a040ad5dc',
        cells=288,clean_cells=48,corrupt_cells=240,target_sources={'vidstg':24,'hc2':21},
        target_panels={'vidstg':{'search':16,'confirm':8},'hc2':{'search':14,'confirm':7}},
        original_panel_sources_each_dataset={'search':32,'confirm':16},
        unique_swap_encoder_inputs={'vidstg':144,'hc2':126},query_derangement='SHA source order / first valid cyclic shift',
        donor_GT_screening=False,donor_mismatch_certified=False,donors_fixed_in_bootstrap=True,
        source_checkpoint_sha256=checkpoints,frame_task='event',candidate_tasks=['precision','recall','tiou'],
        candidate_views=['Endpoint','Inside','Context','Contrast','Full','Geometry'],
        source_frozen_probes={ds:sha(ATLAS/ds/'FROZEN_ATLAS.pt') for ds in DATASETS},
        probes_retrained=0,focused_probes_per_dataset=40,support=32,old8_preserved=True,
        A_states_unchanged=True,expert_schedule_unchanged=True,new_expert_calls=0,new_candidates=0,
        GT_for_donor_or_probe_selection=False,bootstrap_draws=10000,seed=20261003,
        primary=['frame/Hidden/event/real:auc','candidate/Full/precision/real:r2',
                 'candidate/Full/recall/real:r2','candidate/Full/tiou/real:r2'],
        primary_panel='corrupted expert cells, source balanced',raw_annotations_read=False,
        historical_exposure=True,no_new_top1_output=True,production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        time=time.time())
    write(PUBLIC/'CONFIG.json',cfg)
    for p in [BASE/'PRIVATE_QUERY_DONORS.json',BASE/'COHORT.json',PUBLIC/'QUERY_MAPPING.json',PUBLIC/'CONFIG.json',
              ATLAS/'FINAL_COMPLETION.json',ROOT/'methods/CURRENT_METHOD.json']:
        metadata[str(p.relative_to(ROOT))]=sha(p)
    write(BASE/'RUNTIME_LOCK.json',dict(code={p:sha(ROOT/p) for p in OWN+DEPS},metadata=metadata,
        payloads=payloads,time=time.time()))
    st('prepared_pending_no_GT_smoke',cells=288,GT_read=False)
    print('QUERY_SWAP_LOCKED',len(cohort),'cells',len(payloads),'payloads',flush=True)

def cuda_setup(ds):
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_final_simplification_v1 import lease
    install_clean_loader();sys.addaudithook(guard);l=lease()
    torch.set_num_threads(4);torch.manual_seed(20261003);np.random.seed(20261003)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from methods.decota_final_simplified_v1.tensors import state_hash
    model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False)
    assert state_hash(model.state_dict())==read(PUBLIC/'CONFIG.json')['source_checkpoint_sha256'][ds]
    return model,l

def observations(ds,row,condition):
    from vg_tta.exact_frame_decode_audit_v2 import decode as vid_decode
    from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as hc_decode
    if ds=='hc2':
        from vg_tta import exact_frame_decode_audit_v2 as binding
        binding.decode=hc_decode
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    frames,ids=(vid_decode if ds=='vidstg' else hc_decode)(row['input'])
    assert ids==row['frame_ids']
    shifted,pixel,spec=observation(row,condition,frames)
    return shifted,pixel,spec

def input_data(model,actor,frames,row,query,subject):
    from scripts.run_tastvg_paper48_p5_online_v1 import source_capture
    r=copy.deepcopy(row);r['input']['caption']=query
    actor.restore(actor.initial)
    return source_capture(model,frames,r,subject)

def replay(actor,data,c,original):
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from vg_tta.tastvg_temporal_latent_quality_v1 import observe
    from methods.decota_final_simplified_v1.tensors import state_hash
    old=load(ROOT/c['state_payload']);assert old['pre_sha']==original['A_state_pre_sha256']
    actor.restore(device_tree(old['pre_state'],'cuda'));before=state_hash(actor.state())
    assert before==c['pre_sha'];h,p,_=observe(actor,device_tree(data,'cuda'),False)
    assert state_hash(actor.state())==before
    return h,p

def smoke(ds):
    verify(True);tick=time.time();model,lease=cuda_setup(ds)
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor
    from methods.decota_final_simplified_v1.tensors import state_hash
    actor=SpatialActor(model);initial=state_hash(model.state_dict());out=[]
    co=read(BASE/'COHORT.json');rows=read(PLAN/ds/'PLAN.json')['rows'];donors=read(BASE/'PRIVATE_QUERY_DONORS.json')[ds]
    try:
        cells=[c for c in co['cells'] if c['dataset']==ds and prefix(c) in co['smokes'][ds]]
        for c in cells:
            z=checked(ROOT/c['feature']);r=rows[c['parent']];frames,pixel,_=observations(ds,r,c['condition'])
            assert pixel==z['pixel_sha256']
            subject=read(POOL/ds/'subjects'/f'{c["parent"]:05}.json')['parses']['subject']
            data=input_data(model,actor,frames,r,r['input']['caption'],subject)
            h,p=replay(actor,data,c,z);old=load(ROOT/c['state_payload'])
            x,ctx=interval_features(h.numpy(),z['frame_ids'],z['candidate_indices'],r['input']['fps'])
            assert torch.equal(h,z['hidden']), 'True-query recomputation hidden parity'
            assert np.array_equal(x,z['x']) and ctx==z['context'], 'True feature parity'
            assert torch.equal(p['boxes'],old['slow']['boxes']), 'True A spatial output parity'
            assert p['indices']==old['slow']['indices']
            true_h=h.clone();del data,h,p;gc.collect();torch.cuda.empty_cache()
            d=donors[str(c['parent'])];data=input_data(model,actor,frames,r,d['caption'],d['subject'])
            h,p=replay(actor,data,c,z)
            assert h.shape==true_h.shape and torch.isfinite(h).all()
            commit(BASE/ds/'smoke'/f'{prefix(c)}.pt',dict(hidden_true=true_h,hidden_swap=h,
                pixel_sha256=pixel,cell_key=z['cell_key'],GT_read=False))
            out.append(dict(cell=prefix(c),pixel_parity=True,hidden_bitwise_parity=True,
                feature_bitwise_parity=True,A_spatial_bitwise_parity=True,A_interval_parity=True,
                head_identity_checks=4,swap_hidden_changed=not torch.equal(true_h,h),
                no_GT=True,no_candidates=True,no_update=True))
            print('QUERY_SWAP_SMOKE',ds,len(out),2,flush=True)
            del data,frames,h;gc.collect();torch.cuda.empty_cache()
        actor.close();assert state_hash(model.state_dict())==initial
        write(BASE/ds/'SMOKE.json',dict(status='passed',rows=out,GT_read=False,model_restored=True,
            worker_wall_seconds=time.time()-tick,new_encoder_inputs=4,new_backbone_offset_forwards=8,
            latent_replay_calls=4,head_identity_checks=8,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),time=time.time()))
        st('smoke_complete_pending_root_acceptance',dataset=ds,GT_read=False)
    finally:actor.close();lease.close()

def capture(ds):
    verify(True);assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='passed'
    tick=time.time();model,lease=cuda_setup(ds)
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor
    from methods.decota_final_simplified_v1.tensors import state_hash
    actor=SpatialActor(model);initial=state_hash(model.state_dict());done=enc=0
    rows=read(PLAN/ds/'PLAN.json')['rows'];donors=read(BASE/'PRIVATE_QUERY_DONORS.json')[ds]
    cells=[c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds]
    cells.sort(key=lambda c:(c['parent'],c['condition'],c['split'],c['order'],c['arrival']))
    cached=None;data=None;frames=None
    try:
        for c in cells:
            assert shutil.disk_usage(ROOT).free>8*2**30
            f=BASE/ds/'swap_features'/f'{prefix(c)}.pt'
            assert not f.exists(), 'Preserve existing run; no implicit retry'
            z=checked(ROOT/c['feature']);r=rows[c['parent']];d=donors[str(c['parent'])]
            marker=(c['parent'],c['condition'])
            if marker!=cached:
                del data,frames;gc.collect();torch.cuda.empty_cache()
                frames,pixel,spec=observations(ds,r,c['condition']);assert pixel==z['pixel_sha256']
                data=input_data(model,actor,frames,r,d['caption'],d['subject']);cached=marker;enc+=1
            assert data['frame_ids']==z['frame_ids']
            h,_=replay(actor,data,c,z)
            x,ctx=interval_features(h.numpy(),z['frame_ids'],z['candidate_indices'],r['input']['fps'])
            geom=geometry_features(z['frame_ids'],z['candidate_indices']);assert np.array_equal(geom,z['geometry'])
            payload={k:z[k] for k in ['cell_key','dataset','split','source_id','condition','order','arrival',
                'anchor_index','frame_ids','candidate_indices','intervals','A_state_pre_sha256',
                'A_state_post_sha256','pixel_sha256']}
            payload.update(hidden=h,x=x,geometry=geom,context=ctx,donor_index=d['donor_index'],
                donor_caption_sha256=hashlib.sha256(d['caption'].encode()).hexdigest(),
                true_feature_sha256=c['feature_sha256'],candidate_support_changed=False,
                GT_read=False,parameter_updates=0,encoder_cache_key=list(marker))
            commit(f,payload);done+=1
            st('swap_capture_running',dataset=ds,done=done,total=144,encoder_inputs=enc,GT_read=False)
            if done%12==0:print('QUERY_SWAP_CAPTURE',ds,done,144,'ENCODERS',enc,flush=True)
            del h,x,payload;gc.collect()
        actor.close();assert state_hash(model.state_dict())==initial
        assert done==144 and enc==read(PUBLIC/'CONFIG.json')['unique_swap_encoder_inputs'][ds]
        files={str(p.relative_to(BASE)):sha(p) for p in sorted((BASE/ds/'swap_features').glob('*.pt'))}
        write(BASE/ds/'CAPTURE_BARRIER.json',dict(status='completed',files=files,cells=done,
            GT_read=False,model_restored=True,time=time.time(),worker_wall_seconds=time.time()-tick,
            new_encoder_inputs=enc,new_backbone_offset_forwards=enc*2,latent_replay_calls=done,
            head_identity_checks=done*2,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated()))
        st('swap_capture_complete_pending_next_stage',dataset=ds,done=done,GT_read=False)
    finally:actor.close();lease.close()

def seal():
    verify(True);assert not torch.cuda.is_initialized();sys.addaudithook(guard);tick=time.time();files={}
    for ds in DATASETS:
        barrier=read(BASE/ds/'CAPTURE_BARRIER.json');assert len(barrier['files'])==144
        models=checked(ATLAS/ds/'FROZEN_ATLAS.pt')['models']
        oldscores={r['cell']:r['predictions'] for r in checked(ATLAS/ds/'SEALED_READOUT.pt')}
        rows=[]
        for c in read(BASE/'COHORT.json')['cells']:
            if c['dataset']!=ds:continue
            z=checked(ROOT/c['feature']);sw=checked(BASE/ds/'swap_features'/f'{prefix(c)}.pt')
            tr=readout(z,models);sp=readout(sw,models)
            for k,v in tr.items():assert np.array_equal(v,oldscores[z['cell_key']][k]),k
            for k,v in tr.items():
                if '/Geometry/' in k:assert np.array_equal(v,sp[k]),'Fixed geometry control'
            rows.append(dict(cell=z['cell_key'],true=tr,swap=sp))
        assert len(rows)==144;f=BASE/ds/'SEALED_READOUT.pt';commit(f,rows);files[ds]=sha(f)
        print('QUERY_SWAP_READOUT_SEALED',ds,len(rows),flush=True)
    receipt=dict(status='all_query_swap_readouts_sealed',time=time.time(),files=files,
        source_probe_hashes=read(PUBLIC/'CONFIG.json')['source_frozen_probes'],
        GT_read=False,probes_retrained=0,cells=288,CPU_wall_seconds=time.time()-tick,
        candidate_support_changed=False)
    write(BASE/'GLOBAL_READOUT_SEAL.json',receipt);write(PUBLIC/'GLOBAL_READOUT_SEAL.json',receipt)
    st('sealed_pending_original_GT_join',GT_read=False)

def metric_rows(ds):
    # Called only after all prediction barriers; GT is the ORIGINAL query's span.
    gts={s:read(POOL/ds/f'GT_LABELS_{s}.json') for s in ['search','confirm']}
    readouts={r['cell']:r for r in checked(BASE/ds/'SEALED_READOUT.pt')}
    frozen=checked(ATLAS/ds/'FROZEN_ATLAS.pt');rows=[]
    for c in read(BASE/'COHORT.json')['cells']:
        if c['dataset']!=ds:continue
        z=checked(ROOT/c['feature']);gt=gts[c['split']][str(c['parent'])]['span'];ids=np.asarray(z['frame_ids'])
        labels={'frame':frame_labels(ids,gt), 'candidate':candidate_labels(
            [[ids[i],ids[j]+1] for i,j in z['candidate_indices']],gt,z['anchor_index'])}
        scores=readouts[z['cell_key']];metrics={}
        for arm in ['true','swap']:
            for name,p in scores[arm].items():
                if name.endswith('/logit'):continue
                fam,view,task,control=name.split('/');y=labels[fam][task]
                metrics[arm+'/'+name]=moments(y,p,task=='event',scores[arm].get(name+'/logit'))
            for family,tasks in [('frame',['event']),('candidate',['precision','recall','tiou'])]:
                for task in tasks:
                    y=labels[family][task];p=np.full(len(y),frozen['null'][family+'/'+task])
                    metrics[arm+'/'+family+'/Null/'+task+'/real']=moments(y,p,task=='event')
        rows.append(dict(dataset=ds,panel=c['split'],source_index=c['parent'],order=c['order'],
            condition=c['condition'],cell=z['cell_key'],frames=len(ids),candidates=32,
            donor_index=c['query_swap_donor'],event_frames=int(labels['frame']['event'].sum()),
            event_both_classes=len(np.unique(labels['frame']['event']))==2,metrics=metrics))
    return rows

def groups(rows):
    out={}
    for mode in ['corrupt','clean']:
        rr=[r for r in rows if (r['condition']=='clean')==(mode=='clean')]
        out['target_'+mode]=rr
        for panel in ['search','confirm']:out['target_'+mode+'_'+panel]=[r for r in rr if r['panel']==panel]
        for order in sorted({r['order'] for r in rr}):out['target_'+mode+'_'+order]=[r for r in rr if r['order']==order]
    return out

def summarize(rows):
    summary={};names=list(rows[0]['metrics'])
    for label,rr in groups(rows).items():
        coverage=dict(cells=len(rr),sources=len({r['source_index'] for r in rr}),
            frames=sum(r['frames'] for r in rr),candidates=sum(r['candidates'] for r in rr),
            event_single_class=sum(not r['event_both_classes'] for r in rr))
        metrics={n:bootstrap_summary(rr,n) for n in names};paired={}
        for name in names:
            if not name.startswith('true/'):continue
            task=name.split('/')[3];fields=['auc','ap','logloss','r2','mse'] if task=='event' else ['r2','mse','mae','within_r2']
            paired[name[5:]]={f:paired_difference(rr,name,'swap/'+name[5:],f) for f in fields}
        summary[label]=dict(coverage=coverage,metrics=metrics,paired_true_minus_swap=paired)
        print('QUERY_SWAP_SUMMARY',rows[0]['dataset'],label,len(rr),flush=True)
    return summary

def diagnose():
    verify(True);assert not torch.cuda.is_initialized();torch.set_num_threads(2);tick=time.time()
    seal=read(BASE/'GLOBAL_READOUT_SEAL.json');assert seal['cells']==288 and not seal['GT_read']
    for ds,h in seal['files'].items():assert sha(BASE/ds/'SEALED_READOUT.pt')==h
    join_at=time.time();coverage={}
    for ds in DATASETS:
        rows=metric_rows(ds);write(PUBLIC/ds/'ROWS.json',rows);write(PUBLIC/ds/'SUMMARY.json',summarize(rows))
        coverage[ds]=dict(cells=len(rows),sources=len({r['source_index'] for r in rows}))
    write(PUBLIC/'LABEL_JOIN.json',dict(time=join_at,readout_seal_time=seal['time'],
        only_original_query_cached_span=True,raw_annotations_read=False,donor_GT_read=False,
        target_GT_used_for_selection=False,historical_exposure=True))
    rs={ds:{stage:read(BASE/ds/(stage+'.json')) for stage in ['SMOKE','CAPTURE_BARRIER']} for ds in DATASETS}
    write(PUBLIC/'RESOURCES.json',dict(workers=rs,CPU_readout_seconds=seal['CPU_wall_seconds'],
        CPU_diagnosis_seconds=time.time()-tick,total_new_encoder_inputs=sum(
            rs[d][s]['new_encoder_inputs'] for d in DATASETS for s in ['SMOKE','CAPTURE_BARRIER']),
        total_new_backbone_offset_forwards=sum(rs[d][s]['new_backbone_offset_forwards'] for d in DATASETS for s in ['SMOKE','CAPTURE_BARRIER']),
        new_experts=0,training_calls=0,backward_calls=0,new_candidates=0,
        original_readout_reused_cells=288,CUDA_initialized_in_scoring=torch.cuda.is_initialized()))
    st('completed_pending_root_audit_report_publication',coverage=coverage)

def main():
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='smoke':smoke(sys.argv[2])
    elif action=='capture':capture(sys.argv[2])
    elif action=='seal':seal()
    elif action=='diagnose':diagnose()
    else:raise ValueError(action)
if __name__=='__main__':
    try:main()
    except BaseException:
        if BASE.exists():
            p=BASE/'failures'/f'{time.time_ns()}.json'
            write(p,dict(action=sys.argv[1:],time=time.time(),traceback=traceback.format_exc()))
            st('failed',action=sys.argv[1:],failure=str(p.relative_to(BASE)))
        raise
