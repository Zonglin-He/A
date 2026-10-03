"""Source-only linear fitting, then immutable cached target suffix readout."""
import os,sys,time,json,hashlib,collections,traceback,gc,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,save,load,status as atomic_status
from scripts.tastvg_latent_quality_math_v1 import *
BASE=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
PUBLIC=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
PRIOR=ROOT/'artifacts/tastvg_temporal_boundary_support_v1'
LABELS=ROOT/'results/tastvg_temporal_boundary_support/2026-10-03'
OLD=ROOT/'artifacts/tastvg_current_correction_views_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
SCALAR=ROOT/'results/tastvg_large_correction_evidence/2026-10-03'
DATASETS=['vidstg','hc2']; ARMS=['L8','L32','G8','G32']
OWN=['scripts/run_tastvg_temporal_latent_quality_v1.py',
 'scripts/tastvg_latent_quality_math_v1.py','vg_tta/tastvg_temporal_latent_quality_v1.py',
 'scripts/test_tastvg_latent_quality_v1.py','protocols/tastvg_temporal_latent_quality_v1.md',
 'docs/tastvg_temporal_latent_quality_v1/EXECUTION.md']

def st(s,**x):atomic_status(BASE/'STATUS.json',dict(status=s,time=time.time(),pid=os.getpid(),**x))
def prefix(c):return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def key(c):return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])
def predfile(c):return PRIOR/c['dataset']/'predictions'/f'{prefix(c)}.json'
def canonical(source,ds):return str(source)
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
    p=str(args[0])
    if any(s in p for s in ['/annos/','/annotations/','SOURCE_GT.json','GT_LABELS',
          '/ROWS.json','/SUMMARY.json','EVIDENCE_ROWS.json','ORACLE_ROWS.json']):
        raise PermissionError('Latent inference cannot read annotations or labelled results')

def verify():
    r=read(BASE/'RUNTIME_LOCK.json')
    for f in sorted((BASE/'revisions').glob('*.json')):
        z=read(f);assert z['science_changed'] is False;r['pins'].update(z['pin_overrides'])
    for f,h in {**r['pins'],**r['metadata']}.items():assert sha(ROOT/f)==h,f
    return r

def commit(f,value):
    save(f,value);write(f.with_suffix('.json'),dict(sha256=sha(f),time=time.time(),
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))

def checked(f):
    r=read(f.with_suffix('.json'));assert sha(f)==r['sha256'];return load(f)

def prepare():
    import ijson
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    cells=read(PRIOR/'COHORT.json')['cells'];assert len(cells)==1152
    assert sum(c['scheduled'] for c in cells)==288
    assert read(ROOT/'artifacts/tastvg_large_correction_evidence_v1/FINAL_COMPLETION.json')['commit']=='83023d8850607ef508a6d56c8d5b4349b17dc7d5'
    from scripts.evaluate_fullspan_scale_shift_corruptions_v1 import _sample_frame_ids
    from scripts.tastvg_paper48_p5_common_v1 import frame_ids as hc_ids
    rows={};source_gt={};intake={};source_pins={}
    vid=read(ROOT/'artifacts/desta3d_v1/SOURCE_INITIAL_INPUTS.json')
    hcpath=ROOT/'artifacts/decota_paper_execution_20260917/vitta/hc2_to_vid/SOURCE_MANIFEST.json'
    hc=read(hcpath);assert hc['official_split']=='train'
    validation=set(sorted({r['input']['source'] for r in hc['records']},
        key=lambda s:hashlib.sha256(('temporal-latent-v1|'+s).encode()).hexdigest())[:16])
    source_pins[str(hcpath.relative_to(ROOT))]=sha(hcpath)
    for ds,raw in [('vidstg',vid),('hc2',hc['records'])]:
        target=read(OLD/ds/'PLAN.json')['rows']
        excluded_sources={canonical(r['source'],ds) for r in target}
        target_media={r['input']['video_sha256'] for r in target}
        rr=[]
        for i,r in enumerate(raw):
            q=copy.deepcopy(r['input']);q.pop('video_id',None)
            assert canonical(q['source'],ds) not in excluded_sources
            assert q['video_sha256'] not in target_media
            assert Path(q['video_path']).is_file()
            ids=_sample_frame_ids(q) if ds=='vidstg' else hc_ids(q['frame_count'])
            assert len(ids)>=9;q['frame_ids']=ids
            if ds=='hc2':q.update(fps=q['frame_count']/20.,duration=20.)
            split=r['split'] if ds=='vidstg' else ('validation' if q['source'] in validation else 'train')
            rr.append(dict(index=i,source=q['source'],split=split,input=q,frame_ids=ids,
                official_annotation_key=r.get('official_annotation_key')))
        assert len({canonical(r['source'],ds) for r in rr})==len(rr)
        rows[ds]=rr;source_gt[ds]={}
        annotation=ROOT/'data/attribute_tta_official_release'/('vidstg' if ds=='vidstg' else 'hc-stvg2')/'annos/train.json'
        lookup={(r['source'],r['input']['caption'].lower(),r['input'].get('start_frame'),
                 r['input'].get('end_frame')):r['index'] for r in rr}
        wanted={r['official_annotation_key']:r['index'] for r in rr} if ds=='hc2' else {}
        parsed=retained=0
        with annotation.open('rb') as f:
            for k,a in ijson.kvitems(f,'',use_float=True):
                parsed+=1
                if ds=='vidstg':
                    if a['qtype']!='declar':continue
                    at=lookup.get((a['vid'],a['sentence']['description'].lower(),
                        int(a['used_segment']['begin_fid']),int(a['used_segment']['end_fid'])))
                    if at is None:continue
                    event=[int(a['ori_temp_gt']['begin_fid']),int(a['ori_temp_gt']['end_fid'])+1]
                else:
                    at=wanted.get(k)
                    if at is None:continue
                    assert a['English'].lower()==rr[at]['input']['caption'].lower()
                    start=int(a['st_frame'])-1;event=[start,start+len(a['bbox'])]
                    assert event[1]==int(a['ed_frame'])
                assert event[0]<event[1]
                if str(at) in source_gt[ds]:assert source_gt[ds][str(at)]==event
                source_gt[ds][str(at)]=event;retained+=1
        assert len(source_gt[ds])==len(rr),(ds,len(source_gt[ds]),len(rr))
        intake[ds]=dict(annotation_sha256=sha(annotation),parsed_source_training_rows=parsed,
            matching_records=retained,retained_sources=len(rr),target_source_overlap=0,target_media_overlap=0)
        source_pins[str(annotation.relative_to(ROOT))]=sha(annotation)
    from methods.decota_final_simplified_v1.observations import QuerySubjectParser
    parser=QuerySubjectParser(ROOT/'.cache/stanza')
    for ds in DATASETS:
        for i,r in enumerate(rows[ds]):
            r['subject']=parser(r['input']['caption'])['subject']
            assert sha(r['input']['video_path'])==r['input']['video_sha256']
            if i%32==0:print('SOURCE_METADATA',ds,i,len(rows[ds]),flush=True)
    metadata={};bindings={}
    for c in cells:
        if not c['scheduled']:continue
        ds=c['dataset'];rf=POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json';r=read(rf)
        assert r['pixel_sha256']==c['pixel_sha256'];f=POOL/ds/r['cache']
        bindings[str(f.relative_to(ROOT))]=r['sha256']
        from scripts.tastvg_correction_views_common_v1 import oldfile
        af=oldfile(ds,c['split'],c['condition'],c['order'],c['arrival'])
        bindings[str(af.relative_to(ROOT))]=read(af.with_suffix('.json'))['sha256']
        for f in [predfile(c),rf]:metadata[str(f.relative_to(ROOT))]=sha(f)
    write(BASE/'SOURCE_INPUTS.json',rows);write(BASE/'SOURCE_GT.json',source_gt)
    write(BASE/'TARGET_INPUT_BINDINGS.json',bindings);write(BASE/'COHORT.json',dict(cells=cells))
    counts={ds:{sp:sum(r['split']==sp for r in rows[ds]) for sp in ['train','validation']} for ds in DATASETS}
    cfg=dict(version='tastvg_temporal_latent_quality_v1',predecessor_commit='83023d8850607ef508a6d56c8d5b4349b17dc7d5',
        source_counts=counts,source_GT=True,target_GT_training=False,target_GT_selection=False,
        source_intake=intake,source_supervision='physical tIoU of fixed native/maximin source candidates',
        probe='1792D ridge with intercept, source training standardization',geometry_control='normalized start/end/length',
        window_seconds=1.,alphas=ALPHAS,source_selection='validation top1 tIoU then MSE then smallest alpha',
        checkpoint_state_sha256={d:read(POOL/d/'CAPTURE_BARRIER.json')['checkpoint_state_sha256'] for d in DATASETS},
        hidden='sixth temporal decoder layer input to temp_embed after second native pass',
        targets=dict(arrivals=1152,expert=288,nonexpert=864,clean_expert=48,corrupt_expert=240),
        supports=[8,32],arms=ARMS,decision='unique strict improvement over fixed A8 else A8',
        large_radius=.5,dominant_balance=.25,bootstrap=10000,seed=20261003,
        target_historical_exposure=True,production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        private_weights_and_latents=True,new_experts=0,temporal_parameter_updates=0,time=time.time())
    write(PUBLIC/'CONFIG.json',cfg)
    for f in [BASE/'SOURCE_INPUTS.json',BASE/'SOURCE_GT.json',BASE/'TARGET_INPUT_BINDINGS.json',
              BASE/'COHORT.json',PUBLIC/'CONFIG.json',ROOT/'methods/CURRENT_METHOD.json']:
        metadata[str(f.relative_to(ROOT))]=sha(f)
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in OWN},metadata=metadata,
        source_annotation_pins=source_pins,time=time.time()))
    st('prepared_pending_source_capture',source_counts=counts,target_GT_read=False)
    print('LATENT_PROTOCOL_LOCKED',counts,flush=True)

def cuda_setup(ds):
    import torch
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_final_simplification_v1 import lease
    install_clean_loader();sys.addaudithook(guard)
    l=lease();torch.set_num_threads(4);torch.manual_seed(20261003)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from scripts.run_spatial_regression_alignment_v1 import model_load
    model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False)
    from methods.decota_final_simplified_v1.tensors import state_hash
    assert state_hash(model.state_dict())==read(PUBLIC/'CONFIG.json')['checkpoint_state_sha256'][ds]
    return model,l

def capture_source(ds):
    verify();tick=time.time();model,lease=cuda_setup(ds)
    import torch
    from scripts.run_tastvg_paper48_p5_online_v1 import source_capture
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor
    from vg_tta.tastvg_temporal_latent_quality_v1 import observe
    from vg_tta.tastvg_temporal_boundary_support_v1 import expanded
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.exact_frame_decode_audit_v2 import decode as vid_decode
    from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as hc_decode
    rows=read(BASE/'SOURCE_INPUTS.json')[ds];mh=state_hash(model.state_dict());done=0;actor=None
    try:
        for r in rows:
            f=BASE/ds/'source_features'/f'{r["index"]:04}.pt'
            if f.with_suffix('.json').exists():checked(f);done+=1;continue
            st('source_capture_running',dataset=ds,done=done,total=len(rows),target_GT_read=False)
            frames,ids=(vid_decode if ds=='vidstg' else hc_decode)(r['input']);assert ids==r['frame_ids']
            data=source_capture(model,frames,r,r['subject']);actor=SpatialActor(model)
            h,p,old=observe(actor,device_tree(data,'cuda'),True)
            assert torch.equal(p['boxes'],data['prediction']['boxes'])
            assert all(torch.equal(a,b) for a,b in zip(p['logits'],data['prediction']['logits']))
            cs=expanded(old,ids);pairs=[x['indices'] for x in cs]
            x,context=interval_features(h.numpy(),ids,pairs,r['input']['fps'])
            geom=geometry_features(ids,pairs)
            commit(f,dict(dataset=ds,index=r['index'],split=r['split'],hidden=h,frame_ids=ids,
                candidates=cs,x=x,geometry=geom,context=context,head_exact=True,
                checkpoint_state_sha256=mh,source_video_sha256=r['input']['video_sha256'],GT_read=False))
            actor.close();actor=None;done+=1
            print('SOURCE_LATENT',ds,done,len(rows),flush=True)
            del data,h,x,frames,p;gc.collect();torch.cuda.empty_cache()
        assert state_hash(model.state_dict())==mh
        files={str(f.relative_to(BASE)):sha(f) for f in (BASE/ds/'source_features').glob('*.json')}
        assert len(files)==len(rows)
        write(BASE/ds/'SOURCE_FEATURE_BARRIER.json',dict(files=files,queries=done,time=time.time(),
            model_restored=True,GT_read=False,worker_wall_seconds=time.time()-tick,
            new_backbone_offset_forwards=2*done,native_suffix_calls=2*done,head_identity_checks=2*done,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated()))
        st('source_capture_complete_pending_next_stage',dataset=ds,done=done,target_GT_read=False)
    finally:
        if actor:actor.close()
        lease.close()

def fit():
    verify();tick=time.time();labels=read(BASE/'SOURCE_GT.json');summary={};barriers={}
    for ds in DATASETS:
        r=read(BASE/ds/'SOURCE_FEATURE_BARRIER.json');barriers[ds]=sha(BASE/ds/'SOURCE_FEATURE_BARRIER.json')
        items=[checked(BASE/f.replace('.json','.pt')) for f in sorted(r['files'])]
        rr=read(BASE/'SOURCE_INPUTS.json')[ds]
        yy=[temporal_iou([c['physical_interval'] for c in z['candidates']],labels[ds][str(z['index'])]) for z in items]
        train=[j for j,z in enumerate(items) if z['split']=='train'];val=[j for j,z in enumerate(items) if z['split']=='validation']
        y=np.concatenate([yy[j] for j in train]);vy=np.concatenate([yy[j] for j in val]);groups=np.repeat(val,32)
        models={};details={}
        for name,field in [('L','x'),('G','geometry')]:
            xx=np.concatenate([items[j][field] for j in train]);vx=np.concatenate([items[j][field] for j in val])
            model,path,at=fit_path(xx,y,vx,vy,groups);models[name]=model
            details[name]=dict(path=path,selected_index=at,selected_alpha=model['alpha'],
                training_queries=len(train),validation_queries=len(val),dimensions=xx.shape[1],
                training_candidates=len(y),validation_candidates=len(vy),
                training_mean_target=float(y.mean()),validation_mean_target=float(vy.mean()),
                validation_oracle=float(np.mean([max(yy[j]) for j in val])),
                validation_native=float(np.mean([yy[j][0] for j in val])),
                coefficients_sha256=hashlib.sha256(np.asarray(model['weight'],dtype='<f8').tobytes()).hexdigest())
        commit(BASE/ds/'FROZEN_PROBE.pt',models);summary[ds]=details
        write(BASE/ds/'SOURCE_FIT_SUMMARY.json',details)
        print('FROZEN_LINEAR_PROBE',ds,{n:details[n]['selected_alpha'] for n in models},flush=True)
    write(PUBLIC/'SOURCE_FIT_SUMMARY.json',summary)
    write(BASE/'FIT_BARRIER.json',dict(status='both_source_probes_frozen',time=time.time(),source_GT=True,
        target_GT_read=False,source_features=barriers,probes={d:sha(BASE/d/'FROZEN_PROBE.pt') for d in DATASETS},
        source_GT_sha256=sha(BASE/'SOURCE_GT.json'),CPU_wall_seconds=time.time()-tick))
    st('source_fit_complete_pending_target_replay',target_GT_read=False)

def capture_target(ds):
    verify();tick=time.time();fit=read(BASE/'FIT_BARRIER.json')
    assert fit['target_GT_read'] is False
    model,lease=cuda_setup(ds)
    import torch
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from scripts.tastvg_correction_views_common_v1 import oldfile
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor
    from vg_tta.tastvg_temporal_latent_quality_v1 import observe
    from methods.decota_final_simplified_v1.tensors import state_hash
    cfg=read(PUBLIC/'CONFIG.json');probes=checked(BASE/ds/'FROZEN_PROBE.pt')
    cells=[c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds and c['scheduled']]
    bindings=read(BASE/'TARGET_INPUT_BINDINGS.json');plan=read(OLD/ds/'PLAN.json');actor=SpatialActor(model)
    initial=state_hash(model.state_dict());done=0
    try:
        for c in cells:
            f=BASE/ds/'target_features'/f'{prefix(c)}.pt'
            if f.with_suffix('.json').exists():checked(f);done+=1;continue
            z=read(predfile(c));rf=POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json';rec=read(rf)
            cf=POOL/ds/rec['cache'];af=oldfile(ds,c['split'],c['condition'],c['order'],c['arrival'])
            for p in [cf,af]:assert sha(p)==bindings[str(p.relative_to(ROOT))]
            data=load(cf);old=load(af);assert old['pre_sha']==z['persistent_pre_sha']
            actor.restore(device_tree(old['pre_state'],'cuda'));pre=state_hash(actor.state())
            assert pre==z['persistent_pre_sha'];h,p,_=observe(actor,device_tree(data,'cuda'))
            assert torch.equal(p['boxes'],old['slow']['boxes']) and p['indices']==z['native_indices']
            assert all(torch.equal(a,b) for a,b in zip(p['logits'],data['prediction']['logits'])), 'A/source temporal head parity'
            assert state_hash(actor.state())==pre
            ids=data['frame_ids'];pairs=[q['indices'] for q in z['candidates']]
            fps=plan['rows'][c['parent']]['input']['fps'];xx,context=interval_features(h.numpy(),ids,pairs,fps)
            g=geometry_features(ids,pairs);scores={n:predict(probes[n],v).tolist() for n,v in [('L',xx),('G',g)]}
            a=z['choices']['A8'];choices={name+str(n):pick(scores[name],a,n) for name in ['L','G'] for n in [8,32]}
            payload=dict(cell_key=key(c),dataset=ds,split=c['split'],source_id=c['parent'],condition=c['condition'],
                order=c['order'],arrival=c['arrival'],anchor_index=a,frame_ids=ids,candidate_indices=pairs,
                intervals=z['intervals_normalized'],hidden=h,x=xx,geometry=g,context=context,scores=scores,choices=choices,
                A_state_pre_sha256=z['persistent_pre_sha'],A_state_post_sha256=z['persistent_post_sha'],
                source_A_temporal_bitwise_parity=True,A_spatial_bitwise_parity=True,pixel_sha256=c['pixel_sha256'],
                probe_sha256=fit['probes'][ds],GT_read=False)
            commit(f,payload);done+=1
            st('target_replay_running',dataset=ds,done=done,total=len(cells),target_GT_read=False)
            if done%12==0:print('TARGET_LATENT',ds,done,len(cells),flush=True)
            del data,old,h,p,xx;gc.collect();torch.cuda.empty_cache()
        actor.close();assert state_hash(model.state_dict())==initial
        files={str(f.relative_to(BASE)):sha(f) for f in (BASE/ds/'target_features').glob('*.json')}
        assert len(files)==144
        write(BASE/ds/'TARGET_FEATURE_BARRIER.json',dict(files=files,cells=done,time=time.time(),
            source_probe_frozen_at=fit['time'],target_GT_read=False,worker_wall_seconds=time.time()-tick,
            model_restored=True,new_backbone_forwards=0,new_suffix_replays=done,
            head_identity_checks=2*done,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated()))
        st('target_replay_complete_pending_next_stage',dataset=ds,done=done,target_GT_read=False)
    finally:
        actor.close();lease.close()

def seal():
    verify();rows=[]
    for ds in DATASETS:
        b=read(BASE/ds/'TARGET_FEATURE_BARRIER.json')
        for f,h in sorted(b['files'].items()):
            assert sha(BASE/f)==h;z=checked((BASE/f).with_suffix('.pt'))
            rows.append({k:z[k] for k in ['cell_key','dataset','split','source_id','condition','order','arrival',
                'anchor_index','candidate_indices','intervals','scores','choices','context',
                'A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256',
                'source_A_temporal_bitwise_parity','A_spatial_bitwise_parity','GT_read']})
    assert len(rows)==288
    write(PUBLIC/'SCORE_ROWS.json',rows)
    write(PUBLIC/'GLOBAL_SCORE_SEAL.json',dict(status='sealed_before_target_metric_join',time=time.time(),
        score_rows_sha256=sha(PUBLIC/'SCORE_ROWS.json'),source_fit_barrier_sha256=sha(BASE/'FIT_BARRIER.json'),
        cells=288,target_GT_read=False,source_GT_used_for_fit=True))
    st('sealed_pending_target_metric_join',target_GT_read=False)

def diagnose():
    from scripts.tastvg_large_evidence_math_v1 import geometry as operation,aggregate,binary_cell,candidate_mask,GROUPS
    verify();seal=read(PUBLIC/'GLOBAL_SCORE_SEAL.json');assert sha(PUBLIC/'SCORE_ROWS.json')==seal['score_rows_sha256']
    ev={z['cell_key']:z for z in read(PUBLIC/'SCORE_ROWS.json')};tot=collections.Counter();label_files={}
    for ds in DATASETS:
        for sp in ['search','confirm']:
            lf=LABELS/sp/ds/'ROWS.json';label_files[str(lf.relative_to(ROOT))]=sha(lf)
            legacy={key(z):z for z in read(SCALAR/sp/ds/'ROWS.json')};raw=read(lf);rows=[];binary=[]
            for r in raw:
                out={k:r[k] for k in ['dataset','split','source_id','condition','order','arrival','expert_scheduled',
                                      'A_state_pre_sha256','A_state_post_sha256','A8_v','A8_t']};tot['arrivals']+=1
                if r['expert_scheduled']:
                    e=ev[key(r)];a=e['anchor_index'];v=np.asarray(r['candidate_v']);t=np.asarray(r['candidate_t'])
                    assert e['candidate_indices']==r['candidate_indices'] and e['intervals']==r['intervals_normalized']
                    assert out['A_state_pre_sha256']==e['A_state_pre_sha256'] and out['A_state_post_sha256']==e['A_state_post_sha256']
                    assert abs(v[a]-r['A8_v'])<EPS
                    out.update(anchor_index=a,candidate_v=v.tolist(),candidate_t=t.tolist());tot['experts']+=1
                    geos=[operation(e['intervals'][a],q) for q in e['intervals']]
                    for signal in ['L','G']:
                        delta=np.asarray(e['scores'][signal])-e['scores'][signal][a]
                        for group in GROUPS:
                            val=binary_cell(v-v[a],delta,candidate_mask(geos,a,group))
                            if val:binary.append(dict(**{k:out[k] for k in ['dataset','split','source_id','condition','order','arrival']},signal=signal,group=group,**val))
                else:tot['nonexperts']+=1
                for arm in ARMS:
                    n=int(arm[1:]);i=e['choices'][arm] if r['expert_scheduled'] else None
                    sv=float(v[i]) if i is not None else r['A8_v'];stt=float(t[i]) if i is not None else r['A8_t'];d=sv-r['A8_v']
                    ov=float(v[:n].max()) if i is not None else sv;ot=float(t[:n].max()) if i is not None else stt
                    out.update({arm+'_v':sv,arm+'_t':stt,arm+'_gain':d,arm+'_t_gain':stt-r['A8_t'],
                        arm+'_regret':ov-sv,arm+'_t_regret':ot-stt,arm+'_gross_gain':max(d,0.),arm+'_gross_loss':max(-d,0.),
                        arm+'_severe':float(d<-.05),arm+'_changed':float(i is not None and i!=a),
                        arm+'_large_selected':float(i is not None and geos[i]['large']),
                        arm+'_destroyed_old_fast':float(i is not None and r['old_fast_gain']>EPS and d< -EPS)})
                    for signal in ['N','U','S']:
                        for met in ['v','t']:
                            out[arm+'_vs_'+signal+str(n)+'_'+met]=out[arm+'_'+met]-legacy[key(r)][signal+str(n)+'_'+met]
                out['L32_vs_G32_v']=out['L32_v']-out['G32_v'];out['L32_vs_G32_t']=out['L32_t']-out['G32_t'];rows.append(out)
            fields=[k for k,vv in rows[0].items() if isinstance(vv,(int,float)) and not isinstance(vv,bool) and k not in ['source_id','arrival','anchor_index']]
            summaries={};bs={}
            for cat in ['corruption','clean']:
                summaries[cat]={};bs[cat]={}
                for subset in ['all','expert','nonexpert']:
                    rr=[r for r in rows if (r['condition']=='clean')==(cat=='clean') and (subset=='all' or r['expert_scheduled']==(subset=='expert'))]
                    z=aggregate(rr,fields);z['transitions']={a:dict(improved=sum(r[a+'_gain']>EPS for r in rr),harmed=sum(r[a+'_gain']< -EPS for r in rr),severe_gt5pp=sum(r[a+'_gain']<-.05 for r in rr),large_selected=sum(r[a+'_large_selected'] for r in rr)) for a in ARMS};summaries[cat][subset]=z
                for group in GROUPS:
                    bs[cat][group]={}
                    for signal in ['L','G']:
                        rr=[r for r in binary if r['signal']==signal and r['group']==group and (r['condition']=='clean')==(cat=='clean')]
                        z=aggregate(rr,['tp','fp','fn','tn','positive','negative','accepted'],dict(precision=('tp','accepted'),benefit_recall=('tp','positive'),harm_acceptance=('fp','negative')))
                        valid=[r for r in rr if r['auc'] is not None];z['auc']=aggregate(valid,['auc'])
                        z['raw_counts']={k:sum(r[k] for r in rr) for k in ['candidates','positives','negatives','TP','FP','FN','TN']};z['eligible_auc_cells']=len(valid);bs[cat][group][signal]=z
            dest=PUBLIC/sp/ds;write(dest/'ROWS.json',rows);write(dest/'SUMMARY.json',summaries)
            write(dest/'BINARY_ROWS.json',binary);write(dest/'DISCRIMINATION.json',bs)
            experts=[r for r in rows if r['expert_scheduled'] and r['condition']!='clean']
            write(dest/'CASES.json',{a:dict(positive=sorted(experts,key=lambda r:-r[a+'_gain'])[:3],negative=sorted(experts,key=lambda r:r[a+'_gain'])[:3]) for a in ARMS})
            print('LATENT_LABEL_PANEL',ds,sp,len(rows),flush=True)
    assert dict(tot)==dict(arrivals=1152,experts=288,nonexperts=864)
    write(PUBLIC/'COVERAGE.json',dict(tot));write(PUBLIC/'LABEL_JOIN.json',dict(time=time.time(),seal_time=seal['time'],label_files=label_files,raw_target_annotations_read=False,target_GT_used_for_fit=False))
    resources={d:{k:read(BASE/d/k) for k in ['SOURCE_FEATURE_BARRIER.json','TARGET_FEATURE_BARRIER.json']} for d in DATASETS}
    write(PUBLIC/'RESOURCES.json',dict(stages={d:{k:{x:v for x,v in r.items() if x!='files'} for k,r in stages.items()} for d,stages in resources.items()},
        source_feature_bytes=sum(f.stat().st_size for f in BASE.glob('*/source_features/*.pt')),
        target_feature_bytes=sum(f.stat().st_size for f in BASE.glob('*/target_features/*.pt')),
        new_experts=0,spatial_updates=0,temporal_updates=0,worker_wall_is_not_pure_GPU_kernel=True))
    st('completed_pending_root_audit_report_publication',coverage=dict(tot))

if __name__=='__main__':
    try:
        name=sys.argv[1]
        if name in ['capture_source','capture_target']:globals()[name](sys.argv[2])
        else:globals()[name]()
    except BaseException as e:
        st('failed',stage=sys.argv[1:],error=repr(e),traceback=traceback.format_exc());raise
