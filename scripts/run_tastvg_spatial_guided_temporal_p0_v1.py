"""Bounded frozen UVTG input audit. Deployable inference never reads GT."""
import os,sys,time,json,hashlib,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts.spatial_guided_temporal_import_v1 import core
BASE=ROOT/'artifacts/tastvg_spatial_guided_temporal_p0_v1'
PUB=ROOT/'results/tastvg_spatial_guided_temporal_p0/2026-10-04'
R2=ROOT/'artifacts/tastvg_dta_expert_r2_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
VIEW=ROOT/'artifacts/tastvg_current_correction_views_v1'
OWN=['vg_tta/tastvg_spatial_guided_temporal_p0_v1.py','scripts/spatial_guided_temporal_import_v1.py',
 'scripts/run_tastvg_spatial_guided_temporal_p0_v1.py','scripts/score_tastvg_spatial_guided_temporal_p0_v1.py',
 'scripts/audit_tastvg_spatial_guided_temporal_p0_v1.py','scripts/report_tastvg_spatial_guided_temporal_p0_v1.py',
 'scripts/test_tastvg_spatial_guided_temporal_p0_v1.py','protocols/tastvg_spatial_guided_temporal_p0_v1.md',
 'docs/tastvg_spatial_guided_temporal_p0_v1/EXECUTION.md','scripts/decota_matrix_common_v1.py',
 'scripts/audit_tastvg_dta_oracle_r1_v1.py']
def key(c):return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])
def name(c):return f'{c["dataset"]}_{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def budget():assert shutil.disk_usage(ROOT).free>8*2**30,'8GiB free-space floor'
def state(s,**kw):status(BASE/'STATUS.json',dict(status=s,pid=os.getpid(),time=time.time(),**kw))
def verify(all_inputs=False):
    z=read(BASE/'RUNTIME_LOCK.json')
    code=dict(z['code'])
    for f in sorted((BASE/'revisions').glob('*.json')):code.update(read(f)['pin_overrides'])
    for f,h in {**code,**z['assets'],**(z['inputs'] if all_inputs else {})}.items():assert sha(f if f.startswith('/') else ROOT/f)==h,f
    return z

def prepare():
    import numpy as np
    sys.addaudithook(core.data_guard)
    assert not (BASE/'RUNTIME_LOCK.json').exists();budget()
    assert read(R2/'FINAL_COMPLETION.json')['status']=='verified_complete'
    cells=[c for c in read(R2/'COHORT.json')['cells'] if c['scheduled']];assert len(cells)==288
    support=read(R2/'EXPERT_SUPPORT.json');inputs={};pack={};boxes={}
    def pin(f):inputs[str(f.relative_to(ROOT))]=sha(f)
    for f in [R2/'COHORT.json',R2/'EXPERT_SUPPORT.json',R2/'FINAL_COMPLETION.json',ROOT/'methods/CURRENT_METHOD.json']:pin(f)
    plans={ds:read(VIEW/ds/'PLAN.json') for ds in ['vidstg','hc2']}
    for ds in plans:pin(VIEW/ds/'PLAN.json')
    for c in cells:
        k=key(c);row=plans[c['dataset']]['rows'][c['parent']];s=support[k]
        f=ROOT/c['old_payload'];receipt=read(f.with_suffix('.json'));pin(f.with_suffix('.json'))
        assert sha(f)==receipt['sha256'];inputs[str(f.relative_to(ROOT))]=receipt['sha256']
        z=load(f);assert z['pre_sha']==c['pre_sha'] and z['post_sha']==c['post_sha'] and z['pixel_sha256']==c['pixel_sha256']
        b=core.a_xyxy(z['slow']['boxes'].numpy(),row['input']['width'],row['input']['height'])
        assert len(b)==len(row['frame_ids']);boxes[k]=b.tolist()
        ef=POOL/c['dataset']/'experts/temporal'/c['condition']/f'{c["parent"]:05}.json';pin(ef);e=read(ef)
        cf=POOL/c['dataset']/'experts'/e['cache'];assert sha(cf)==e['cache_sha256']==s['expert_cache_sha256']
        inputs[str(cf.relative_to(ROOT))]=e['cache_sha256'];full=load(cf)
        assert not full['GT_read'] and full['proposals']==s['proposals'] and full['proposal_confidence']==s['confidence']
        top=core.top(s['proposals'],s['confidence']);assert top['index']==s['deploy']['index']
        pick,duration=core.sample_indices(row['frame_ids'],row['input']['fps'])
        assert pick.tolist()==full['picked_observations'] and duration==full['duration']
        pack[k]=dict(full_cache=str(cf.relative_to(ROOT)),full_cache_sha256=e['cache_sha256'],
            Full=top,full_input_sha256=full['input_sha256'],pixel_sha256=c['pixel_sha256'],
            A_state_pre_sha256=c['pre_sha'],A_state_post_sha256=c['post_sha'])
    write(BASE/'COHORT.json',dict(cells=cells));write(BASE/'A_BOXES_INPUT.json',boxes);write(BASE/'FULL_INPUT.json',pack)
    for f in [BASE/'COHORT.json',BASE/'A_BOXES_INPUT.json',BASE/'FULL_INPUT.json']:pin(f)
    assets={}
    for f in [ROOT/'checkpoints/universalvtg/models/best.pth',ROOT/'checkpoints/universalvtg/opt.yaml']:
        h=sha(f);assert h==read(f.with_suffix(f.suffix+'.receipt.json'))['sha256'];assets[str(f.relative_to(ROOT))]=h
    pe=Path('/home/wwww/.cache/huggingface/hub/models--facebook--PE-Core-L14-336/blobs/0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8')
    assets[str(pe)]=sha(pe)
    dependencies=['scripts/run_tastvg_full_b1_experts_v1.py','scripts/run_final_simplification_v1.py',
        'vg_tta/exact_frame_decode_audit_v2.py','vg_tta/tastvg_paper48_hc2_decode_v1.py',
        'vg_tta/tastvg_deployment_corruption_v2.py','scripts/c1_controlled_corruption_v1.py']
    dependencies += [str(f.relative_to(ROOT)) for d in ['external/UniversalVTG/model','external/UniversalVTG/perception_models/core'] for f in (ROOT/d).rglob('*.py')]
    dependencies += ['external/UniversalVTG/universal_vtg_inference.py','external/UniversalVTG/feature_extraction/extract_visual_features.py']
    code={f:sha(ROOT/f) for f in sorted(set(OWN+dependencies))}
    gt_inputs=read(R2/'RUNTIME_LOCK.json')['oracle_label_inputs']
    cfg=dict(version='tastvg_spatial_guided_temporal_p0_v1',arms=core.ARMS,
        context_ratio=1.5,crop='centred square, ceil(side), edge replication, min2; invalid A full-frame fallback',
        GT_ROI='spatial-coordinate interpolation plus nearest-box extrapolation; oracle diagnostic, not full-video perfect tracking',
        sampling='exact original full observations and nearest-original-observation 2Hz feature grid',
        expert='same frozen UniversalVTG + PE-Core-L14-336; confidence first raw argmax; query unifier disabled',
        parent_arrivals=1152,expert_cells=288,corrupt_cells=240,clean_cells=48,nonexpert_cells_scored=0,
        design_sources_per_dataset=dict(search=32,confirm=16),
        expert_sources=dict(vidstg=dict(search=16,confirm=8),hc2=dict(search=14,confirm=7)),
        conditions=sorted({c['condition'] for c in cells}),orders=['order1','order2'],source_queries=1,
        historical_exposure=True,fresh_test=False,preliminary_GT_support_metadata_inspection=True,
        spatial_A_fixed=True,current_pre_update_boxes=True,backbone_calls=0,spatial_expert_calls=0,backward_calls=0,parameter_updates=0,
        DTA_started=False,production_promoted=False,production_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        expert_checkpoint_sha256=assets['checkpoints/universalvtg/models/best.pth'],expert_opt_sha256=assets['checkpoints/universalvtg/opt.yaml'],PE_sha256=assets[str(pe)],
        prior_R2_commit='324b3357dbd67b435d1fb507ae516a44ecf53961',prior_medoid_commit='522285ed6cac8e46ed2efdc6635991201a271a75',
        bootstrap_draws=10000,bootstrap_seed=20261004,aggregation='source macro, same-source paired bootstrap',
        GO_rule='all four corruption panels A_ROI-Full paired lower95CI>0; no automatic DTA',
        max_new_ROI_requests=576,smoke_Full_controls=2,network_downloads=0)
    write(PUB/'CONFIG.json',cfg);write(PUB/'CODE_BINDINGS.json',{f:code[f] for f in OWN})
    write(BASE/'RUNTIME_LOCK.json',dict(code=code,assets=assets,inputs=inputs,GT_inputs=gt_inputs,time=time.time()))
    state('prepared_pending_smoke',cells=288,GT_in_deployable_inputs=False)

def full_replay(model,cache,ids,fps):
    import torch,numpy as np
    with torch.inference_mode():raw=model.predict(cache['video_features'],cache['text_features'],return_raw=True,fps=None,feature_fps=2.,duration=cache['duration'],use_unifier=False)
    seg,score=raw['raw_segments'],raw['raw_scores']
    if isinstance(seg,list):seg,score=seg[0],score[0]
    sec=model._convert_segments_to_seconds(seg,fps=None,feature_fps=2.).clamp(0,cache['duration']).float().cpu().numpy()
    conf=score.float().cpu().numpy();ok=sec[:,1]>sec[:,0];p=sec[ok]*fps+ids[0];cf=conf[ok]
    return p,cf

def encode(model,images,pick):
    import numpy as np,torch
    from PIL import Image
    unique,inverse=np.unique(pick,return_inverse=True);features=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for at in range(0,len(unique),16):
            pixels=torch.stack([model._video_preprocess(Image.fromarray(images[i])) for i in unique[at:at+16]]).cuda().half()
            features.append(model._video_extractor(pixels).float().cpu())
    return torch.cat(features)[inverse].T.contiguous()

def model_worker(phase):
    import numpy as np,torch
    from vg_tta.exact_frame_decode_audit_v2 import decode as vd
    from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as hd
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from scripts.run_final_simplification_v1 import lease
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';budget();verify()
    sys.addaudithook(core.data_guard if phase!='GT_ROI' else lambda event,args: gt_worker_guard(event,args))
    if phase=='GT_ROI':
        assert read(BASE/'DEPLOYABLE_BARRIER.json')['status']=='sealed'
        assert read(BASE/'GT_INPUT_RECEIPT.json')['temporal_span_used'] is False
        assert sha(BASE/'GT_BOXES_INPUT.json')==read(BASE/'GT_INPUT_RECEIPT.json')['box_atlas_sha256']
        atlas=read(BASE/'GT_BOXES_INPUT.json')['boxes']
    else:atlas=read(BASE/'A_BOXES_INPUT.json')
    pins=read(BASE/'RUNTIME_LOCK.json')['inputs']
    for f in ['COHORT.json','FULL_INPUT.json','A_BOXES_INPUT.json']:
        assert sha(BASE/f)==pins[str((BASE/f).relative_to(ROOT))]
    if phase!='smoke':assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass'
    lock=lease();tick=time.monotonic();done=created=0;failure=None
    torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    repo=ROOT/'external/UniversalVTG';sys.path[:0]=[str(repo),str(repo/'perception_models')]
    from universal_vtg_inference import UniversalVTG
    model=UniversalVTG(experiment_name=str(ROOT/'checkpoints/universalvtg'),device='cuda',enable_query_unifier=False)
    model._ensure_video_encoder()
    for m in [model.model,model._video_extractor]:m.eval().requires_grad_(False)
    assert not any(p.requires_grad for m in [model.model,model._video_extractor] for p in m.parameters())
    cells=read(BASE/'COHORT.json')['cells'];fullpack=read(BASE/'FULL_INPUT.json')
    plans={ds:read(VIEW/ds/'PLAN.json') for ds in ['vidstg','hc2']}
    if phase=='smoke':cells=[next(c for c in cells if c['dataset']==ds and c['condition']=='clean') for ds in ['vidstg','hc2']]
    else:cells=sorted(cells,key=lambda c:(c['dataset'],c['parent'],c['condition'],c['order']))
    # Reuse each identical decoded observation, without persisting crop videos.
    frames=None;last=None;smoke=[]
    try:
        for c in cells:
            k=key(c);nm=name(c);dst=BASE/phase/(nm+'.json')
            if dst.exists():assert sha(BASE/read(dst)['cache'])==read(dst)['cache_sha256'];done+=1;continue
            budget();row=plans[c['dataset']]['rows'][c['parent']];old=fullpack[k];source=(c['dataset'],c['parent'])
            if source!=last:
                # Same binding as the original HC expert controller; it also
                # controls the physical donor frame used by frame_freeze.
                from vg_tta import exact_frame_decode_audit_v2 as decoder_binding
                decoder_binding.decode=hd if c['dataset']=='hc2' else vd
                frames,ids=(vd if c['dataset']=='vidstg' else hd)(row['input']);assert ids==row['frame_ids'];last=source
            shifted,pixel,spec=observation(row,c['condition'],frames);assert pixel==c['pixel_sha256']
            pick,duration=core.sample_indices(ids,row['input']['fps']);unique=np.unique(pick)
            full=load(ROOT/old['full_cache']);assert sha(ROOT/old['full_cache'])==old['full_cache_sha256']
            assert pick.tolist()==full['picked_observations']
            h=hashlib.sha256(row['input']['caption'].encode()+str(ids).encode()+str(row['input']['fps']).encode()+str(shifted.shape).encode()+b''.join(shifted[i].tobytes() for i in unique)).hexdigest()
            assert h==old['full_input_sha256']
            windows=[core.window(b,row['input']['width'],row['input']['height']) for b in atlas[k]]
            assert len(windows)==len(ids)
            if phase=='smoke':
                video=encode(model,shifted,pick)
                assert video.shape==full['video_features'].shape
                feature_error=float((video-full['video_features']).abs().max());assert feature_error<=.02,feature_error
                p,cf=full_replay(model,dict(full,video_features=video),ids,row['input']['fps'])
                assert p.shape==np.asarray(full['proposals']).shape and cf.shape==np.asarray(full['proposal_confidence']).shape
                pe=float(np.abs(p-np.asarray(full['proposals'])).max());ce=float(np.abs(cf-full['proposal_confidence']).max())
                assert pe<.1 and ce<1e-4 and core.top(p,cf)['index']==old['Full']['index'],(pe,ce)
                # Crop format/resource smoke only; no additional scientific prediction.
                ims=[core.crop(shifted[i],windows[i]) for i in unique[:2]]
                assert all(im.ndim==3 and im.shape[2]==3 for im in ims)
                import PIL.Image
                model._video_preprocess(PIL.Image.fromarray(ims[0]))
                smoke.append(dict(cell_key=k,pixel_hash_exact=True,pick_exact=True,Full_top1_exact=True,
                    max_feature_error=feature_error,max_proposal_frame_error=pe,max_confidence_error=ce,
                    first_crop_shape=list(ims[0].shape),GT_read=False,free_disk_bytes=shutil.disk_usage(ROOT).free,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                done+=1;print('SMOKE',done,2,flush=True);continue
            images={int(i):core.crop(shifted[i],windows[i]) for i in unique}
            image_hashes={str(i):hashlib.sha256(im.tobytes()).hexdigest() for i,im in images.items()}
            feature_key=core.digest(dict(caption=row['input']['caption'],ids=ids,fps=row['input']['fps'],duration=duration,
                windows=windows,picked=pick.tolist(),image_hashes=image_hashes,transform='stock_PE336_FP16_batch16'))
            cache=BASE/'cache'/phase/(feature_key+'.pt');start=time.monotonic();fresh=not cache.exists()
            if fresh:
                video=encode(model,images,pick)
                p,cf=full_replay(model,dict(full,video_features=video),ids,row['input']['fps'])
                choice=core.top(p,cf)
                save(cache,dict(proposals=p.tolist(),confidence=cf.tolist(),selection=choice,video_features=video,
                    feature_input_sha256=feature_key,GT_spatial_used=phase=='GT_ROI',GT_temporal_span_used=False,
                    duration=duration,picked=pick.tolist(),windows=windows,crop_pixel_sha256=image_hashes,
                    seconds=time.monotonic()-start));created+=1
            v=load(cache)
            write(dst,dict(cell_key=k,arm=phase,cache=str(cache.relative_to(BASE)),cache_sha256=sha(cache),
                feature_input_sha256=feature_key,selection=v['selection'],new_call=fresh,seconds=time.monotonic()-start,
                original_pixel_sha256=pixel,A_state_pre_sha256=c['pre_sha'],A_state_post_sha256=c['post_sha'],
                invalid_box_full_fallbacks=sum(w['fallback_full'] for w in windows),GT_spatial_used=phase=='GT_ROI',
                GT_temporal_span_used=False,original_frame_count=len(ids),feature_frame_count=len(pick)))
            done+=1;state(phase+'_running',done=done,total=288,new_calls=created,seconds=time.monotonic()-tick)
            print(phase,done,288,'new',created,'sec',round(time.monotonic()-tick,2),flush=True)
        if phase=='smoke':
            write(BASE/'SMOKE.json',dict(status='pass',rows=smoke,cells=2,new_Full_control_calls=2,
                GPU_worker_wall_seconds=time.monotonic()-tick,GT_read=False,time=time.time()))
            state('smoke_completed_pending_root_acceptance')
        else:
            receipts={str(p.relative_to(BASE)):sha(p) for p in (BASE/phase).glob('*.json')}
            assert len(receipts)==288
            previous=[read(f) for f in (BASE/'allocations').glob('*.json') if read(f)['phase']==phase]
            barrier=dict(status='sealed',arm=phase,cells=288,GT_spatial_used=phase=='GT_ROI',GT_temporal_span_used=False,
                receipts=receipts,new_calls=created+sum(r['new_calls'] for r in previous),new_calls_this_attempt=created,
                prior_attempts=len(previous),GPU_worker_wall_seconds=time.monotonic()-tick+sum(r['wall_seconds'] for r in previous),
                backward_calls=0,parameter_updates=0,time=time.time(),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'))
            write(BASE/(phase+'_BARRIER.json'),barrier)
            if phase=='A_ROI':
                write(BASE/'DEPLOYABLE_BARRIER.json',dict(status='sealed',GT_read=False,cells=288,
                    A_ROI_barrier_sha256=sha(BASE/'A_ROI_BARRIER.json'),Full_input_sha256=sha(BASE/'FULL_INPUT.json'),time=time.time()))
                state('deployable_sealed_pending_GT_spatial_oracle')
            else:
                write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',cells=288,arm_predictions=864,
                    deployable_barrier_sha256=sha(BASE/'DEPLOYABLE_BARRIER.json'),GT_ROI_barrier_sha256=sha(BASE/'GT_ROI_BARRIER.json'),
                    GT_input_receipt_sha256=sha(BASE/'GT_INPUT_RECEIPT.json'),GT_spatial_oracle=True,temporal_GT_scoring_started=False,time=time.time()))
                state('all_predictions_sealed_pending_CPU_score')
    except BaseException as e:
        failure=dict(error=repr(e),traceback=traceback.format_exc());state('failed',phase=phase,failure=failure,done=done);raise
    finally:
        write(BASE/'allocations'/f'{time.time_ns()}.json',dict(phase=phase,done=done,new_calls=created,
            wall_seconds=time.monotonic()-tick,failure=failure,time=time.time()))
        lock.close()

def gt_worker_guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        s=str(args[0])
        if any(k in s for k in ['GT_LABELS','GT_EXPOSURE','/ROWS.json','/SUMMARY.json','test_annotations.json','valv2_proc.json']):
            raise PermissionError('GT crop worker accepts sealed spatial-only atlas, not temporal labels or scores')

def prepare_GT():
    assert read(BASE/'DEPLOYABLE_BARRIER.json')['status']=='sealed';verify()
    cells=read(BASE/'COHORT.json')['cells'];labels={};boxes={};metadata={}
    plans={ds:read(VIEW/ds/'PLAN.json') for ds in ['vidstg','hc2']}
    gt_pins=read(BASE/'RUNTIME_LOCK.json')['GT_inputs']
    for ds in plans:
        for sp in ['search','confirm']:
            f=POOL/ds/f'GT_LABELS_{sp}.json';assert sha(f)==gt_pins[str(f.relative_to(ROOT))]
            labels[(ds,sp)]=read(f)
    for c in cells:
        # Access only spatial truth; the annotated temporal span is never an argument.
        b,m=core.extend_gt(labels[(c['dataset'],c['split'])][str(c['parent'])]['truth'],plans[c['dataset']]['rows'][c['parent']]['frame_ids'])
        boxes[key(c)]=b.tolist();metadata[key(c)]=m
    write(BASE/'GT_BOXES_INPUT.json',dict(boxes=boxes,metadata=metadata,temporal_span_used=False))
    write(BASE/'GT_INPUT_RECEIPT.json',dict(GT_spatial_used=True,temporal_span_used=False,cells=288,
        box_atlas_sha256=sha(BASE/'GT_BOXES_INPUT.json'),GT_inputs=gt_pins,
        deployable_barrier_sha256=sha(BASE/'DEPLOYABLE_BARRIER.json'),
        caveat='Outside annotated spatial support nearest-coordinate extension, not GT full-video tracking',time=time.time()))
    state('GT_spatial_atlas_prepared_pending_GT_ROI')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','smoke','A_ROI','prepare_GT','GT_ROI']);a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='prepare_GT':prepare_GT()
    else:
        try:model_worker(a.phase)
        except BaseException as e:
            state('failed',phase=a.phase,error=repr(e),traceback=traceback.format_exc());raise
