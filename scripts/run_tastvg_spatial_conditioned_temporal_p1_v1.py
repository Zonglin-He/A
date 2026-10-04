"""Finite frozen soft spatial-conditioning study. No GT before prediction seal."""
import os,sys,time,hashlib,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts.spatial_guided_temporal_import_v1 import core
BASE=ROOT/'artifacts/tastvg_spatial_conditioned_temporal_p1_v1'
PUB=ROOT/'results/tastvg_spatial_conditioned_temporal_p1/2026-10-04'
P0=ROOT/'artifacts/tastvg_spatial_guided_temporal_p0_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
VIEW=ROOT/'artifacts/tastvg_current_correction_views_v1'
OWN=['vg_tta/tastvg_spatial_conditioned_temporal_p1_v1.py',
 'scripts/run_tastvg_spatial_conditioned_temporal_p1_v1.py',
 'scripts/score_tastvg_spatial_conditioned_temporal_p1_v1.py',
 'scripts/audit_tastvg_spatial_conditioned_temporal_p1_v1.py',
 'scripts/report_tastvg_spatial_conditioned_temporal_p1_v1.py',
 'scripts/test_tastvg_spatial_conditioned_temporal_p1_v1.py',
 'protocols/tastvg_spatial_conditioned_temporal_p1_v1.md',
 'docs/tastvg_spatial_conditioned_temporal_p1_v1/EXECUTION.md']
def key(c):return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])
def name(c):return f'{c["dataset"]}_{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def state(s,**kw):status(BASE/'STATUS.json',dict(status=s,pid=os.getpid(),time=time.time(),**kw))
def budget():assert shutil.disk_usage(ROOT).free>8*2**30,'8GiB free-space floor'
def verify(all_inputs=False):
    z=read(BASE/'RUNTIME_LOCK.json');code=dict(z['code'])
    for f in sorted((BASE/'revisions').glob('*.json')):code.update(read(f)['pin_overrides'])
    for f,h in {**code,**z['assets'],**(z['inputs'] if all_inputs else {})}.items():
        assert sha(f if f.startswith('/') else ROOT/f)==h,f
    return z
def prepare():
    sys.addaudithook(core.data_guard);budget();assert not (BASE/'RUNTIME_LOCK.json').exists()
    old=read(P0/'RUNTIME_LOCK.json');code=dict(old['code'])
    for f in sorted((P0/'revisions').glob('*.json')):code.update(read(f)['pin_overrides'])
    for f,h in {**code,**old['assets'],**old['inputs']}.items():assert sha(f if f.startswith('/') else ROOT/f)==h,f
    cells=read(P0/'COHORT.json')['cells'];assert len(cells)==288
    inputs=dict(old['inputs'])
    for f in ['COHORT.json','A_BOXES_INPUT.json','FULL_INPUT.json']:
        write(BASE/f,read(P0/f));inputs[str((BASE/f).relative_to(ROOT))]=sha(BASE/f)
        inputs[str((P0/f).relative_to(ROOT))]=sha(P0/f)
    for c in cells:
        f=P0/'A_ROI'/(name(c)+'.json');r=read(f)
        assert sha(P0/r['cache'])==r['cache_sha256']
        inputs[str(f.relative_to(ROOT))]=sha(f);inputs[str((P0/r['cache']).relative_to(ROOT))]=r['cache_sha256']
    for f in OWN:code[f]=sha(ROOT/f)
    cfg=dict(version='tastvg_spatial_conditioned_temporal_p1_v1',arms=['Full','Soft','A_ROI'],alpha=.5,
        mechanism='Frozen attention-pool additive log spatial prior; CLS=1, patch=.5+.5*fractional_A_box_mask',
        full_context='All complete-frame tokens retained; no pixel cropping, no token deletion; outside prior=.5',
        transform='Stock PE336 bilinear squash resize; patch14, 24x24; no centre crop',
        projection='Original learned attention pool, residual MLP and visual projection; no extra L2 normalization',
        invalid_box='Zero/nonfinite/inverted/clipped-empty mask uses exact original global pool',
        sampling='Exact original full observations, nearest-original-observation 2Hz feature grid, FP16 batch16',
        expert='Same frozen UniversalVTG + PE-Core-L14-336; first raw confidence argmax; no query unifier',
        parent_arrivals=1152,expert_cells=288,corrupt_cells=240,clean_cells=48,nonexpert_cells_scored=0,
        design_sources_per_dataset=dict(search=32,confirm=16),
        expert_sources=dict(vidstg=dict(search=16,confirm=8),hc2=dict(search=14,confirm=7)),
        conditions=sorted({c['condition'] for c in cells}),orders=['order1','order2'],source_queries=1,
        historical_exposure=True,fresh_test=False,spatial_A_fixed=True,current_pre_update_boxes=True,
        backbone_calls=0,spatial_expert_calls=0,backward_calls=0,parameter_updates=0,DTA_started=False,
        production_promoted=False,production_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        expert_checkpoint_sha256=old['assets']['checkpoints/universalvtg/models/best.pth'],
        PE_sha256=next(v for k,v in old['assets'].items() if 'PE-Core' in k),
        bootstrap_draws=10000,bootstrap_seed=20261004,aggregation='Source macro; same-source paired bootstrap',
        primary='Soft minus cached Full teacher tIoU; all four corrupt panels lower95CI>0 for teacher-only GO',
        GO_rule='all four corruption panels Soft-Full paired lower95CI>0; no automatic DTA',
        max_new_Soft_requests=288,smoke_Full_controls=2,smoke_Soft_format_calls=2,max_new_Soft_calls_including_smoke=290,network_downloads=0,
        previous_crop_commit='39e060b8a7821916194bcf03c6f510e685f6ad0a',
        source_paper='CoSD CVPR2023 soft modulation is jointly trained, not a guarantee for this frozen plugin',
        stopping_scope='If not stable, stop this UVTG coupling line as resource decision; no universal impossibility claim')
    write(PUB/'CONFIG.json',cfg);write(PUB/'CODE_BINDINGS.json',{f:code[f] for f in OWN})
    write(BASE/'RUNTIME_LOCK.json',dict(code=code,assets=old['assets'],inputs=inputs,GT_inputs=old['GT_inputs'],time=time.time()))
    state('prepared_pending_smoke',cells=288,GT_read=False)

def replay(model,cache,ids,fps):
    import numpy as np,torch
    with torch.inference_mode():raw=model.predict(cache['video_features'],cache['text_features'],return_raw=True,
        fps=None,feature_fps=2.,duration=cache['duration'],use_unifier=False)
    seg,score=raw['raw_segments'],raw['raw_scores']
    if isinstance(seg,list):seg=seg[0];score=score[0]
    sec=model._convert_segments_to_seconds(seg,fps=None,feature_fps=2.).clamp(0,cache['duration']).float().cpu().numpy()
    cf=score.float().cpu().numpy();valid=sec[:,1]>sec[:,0]
    return sec[valid]*fps+ids[0],cf[valid]

def encode(model,images,pick,boxes,width,height,smoke=False):
    import numpy as np,torch,PIL.Image
    from vg_tta.tastvg_spatial_conditioned_temporal_p1_v1 import fractional_mask,pool_features
    unique,inverse=np.unique(pick,return_inverse=True)
    masks=np.stack([fractional_mask(boxes[i],width,height) for i in unique])
    visual=model._video_extractor.model.visual;global_batches=[];soft_batches=[];errors=[]
    assert visual.image_size==336 or visual.image_size==(336,336)
    for i in range(0,len(unique),16):
        indexes=unique[i:i+16]
        pixels=torch.stack([model._video_preprocess(PIL.Image.fromarray(images[j])) for j in indexes]).cuda().half()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            tokens=visual.forward_features(pixels,norm=True)
            assert tokens.shape[1:]==(577,1024)
            g,s,_=pool_features(visual,tokens,masks[i:i+16])
            if smoke:
                stock=model._video_extractor(pixels)
                g0,s0,_=pool_features(visual,tokens,masks[i:i+16],alpha=0.)
                gf,sf,_=pool_features(visual,tokens,np.ones_like(masks[i:i+16]))
                assert torch.equal(g,g0) and torch.equal(g,s0) and torch.equal(g,gf) and torch.equal(g,sf)
                error=float((stock-g).abs().max());assert error==0.,error;errors.append(error)
        global_batches.append(g.float().cpu());soft_batches.append(s.float().cpu())
    g=torch.cat(global_batches)[inverse];s=torch.cat(soft_batches)[inverse]
    delta=s-g
    stats=dict(mask_mean=float(masks.mean()),mask_empty_frames=int((masks.sum((1,2))==0).sum()),
        feature_relative_L2=float((delta.norm(dim=1)/g.norm(dim=1).clamp_min(1e-12)).mean()),
        global_feature_mean_norm=float(g.norm(dim=1).mean()),soft_feature_mean_norm=float(s.norm(dim=1).mean()),
        feature_cosine=float(torch.nn.functional.cosine_similarity(g,s,dim=1).mean()),
        alpha0_fullmask_bitwise_pass=smoke,max_stock_feature_error=max(errors) if errors else None)
    return g.T.contiguous(),s.T.contiguous(),masks,stats

def worker(phase):
    import numpy as np,torch
    budget();verify(all_inputs=True);sys.addaudithook(core.data_guard)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from scripts.run_final_simplification_v1 import lease
    lock=lease();tick=time.monotonic();created=done=0;failure=None
    try:
        torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
        for f in [ROOT/'external/UniversalVTG',ROOT/'external/UniversalVTG/perception_models']:sys.path.insert(0,str(f))
        from universal_vtg_inference import UniversalVTG
        from vg_tta.exact_frame_decode_audit_v2 import decode as vd
        from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as hd
        from scripts.run_tastvg_full_b1_experts_v1 import observation
        model=UniversalVTG(experiment_name=str(ROOT/'checkpoints/universalvtg'),device='cuda',enable_query_unifier=False)
        model._ensure_video_encoder()
        for m in [model.model,model._video_extractor]:
            m.eval()
            for p in m.parameters():p.requires_grad_(False)
        cells=read(BASE/'COHORT.json')['cells'];fulls=read(BASE/'FULL_INPUT.json');atlas=read(BASE/'A_BOXES_INPUT.json')
        plans={ds:read(VIEW/ds/'PLAN.json') for ds in ['vidstg','hc2']}
        if phase=='smoke':cells=[next(c for c in cells if c['dataset']==ds and c['condition']=='clean') for ds in plans]
        else:
            assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass'
            cells=sorted(cells,key=lambda c:(c['dataset'],c['parent'],c['condition'],c['order']))
        source=None;smokes=[]
        for c in cells:
            budget();k=key(c);dst=BASE/'Soft'/(name(c)+'.json')
            if phase!='smoke' and dst.exists():
                r=read(dst);assert r['cell_key']==k and r['GT_read'] is False
                assert sha(BASE/r['cache'])==r['cache_sha256']
                v=load(BASE/r['cache']);assert v['feature_input_sha256']==r['feature_input_sha256'] and not v['GT_read']
                assert (r['A_state_pre_sha256'],r['A_state_post_sha256'],r['original_pixel_sha256'])==(c['pre_sha'],c['post_sha'],c['pixel_sha256'])
                done+=1;continue
            row=plans[c['dataset']]['rows'][c['parent']]
            src=(c['dataset'],c['parent'])
            if src!=source:
                from vg_tta import exact_frame_decode_audit_v2 as binding
                binding.decode=hd if c['dataset']=='hc2' else vd
                frames,ids=(hd if c['dataset']=='hc2' else vd)(row['input']);source=src
            assert ids==row['frame_ids']
            shifted,pixel,_=observation(row,c['condition'],frames);assert pixel==c['pixel_sha256']
            old=fulls[k];assert sha(ROOT/old['full_cache'])==old['full_cache_sha256'];full=load(ROOT/old['full_cache'])
            pick,duration=core.sample_indices(ids,row['input']['fps']);assert pick.tolist()==full['picked_observations']
            assert duration==full['duration'];unique=np.unique(pick)
            h=hashlib.sha256(row['input']['caption'].encode()+str(ids).encode()+str(row['input']['fps']).encode()+str(shifted.shape).encode()+b''.join(shifted[i].tobytes() for i in unique)).hexdigest()
            assert h==old['full_input_sha256']
            boxes=np.asarray(atlas[k],float);images={int(i):shifted[i] for i in unique}
            feature_key=core.digest(dict(pixel_sha256=pixel,caption=row['input']['caption'],ids=ids,
                fps=row['input']['fps'],duration=duration,picked=pick.tolist(),boxes=boxes[unique].tolist(),
                alpha=.5,transform='stock_PE336_FP16_batch16_squash',pool='CLS1_patch.5+.5fractional_mask_original_projection',
                pool_impl_sha256=sha(ROOT/'vg_tta/tastvg_spatial_conditioned_temporal_p1_v1.py')))
            cache=BASE/'cache/Soft'/(feature_key+'.pt');start=time.monotonic();fresh=not cache.exists()
            if phase=='smoke' or fresh:
                global_f,soft_f,masks,stats=encode(model,images,pick,boxes,row['input']['width'],row['input']['height'],phase=='smoke')
            if phase=='smoke':
                assert global_f.shape==full['video_features'].shape
                error=float((global_f-full['video_features']).abs().max());assert error<=.02,error
                p,cf=replay(model,dict(full,video_features=global_f),ids,row['input']['fps'])
                pe=float(np.abs(p-np.asarray(full['proposals'])).max());ce=float(np.abs(cf-full['proposal_confidence']).max())
                assert pe<.1 and ce<1e-4 and core.top(p,cf)['index']==old['Full']['index'],(pe,ce)
                sp,sc=replay(model,dict(full,video_features=soft_f),ids,row['input']['fps']);core.top(sp,sc)
                smokes.append(dict(cell_key=k,pixel_hash_exact=True,pick_exact=True,Full_top1_exact=True,
                    max_historical_feature_error=error,max_proposal_frame_error=pe,max_confidence_error=ce,
                    **stats,GT_read=False,free_disk_bytes=shutil.disk_usage(ROOT).free,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                done+=1;print('SMOKE',done,2,flush=True);continue
            if fresh:
                p,cf=replay(model,dict(full,video_features=soft_f),ids,row['input']['fps']);choice=core.top(p,cf)
                save(cache,dict(proposals=p.tolist(),confidence=cf.tolist(),selection=choice,video_features=soft_f,
                    feature_input_sha256=feature_key,duration=duration,picked=pick.tolist(),masks=masks,stats=stats,
                    GT_read=False,seconds=time.monotonic()-start));created+=1
            v=load(cache)
            assert v['feature_input_sha256']==feature_key and v['duration']==duration and v['picked']==pick.tolist() and not v['GT_read']
            assert core.top(v['proposals'],v['confidence'])==v['selection']
            write(dst,dict(cell_key=k,arm='Soft',cache=str(cache.relative_to(BASE)),cache_sha256=sha(cache),
                feature_input_sha256=feature_key,selection=v['selection'],new_call=fresh,seconds=time.monotonic()-start,
                original_pixel_sha256=pixel,A_state_pre_sha256=c['pre_sha'],A_state_post_sha256=c['post_sha'],
                mask_sha256=hashlib.sha256(v['masks'].tobytes()).hexdigest(),GT_read=False,
                original_frame_count=len(ids),feature_frame_count=len(pick),stats=v['stats']))
            done+=1;state('Soft_running',done=done,total=288,new_calls=created,seconds=time.monotonic()-tick)
            print('Soft',done,288,'new',created,'sec',round(time.monotonic()-tick,2),flush=True)
        if phase=='smoke':
            write(BASE/'SMOKE.json',dict(status='pass',rows=smokes,cells=2,new_Full_control_calls=2,
                new_Soft_format_calls=2,GPU_worker_wall_seconds=time.monotonic()-tick,GT_read=False,time=time.time()))
            state('smoke_completed_pending_root_acceptance')
        else:
            verify(all_inputs=True)
            receipts={str(p.relative_to(BASE)):sha(p) for p in (BASE/'Soft').glob('*.json')};assert len(receipts)==288
            assert set(receipts)=={'Soft/'+name(c)+'.json' for c in cells}
            for c in cells:
                r=read(BASE/'Soft'/(name(c)+'.json'));assert r['cell_key']==key(c) and not r['GT_read']
                assert sha(BASE/r['cache'])==r['cache_sha256'];v=load(BASE/r['cache'])
                assert v['feature_input_sha256']==r['feature_input_sha256'] and not v['GT_read']
                assert core.top(v['proposals'],v['confidence'])==r['selection']==v['selection']
            write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',cells=288,arm_predictions=864,new_arm_predictions=288,
                receipts=receipts,GT_read=False,temporal_GT_scoring_started=False,new_calls=created,
                A_ROI_receipts={str((P0/'A_ROI'/(name(c)+'.json')).relative_to(ROOT)):sha(P0/'A_ROI'/(name(c)+'.json')) for c in cells},
                full_input_sha256=sha(BASE/'FULL_INPUT.json'),
                GPU_worker_wall_seconds=time.monotonic()-tick,backward_calls=0,parameter_updates=0,
                runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))
            state('all_predictions_sealed_pending_CPU_score',cells=288)
    except BaseException as e:
        failure=dict(error=repr(e),traceback=traceback.format_exc());state('failed',phase=phase,failure=failure,done=done);raise
    finally:
        write(BASE/'allocations'/f'{time.time_ns()}.json',dict(phase=phase,done=done,new_calls=created,
            wall_seconds=time.monotonic()-tick,failure=failure,time=time.time()));lock.close()
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','smoke','Soft']);a=p.parse_args()
    if a.phase=='prepare':prepare()
    else:worker(a.phase)
