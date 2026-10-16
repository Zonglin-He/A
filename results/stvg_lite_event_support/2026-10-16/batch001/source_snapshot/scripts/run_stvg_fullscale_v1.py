"""Resumable full-split replication, separate diagnostic and label-free stages.

Uses existing frozen adapters and the released method without editing them.
One GPU worker at a time; per-query receipts make interrupted runs auditable.
"""
import argparse, fcntl, gc, json, math, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
OUT=ROOT/'artifacts/stvg_fullscale_diagnostics_v1'


def plan():
    p=read(OUT/'lock.json')
    for f,h in p['protected_pins'].items():assert sha(ROOT/f)==h,f
    implementation=OUT/'implementation_lock.json'
    if implementation.exists():
        for f,h in read(implementation)['files'].items():assert sha(ROOT/f)==h,f
    return p


def label_payload(p):
    assert sha(p['labels'])==p['labels_sha256']
    return read(p['labels'])


def temporal_specs(replay,row,gt,p):
    import numpy as np
    from vg_tta.st_component_diagnostics_v1 import contiguous_controls
    from vg_tta.matched_direction_diagnostics_v1 import equal_length_center_pair
    n=len(row['input']['frame_ids']);allkeys=np.ones(n,bool);ni=replay.base['indices'];native=np.zeros(n,bool);native[ni[0]:ni[1]+1]=True
    specs={'noop_temporal':dict(allowed=allkeys,zero=True),'time_all_keys_explicit':dict(allowed=allkeys),
        'time_self_only':dict(allowed=allkeys,mode='self'),'time_native':dict(allowed=native)}
    event=np.array(gt['event_mask'],bool)
    if event.any():
        for stage in ['all','early','middle','late']:specs['time_GT_'+stage]=dict(allowed=event,stage=stage)
        specs['time_GT_soft']=dict(allowed=event,mode='soft',gain=4.)
        for name,z in contiguous_controls(event,row['input']['frame_ids'],p['seeds']).items():
            specs['time_'+name]=dict(allowed=z['mask'],control_audit={k:v for k,v in z.items() if k!='mask'})
        pair,audit=equal_length_center_pair(ni,gt['interval'],row['input']['frame_ids'])
        for name,z in pair.items():specs['time_'+name]=dict(allowed=z['mask'],control_audit=audit)
    return specs


def spatial_specs(replay,row,gt):
    import numpy as np
    from vg_tta.st_causal_audit_v2 import center_roi
    from vg_tta.matched_direction_diagnostics_v1 import spatial_directions
    from scripts.run_gt_direction_focus_v2 import pixel_directions
    centers=gt['full_track_centers']
    if centers is None:return {}
    pred=replay.base['boxes'][:,:2].numpy();target=np.asarray(centers);q=row['input']
    norm,na=spatial_directions(pred,target);pixel,pa=pixel_directions(pred,target,q['width'],q['height'])
    specs={}
    def roi(c):return [center_roi(c[pos],grid,.25) for pos,grid in zip(replay.offsets,replay.grids)]
    for tag,dc,a in [('normalized',norm,na),('pixels',pixel,pa)]:
        for name,c in dc.items():specs['space_'+tag+'_'+name]=dict(roi=roi(c),gain=4.,stage='all',control_audit=a)
    specs['space_native_center']=dict(roi=roi(pred),gain=4.)
    specs['space_float_gain1']=dict(roi=roi(pred),gain=1.)
    specs['space_uniform_mass']=dict(roi=roi(target),gain=4.,mode='uniform')
    return specs


def query_parser():
    from vg_tta.foreground_runtime import QuerySubjectParser
    return QuerySubjectParser(ROOT/'.cache/stanza')


def unavailable(row,path,locksha):
    """Account for an unusable published input, never silently shrink a split."""
    save(path,dict(key=row['key'],input=row['input'],input_unavailable=row['input_unavailable'],
        status='input_unavailable',lock_sha256=locksha,seconds=0.))
    rr=dict(path=str(path),sha256=sha(path),key=row['key'],source=row['input']['source'],
        lock_sha256=locksha,status='input_unavailable',seconds=0.)
    write(path.with_suffix('.json'),rr);return rr


def run(stage,b,cohort,limit=0):
    import torch,numpy as np
    from scripts.run_decota_refine_v1 import configure,student
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.exact_frame_decode_audit_v2 import decode
    p=plan();g=p['legacy_checkpoint_mapping'][cohort];configure();locksha=sha(OUT/'lock.json')
    lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX)
    dest=OUT/stage/b/cohort;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'barrier.json').exists():return
    if b=='ptd':return run_ptd(p,cohort,limit,dest)
    old=read(ROOT/'artifacts/decota_refine_v1/lock.json');m=student(old,b,g);state0=state_digest(m);parser=query_parser()
    labels=label_payload(p) if stage=='causal' else None
    predictor=expert=None
    if stage=='method':
        from methods.decota_refine_uniform_v1.predictor import UniformDeCoTARefinePredictor
        from vg_tta.tg_spatial_tta_v1 import SpatialExpert
        assert sha(Path(old['expert_snapshot'])/'model.safetensors')==old['expert_sha256']
        expert=SpatialExpert(old['expert_snapshot']);cfg=read(ROOT/'methods/decota_refine_uniform_v1/configs.json')[b][g]
        predictor=UniformDeCoTARefinePredictor(m,expert,parser,backbone=b,config=cfg)
    receipts=[];done_new=0
    for row in p['rows'][cohort]:
        j=row['ordinal'];q=row['input'];path=dest/f'{j:06d}.pt';rp=path.with_suffix('.json')
        if rp.exists():
            rr=read(rp);assert sha(path)==rr['sha256'] and rr['lock_sha256']==locksha
            receipts.append(rr);continue
        if path.exists():raise RuntimeError('Unreceipted prediction preserved: '+str(path))
        if row['input_unavailable'] is not None:
            receipts.append(unavailable(row,path,locksha));continue
        started=time.perf_counter();preds=None
        raw,ids=decode(q);subject=parser(q['caption'])['subject']
        if len(ids)<4:raise RuntimeError('Too few positions for two-offset protocol; explicit repair required: '+row['key'])
        if stage=='causal':
            from vg_tta.st_causal_audit_v2 import TubeReplay,TAReplay
            replay=TubeReplay(m,raw,ids,q) if b=='tubedetr' else TAReplay(m,raw,ids,q,subject)
            base=replay.base;gt=labels[row['key']];specs=temporal_specs(replay,row,gt,p);specs.update(spatial_specs(replay,row,gt))
            preds={'native':base}
            for name,spec in specs.items():
                z=replay.run(spec)
                if name in ('noop_temporal','time_all_keys_explicit'):
                    assert torch.equal(z['boxes'],base['boxes']) and torch.equal(z['logits'],base['logits']),name
                z['control_audit']=spec.get('control_audit');preds[name]=z
            repeat=replay.run();assert torch.equal(repeat['boxes'],base['boxes']) and torch.equal(repeat['logits'],base['logits'])
            result=dict(predictions=preds,grids=replay.grids,offsets=replay.offsets,repeat_exact=True,noop_exact=True,
                full_track_eligible=gt['full_track_centers'] is not None,GT_diagnostic_only=True,formal_TTA=False)
            del replay,base,repeat,z
        elif stage=='method':
            from vg_tta.tg_spatial_tta_v1 import visual_query
            r=predictor.predict(raw,ids,q,subject=subject,parsed_visual_query=visual_query(parser,q['caption']),audit_state=True)
            native=r['runtime'];preds=dict(frozen=dict(boxes=native['boxes'],indices=native['native_indices']),
                temporal_only=dict(boxes=native['boxes'],indices=r['indices']),mymethod=dict(boxes=r['boxes'],indices=r['indices']))
            result=dict(predictions=preds,expert=r['expert'],pseudo=r['pseudo'],keyframes=r['keyframes'],config=r['config'],
                temporal_audit=native.get('temporal_audit'),spatial_audit=r['spatial_audit'],GT_used=False,state_reset_checked=True)
            del r,native
        else:
            from vg_tta.unanchored_shift_predictor_v1 import DEFAULT_BASELINE_CONFIGS,predict_episode
            if b=='tastvg':
                from vg_tta.decota_tastvg_episode_v1 import make_batch,episode
                cfg=read(ROOT/'methods/decota_refine_uniform_v1/configs.json')[b][g]
                r=episode(m,make_batch(raw,ids,q,subject,m),dict(lr=cfg['lr'],steps=cfg['steps'],gamma=0.),DEFAULT_BASELINE_CONFIGS,baselines=True,ablations=False)
                result=dict(predictions={k:v for k,v in r['predictions'].items() if k in ['frozen','tent','memo','sar']},audits=r['audits'],GT_used=False)
            else:
                from vg_tta.metrics import interval_from_logits
                from scripts.evaluate_fullspan_scale_shift_external_v1 import _snapshot_full
                cfg=read(ROOT/'methods/decota_refine_uniform_v1/configs.json')[b][g]
                r=predict_episode(m,raw,ids,q['caption'],row['key'],selected_config=dict(lr=cfg['lr'],steps=cfg['steps'],gamma=0.),
                    source_snapshot=_snapshot_full(m),baselines=True,check_controls=True,device='cuda:0',sar_entropy_margin_fraction=.4)
                result=dict(predictions={k:dict(boxes=r['native_predictions'][k]['pred_boxes'],indices=list(interval_from_logits(r['native_predictions'][k]['pred_sted']))) for k in ['frozen','tent','memo','sar']},audits=r['audits'],GT_used=False)
            del r
        result.update(input=q,key=row['key'],query_type=row['query_type'],backbone=b,cohort=cohort,frame_ids=ids,
            lock_sha256=locksha,runner_sha256=sha(__file__),exact_original_pixels=True,seconds=time.perf_counter()-started)
        save(path,result);rr=dict(path=str(path),sha256=sha(path),key=row['key'],source=q['source'],lock_sha256=locksha,seconds=result['seconds'])
        write(rp,rr);receipts.append(rr);done_new+=1
        status(dest/'progress.json',dict(done=len(receipts),total=len(p['rows'][cohort]),last_key=row['key'],last_seconds=result['seconds'],updated=time.time()))
        print('DONE',stage,b,cohort,j+1,len(p['rows'][cohort]),round(result['seconds'],2),flush=True)
        del raw,result,preds
        gc.collect();torch.cuda.empty_cache()
        if limit and done_new>=limit:break
    assert state0==state_digest(m);plan()
    if len(receipts)==len(p['rows'][cohort]):
        missing=sum(r.get('status')=='input_unavailable' for r in receipts)
        write(dest/'barrier.json',dict(receipts=receipts,queries=len(receipts),full_split_complete=not missing,
            all_available_complete=True,unavailable_queries=missing,model_state_unchanged=True,lock_sha256=locksha))


def run_ptd(p,cohort,limit,dest):
    import torch,numpy as np
    from scripts import run_ptd_causal_p0_v2 as pt
    from vg_tta.st_component_diagnostics_v1 import contiguous_controls
    from vg_tta.st_causal_audit_v2 import center_roi
    from vg_tta.matched_direction_diagnostics_v1 import spatial_directions
    from scripts.run_gt_direction_focus_v2 import pixel_directions
    from transformers import AutoProcessor,AutoModelForImageTextToText
    d=pt.utilities();processor=AutoProcessor.from_pretrained(pt.CK,local_files_only=True)
    d.patch_qwen3_video_processor(processor);d.patch_processor_with_time_tokens(processor)
    receipt=read(pt.CK/'OFFICIAL_RECEIPT.json');assert sha(pt.CK/'model.safetensors')==receipt['sha256']
    from train.monkey_patch_forward import replace_qwen3_with_ptd_forward
    replace_qwen3_with_ptd_forward();torch.manual_seed(20260910);torch.set_num_threads(4)
    model,info=AutoModelForImageTextToText.from_pretrained(pt.CK,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda',output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'];model.eval().requires_grad_(False)
    labels=label_payload(p);locksha=sha(OUT/'lock.json');receipts=[];done=0
    for row0 in p['rows'][cohort]:
        j=row0['ordinal'];path=dest/f'{j:06d}.pt';rp=path.with_suffix('.json')
        if rp.exists():
            rr=read(rp);assert sha(path)==rr['sha256'] and rr['lock_sha256']==locksha;receipts.append(rr);continue
        if path.exists():raise RuntimeError('Unreceipted output: '+str(path))
        if row0['input_unavailable'] is not None:
            receipts.append(unavailable(row0,path,locksha));continue
        started=time.perf_counter();q=dict(row0['input']);n=len(q['frame_ids']);pos=np.unique(np.linspace(0,n-1,min(64,n)).round().astype(int))
        q['frame_ids']=[q['frame_ids'][i] for i in pos];row={**row0,'input':q};gt=labels[row['key']];event=np.asarray(gt['event_mask'])[pos]
        x=pt.inputs_for(row,processor);n=len(pos)
        preds={'native_sparse':pt.infer(model,processor,x),'native':pt.infer(model,processor,x,dense=True)}
        repeat=pt.infer(model,processor,x,dense=True);assert repeat['text']==preds['native']['text']
        controls={}
        if event.any():
            ii=np.flatnonzero(event);preds['time_GT_all']=pt.infer(model,processor,x,interval=[int(ii[0]),int(ii[-1])],dense=True)
            for name,z in contiguous_controls(event,q['frame_ids'],p['seeds']).items():
                ii=np.flatnonzero(z['mask']);preds['time_'+name]=pt.infer(model,processor,x,interval=[int(ii[0]),int(ii[-1])],dense=True)
                controls[name]={k:v for k,v in z.items() if k!='mask'}
        centers=gt['full_track_centers'];spatial={}
        if centers is not None and bool(preds['native']['present'].all()):
            target=np.asarray(centers)[pos];pred=preds['native']['boxes'][:,:2].numpy()
            _,hh,ww=map(int,x['video_grid_thw'][0]);grid=(hh//2,ww//2)
            vp=torch.where(x['input_ids'][0]==model.config.video_token_id)[0];assert len(vp)==n*grid[0]*grid[1]
            nd,na=spatial_directions(pred,target);pd,pa=pixel_directions(pred,target,q['width'],q['height'])
            zero=torch.zeros(x['input_ids'].shape[1],device='cuda');spatial['zero_bias']=pt.infer(model,processor,x,prefix_bias=zero,temporal_only=True)
            assert spatial['zero_bias']['indices']==preds['native']['indices']
            for tag,dc in [('normalized',nd),('pixels',pd)]:
                for name,c in dc.items():
                    roi=center_roi(c,grid,.25).flatten().cuda();bias=zero.clone();bias[vp]=roi.float()*math.log(4.)
                    spatial['space_'+tag+'_'+name]=pt.infer(model,processor,x,prefix_bias=bias,temporal_only=True)
            semantic=lambda s:s.split('<|object_ref_end|>')[0]
            assert all(semantic(z['text'])==semantic(preds['native']['text']) for z in spatial.values())
        save(path,dict(input=q,key=row['key'],query_type=row['query_type'],predictions=preds,spatial=spatial,controls=controls,
            parent_positions=pos.tolist(),frame_ids=q['frame_ids'],backbone='ptd',cohort=cohort,repeat_exact=True,GT_diagnostic_only=True,
            mixed_source_training=True,formal_TTA=False,lock_sha256=locksha,seconds=time.perf_counter()-started,checkpoint_sha256=receipt['sha256']))
        rr=dict(path=str(path),sha256=sha(path),key=row['key'],source=q['source'],lock_sha256=locksha,seconds=time.perf_counter()-started)
        write(rp,rr);receipts.append(rr);done+=1
        status(dest/'progress.json',dict(done=len(receipts),total=len(p['rows'][cohort]),last_seconds=rr['seconds'],updated=time.time()))
        print('DONE causal ptd',cohort,j+1,len(p['rows'][cohort]),round(rr['seconds'],2),flush=True)
        del x,preds,spatial;gc.collect();torch.cuda.empty_cache()
        if limit and done>=limit:break
    if len(receipts)==len(p['rows'][cohort]):
        missing=sum(r.get('status')=='input_unavailable' for r in receipts)
        write(dest/'barrier.json',dict(receipts=receipts,full_split_complete=not missing,all_available_complete=True,
            unavailable_queries=missing,mixed_source_training=True,lock_sha256=locksha))


def batch():
    lease=open(OUT/'batch.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=plan();jobs=[]
    # Each cohort is complete before the next; no test-label feedback changes configuration.
    for b in ['tubedetr','tastvg']:
        for c in p['rows']:jobs.append(('causal',b,c))
    for c in p['rows']:jobs.append(('causal','ptd',c))
    for stage in ['method','baselines']:
        for b in ['tubedetr','tastvg']:
            for c in p['rows']:jobs.append((stage,b,c))
    writepath=OUT/'queue.json'
    if not writepath.exists():write(writepath,dict(jobs=jobs,created=time.time(),corruption=False,no_retuning=True))
    for stage,b,c in jobs:
        dest=OUT/stage/b/c
        if (dest/'barrier.json').exists():continue
        py=ROOT/('.venv-ptd-audit/bin/python' if b=='ptd' else '.conda/tubedetr/bin/python')
        status(OUT/'batch_progress.json',dict(state='running',stage=stage,backbone=b,cohort=c,updated=time.time(),pid=os.getpid()))
        cmd=['bash',str(ROOT/'scripts/with_local_cuda.sh'),str(py),'-B','-u',str(Path(__file__).resolve()),stage,'--backbone',b,'--cohort',c]
        subprocess.run(cmd,check=True,cwd=ROOT)
        subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts/score_stvg_fullscale_v1.py'),'--stage',stage,'--backbone',b,'--cohort',c],check=True,cwd=ROOT)
    status(OUT/'batch_progress.json',dict(state='complete',updated=time.time(),pid=os.getpid()))


def launch():
    plan();existing=OUT/'process.json'
    if existing.exists():
        old=read(existing);proc=Path('/proc')/str(old['pid'])/'cmdline'
        if proc.exists() and b'run_stvg_fullscale_v1.py\x00batch' in proc.read_bytes():
            print('ALREADY_RUNNING',old['pid'],flush=True);return
    with open(OUT/'batch.log','ab',buffering=0) as log:
        cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B','-u',str(Path(__file__).resolve()),'batch']
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    status(existing,dict(pid=child.pid,command=cmd,started=time.time(),log=str(OUT/'batch.log')))
    print('STARTED_BATCH',child.pid,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['causal','method','baselines','batch','launch']);ap.add_argument('--backbone',choices=['tubedetr','tastvg','ptd']);ap.add_argument('--cohort',choices=['hcstvg1_test','vidstg_test']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    try:
        if a.stage=='batch':batch()
        elif a.stage=='launch':launch()
        else:run(a.stage,a.backbone,a.cohort,a.limit)
    except Exception:
        status(OUT/'last_failure.json',dict(stage=a.stage,backbone=a.backbone,cohort=a.cohort,error=traceback.format_exc(),updated=time.time()));raise
