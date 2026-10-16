"""Cross-architecture causal-path audit on historically exposed dev sources."""
import argparse,sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
OUT=ROOT/'artifacts/st_causal_backbone_audit_v2'
PARENT=ROOT/'artifacts/decota_heuristic_study_v1/lock.json'
GROUPS=['hc_to_vid','vid_to_hc']


def prepare():
    import numpy as np
    from scripts.score_decota_heuristic_study_v1 import gt_payload,gt_array
    p=read(PARENT);rows={}
    full=read(ROOT/'artifacts/st_error_direction_dev_v1/spatial_context/lock.json')
    full={x['input']['source']:x for x in full['queries']}
    for g in GROUPS:
        rows[g]=[]
        for r in p['rows']['development'][g]:
            q=r['input'];targets,gt=gt_payload(p,r,g,'development');truth,valid=gt_array(targets,q['frame_ids'],gt)
            centers=None
            if q['source'] in full:
                x=full[q['source']];assert x['input']['frame_ids']==q['frame_ids']
                centers=x['centers_all_frames']
            rows[g].append(dict(**r,diagnostic_GT=dict(interval=gt,boxes=truth.tolist(),valid=valid.tolist(),full_track_centers=centers)))
        for k in ['source','video_sha256']:
            assert not {r['input'][k] for r in rows[g]} & {r['input'][k] for r in p['rows']['evaluation'][g]}
    pins={k:sha(ROOT/k) for k in p['protected_pins']}
    for k in ['methods/CURRENT_METHOD.json','methods/decota_refine_uniform_v1/configs.json']:pins[k]=sha(ROOT/k)
    write(OUT/'lock.json',dict(version='st_causal_backbone_audit_v2',created=time.time(),parent_sha256=sha(PARENT),rows=rows,
        primary_backbones=['tubedetr','tastvg'],third_backbone='official STCAT acquisition in progress; add explicit checkpoint receipt before execution',
        resolution=224,development_only=True,historically_exposed=True,corruption=False,main_method_unchanged=True,
        seeds=[20260910,20260911,20260912],stages=['all','early','middle','late'],
        spatial_oracle='Full-track GT centers only: identical fixed token budget at EVERY input frame, no event-validity pattern',
        temporal_oracle='GT event keys only; all query/output positions and original frames/time IDs retained',
        inference_scope='Frozen encoded context, final native decoder pass. Upstream re-encoding not implied.',
        primary_tests=['GT temporal context vs mean matched random sIoU','full-track GT vs anti-GT direction tIoU'],
        statistics='source paired bootstrap 10000; Holm across available backbone x direction primary tests; exploratory layer/strength tests separate',
        strengths=[2.,4.],no_parameter_updates=True,protected_pins=pins))
    print('LOCKED', {g:dict(n=len(rr),full_track=sum(r['diagnostic_GT']['full_track_centers'] is not None for r in rr)) for g,rr in rows.items()},flush=True)


def plan():
    p=read(OUT/'lock.json');assert sha(PARENT)==p['parent_sha256']
    for f,h in p['protected_pins'].items():assert sha(ROOT/f)==h,f
    return p


def specs(replay,row,p):
    import numpy as np,torch
    from vg_tta.st_component_diagnostics_v1 import contiguous_controls
    from vg_tta.matched_direction_diagnostics_v1 import spatial_directions,equal_length_center_pair
    from vg_tta.st_causal_audit_v2 import center_roi
    from vg_tta.expert_space_time_probe_v1 import region_masks
    from methods.decota_refine_uniform_v1.api import uniform_positions
    from scripts.tune_decota_refine_v1 import anchors,SPATIAL
    ids=row['input']['frame_ids'];n=len(ids);valid=np.asarray(row['diagnostic_GT']['valid']);base=replay.base
    out={'noop_temporal':dict(allowed=np.ones(n,bool),zero=True),
         'time_self_only':dict(allowed=np.ones(n,bool),mode='self')}
    out['time_all_keys_explicit']=dict(allowed=np.ones(n,bool))
    ni=base['indices'];native=np.zeros(n,bool);native[ni[0]:ni[1]+1]=True
    out['time_native']=dict(allowed=native)
    for stage in p['stages']:
        out['time_GT_'+stage]=dict(allowed=valid,stage=stage)
    out['time_GT_soft']=dict(allowed=valid,mode='soft',gain=4.)
    for name,c in contiguous_controls(valid,ids,p['seeds']).items():
        out['time_'+name]=dict(allowed=c['mask'],control_audit={k:v for k,v in c.items() if k!='mask'})
    pair,pa=equal_length_center_pair(ni,row['diagnostic_GT']['interval'],ids)
    for k,v in pair.items():
        out['time_'+k]=dict(allowed=v['mask'],control_audit=pa)
    ex=load(ROOT/f"artifacts/decota_refine_tuning_v1/expert/{row['group']}/{row['input']['index']:06d}.pt")
    assert not ex['GT_used'] and ex['frame_ids']==ids
    pp=anchors(ex['probes'],uniform_positions(ids,[0,n-1],8),SPATIAL)
    for gain in p['strengths']:
        for kind,seed in [('expert',0),('native_center',0),('uniform_mass',0)]+[('random',s) for s in p['seeds']]:
            roi=[]
            for positions,grid in zip(replay.offsets,replay.grids):
                mask,_=region_masks([ids[i] for i in positions],pp,base['boxes'][positions],grid,kind=kind,seed=seed or p['seeds'][0]);roi.append(mask)
            stages=p['stages'] if kind=='expert' and gain==4 else ['all']
            for stage in stages:
                name=f'space_{kind}'+(str(seed) if seed else '')+f'_g{int(gain)}_{stage}'
                out[name]=dict(roi=roi,gain=gain,stage=stage,mode='uniform' if kind=='uniform_mass' else 'hard')
    out['noop_spatial']={**out['space_expert_g4_all'],'zero':True}
    centers=row['diagnostic_GT']['full_track_centers']
    if centers is not None:
        directions,da=spatial_directions(base['boxes'][:,:2].numpy(),np.asarray(centers))
        for k,c in directions.items():
            for stage in (p['stages'] if k=='GT_direction' else ['all']):
                out['oracle_space_'+k+'_'+stage]=dict(roi=[center_roi(np.asarray(c)[pos],grid) for pos,grid in zip(replay.offsets,replay.grids)],gain=4.,stage=stage,geometry_audit=da)
    return out,pp


def run(b,g,limit=None,pixel_mode='native'):
    import torch,numpy as np
    from scripts.run_decota_refine_v1 import student,configure
    from vg_tta.st_causal_audit_v2 import TubeReplay,TAReplay
    from vg_tta.dense_expansion_data_v1 import decode_raw
    from vg_tta.foreground_runtime import state_digest
    p=plan();configure();root=OUT if pixel_mode=='native' else OUT/'exact_pixels'
    if pixel_mode=='exact':
        from vg_tta.exact_frame_decode_audit_v2 import decode as decode_raw
    if b=='stcat':
        from vg_tta.stcat_audit_runtime_v2 import load_model,STCATReplay
        m=load_model(g);adapter=STCATReplay
    else:m=student(read(ROOT/'artifacts/decota_refine_v1/lock.json'),b,g);adapter=TubeReplay if b=='tubedetr' else TAReplay
    state0=state_digest(m);receipts=[]
    for row in p['rows'][g][:limit]:
        row={**row,'group':g};j=row['ordinal'];q=row['input'];dest=root/'runs'/b/g/f'{j:03d}.pt'
        if not dest.exists():
            start=time.perf_counter();raw,ids=decode_raw(q);assert ids==q['frame_ids'] and sha(q['video_path'])==q['video_sha256']
            rr=adapter(m,raw,ids,q) if b in ['tubedetr','stcat'] else adapter(m,raw,ids,q,row['student_subject']['subject'])
            base=rr.base
            if b in row['native'] and pixel_mode=='native':
                old=load(row['native'][b]['path']);assert sha(row['native'][b]['path'])==row['native'][b]['sha256']
                assert torch.equal(base['boxes'],old['predictions']['frozen']['boxes'].float())
                assert torch.equal(base['logits'],old['native_logits'].reshape(len(ids),2).float())
                assert base['indices']==list(old['predictions']['frozen']['indices'])
            ss,pp=specs(rr,row,p);preds={'native':base}
            for name,spec in ss.items():
                pred=rr.run(spec)
                if name.startswith('noop') or name=='time_all_keys_explicit':
                    assert all(torch.equal(pred[k],base[k]) for k in ['boxes','logits']),name
                pred['delta']={k:float((pred[k]-base[k]).abs().max()) for k in ['boxes','logits']}
                pred['control_audit']=spec.get('control_audit',spec.get('geometry_audit'))
                preds[name]=pred
            repeat=rr.run();assert torch.equal(repeat['boxes'],base['boxes']) and torch.equal(repeat['logits'],base['logits'])
            save(dest,dict(backbone=b,group=g,ordinal=j,input=q,frame_ids=ids,grids=rr.grids,predictions=preds,expert_anchors=pp,
                GT_used=True,GT_used_for=['temporal context oracle','full-track spatial oracle if eligible'],formal_TTA=False,
                repeat_exact=True,no_op_exact=True,pixel_mode=pixel_mode,all_input_frames_preserved=True,seconds=time.perf_counter()-start,
                source_hashes={str(ROOT/'vg_tta/st_causal_audit_v2.py'):sha(ROOT/'vg_tta/st_causal_audit_v2.py'),str(Path(__file__)):sha(__file__)},lock_sha256=sha(OUT/'lock.json')))
            print('DONE',b,g,j+1,len(p['rows'][g]),len(preds),round(time.perf_counter()-start,2),flush=True)
            del rr,preds,base,repeat,raw;gc.collect();torch.cuda.empty_cache()
        receipts.append(dict(path=str(dest),sha256=sha(dest),ordinal=j,source=q['source']))
        status(root/f'progress_{b}_{g}.json',dict(done=len(receipts),total=len(p['rows'][g]),unix=time.time()))
    assert state0==state_digest(m)
    if limit is None:write(root/f'barrier_{b}_{g}.json',dict(receipts=receipts,model_state_unchanged=True,GT_used=True,formal_TTA=False,pixel_mode=pixel_mode))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--backbone',choices=['tubedetr','tastvg','stcat']);ap.add_argument('--group',choices=GROUPS);ap.add_argument('--limit',type=int);ap.add_argument('--pixel-mode',choices=['native','exact'],default='native');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    else:run(a.backbone,a.group,a.limit,a.pixel_mode)
