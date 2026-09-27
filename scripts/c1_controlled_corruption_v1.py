"""Input-only preparation and bounded same/cross-checkpoint corruption streams."""
import argparse
import collections
import copy
import gc
import hashlib
import io
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
OUT=ROOT/'artifacts/c1_controlled_corruption_v1'
BASE=ROOT/'artifacts/spatial_consolidation_roles_v1'
CONDITIONS=['clean','noise_light','noise_medium','defocus_light','defocus_medium','jpeg_light','jpeg_medium']
MODELS=['vid_source','hc2_source']


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def pixelhash(x):return hashlib.sha256(x.tobytes()).hexdigest()


def configuration(model):
    from scripts.run_spatial_ssl_gpu_v1 import config
    target=config('vidstg_test')
    if model=='hc2_source':return target
    source=config('hcstvg1_test')
    return replace(target,direction='vid_to_vid_control',source_dataset=source.source_dataset,
                   checkpoint=source.checkpoint,checkpoint_sha256=source.checkpoint_sha256)


def corrupt(frames,ids,source,condition):
    import cv2
    from PIL import Image
    assert frames.dtype==np.uint8 and len(frames)==len(ids)
    if condition=='clean':return frames.copy()
    result=np.empty_like(frames)
    kernel=None
    if condition.startswith('defocus'):
        radius=(2 if condition.endswith('light') else 3)*min(frames.shape[1:3])/224
        extent=math.ceil(radius);grid=np.arange(-extent,extent+1);yy,xx=np.meshgrid(grid,grid,indexing='ij')
        kernel=(xx*xx+yy*yy<=radius*radius).astype(np.float32);kernel/=kernel.sum()
        kernel=cv2.GaussianBlur(kernel,(3,3),.1);kernel/=kernel.sum()
    for i,(frame,fid) in enumerate(zip(frames,ids)):
        if condition.startswith('noise'):
            seed=int(digest(f'C1-corruption-20260923|{source}|{condition}|{fid}')[:16],16)
            rng=np.random.default_rng(seed);sigma=.04 if condition.endswith('light') else .08
            result[i]=np.clip(np.rint(frame.astype(np.float32)+rng.normal(0,sigma*255,frame.shape)),0,255).astype(np.uint8)
        elif condition.startswith('defocus'):
            result[i]=cv2.filter2D(frame,-1,kernel,borderType=cv2.BORDER_REFLECT_101)
        elif condition.startswith('jpeg'):
            buffer=io.BytesIO();Image.fromarray(frame).save(buffer,format='JPEG',quality=70 if condition.endswith('light') else 35,subsampling=2,optimize=False,progressive=False)
            buffer.seek(0);result[i]=np.asarray(Image.open(buffer).convert('RGB'))
        else:raise ValueError(condition)
    assert result.shape==frames.shape and result.dtype==frames.dtype
    return result


def prepare():
    parent=read(BASE/'LOCK.json');splits={};annfiles={}
    for split in ['train','val','test']:
        f=ROOT/f'external/VidSTG-Dataset/annotations/{split}_annotations.json';a=read(f)
        splits[split]=set(str(z['vid']) for z in a);annfiles[split]=dict(path=str(f),sha256=sha(f),entries=len(a),parents=len(splits[split]))
    assert not splits['train']&splits['test'] and not splits['train']&splits['val'] and not splits['val']&splits['test']
    pool=[r for r in parent['rows'].values() if r['cohort']=='vidstg_test']
    assert len(pool)==len({r['source'] for r in pool})==732 and {r['source'] for r in pool}==splits['test']
    ordered=sorted(pool,key=lambda r:digest('C1-corruption-20260923-source-'+r['source']))
    dev=sorted(ordered[:32],key=lambda r:digest('C1-corruption-20260923-order-'+r['source']));follow=ordered[32:64]
    rows=[];followrows=[]
    for role,rr,dest in [('development',dev,rows),('historical_followup_not_fresh',follow,followrows)]:
        for i,r in enumerate(rr):
            assert sha(r['path'])==r['sha256'];x=load(r['path']);q=x['input']
            assert str(q['source'])==r['source'] and str(q['original_video_id'])==r['source']
            assert sha(q['video_path'])==q['video_sha256']
            dest.append(dict(**r,ordinal=i,role=role,input=q,frame_ids=x['frame_ids'],parses=x['parses']))
    assert len({r['input']['video_sha256'] for r in rows+followrows})==64
    sourcefile=ROOT/'artifacts/decota_paper_execution_20260917/vitta/vid_to_hc1/SOURCE_MANIFEST.json'
    sm=read(sourcefile);source_hashes={r['input']['video_sha256'] for r in sm['records']}
    assert all(r['input']['video_sha256'] not in source_hashes for r in rows+followrows)
    reserved=ROOT/'artifacts/c1_fresh_confirmation_v1/vid_train_audit/PROSPECTIVE_SOURCE_RESERVATION.json'
    write(OUT/'DATA_AUDIT.json',dict(status='passed_for_historical_development',annotations=annfiles,train_test_parent_overlap=0,
          train_val_parent_overlap=0,val_test_parent_overlap=0,development_sources=[r['source'] for r in rows],
          followup_sources=[r['source'] for r in followrows],development_followup_parent_overlap=0,
          selected_duplicate_media_hashes=0,available_source_media_manifest=str(sourcefile),source_manifest_sha=sha(sourcefile),
          source_media_comparison_count=len(source_hashes),source_media_exact_overlap=0,
          fresh_confirmation_available=False,all_test_previously_executed=True,
          unsuitable_train_reservation=str(reserved),unsuitable_train_reservation_sha=sha(reserved),
          scope='official parent IDs and available exact bytes; no whole-training-corpus perceptual dedup or unknown pretraining guarantee'))
    write(OUT/'INPUTS.json',rows);write(OUT/'HISTORICAL_FOLLOWUP_UNRUN.json',followrows)
    owned=['scripts/c1_controlled_corruption_v1.py','scripts/analyze_c1_controlled_corruption_v1.py','protocols/c1_controlled_corruption_v1.md']
    pins={str(f):sha(f) for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py')}
    for rel in ['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json','scripts/run_spatial_ssl_gpu_v1.py',
                'scripts/run_spatial_regression_alignment_v1.py','vg_tta/spatial_online_state_v1.py','vg_tta/spatial_consolidation_v1.py',
                'methods/decota_refine_uniform_v1/api.py','methods/decota_refine_v1/api.py',
                'external/TA-STVG/datasets/build.py','external/TA-STVG/experiments/vidstg.yaml','external/TA-STVG/models/pipeline.py']:
        pins[str(ROOT/rel)]=sha(ROOT/rel)
    write(OUT/'LOCK.json',dict(created=time.time(),source_count=32,query_count=32,conditions=CONDITIONS,models=MODELS,
        configs={m:configuration(m).to_dict() for m in MODELS},inputs_sha=sha(OUT/'INPUTS.json'),data_audit_sha=sha(OUT/'DATA_AUDIT.json'),
        labels=parent['labels'],labels_sha=parent['labels_sha256'],parent_lock_sha=sha(BASE/'LOCK.json'),
        pins=pins,code_sha={str(ROOT/f):sha(ROOT/f) for f in owned},max_seconds=7200,max_bytes=8*1024**3,
        max_arrivals=512,max_DINO=1792,max_backwards=12000,GT_adaptation=False,GT_condition_selection=False,
        primary_all32=True,alpha=1/16,one_order=True,independent_confirmation=False))
    status(OUT/'STATUS.json',dict(status='prepared_before_QA',sources=32,conditions=7,models=2))
    print('PREPARED',len(rows),'sources',annfiles,flush=True)


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in {**p['pins'],**p['code_sha']}.items():assert sha(f)==h,f
    assert sha(OUT/'INPUTS.json')==p['inputs_sha'] and sha(OUT/'DATA_AUDIT.json')==p['data_audit_sha']
    return p


def qa():
    from PIL import Image,ImageDraw,ImageFont
    from vg_tta.exact_frame_decode_audit_v2 import decode
    p=verify();rows=read(OUT/'INPUTS.json');records=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for r in rows[:4]:
        frames,ids=decode(r['input']);positions=[int((len(ids)-1)*v) for v in [.25,.5,.75]]
        thumb=frames[positions];fids=[ids[i] for i in positions]
        panel=Image.new('RGB',(1050,100+7*205),'white');draw=ImageDraw.Draw(panel)
        import textwrap
        draw.multiline_text((10,8),'\n'.join(textwrap.wrap(r['input']['caption'],115)),fill='black',font=font)
        for j,c in enumerate(CONDITIONS):
            z=corrupt(thumb,fids,r['source'],c);assert np.array_equal(z,corrupt(thumb,fids,r['source'],c))
            if c=='clean':assert np.array_equal(z,thumb)
            else:assert not np.array_equal(z,thumb)
            records.append(dict(key=r['key'],condition=c,frame_ids=fids,pixel_sha=pixelhash(z),shape=list(z.shape),deterministic=True))
            for k,rgb in enumerate(z):
                im=Image.fromarray(rgb);im.thumbnail((338,180));panel.paste(im,(k*350+5,100+j*205+22))
                draw.text((k*350+5,100+j*205),f'{c} | frame {fids[k]}',fill='black',font=font)
        f=OUT/'qa'/f"{r['key'].replace(':','_')}.jpg";f.parent.mkdir(exist_ok=True);panel.save(f,quality=94)
    write(OUT/'PIXEL_QA.json',dict(status='mathematical_checks_passed_visual_review_pending',records=records,GT_used=False,created=time.time()))
    print('QA_IMAGES_READY',flush=True)


def run(stage):
    import torch
    from scripts.spatial_consolidation_roles_v1 import setup
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.spatial_online_state_v1 import arrival,QUERY
    from vg_tta.spatial_consolidation_v1 import consolidate
    from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.optim import fit_spatial,fit_temporal
    from methods.decota_final_simplified_v1.replay import TemporalReplay
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_refine_uniform_v1.api import reconstruct
    p=verify();assert read(OUT/'VISUAL_QA_REVIEW.json')['ready']
    if stage=='changing':assert read(OUT/'homogeneous/AUDIT.json')['status']=='passed'
    rows=read(OUT/'INPUTS.json');setup();guard=lease();start=time.time();counter=collections.Counter();files={};bytes_=0
    prior=read(OUT/'homogeneous/BARRIER.json') if stage=='changing' else dict(seconds=0,counts={})
    expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);eh=state_hash(expert.model.state_dict());model=None
    try:
        for m in MODELS:
            if model is not None:del model;gc.collect();torch.cuda.empty_cache()
            model=model_load('hcstvg1_test' if m=='vid_source' else 'vidstg_test');mh=state_hash(model.state_dict());cfg=configuration(m)
            for stream in (CONDITIONS if stage=='homogeneous' else ['changing']):
                previous=None;prevhash=None
                for i,r in enumerate(rows):
                    cond=stream if stage=='homogeneous' else ['clean','defocus_medium','jpeg_medium','clean'][i//8]
                    f=OUT/stage/m/stream/f'{i:03d}.pt';assert not f.exists()
                    assert time.time()-start+prior['seconds']<p['max_seconds']
                    frames,ids=decode(r['input']);clean_sha=pixelhash(frames);shifted=corrupt(frames,ids,r['source'],cond)
                    input_sha=pixelhash(shifted);assert (input_sha==clean_sha)==(cond=='clean')
                    del frames
                    batch,records,s=frozen_forward(model,shifted,r);source=detached(s.initial)
                    native=prediction(s.zero['logits'],s.zero['boxes'],records,ids)
                    cc=collections.Counter(arrivals=1,DINO=0,spatial_backwards=0,temporal_backwards=0,full_reinsertions=0,reference_reuses=0)
                    if stage=='homogeneous':
                        ex=observations(expert,r['parses'],shifted,ids,native['indices'],audit=True);anchors=ex['anchors']['single4']
                        cc['DINO']=ex['new_DINO'];cc['actual_observations']=len(ex['actual_observation_positions'])
                        ep=fit_spatial(s,s.zero,anchors,cfg,trace=True);cc['spatial_backwards']+=ep['backwards']
                        temporal=TemporalReplay(s.head,s.temporal_inputs,{**s.zero,'boxes':ep['final']['boxes']})
                        tt=fit_temporal(temporal,records,cfg,trace=True);cc['temporal_backwards']+=tt['backwards'];del temporal
                        direct,da=reconstruct(s.zero['boxes'],anchors,ids,'absolute');reused=None
                    else:
                        ff=OUT/'homogeneous'/m/cond/f'{i:03d}.pt'
                        assert sha(ff)==prior['files'][str(ff)];saved=load(ff)
                        assert saved['pixel_sha']==input_sha and saved['key']==r['key']
                        assert torch.equal(saved['native']['boxes'],native['boxes'])
                        assert all(torch.equal(a,b) for a,b in zip(saved['native']['logits'],native['logits']))
                        ex=saved['expert'];anchors=saved['anchors'];ep=saved['episodic'];tt=saved['temporal'];direct=saved['direct'];da=saved['direct_audit']
                        reused=dict(path=str(ff),sha256=sha(ff));cc['reference_reuses']=len(ex['actual_observation_positions']);cc['actual_observations']=len(ex['actual_observation_positions'])
                    for (_,pos),o in ex['observations'].items():assert o['receipt']['rgb_sha256']==pixelhash(shifted[pos])
                    initial=arrival(source,previous,'O-split');s.initial=detached(initial,'cuda');s.restore(s.initial)
                    with torch.no_grad():before=detached(s.values())
                    online=fit_spatial(s,before,anchors,cfg,trace=True);cc['spatial_backwards']+=online['backwards']
                    assert online['failure'] is None and ep['failure'] is None and tt['failure'] is None
                    assert torch.count_nonzero(initial[QUERY])==0
                    committed=consolidate(online['initial_state'],online['state'],1/16)
                    assert torch.count_nonzero(committed[QUERY])==0
                    if i==0:
                        assert all(torch.equal(v,source[k]) for k,v in initial.items())
                        assert online['losses']==ep['losses'] and torch.equal(online['final']['boxes'].cpu(),ep['final']['boxes'].cpu())
                    formal=prediction(tt['shrunk']['logits'],online['final']['boxes'],records,ids)
                    if m=='hc2_source' and cond=='clean' and stage=='homogeneous':
                        old=load(r['path']);assert torch.equal(native['boxes'],old['native_boxes'])
                        assert native['indices']==old['predictions']['Frozen']['indices']
                        assert torch.equal(ep['final']['boxes'].cpu(),old['spatial']['final']['boxes'])
                        assert formal['indices']==old['predictions']['Full_DeCoTA']['indices'];cc['old_clean_parity']=1
                    # Real end-to-end reinsertion on fixed positions including every boundary.
                    if i in [0,1,7,8,15,16,23,24,31]:
                        for st,boxes in [(initial,before['boxes']),(online['state'],online['final']['boxes'])]:
                            state={**detached(st,'cuda'),**detached(tt['shrunk_state'],'cuda')}
                            expected=dict(boxes=boxes,logits=detached(tt['shrunk']['logits'],'cuda'))
                            with query_subject(model,batch,r['parses']['subject']):full_prediction(model,batch,ids,records,state,expected)
                            cc['full_reinsertions']+=1
                    assert state_hash(model.state_dict())==mh
                    payload=dict(key=r['key'],source=r['source'],ordinal=i,model=m,stream=stream,condition=cond,
                         input=r['input'],pixel_sha=input_sha,clean_pixel_sha=clean_sha,frame_ids=ids,parses=r['parses'],
                         native=native,expert=ex,anchors=anchors,observed=ex['actual_observation_positions'],
                         episodic=ep,temporal=tt,direct=direct,direct_audit=da,before=before['boxes'],online=online,
                         committed=committed,source_state=source,previous_sha=prevhash,formal_indices=formal['indices'],
                         cost=dict(cc),input_reuse=reused,GT_used=False,corruption_label_used_by_method=False,
                         batch_pixels_sha=hashlib.sha256(batch['videos'].tensors.detach().cpu().numpy().tobytes()).hexdigest())
                    save(f,detached(payload,'cpu'));prevhash=sha(f);files[str(f)]=prevhash;bytes_+=f.stat().st_size
                    write(f.with_suffix('.json'),dict(sha256=prevhash,lock_sha=sha(OUT/'LOCK.json'),pixel_sha=input_sha))
                    previous=detached(committed,'cpu');counter.update(cc)
                    assert bytes_<p['max_bytes']
                    assert counter['DINO']+prior['counts'].get('DINO',0)<=p['max_DINO']
                    assert sum(counter[k]+prior['counts'].get(k,0) for k in ['spatial_backwards','temporal_backwards'])<=p['max_backwards']
                    status(OUT/'STATUS.json',dict(status=stage+'_running',model=m,stream=stream,ordinal=i,counts=dict(counter),seconds=time.time()-start))
                    print(stage,m,stream,i,r['key'],'DINO',cc['DINO'],'seconds',round(time.time()-start,2),flush=True)
                    del shifted,batch,records,s,source,before,online,ex,ep,tt,payload,anchors
                    gc.collect();torch.cuda.empty_cache()
        assert state_hash(expert.model.state_dict())==eh
        write(OUT/stage/'BARRIER.json',dict(counts=dict(counter),seconds=time.time()-start,files=files,bytes=bytes_,created=time.time()))
        status(OUT/'STATUS.json',dict(status=stage+'_complete_unscored',counts=dict(counter)))
    except BaseException as e:
        write(OUT/f'FAILURE_{time.time_ns()}.json',dict(stage=stage,error=repr(e),counts=dict(counter),seconds=time.time()-start));raise
    finally:guard.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','qa','homogeneous','changing']);a=ap.parse_args()
    globals()[a.phase]() if a.phase in ['prepare','qa'] else run(a.phase)
