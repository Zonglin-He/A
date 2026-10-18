"""CPU-only reference-offset and net-decoder diagnostics; no new model/GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,time,collections,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from vg_tta.stvg_dino_feedback_cache32_v1 import iou,quant,correlation,public_value
from scripts.score_stvg_joint_robust_correction32_v1 import paired
D=ROOT/'artifacts/stvg_dino_reference_path_cache32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_reference_path_cache32'
PRE=ROOT/'artifacts/stvg_dino_content_match32_v1'
HF='.conda/tubedetr/lib/python3.11/site-packages/transformers/models/grounding_dino/modeling_grounding_dino.py'


def export(path,x):put(path,public_value(x))


def verify():
    p=read(D/'PLAN_LOCK.json')
    repair=read(D/'CPU_EXPORT_REPAIR003.json')if (D/'CPU_EXPORT_REPAIR003.json').exists()else None
    if repair:
        assert sha(D/'RUNTIME_LOCK.json')==repair['runtime_sha256']
        assert sha(ROOT/repair['preserved_source'])==repair['original_sha256']
    for f,h in read(D/'RUNTIME_LOCK.json')['pins'].items():
        expected=repair['repaired_sha256']if repair and f==repair['source']else h
        assert sha(ROOT/f)==expected,f
    return p


def read_guard():
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        s=os.fsdecode(args[0]).lower()
        if Path(s).suffix in {'.mp4','.avi','.mkv','.jpg','.jpeg','.png','.webp','.safetensors','.pth'}or '/annotations/'in s:
            raise PermissionError('cached CPU-only diagnosis blocks media/weights/annotation: '+s)
    sys.addaudithook(audit)


def prepare():
    old=read(PRE/'PLAN_LOCK.json');assert read(PRE/'FINAL_COMPLETION.json')['status']=='complete'
    files=['scripts/diagnose_stvg_dino_reference_path_cache32_v1.py','scripts/audit_stvg_dino_reference_path_cache32_v1.py','scripts/report_stvg_dino_reference_path_cache32_v1.py','protocols/stvg_dino_reference_path_cache32_v1.md',HF,'vg_tta/stvg_dino_content_match32_v1.py','vg_tta/stvg_dino_conditional_opd32_v1.py','methods/CURRENT_METHOD.json',
        'artifacts/stvg_spatial_readout32_v1/predictions/H009.pt','artifacts/stvg_dino_feedback_cache32_v1/SUPPLEMENTAL_INPUT_LOCK.json']
    for r in old['records']:files.extend([r['prior_Source'],r['prior_prediction'],str((PRE/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))])
    export(D/'PLAN_LOCK.json',dict(status='locked_cached_diagnosis',records=old['records'],scale=old['scale'],count=32,
        threshold=.35,no_model_GT_media_weights_reads=True,CPU_seconds_cap=600,
        bins=dict(native_final_IoU=[0,.25,.5,.75,.9,1],center_relative=[0,.5,1,2,'inf'],log_size=[0,.25,.5,1,'inf']),
        position_control='same frame/high content/mask slot; unique FP32 references',
        native_box='exact slot and score match only, final output not initial proposal or GT',
        no_layer_measurement_without_saved_trace=True,no_new_predictions=True,created_epoch=time.time()))
    export(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in files},plan_sha256=sha(D/'PLAN_LOCK.json')))
    export(D/'STATUS.json',dict(status='prepared',count=0,total=32))


def binof(v,edges):
    for a,b in zip(edges[:-1],edges[1:]):
        if a<=v<b or v==b==edges[-1]:return f'[{a},{b}{"]"if b==edges[-1]else")"}'
    raise ValueError(v)


def summary(rr):
    return dict(references=len(rr),queries=len(set(x['alias']for x in rr)),frames=len(set((x['alias'],x['position'])for x in rr)),
        admitted=sum(x['accepted']for x in rr),native_frame_score_ge035=sum(x['native_high_score']>=.35 for x in rr),
        conditional_score=quant([x['score']for x in rr]),native_high_score=quant([x['native_high_score']for x in rr]),
        input_native_final_IoU=quant([x['input_native_iou']for x in rr]),output_native_final_IoU=quant([x['output_native_iou']for x in rr]),
        net_IoU_gain_pp=quant([100*(x['output_native_iou']-x['input_native_iou'])for x in rr]),
        net_toward_native=sum(x['output_native_iou']>x['input_native_iou']+1e-6 for x in rr),net_away_native=sum(x['output_native_iou']<x['input_native_iou']-1e-6 for x in rr),
        input_center_relative=quant([x['input_center_relative']for x in rr]),input_log_size=quant([x['input_log_size']for x in rr]),
        output_center_relative=quant([x['output_center_relative']for x in rr]),output_log_size=quant([x['output_log_size']for x in rr]))


def analyze():
    import torch
    p=verify();read_guard();start=time.perf_counter();rows=[];frames=[];inventory=[];missing=[];full=load(ROOT/'artifacts/stvg_spatial_readout32_v1/predictions/H009.pt');fullchecks=[]
    for ix,r in enumerate(p['records']):
        z=load(PRE/'predictions'/(r['alias']+'.pt'));s=load(ROOT/r['prior_Source']);e=load(ROOT/r['prior_prediction']);base=np.asarray(e['outputs']['Source_no_correction'],float)
        chart=np.c_[base[:,:2],np.log(base[:,2:])];scale=np.array(p['scale'][r['dataset']]);native={}
        inventory.append(dict(alias=r['alias'],dataset=r['dataset'],active=z['active'],saved_final_conditional=True,saved_actions=True,
            native_initial_reference=False,conditional_layer_references=False,conditional_layer_hidden=False,conditional_layer_logits=False,
            decoder_sampling_locations=False,decoder_attention_weights=False,student_layers_are_not_teacher_layers=True))
        for control in z['controls']:
            pos=control['position'];obs=s['expert']['observations'][('original',pos)];probe=obs['probe'];slot=control['native_top_slot'];score=control['native_top_score'];ids=list(probe['candidate_ids']);box=None
            reason='fallback_placeholder_slot_not_recoverable'if not obs['receipt']['context_active']else'native_top_slot_not_in_retained_pool'
            if obs['receipt']['context_active']and slot in ids:
                j=ids.index(slot);assert float(probe['target_scores'][j])==score
                box=np.asarray(probe['boxes'][j],float);reason='exact_native_slot_and_target_score';native[pos]=box
            if r['alias']=='H018':
                assert list(full['frame_ids'])==list(s['frame_ids'])
                previous=full['old_expert']['observations'][('original',pos)]
                assert all(previous['receipt'][key]==obs['receipt'][key]for key in ['text','rgb_sha256','inputs','target_span'])
                raw=full['raw'][pos]['detection'];assert float(raw['target_all_scores'][slot])==score
                if box is not None:
                    rawbox=np.asarray(raw['all_boxes'][slot],np.float32);lo=np.clip(rawbox[:2]-rawbox[2:]/2,0,1);hi=np.clip(rawbox[:2]+rawbox[2:]/2,0,1)
                    assert np.array_equal(box,np.r_[(lo+hi)/2,hi-lo])
                fullchecks.append(dict(alias=r['alias'],position=pos,slot=slot,exact_score_and_original_clipped_roundtrip_box=True,initial_reference_available=False,layer_trace_available=False))
            f=dict(alias=r['alias'],dataset=r['dataset'],position=pos,native_top_slot=slot,native_top_score=score,native_final_box_available=box is not None,reason=reason)
            frames.append(f)
            if box is None:missing.append(f)
        for k,action in enumerate(z['actions']):
            zz=chart+action*scale;allrefs=np.c_[zz[:,:2],np.exp(zz[:,2:])]
            for pos in z['planned_positions']:
                a=z['matched']['target_high']['details'][k]['anchors'].get(pos)
                if a is None or pos not in native:continue
                ref=allrefs[pos].astype(np.float32).astype(float);out=np.asarray(a['box'],float);n=native[pos];control=next(c for c in z['controls']if c['position']==pos)
                def errors(b):return float(np.linalg.norm((b[:2]-n[:2])/n[2:])),float(np.linalg.norm(np.log(b[2:]/n[2:])))
                ic,isize=errors(ref);oc,osize=errors(out);inIoU=float(iou(ref,n)[0,0]);outIoU=float(iou(out,n)[0,0])
                rows.append(dict(alias=r['alias'],dataset=r['dataset'],position=pos,sample=k,content_slot=a['slot'],mask_slot=a['mask_slot'],score=a['score'],accepted=a['accepted'],
                    native_high_score=control['native_top_score'],input_native_iou=inIoU,output_native_iou=outIoU,
                    input_center_relative=ic,output_center_relative=oc,input_log_size=isize,output_log_size=osize,
                    center_motion_relative=float(np.linalg.norm((out[:2]-ref[:2])/n[2:])),log_size_motion=float(np.linalg.norm(np.log(out[2:]/ref[2:]))),
                    input_digest=hashlib.sha256(ref.astype(np.float32).tobytes()).hexdigest(),input_iou_bin=binof(inIoU,[0,.25,.5,.75,.9,1]),
                    input_center_bin=binof(ic,[0,.5,1,2,float('inf')]),input_log_size_bin=binof(isize,[0,.25,.5,1,float('inf')]),
                    output_iou_bin=binof(outIoU,[0,.25,.5,.75,.9,1])))
        assert time.perf_counter()-start<p['CPU_seconds_cap']
    groups={};fixed=collections.defaultdict(list)
    for x in rows:fixed[(x['alias'],x['position'],x['content_slot'],x['mask_slot'])].append(x)
    within=[]
    for key,rr in fixed.items():
        unique={x['input_digest']:x for x in rr};rr=list(unique.values())
        if len(rr)<2:continue
        near=max(rr,key=lambda x:x['input_native_iou']);far=min(rr,key=lambda x:x['input_native_iou'])
        within.append(dict(alias=key[0],dataset=rr[0]['dataset'],position=key[1],content_slot=key[2],mask_slot=key[3],unique_refs=len(rr),
            input_IoU_span=near['input_native_iou']-far['input_native_iou'],near_score=near['score'],far_score=far['score'],
            near_minus_far_score=near['score']-far['score'],near_minus_far_output_IoU=near['output_native_iou']-far['output_native_iou'],
            input_IoU_score_correlation=correlation([x['input_native_iou']for x in rr],[x['score']for x in rr]),
            crosses035=min(x['score']for x in rr)<.35<=max(x['score']for x in rr)))
    queries=[]
    for r in p['records']:
        rr=[x for x in rows if x['alias']==r['alias']];ww=[x for x in within if x['alias']==r['alias']]
        queries.append(dict(alias=r['alias'],dataset=r['dataset'],matched_refs=len(rr),fixed_mask_groups=len(ww),
            near_minus_far_score=None if not ww else float(np.mean([x['near_minus_far_score']for x in ww])),
            near_minus_far_output_IoU=None if not ww else float(np.mean([x['near_minus_far_output_IoU']for x in ww])),
            mean_net_native_IoU_gain_pp=None if not rr else float(np.mean([100*(x['output_native_iou']-x['input_native_iou'])for x in rr])) ))
    for ds in ['hc2','vidstg']:
        rr=[x for x in rows if x['dataset']==ds];ff=[x for x in frames if x['dataset']==ds];qq=[x for x in queries if x['dataset']==ds];ww=[x for x in within if x['dataset']==ds]
        groups[ds]=dict(queries=16,active=sum(x['active']for x in inventory if x['dataset']==ds),native_frames=len(ff),native_final_frames_recovered=sum(x['native_final_box_available']for x in ff),
            missing_reasons=dict(collections.Counter(x['reason']for x in ff if not x['native_final_box_available'])),all_matched=summary(rr),
            input_IoU_bins={key:summary([x for x in rr if x['input_iou_bin']==key])for key in sorted(set(x['input_iou_bin']for x in rr))},
            input_center_bins={key:summary([x for x in rr if x['input_center_bin']==key])for key in sorted(set(x['input_center_bin']for x in rr))},
            input_size_bins={key:summary([x for x in rr if x['input_log_size_bin']==key])for key in sorted(set(x['input_log_size_bin']for x in rr))},
            recovered_ge075=summary([x for x in rr if x['output_native_iou']>=.75]),recovered_ge09=summary([x for x in rr if x['output_native_iou']>=.9]),
            input_ge075=summary([x for x in rr if x['input_native_iou']>=.75]),native_pass_frames_only=summary([x for x in rr if x['native_high_score']>=.35]),
            query_net_native_IoU_gain_pp=paired([x['mean_net_native_IoU_gain_pp']for x in qq]),
            fixed_content_and_mask=dict(groups=len(ww),crosses035=sum(x['crosses035']for x in ww),
                query_near_minus_far_score=paired([x['near_minus_far_score']for x in qq]),
                query_near_minus_far_output_IoU=paired([x['near_minus_far_output_IoU']for x in qq])))
    private=save(D/'SCALAR_ROWS_PRIVATE.pt',dict(rows=rows,frames=frames,within=within,queries=queries))
    export(P/'DIAGNOSIS.json',dict(status='complete_cached_only',count=32,groups=groups,query_rows=queries,
        no_new_model_or_GT_scores=True,CUDA_initialized=torch.cuda.is_initialized(),CPU_seconds=time.perf_counter()-start,
        comparison_native_is_clipped_final_candidate_not_GT_or_initial_proposal=True,private_rows_receipt=private))
    export(P/'CACHE_AVAILABILITY.json',dict(status='complete',count=32,rows=inventory,frame_recovery=frames,full900_positive_control=fullchecks,
        all_DINO_intermediate_boxes_hidden_logits_initial_reference_attention_missing=True,
        no_measured_layer_of_failure=True,no_center_size_causal_percentages=True))
    export(P/'WITHIN_FIXED_CONTENT_MASK.json',dict(status='observed_reference_variation_same_other_inputs',rows=within,query_rows=queries,
        geometry_relative_to_native_final=True,no_GT_used=True,not_separate_independent_samples=True))
    assert not torch.cuda.is_initialized()
    export(D/'CPU_COMPLETION.json',dict(status='complete',count=32,CPU_seconds=time.perf_counter()-start,CUDA_initialized=False,new_model_calls=0,new_GT_scores=0,new_predictions=0,scalar_refs=len(rows),private_receipt=private))
    put(D/'STATUS.json',dict(status='complete_CPU_pending_audit_publication',count=32,total=32),True)
    print({ds:dict(recovered=g['native_final_frames_recovered'],frames=g['native_frames'],all=g['all_matched'],fixed=g['fixed_content_and_mask'])for ds,g in groups.items()})


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','analyze']);globals()[a.parse_args().stage]()
