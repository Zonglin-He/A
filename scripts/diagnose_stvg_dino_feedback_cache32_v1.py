"""Cached-only admission/slot/reference diagnosis. No teacher, GT or new output."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,time,hashlib,collections,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from vg_tta.stvg_dino_feedback_cache32_v1 import box_stats,iou,quant,slotbin,overlapbin,visiblebin,correlation,score_summary,public_value
D=ROOT/'artifacts/stvg_dino_feedback_cache32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_feedback_cache32'
PRE=ROOT/'artifacts/stvg_dino_conditional_delta32_v2'
ARMS=['One_at_E','Fixed24','Dynamic3']
ARM_MAP=dict(One_at_E='One_correction',Fixed24='Fixed_sample_distill',Dynamic3='On_policy_distill')
HF='.conda/tubedetr/lib/python3.11/site-packages/transformers/models/grounding_dino/modeling_grounding_dino.py'
CFG='.cache/grounding_dino_http_v1/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/config.json'


def export(path,x):put(path,public_value(x))


def verify():
    p=read(D/'PLAN_LOCK.json')
    for f,h in read(D/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f
    return p


def prepare():
    p=read(PRE/'PLAN_LOCK.json');assert len(p['records'])==32 and read(PRE/'FINAL_COMPLETION.json')['count']==32
    files=['vg_tta/stvg_dino_feedback_cache32_v1.py','scripts/diagnose_stvg_dino_feedback_cache32_v1.py','protocols/stvg_dino_feedback_cache32_v1.md',HF,CFG,
        'vg_tta/stvg_dino_conditional_opd32_v1.py','vg_tta/stvg_dino_conditional_delta32_v2.py','methods/decota_final_simplified_v1/observations.py','methods/CURRENT_METHOD.json',
        str((PRE/'PLAN_LOCK.json').relative_to(ROOT)),str((PRE/'GLOBAL_PREDICTION_BARRIER.json').relative_to(ROOT))]
    for r in p['records']:
        files.extend([r['prior_Source'],r['prior_prediction'],str((PRE/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))])
    pins={f:sha(ROOT/f)for f in files}
    old=read(ROOT/'artifacts/stvg_spatial_readout32_v1/PLAN_LOCK.json');wanted={(r['dataset'],r['query_ordinal'])for r in p['records']}
    overlaps=[r for r in old['records']if (r['dataset'],r['query_ordinal'])in wanted]
    export(D/'PLAN_LOCK.json',dict(status='cached_only_locked_before_new_diagnosis',records=p['records'],count=32,per_domain=16,
        primary='repaired fixed24 and dynamic8x3 separately; one-at-E unique anchors as invariant baseline',
        no_new_model=True,no_GT_or_media_or_weights_reads=True,no_threshold_or_parameter_change=True,no_new_predictions=True,
        bins=dict(score=[0,.05,.10,.20,.25,.30,.35,.40,.50,1],geometry_native_final_maxIoU=[0,.25,.5,.75,1],slot=[0,10,50,200,500,900]),
        score_histogram_only_not_threshold_selection=True,exact_gate=.35,within_slot='same frame/text/content index; unique FP32 input references, duplicate score roundoff retained',
        content_causal_control='same reference with alternative content not cached; do not claim causal percentage',
        weights_and_initial_native900_states_not_saved_in_current_predictions=True,
        other_full_native_raw_cache_metadata_match_count=len(overlaps),other_raw_cache_matching_aliases=[r['alias']for r in overlaps],
        shared32_only=True,historical_development_exposure=True,created_epoch=time.time(),CPU_seconds_cap=600))
    export(D/'RUNTIME_LOCK.json',dict(pins=pins,plan_sha256=sha(D/'PLAN_LOCK.json')))
    export(D/'STATUS.json',dict(status='prepared',count=0,total=32))


def read_guard():
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.fsdecode(args[0]);low=path.lower()
        suffix=Path(low).suffix
        if suffix in {'.mp4','.avi','.mkv','.jpg','.jpeg','.png','.webp','.safetensors','.pth'}or '/annotations/'in low:
            raise PermissionError('cached-only diagnosis forbids media/weights/annotations read: '+path)
    sys.addaudithook(audit)


def analyze():
    import torch
    p=verify();read_guard();started=time.perf_counter();anchors=[];candidates=[];native=[];availability=[];byquery=[]
    def chart(b):b=np.asarray(b,np.float64);return np.c_[b[:,:2],np.log(b[:,2:])]
    def tube(z):return np.c_[z[:,:2],np.exp(z[:,2:])]
    for ix,r in enumerate(p['records']):
        z=load(PRE/'predictions'/(r['alias']+'.pt'));s=load(ROOT/r['prior_Source']);e=load(ROOT/r['prior_prediction']);b=np.asarray(e['outputs']['Source_no_correction'],float)
        scale=np.array(read(PRE/'PLAN_LOCK.json')['scale'][r['dataset']]);positions=z['planned_positions'];assert positions==s['expert']['positions4']
        availability.append(dict(alias=r['alias'],dataset=r['dataset'],active=z['active'],observed_anchor_records=len(s['expert']['observations']),
            saved_conditional_fields=['box','score','slot','accepted'],native_probe_maximum_candidates=3,
            initial900_reference_boxes_saved=False,content_embedding_vectors_saved=False,full900_native_target_logits_saved=False,
            original_conditional_input_frames_or_encoder_kwargs_saved=False,alternative_content_same_reference_saved=False))
        for (_,pos),obs in s['expert']['observations'].items():
            probe=obs['probe'];context=bool(obs['receipt']['context_active']);v=np.asarray(probe['target_scores'],float)
            native.append(dict(alias=r['alias'],dataset=r['dataset'],position=pos,context=context,slot_ids_reliable=context,
                original_frame_admitted=bool(probe['accepted']),native_top_score=float(v.max())if len(v)else None,
                native_scores=v.tolist(),native_ids=list(map(int,probe['candidate_ids'])),cached_candidates=len(v)))
        for arm,stored in ARM_MAP.items():
            fit=z['fits'][stored];qrow=dict(alias=r['alias'],dataset=r['dataset'],arm=arm,active=z['active'],candidate_count=0,legal_count=0,any_feedback_count=0)
            for step in fit['rounds']:
                details=step['details'][:1]if arm=='One_at_E'else step['details']
                if arm=='One_at_E':
                    assert all(np.array_equal(details[0]['action'],d['action'])and np.array_equal(details[0]['teacher_action'],d['teacher_action'])for d in step['details'])
                for k,d in enumerate(details):
                    refs=tube(chart(b)+np.asarray(d['action'])*scale)[positions];stats=box_stats(refs);valid=bool(stats['legal'].all());assert valid==d['legal']
                    row=dict(alias=r['alias'],dataset=r['dataset'],arm=arm,round=int(step['round']),sample=k,
                        legal=valid,legal_anchor_count=int(stats['legal'].sum()),invalid_anchor_count=int((~stats['legal']).sum()),
                        individually_valid_anchors_discarded=int(stats['legal'].sum())if not valid else 0,
                        invalid_cx=int((~stats['coordinate_legal'][:,0]).sum()),invalid_cy=int((~stats['coordinate_legal'][:,1]).sum()),
                        invalid_width=int((~stats['coordinate_legal'][:,2]).sum()),invalid_height=int((~stats['coordinate_legal'][:,3]).sum()),
                        nonfinite_anchors=int((~stats['finite']).sum()),returned_anchors=len(d['anchors']),accepted_anchors=d['accepted_anchors'],
                        mean_visible_fraction=float(stats['visible'].mean()),any_feedback=d['accepted_anchors']>0)
                    assert len(d['anchors'])==(len(positions)if valid else 0);candidates.append(row)
                    qrow['candidate_count']+=1;qrow['legal_count']+=valid;qrow['any_feedback_count']+=row['any_feedback']
                    for j,pos in enumerate(positions):
                        if not valid:continue
                        rec=d['anchors'][pos];obs=s['expert']['observations'][('original',pos)];probe=obs['probe'];context=obs['receipt']['context_active']
                        pool=np.asarray(probe['boxes'],float).reshape(-1,4);scores=np.asarray(probe['target_scores'],float);ids=list(map(int,probe['candidate_ids']));slot=int(rec['slot'])
                        top=int(scores.argmax())if len(scores)else None;match=ids.index(slot)if context and slot in ids else None
                        inpool=iou(refs[j],pool)[0];outpool=iou(rec['box'],pool)[0]
                        score=float(rec['score']);assert rec['accepted']==(score>=.35)
                        anchors.append(dict(alias=r['alias'],dataset=r['dataset'],arm=arm,round=int(step['round']),sample=k,position=int(pos),
                            score=score,accepted=bool(rec['accepted']),slot=slot,slot_band=slotbin(slot),context_active=bool(context),
                            slot_in_native_retained_pool=None if not context else match is not None,
                            matched_native_slot_score=None if match is None else float(scores[match]),
                            input_matched_native_final_iou=None if match is None else float(inpool[match]),
                            native_top_score=None if top is None else float(scores[top]),native_frame_admitted=bool(probe['accepted']),
                            input_native_max_iou=float(inpool.max())if len(pool)else None,input_native_top1_iou=float(inpool[top])if top is not None else None,
                            input_high_native_score_max_iou=float(inpool[scores>=.35].max())if (scores>=.35).any()else None,
                            output_native_max_iou=float(outpool.max())if len(pool)else None,
                            input_E_iou=float(iou(refs[j],np.asarray(e['outputs']['E_DINO_support'])[pos])[0,0]),
                            decoder_output_input_iou=float(iou(refs[j],rec['box'])[0,0]),
                            student_visible_fraction=float(stats['visible'][j]),student_fully_inside=bool(stats['inside'][j]),
                            input_reference_fp32_digest=hashlib.sha256(refs[j].astype(np.float32).tobytes()).hexdigest()))
            qrow['query_has_feedback']=qrow['any_feedback_count']>0;byquery.append(qrow)
        assert time.perf_counter()-started<p['CPU_seconds_cap']
    private=save(D/'CACHED_SCALAR_ROWS.pt',dict(candidates=candidates,anchors=anchors,native=native,availability=availability))
    config=read(ROOT/CFG);code=(ROOT/HF).read_text()
    assert config['embedding_init_target']and config['two_stage']and config['num_queries']==900
    assert 'target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)'in code
    groups={}
    for ds in ['hc2','vidstg']:
        groups[ds]={}
        for arm in ARMS:
            cc=[x for x in candidates if x['dataset']==ds and x['arm']==arm];aa=[x for x in anchors if x['dataset']==ds and x['arm']==arm];qq=[x for x in byquery if x['dataset']==ds and x['arm']==arm]
            inside=[x for x in aa if x['student_fully_inside']];near=[x for x in aa if x['input_native_max_iou']is not None and x['input_native_max_iou']>=.75]
            groups[ds][arm]=dict(queries=16,active_queries=sum(x['active']for x in qq),unsupported_queries=sum(not x['active']for x in qq),
                candidate_trajectories=len(cc),illegal_trajectories=sum(not x['legal']for x in cc),legal_trajectories=sum(x['legal']for x in cc),
                legal_trajectories_without_feedback=sum(x['legal']and not x['any_feedback']for x in cc),
                legal_trajectories_with_feedback=sum(x['legal']and x['any_feedback']for x in cc),
                otherwise_valid_anchor_references_discarded=sum(x['individually_valid_anchors_discarded']for x in cc),
                invalid_anchor_references=sum(x['invalid_anchor_count']for x in cc),
                invalid_coordinate_counts={k:sum(x[k]for x in cc)for k in ['invalid_cx','invalid_cy','invalid_width','invalid_height','nonfinite_anchors']},
                query_with_feedback=sum(x['query_has_feedback']for x in qq),anchor_scores=score_summary(aa),
                fully_inside_image=score_summary(inside),overlap_native_final_ge075=score_summary(near),
                slot_bands={v:score_summary([x for x in aa if x['slot_band']==v])for v in ['000-009','010-049','050-199','200-499','500-899']},
                native_overlap_bands={v:score_summary([x for x in aa if overlapbin(x['input_native_max_iou'])==v])for v in ['unknown','[0,.25)','[.25,.50)','[.50,.75)','[.75,1]']},
                visibility_bands={v:score_summary([x for x in aa if visiblebin(x['student_visible_fraction'])==v])for v in ['[0,.5)','[.5,.9)','[.9,1)','1']})
    unique=collections.defaultdict(list)
    for row in anchors:
        if row['arm']=='One_at_E':continue
        unique[(row['alias'],row['position'],row['slot'],row['input_reference_fp32_digest'])].append(row)
    fixedcontent=collections.defaultdict(list);roundoff=[]
    for key,rr in unique.items():
        fixedcontent[key[:3]].append(rr[0]);roundoff.append(max(x['score']for x in rr)-min(x['score']for x in rr))
    within=[]
    for key,rr in fixedcontent.items():
        if len(rr)<2:continue
        score=[x['score']for x in rr];nativequality=[x['input_native_max_iou']for x in rr]
        within.append(dict(alias=key[0],dataset=rr[0]['dataset'],position=key[1],slot=key[2],unique_references=len(rr),
            score_min=min(score),score_max=max(score),score_range=max(score)-min(score),crosses035=min(score)<.35<=max(score),
            correlation_score_native_final_overlap=None if any(x is None for x in nativequality)else correlation(score,nativequality)))
    evidence=dict(status='complete_cached_only',count=32,groups=groups,query_census=byquery,
        score_threshold_remained=.35,threshold_bins_descriptive_not_tuning=True,
        no_new_GT_or_media_or_weights_or_model_or_predictions=True,CUDA_initialized=torch.cuda.is_initialized(),
        CPU_seconds=time.perf_counter()-started,private_scalar_rows_receipt=private)
    assert not torch.cuda.is_initialized()
    export(P/'DIAGNOSIS.json',evidence)
    export(P/'CONTENT_AND_CACHE_AVAILABILITY.json',dict(status='complete',count=32,rows=availability,
        configured_content='900 learned query_position_embeddings rows; not copied encoder ROI/object-query feature',
        config=dict(embedding_init_target=True,two_stage=True,num_queries=900),
        selection='geometrically nearest initial native proposal; no target-score/content-consistency criterion',
        encoder_topk='ranked by maximum text-token logit, not target-token score',
        same_reference_different_content_counterfactual_available=False,initial_reference_nearest_distance_available=False,
        old_other_full900_cache_metadata_overlap=p['other_full_native_raw_cache_metadata_match_count'],
        native_top3_ID_valid_only_context_branch=True,
        missing_controls='No initial900 references, full900 native scores, copied content tensors, or alternative-content responses in current saved paths. Cannot assign causal percentages to content vs position.'))
    export(P/'WITHIN_FIXED_CONTENT.json',dict(status='observed_reference_variation_only',rows=within,
        unique_frame_slot_input_observations=len(unique),same_input_repeated_score_range=quant(roundoff),
        groups={ds:dict(frame_slot_groups=len([x for x in within if x['dataset']==ds]),
            score_ranges=quant([x['score_range']for x in within if x['dataset']==ds]),
            crossing_035_groups=sum(x['crosses035']for x in within if x['dataset']==ds),
            per_group_correlations=quant([x['correlation_score_native_final_overlap']for x in within if x['dataset']==ds]))for ds in groups},
        no_independent_sample_claim=True,no_causal_content_percentage=True))
    export(P/'NATIVE_RETAINED_PROBE_SUMMARY.json',dict(status='cached_original_top3_only',groups={ds:dict(frames=len([x for x in native if x['dataset']==ds]),
        originally_admitted=sum(x['original_frame_admitted']for x in native if x['dataset']==ds),native_top_score=quant([x['native_top_score']for x in native if x['dataset']==ds]),
        context_frames=sum(x['context']for x in native if x['dataset']==ds),fallback_frames=sum(not x['context']for x in native if x['dataset']==ds))for ds in groups},rows=native,
        all900_native_scores_unknown=True,original_gate_also_has_margin005=True))
    export(D/'CPU_COMPLETION.json',dict(status='complete',count=32,CPU_seconds=time.perf_counter()-started,CUDA_initialized=False,new_model_calls=0,new_GT_scores=0,
        cached_scalar_rows=len(anchors),candidate_trajectories=len(candidates),private_receipt=private))
    print({ds:{a:{k:v for k,v in g.items()if k not in ['slot_bands','native_overlap_bands','visibility_bands','fully_inside_image','overlap_native_final_ge075']}for a,g in gg.items()}for ds,gg in groups.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','analyze']);args=parser.parse_args();globals()[args.stage]()
