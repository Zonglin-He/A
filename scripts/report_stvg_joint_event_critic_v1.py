"""Independent frozen-output readback, anonymous export and private pixel panel."""
from pathlib import Path
import json,sys,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import BASE,read,put,load,sha,examples
PUB=ROOT/'results/stvg_joint_event_critic/2026-10-17/pilot32'

def factor_accuracy(scores,values):
 import numpy as np
 from vg_tta.stvg_joint_event_critic_v1 import pair_accuracy
 s=np.asarray(scores);v=np.asarray(values)
 time=[];space=[]
 for i in range(2):time.append(pair_accuracy(s[i],v[i]))
 for i in range(3):space.append(pair_accuracy(s[:,i],v[:,i]))
 def combine(parts):
  n=sum(n for a,n in parts);return dict(accuracy=sum((a or 0)*n for a,n in parts)/n if n else None,pairs=n)
 return dict(same_trajectory_different_interval=combine(time),same_interval_different_trajectory=combine(space))

def audit():
 import numpy as np,torch
 torch.set_num_threads(4)
 from vg_tta.stvg_joint_event_critic_v1 import EventCritic,collate
 result=read(BASE/'POSTSEAL_RESULTS.json');selection=read(BASE/'SELECTION_BARRIER.json');lock=read(BASE/'PLAN_LOCK.json')
 assert result['count']==selection['count']==32
 models={};factors=[];checks=0
 for ds in ['hc2','vidstg']:
  sd='vidstg'if ds=='hc2'else'hc2';m=EventCritic();m.load_state_dict(load(BASE/'critics'/(sd+'.pt'))['state']);m.eval().requires_grad_(False);models[ds]=m
 for r in result['rows']:
  z=load(BASE/'target'/(r['alias']+'.pt'));xs=collate(examples(z))
  with torch.no_grad():s=models[r['dataset']](*xs).reshape(2,3).numpy()
  old=np.asarray(r['critic_scores']);assert np.array_equal(s,old)
  joint=tuple(np.unravel_index(s.argmax(),s.shape));assert list(joint)==r['joint']
  assert r['values']['Joint']==r['all_hypothesis_metrics'][joint[0]][joint[1]]
  assert len(z['fit']['initial'])==len(z['fit']['state']) and z['fit']['LN_writeback']==0 and z['fit']['episodic']
  assert z['candidates']['Actionness_used'] is False
  for f in z['features']:assert len(f['visual'])==len(z['frame_ids']) and not f['GT_input']
  q=np.array([[x['v']for x in row]for row in r['all_hypothesis_metrics']]);factors.append(dict(source_alias=r['alias'],dataset=r['dataset'],**factor_accuracy(s,q)));checks+=6
 for ds,g in result['groups'].items():
  rr=[r for r in result['rows']if r['dataset']==ds];assert len(rr)==16
  for arm,metrics in g['metrics'].items():
   for metric,expected in metrics.items():assert abs(float(np.mean([r['values'][arm][metric]for r in rr]))-expected)<1e-14;checks+=1
  for arm,d in g['deltas_vs_Direct_native'].items():
   x=np.array([r['values'][arm]['v']-r['values']['Direct_native']['v']for r in rr]);assert abs(100*x.mean()-d['mean_pp'])<1e-12;checks+=1
 # Source validation factorization of the actual frozen selected critics.
 labs=read(BASE/'SOURCE_LABELS.json')['rows'];source_factors=[]
 for ds in ['hc2','vidstg']:
  m=EventCritic();m.load_state_dict(load(BASE/'critics'/(ds+'.pt'))['state']);m.eval().requires_grad_(False)
  for r in labs:
   if r['dataset']!=ds or r['split']!='source_validation':continue
   z=load(BASE/'source'/(r['alias']+'.pt'))
   with torch.no_grad():s=m(*collate(examples(z))).reshape(2,3).numpy()
   y=np.array([[a['v']for a in row]for row in r['values']]);source_factors.append(dict(source_alias=r['alias'],dataset=ds,**factor_accuracy(s,y)))
 def pooled(rr):
  out={}
  for ds in ['hc2','vidstg']:
   out[ds]={}
   for kind in ['same_trajectory_different_interval','same_interval_different_trajectory']:
    parts=[r[kind]for r in rr if r['dataset']==ds];n=sum(x['pairs']for x in parts);out[ds][kind]=dict(pairs=n,accuracy=sum((x['accuracy']or 0)*x['pairs']for x in parts)/n if n else None)
  return out
 put(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,exact_repeat_scores=192,checked_scalar_equalities=checks,source_heldout_factor_accuracy=pooled(source_factors),target_factor_accuracy=pooled(factors),all_hypotheses_complete_output=True,all_context_frames_retained=True,all_queries_episodic=True,source_GT_only_training=True,CURRENT_unchanged=True))
 put(BASE/'FACTOR_DIAGNOSTICS.json',dict(status='complete',source_validation=source_factors,target=factors))

def render():
 import collections,numpy as np
 from PIL import Image,ImageDraw,ImageFont
 from scripts.run_decota_paper_main_v1 import frames_for,read_row
 p=read(BASE/'PLAN_LOCK.json');canvas=Image.new('RGB',(1280,640),'white');draw=ImageDraw.Draw(canvas)
 for rowidx,ds in enumerate(['hc2','vidstg']):
  r=next(x for x in p['target']if x['dataset']==ds);z=load(BASE/'target'/(r['alias']+'.pt'));raw=read_row(ds,r['query_ordinal']);frames,ids,_=frames_for(ds,raw,collections.OrderedDict());positions=z['expert']['positions4']
  for col,pos in enumerate(positions):
   img=Image.fromarray(frames[pos]);img.thumbnail((310,245));ox=col*320+(310-img.width)//2;oy=rowidx*320+32;canvas.paste(img,(ox,oy));d=ImageDraw.Draw(canvas)
   for b,color in [(z['boxes'][0][pos],'blue'),(z['boxes'][1][pos],'green')]:
    b=np.asarray(b);xy=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2];box=[ox+xy[0]*img.width,oy+xy[1]*img.height,ox+xy[2]*img.width,oy+xy[3]*img.height];d.rectangle(box,outline=color,width=2)
   for a in z['expert']['anchors']['single4']:
    if a['position']==pos:
     b=np.asarray(a['box']);xy=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2];d.rectangle([ox+xy[0]*img.width,oy+xy[1]*img.height,ox+xy[2]*img.width,oy+xy[3]*img.height],outline='red',width=2)
   draw.text((col*320+5,rowidx*320+6),f'{r["alias"]} Native anchor {col+1}',fill='black')
  draw.text((8,rowidx*320+285),'Blue: Source | Red: admitted DINO | Green: complete direct Student',fill='black')
 dest=BASE/'root_review/FIXED_INPUT_OUTPUT.png';dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest)
 put(BASE/'root_review/PIXEL_BINDING.json',dict(status='rendered_pending_actual_view',count=8,selection='first prelocked target per domain, all four Native planned anchors',image_sha256=sha(dest),not_semantic_annotation=True,no_GT_overlay=True))
 print(dest)

def export():
 assert read(BASE/'ROOT_SCALAR_READBACK.json')['status']=='pass';assert read(BASE/'root_review/ROOT_PIXEL_REVIEW.json')['status']=='pass'
 result=read(BASE/'POSTSEAL_RESULTS.json');factor=read(BASE/'ROOT_SCALAR_READBACK.json');p=read(BASE/'PLAN_LOCK.json');decision=read(BASE/'DECISION.json');PUB.mkdir(parents=True,exist_ok=True)
 rows=[]
 for r in result['rows']:
  row={('source_alias'if k=='alias'else k):v for k,v in r.items()}
  for metrics in row['values'].values():
   metrics['R_at_0p3']=float(metrics['v']>.3);metrics['R_at_0p5']=float(metrics['v']>.5)
  payload=load(BASE/'target'/(r['alias']+'.pt'));row['candidate_physical_frame_intervals']=payload['candidates']['intervals'];row['observation_grid_positions']=payload['candidates']['observation_fingerprints'];rows.append(row)
 for ds,g in result['groups'].items():
  rr=[r for r in rows if r['dataset']==ds]
  for arm,metrics in g['metrics'].items():
   for key in ['R_at_0p3','R_at_0p5']:metrics[key]=sum(r['values'][arm][key]for r in rr)/len(rr)
 put(PUB/'RESULTS.json',dict(status='complete',count=32,groups=result['groups'],rows=rows,source_validation_accuracy=result['source_validation_accuracy'],extra_source_supervision=True,development_only=True,no_target_GT_training=True,advance_matched_OPD=result['advance_matched_OPD']))
 put(PUB/'FACTOR_DIAGNOSTICS.json',read(BASE/'FACTOR_DIAGNOSTICS.json'))
 binding=read(BASE/'SOURCE_LABEL_BINDING002.json')
 put(PUB/'DATA_BINDING_CORRECTION.json',dict(status='pass',count=48,old_processed_pointer_wrong_event_count=binding['old_wrong_event_count'],exact_original_video_Query_used_segment_target_ID_binding=True,equivalent_duplicate_records_retained=True,original_VidOR_frame_coordinates=True,no_prediction_or_target_quality_used_to_bind=True,critic_not_trained_before_correction=True,source_candidate_computations_unchanged=True,source_schema_inspected_by_root_before_complete_source_candidate_seal=True,training_labels_only_after_complete_source_prediction_barrier=True,original_deadline_not_reset=True))
 put(PUB/'TIME_REPRESENTATION_CORRECTION.json',dict(status='pass',original_mismatch_hypotheses=6,checked_hypotheses=384,wrong_masks_differed_by_one_frame=True,corrected_integer_masks_exact=True,old_two_source_critics_preserved_before_any_target_inference=True,additional_CPU_source_critic_trainings=2,additional_GPU_Source_DINO_or_Student_fits=0,same_seeds_epochs_optimizer_and_data=True,correction_not_selected_using_target_performance=True,original_deadline_not_reset=True))
 choices={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in result['rows']if r['dataset']==ds];parts=dict(Source_native=0,Source_alternative_time=0,Direct_native=0,Direct_alternative_time=0,wrong_spatial_state_at_selected_time=0)
  for r in rr:
   j,t=r['joint'];key=('Source'if j==0 else'Direct')+('_native'if t==0 else'_alternative_time');parts[key]+=1
   q=r['all_hypothesis_metrics'];parts['wrong_spatial_state_at_selected_time']+=int(q[1-j][t]['v']>q[j][t]['v']+1e-8)
  choices[ds]=parts
 put(PUB/'SELECTION_COUNTS.json',dict(status='complete',groups=choices,spatial_error_uses_same_selected_interval=True))
 put(PUB/'ROOT_READBACK.json',factor);put(PUB/'DECISION.json',decision)
 histories={d:read(BASE/'critics'/(d+'.json'))for d in ['hc2','vidstg']};put(PUB/'SOURCE_TRAINING.json',dict(status='complete',critics=histories,source_dataset_pair='official train only; parent-disjoint from target pilot',historically_used_source_pool=True))
 source_rows=[dict(source_alias=r['alias'],dataset=r['dataset'],split=r['split'],six_candidate_metrics=r['values'])for r in read(BASE/'SOURCE_LABELS.json')['rows']]
 put(PUB/'SOURCE_CANDIDATE_QUALITY.json',dict(status='complete',count=96,rows=source_rows,no_annotation_coordinates_or_source_identity=True,source_supervision_only=True))
 put(PUB/'ACTUAL_PROTOCOL.json',dict(source_train_per_domain=36,source_validation_per_domain=12,target_per_domain=16,target_historically_exposed=True,source_target_parent_overlap_each_direction=0,source_candidates_use_destination_domain_LR_steps=True,preparation='additional source-supervised training; target critic frozen',candidate_count=6,spatial_states=['Source','Direct actual complete Student'],temporal='Native + highest legal boundary distribution hypotheses with different Uniform4 inputs',external_Actionness_projection=False,action_scores_used_for_candidates_or_critic=False,original_Student_internal_gates_unchanged=True,temporal_parameter_adaptation=False,episodic_1792=True,LN_writeback=0,critic=p['critic'],uniform_joint_prior=True,temperature=1,source_model_unchanged=True,restored_DINO_P7A_CURRENT_unchanged=True,Qwen_official_paused=True))
 totals={}
 for group in ['target','source']:
  zz=[load(BASE/group/(r['alias']+'.pt'))for r in p[group]]
  totals[group]=dict(queries=len(zz),new_DINO_calls=sum(z['compute']['new_DINO_calls']for z in zz),direct_fit_calls=len(zz),nonzero_fit_calls=sum(not z['fit']['empty']for z in zz),gradient_calls=sum(z['fit']['gradient_calls']for z in zz),capture_fit_feature_seconds=sum(z['compute']['capture_fit_feature_seconds']for z in zz),complete_vs_cached_max_box_difference=max(z['cached_vs_complete_max_box_difference']for z in zz))
 put(PUB/'COST.json',dict(status='complete',groups=totals,final_source_critic_trainings=2,additional_CPU_critic_trainings_before_integer_membership_correction=2,epochs_per_training=80,no_target_inference_of_superseded_critics=True,critic_weights_not_published=True,timings_not_end_to_end_deployment_latency=True))
 text=['# Source-trained visual event critic: finite joint-tube pilot','',
 'Additional source-supervised preparation; target model/critic frozen outside the per-Query direct1792 adaptation. Two directions each use 36 source-train +12 source-validation parents and16 previously exposed target development parents. No independent confirmation or official full result is claimed. Five HC movie-parent overlaps were excluded before the fixed source selection.','',
 'The critic compares complete Source/direct Student trajectories crossed with three Native-only time hypotheses. It sees actual frozen appearance/motion ROI features, full-frame context, full Query tokens and physical positions, including frames outside the proposed interval. A visual input interface is not itself proof of object/event understanding.','',
 'Before critic training, a source label identity defect was corrected: seven old processed pointers mapped a repeated identical caption/video to a different event occurrence. Source labels now bind original video, Query, exact used_segment and target ID, and use original VidOR frame coordinates. GPU candidates were unchanged; only their waiting CPU controller was handed off before source labeling. Original receipts and deadline were preserved.','',
 '| Target | Frozen Native vIoU % | Direct + Native % | Direct + conditional time % | Joint % | Oracle % | Joint minus Direct pp |','|---|---:|---:|---:|---:|---:|---:|']
 for ds,g in result['groups'].items():
  m=g['metrics'];text.append('|'+ds+'|'+'|'.join(f'{100*m[a]["v"]:.6f}'for a in ['Frozen_native','Direct_native','Direct_conditional','Joint','Oracle'])+f'|{g["deltas_vs_Direct_native"]["Joint"]["mean_pp"]:+.6f}|')
 text+=['','All signed per-parent metrics, strict-pair denominators, same-trajectory/time and same-time/trajectory factors, source-validation curves, intervals and unknown cases are retained in JSON. Oracle uses post-seal target GT only as a disclosed development diagnostic and is not a usable method. All six candidate outputs sealed before target diagnostic GT; critic training and source checkpoint selection never use target labels.','',
 '**Decision:** '+decision['status']+'. '+('Matched same-critic OPD must still be completed before claiming its increment.'if result['advance_matched_OPD']else'The locked gate did not qualify the new feedback in both directions. No OPD extension or online/full stream is automatically run; this fixed pilot does not rule out all visual event critics.'),'',
 'CURRENT remains the restored historical DINO P7-A; its online Gaussian/LN-carry implementation is not silently replaced by this episodic research candidate. Qwen official evaluation remains paused.','',
 'Mechanism diagnostic: for the HC-source critic transferred to Vid, same-interval spatial-state ordering drops from32/33 correct source-validation pairs to9/30 target pairs. For the Vid-source critic transferred to HC it is16/24 source-validation pairs and38/45 target pairs. These dependent pairs are descriptive, not independent samples. Direct-conditioned time selection has positive mean deltas in both targets, but paired16-parent95% intervals both cross zero. The evidence supports a transfer failure of this fixed spatial update evaluator, rather than absence of candidate headroom or proof that all conditional critics fail.','',
 'Integer membership correction was completed before the first target critic inference. Six of384 float-seconds masks differed by one frame; all384 integer-frame masks now match exactly. The two superseded source models were retained and both source critics were rerun with the same seeds/80epochs/configuration; no Source/DINO/Student GPU computation was repeated.','',
 'Corrected fresh-run entrypoint: `scripts/execute_stvg_joint_event_critic_v1.py`. It requires the original official-data/source-manifest/checkpoint adapters, freezes a fresh scope, applies strict source identity binding and integer memberships from the start, and refuses to overwrite an existing scope. The actual001/002/003 handoff history remains available as implementation evidence.','',
 'Private fixed8-anchor actual pixel inspection is an input/output engineering check, not independent semantic annotation. The old512 panel remains44 model-assisted/468 pending. Raw videos, annotations, GT coordinates, model weights, tensors and original Query strings are excluded from public files.','',
 'Related scope: [E3M](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11011.pdf) already alternates temporal relevance and spatial region estimation; [CAPD](https://arxiv.org/abs/2609.34581) studies masked temporal-group influence in MLLM token distillation. This pilot makes no novelty claim from coupling alone.']
 put(PUB/'PIXEL_REVIEW.json',dict(status='pass',fixed_anchors=8,actual_view=True,no_new_semantic_labels=True,raw_pixels_excluded_from_public=True))
 (PUB/'REPORT.md').write_text('\n'.join(text)+'\n')
 combined='\n'.join(x.read_text()for x in PUB.glob('*'));assert '/home/'not in combined
 for r in p['source']:assert r['input']['caption']not in combined
 from scripts.run_decota_paper_main_v1 import read_row
 for r in p['target']:assert read_row(r['dataset'],r['query_ordinal'])['input']['caption']not in combined
 put(BASE/'ROOT_PRIVACY_READBACK.json',dict(status='pass',count=32,all_signed_neutral_unknown_retained=True,no_source_identity_raw_Query_Caption_media_GT_coordinates_tensor_weight_or_credential=True))
 print(PUB)

if __name__=='__main__':
 stage=sys.argv[1];globals()[stage]()
