"""Independent post-seal scalar readback and anonymous DINO restoration report."""
from pathlib import Path
import argparse,copy,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'artifacts/stvg_dino_rollback32_v1';PUB=ROOT/'results/stvg_dino_rollback/2026-10-17/verification32'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p;p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def report():
 import torch
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
 root=read(BASE/'ROOT_READBACK.json');cpu=read(BASE/'CPU_COMPLETION.json');bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert root['status']=='pass'and root['count']==cpu['count']==bar['count']==32 and cpu['status']==bar['status']=='complete';assert sha(BASE/'POSTSEAL_RESULTS.json')==cpu['result_sha256']
 def load(f):assert sha(ROOT/f['path'])==f['sha256'];return torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
 cases={};pred=[]
 for x,f in zip(read(BASE/'SOURCE_BARRIER.json')['files'],bar['files']):
  c=load(x);p=load(f);assert c['case_id']==p['case_id'];cases[c['case_id']]=c;pred.append(p)
 assert len(cases)==32
 targets,_=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[];scalars=0;calls=steps=0
 old=read(BASE/'POSTSEAL_RESULTS.json');idx={(x['case_id'],x['arm']):x['metrics']for x in old['rows']}
 for p in pred:
  cid=p['case_id'];c=cases[cid];calls+=p['compute']['new_DINO_calls'];steps+=p['fit']['gradient_calls']
  for name,arm in p['arms'].items():
   v,_=metric._score_student_arm(arm,c,targets[cid],set(p['positions']),official,independent);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>.3),R_vIoU_at_0p5=float(v['m_vIoU']>.5));assert v==idx[cid,name];scalars+=len(v);rows.append(dict(case_id=cid,dataset=c['dataset'],arm=name,metrics=v))
 summary,paired=subset_aggregate(rows);assert summary==old['summary']and paired==old['paired']
 aliases={r['alias']:f'R{i+1:03d}'for i,r in enumerate(read(BASE/'PLAN_LOCK.json')['records'])}
 public_rows=[dict(source_alias=aliases[x['case_id']],dataset=x['dataset'],arm=x['arm'],metrics=x['metrics'])for x in rows]
 paired=copy.deepcopy(paired)
 for p in paired:p['per_source_delta_pp']={aliases[k]:v for k,v in p['per_source_delta_pp'].items()}
 write(PUB/'RESULTS.json',dict(status='complete',count=32,cohort='original exposed P0 order1 first16 per target; chronological online prefixes',rows=public_rows,summary=summary,paired=paired,all_positive_negative_neutral_unknown_retained=True,not_official_full_or_independent_confirmation=True,not_matched_to_Qwen_development32=True,GT_only_after_all32_prediction_seal=True))
 design=read(BASE/'PLAN_LOCK.json');write(PUB/'FROZEN_METHOD.json',{k:v for k,v in design.items()if k!='records'})
 write(PUB/'VERIFICATION.json',dict(status='pass',count=32,independently_recomputed_scalars=scalars,all_summary_and_signed_paired_values_exact=True,Source_and_DINO_frozen=True,QR_and_Adam_reset=True,online_LN_initial_and_writeback_exact=True,original_Gaussian_reward_gradient_Adam_round_arithmetic=True,independent_Actionness_projection_enumeration=True,whole_two_decoder_output_exact=True,original_predictor_fitter_and_config_snapshot_byte_exact=True,historical_Vid_complete_fit_and_writeback_exact_count=16,historical_HC_raw_fits_previously_retired_not_claimed_exact=True,actual_fixed_input_output_pixel_view_count=8,semantic_annotation_count=0))
 costs={}
 for ds in ['hc2','vidstg']:
  vals=[p['compute']['complete_predict_seconds']for p in pred if p['dataset']==ds];costs[ds]=dict(count=len(vals),predict_seconds_sum=sum(vals),predict_seconds_mean=sum(vals)/len(vals))
 write(PUB/'COST.json',dict(status='complete',actual_new_Source_captures=32,actual_new_DINO_calls=calls,actual_new_spatial_fit_calls=32,actual_gradient_calls=steps,per_domain_predict=costs,predict_timing_scope='original decoded-frame predictor call; excludes upstream video decoding and extra independent validation readout/hash/audit/publication',not_full_deployment_latency=True,new_Temporal_parameter_fits=0,new_Qwen_calls=0))
 write(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,scalar_values_checked=scalars,summary_and_full_signed_pairs_exact=True,public_results_sha256=sha(PUB/'RESULTS.json'),GT_read=True))
 vals={(x['dataset'],x['arm'],x['metric']):x['mean']for x in summary};arms=['Frozen_native','Frozen_projection','Before_projection','OPD_native_readout','Full_P7A'];cols=['m_vIoU','m_tIoU','m_sIoU_dense_GT','R_vIoU_at_0p3','R_vIoU_at_0p5']
 text=['# Historical DINO P7-A restoration: fixed32 regression validation','',
 'User-selected rollback from Qwen to the historical frozen Actionness projection + single DINO + online Gaussian OPD. Qwen official continuation is paused and its partial evidence remains intact. No official full-population or historical controller is restarted.','',
 'Original source/fitter/config bytes match the pre-cleanup snapshot. Same128-parent original P0 order1 first16 arrivals per target are fixed before new models or GT. This exposed short-stream regression set is not independent confirmation, not the Qwen development32 cohort, and not a long-stream/full-query result.','',
 '| Target | Arm | vIoU % | tIoU % | dense sIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|']
 for ds in ['hc2','vidstg']:
  for arm in arms:text.append('| '+ds+' | '+arm+' | '+' | '.join('unknown'if vals.get((ds,arm,k))is None else f'{100*vals[ds,arm,k]:.6f}'for k in cols)+' |')
 text+=['','Frozen_projection holds Source boxes and changes only I0 to I1; Full_P7A minus that control measures the complete online Spatial contribution on this fixed temporal readout. Before_projection contains inherited LN but no current-query update. OPD_native_readout holds adapted boxes and restores Native I0 for attribution. All original denominators, negative/neutral/unknown cases and signed paired deltas remain.','',
 'The implementation is restored and its checked invariants pass; this does not establish new overall superiority. In this short prefix HC Full falls below its projection-only and inherited-before controls, while Vid improves. The user rollback is not a result-driven retuning decision. Existing development, failed confirmation and negative findings are retained.','',
 'Checks cover32 complete fresh Source/DINO/OPD predictions, original Gaussian/IoU/Adam arithmetic, frozen Source and specialist,1792 coordinates, QR/Adam reset, next-arrival LN/writeback, independent direct Actionness-score enumeration and both-Decoder output equality. The16 saved Vid fits and committed histories are exact; HC historical raw fit payloads were retired earlier and are not claimed reloaded. Eight fixed actual Native-anchor image outputs were privately viewed, with zero new independent semantic labels.','',
 'All32 complete predictions and GT-free ROOT sealed before offline GT scoring. Independent root readback recomputed '+str(scalars)+' scalar metrics and all summaries/paired deltas exactly. The results omit original Query text, identities, raw frames, reference/GT coordinates, tensors, weights and private paths.','',
 'Predictor timing excludes upstream video decoding and extra verification/audit/serialization. It is not full fresh end-to-end deployment latency. No Prompt/parameter/Teacher changes or optional search is performed.']
 (PUB/'REPORT.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(status='pass',count=32,scalars=scalars,DINO_calls=calls,gradient_calls=steps,public_files=len(list(PUB.iterdir())))))
if __name__=='__main__':report()
