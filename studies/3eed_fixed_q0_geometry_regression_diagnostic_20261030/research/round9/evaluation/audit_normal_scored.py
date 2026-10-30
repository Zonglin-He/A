"""Independent R9 sealed-result reduction/coordinate audit, no new IoU/model/GT metadata."""
from pathlib import Path
import argparse,collections,hashlib,json,math
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mean(v):return math.fsum(v)/len(v) if v else None
def near(a,b):
 if a is None or b is None:assert a is None and b is None,(a,b)
 else:assert np.allclose(a,b,rtol=0,atol=1e-12),(a,b)
def eligible(r,arm,pop):
 if pop=='all':return True
 if pop=='annotated_spatially_confirmed':return r['identity_diagnostic']['annotated_spatially_confirmed']
 if pop=='common_eligible_all_modes':return all(x['accepted_any_axis'] for x in r['G'].values())
 mode=arm.split('__')[0] if '__' in arm else None
 return all(x[pop] for x in r['G'].values()) if mode is None else r['G'][mode][pop]
def scenestats(rows,deltas):
 g=collections.defaultdict(list)
 for row,delta in zip(rows,deltas):g[row['scene']].append(delta)
 gm=np.array([mean(g[k]) for k in sorted(g)]);ci=np.quantile(np.random.default_rng(901).choice(gm,(2000,len(gm))).mean(1),[.025,.975]).tolist() if len(gm)>1 else None
 return len(g),mean(gm.tolist()),ci

def main(a):
 seal=read(a.scored/'SEALED.json');ps=read(a.predictions/'SEALED.json');assert sha(a.predictions/'SEALED.json')==seal['normal_prediction_seal_sha256']
 for root,s in [(a.scored,seal),(a.predictions,ps)]:
  for name,h in s['files'].items():assert sha(root/name)==h
 d=read(a.scored/'RESULTS_PRIVATE.json');rows=d['rows_PRIVATE'];side=read(a.scored/'PRIVATE_GT7D_SIDECAR.json');gt={r['id']:np.array(r['GT7D_model'],float) for r in side['rows']};targetindex=read(a.scored/'GT_CONTROL_TARGET_INDEX_PRIVATE.json');assert all(set(x)=={'id','target6_modelcoords'} for x in targetindex) and set(gt)=={r['id'] for r in rows}=={r['id'] for r in targetindex}
 for t in targetindex:assert np.array_equal(np.asarray(t['target6_modelcoords']),gt[t['id']][:6])
 maxcoord=0.;coordcells=0
 for r in rows:
  with np.load(a.predictions/(r['id']+'.npz'),allow_pickle=False) as z:boxes=z['arm_boxes_raw6'].astype(float);names=z['arm_names'].tolist();q=int(z['q_source']);assert (z['arm_query_index']==q).all() and q==r['q_source']
  for j,name in enumerate(names):
   ce=r['arms'][name]['coordinate_errors'];e=boxes[j]-gt[r['id']][:6];expected={'center_L2_m':float(np.linalg.norm(e[:3])),'size_L2_m':float(np.linalg.norm(e[3:])),'coordinate_Linf_m':float(abs(e).max()),'center_abs_error_m':abs(e[:3]).tolist(),'size_abs_error_m':abs(e[3:]).tolist(),'center_L2_gt_dimension_normalized':float(np.linalg.norm(e[:3]/gt[r['id']][3:6])),'size_relative_abs_mean':float(np.mean(abs(e[3:])/gt[r['id']][3:6]))}
   for k,v in expected.items():near(ce[k],v);maxcoord=max(maxcoord,float(np.max(abs(np.asarray(ce[k])-v))));coordcells+=1
   near(r['arms'][name]['delta_vs_source'],r['arms'][name]['iou']-r['arms']['source']['iou']);near(r['arms'][name]['selected_center_shift_L2_m'],float(np.linalg.norm(boxes[j,:3]-boxes[0,:3])));assert not r['arms'][name]['selected_size_changed'] and np.array_equal(boxes[j,3:],boxes[0,3:])
  ident=r['identity_diagnostic'];assert ident['target_iou']==r['arms']['source']['iou'];other=ident['best_unique_other_iou'];computed=bool(other is not None and ident['target_iou']>.25 and ident['target_iou']-other>.1);assert ident['annotated_spatially_confirmed']==computed
  if other is None:assert ident['no_other_annotation'] and not computed
  else:near(ident['margin_vs_annotated_others'],ident['target_iou']-other)
 cells=0
 for s in d['summary']:
  pp=[r for r in rows if r['platform']==s['platform'] and eligible(r,s['arm'],s['population'])];n=len(pp);assert n==s['n'];v=[r['arms'][s['arm']]['iou'] for r in pp];delta=[r['arms'][s['arm']]['iou']-r['arms']['source']['iou'] for r in pp];ng,macro,ci=scenestats(pp,delta);assert ng==s['scene_clusters'];near(s['mean_iou'],mean(v));near(s['Acc25'],sum(x>.25 for x in v)/n if n else None);near(s['Acc50'],sum(x>.5 for x in v)/n if n else None);near(s['micro_delta_vs_source'],mean(delta));near(s['equal_scene_delta_vs_source'],macro);near(s['descriptive_scene_CI95'],ci);assert s['negative_vs_source_atol1e8']==sum(x< -1e-8 for x in delta)
  for field,key in [('center_L2_m_mean','center_L2_m'),('size_L2_m_mean','size_L2_m')]:near(s[field],mean([r['arms'][s['arm']]['coordinate_errors'][key] for r in pp]))
  cells+=10
 for s in d['correction_and_cancellation_diagnostics']:
  pp=[r for r in rows if r['platform']==s['platform']];v=[r['arms'][s['arm']]['delta_vs_source'] for r in pp];positive=[x for x in v if x>1e-8];negative=[x for x in v if x< -1e-8];assert s['n']==len(v) and s['positive_count_atol1e8']==len(positive) and s['negative_count_atol1e8']==len(negative) and s['ties_atol1e8']==sum(abs(x)<=1e-8 for x in v);near(s['sum_positive_IoU_delta'],math.fsum(positive));near(s['sum_negative_IoU_delta'],math.fsum(negative));near(s['net_all_IoU_delta'],math.fsum(v));near(s['mean_selected_center_shift_m'],mean([r['arms'][s['arm']]['selected_center_shift_L2_m'] for r in pp]));assert s['size_changed_count']==0;align=[r['G'][s['mode']]['raw_consensus_cosine_GT_correction'] for r in pp if r['G'][s['mode']]['raw_consensus_cosine_GT_correction'] is not None];assert s['raw_consensus_alignment_defined']==len(align) and s['raw_consensus_alignment_positive']==sum(x>0 for x in align) and s['raw_consensus_alignment_negative']==sum(x<0 for x in align);near(s['raw_consensus_alignment_mean_defined'],mean(align));cells+=12
 identity=[]
 for platform in ('drone','quad'):
  pp=[r for r in rows if r['platform']==platform];confirmed=[r for r in pp if r['identity_diagnostic']['annotated_spatially_confirmed']];kernels={}
  for mode in ('rgb_fulltext','pc_only'):
   kernels[mode]={k:{'mean_IoU_param_minus_kernel':mean([r['arms'][mode+'__param'+k]['iou']-r['arms'][mode+'__outputspace'+k]['iou'] for r in pp]),'max_abs_IoU_param_minus_kernel':max(abs(r['arms'][mode+'__param'+k]['iou']-r['arms'][mode+'__outputspace'+k]['iou']) for r in pp)} for k in ('1','3')}
  identity.append({'platform':platform,'n':len(pp),'scene_clusters':len({r['scene'] for r in pp}),'annotated_spatially_confirmed':len(confirmed),'no_other_annotation':sum(r['identity_diagnostic']['no_other_annotation'] for r in pp),'source_Acc25_successes':sum(r['arms']['source']['iou']>.25 for r in pp),'confirmed_set_equals_source_Acc25_successes':all(r['identity_diagnostic']['annotated_spatially_confirmed']==(r['arms']['source']['iou']>.25) for r in pp),'kernel_readout_comparison':kernels})
 out={'state':'independent_R9_normal11_scored_reduction_and_coordinate_audit_pass','auditor_sha256':sha(Path(__file__)),'results_sha256':sha(a.scored/'RESULTS_PRIVATE.json'),'scored_seal_sha256':sha(a.scored/'SEALED.json'),'normal_prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'n':len(rows),'summary_and_cancellation_cells':cells,'coordinate_fields_recomputed':coordcells,'max_coordinate_error_difference_m':maxcoord,'all_summary_CI_seed901_draws2000_recomputed':True,'identity_counts_and_kernel_comparisons':identity,'limits':['No new IoU calculations, model, GPU, original metadata or GT annotation-file access. Coordinate errors recomputed from sealed selected predictions and sealed derived GT7D sidecar.','Identity rule arithmetic audited; raw annotation geometric dedup not rerun, its unit fixtures and original scorer retain provenance.','Confirmed subset explicitly selects targetIoU>.25 and margin>.1; unknown/unconfirmed is not proof of wrong identity.','2Drone/3Quad scenes and prior research-exposed targetVAL; not fresh holdout or many independent domains.']};assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scored',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
