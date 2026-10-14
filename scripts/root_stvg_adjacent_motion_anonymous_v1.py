"""Independent exhaustive saved-public/source join and arithmetic; no inference/GT scoring."""
import json,hashlib,math,re
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'artifacts/stvg_adjacent_motion_opd_v1';A=B/'anonymous'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
alias={**{f'hc2_{i:03d}':f'HC{i+1:02d}' for i in range(4)},**{f'vidstg_{i:03d}':f'V{i+1:02d}' for i in range(4)}}
counts=Counter()
def eq(a,b,p=''):
 if isinstance(a,dict):
  assert isinstance(b,dict),p
  for k,v in a.items():assert k in b,(p,k);eq(v,b[k],p+'/'+k)
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),(p,'length')
  for i,(x,y) in enumerate(zip(a,b)):eq(x,y,p+'/'+str(i))
 else:
  if isinstance(b,str):
   for c,d in alias.items():b=b.replace(c,d)
  assert a==b,(p,a,b);counts['exact_public_source_leaves']+=1

def rows(a,b,key):
 bm={tuple(alias.get(r[k],r[k]) for k in key):r for r in b};assert len(a)==len(b)==len(bm)
 for r in a:eq(r,bm[tuple(r[k] for k in key)])
def support(boxes):
 present=sum(x is not None for x in boxes)
 valid=sum(x is not None and len(x)==4 and all(math.isfinite(v) for v in x) and x[0]<x[2] and x[1]<x[3] and min(x)>=-1e-7 and max(x)<=1+1e-7 for x in boxes)
 return dict(present_slots=present,valid_slots=valid,missing_slots=4-present,invalid_present_slots=present-valid,at_least_two_slots=int(valid>=2),all_four_slots=int(valid==4),anchor_only=int(valid==1))
def main():
 r=read(B/'ROOT_POSTSEAL_REPORT.json');t=read(B/'ROOT_TEACHER_READBACK.json');raw=read(B/'CANDIDATE_INPUT_BARRIER.json');m=read(A/'metric_and_case_rows.json');f=read(A/'fit_motion_and_oracle_readback.json');pubt=read(A/'teacher_control_readback.json');g=read(A/'candidate_geometry_and_display.json')
 for k in ['case_rows','metric_rows','arm_comparison_rows','candidate_geometry_rows','candidate_display_rows','candidate_oracle_rows','groups']:
  keys=['case_id'] if k in ['case_rows','candidate_oracle_rows'] else ['case_id','arm'] if k in ['metric_rows','arm_comparison_rows'] else ['cohort','dataset','arm'] if k=='groups' else ['case_id','candidate_id']
  rows(m[k],r[k],keys)
 for group in m['groups']:
  sel=[x for x in m['metric_rows'] if x['dataset']==group['dataset'] and x['cohort']==group['cohort'] and x['arm']==group['arm']];assert len(sel)==4
  for k,v in group['metrics_percent'].items():assert abs(sum(x['metrics'][k] for x in sel)*25-v)<1e-10;counts['recomputed_group_metrics']+=1
 for k in ['fit_movement_rows','motion_round_rows']:eq(f['fit_and_motion'][k],r[k])
 eq(f['fit_and_motion']['V03_prior_control'],r['V03_prior_control'])
 for k in ['candidate_geometry_rows','candidate_display_rows','candidate_oracle_rows']:eq(f['candidate_oracle'][k],r[k])
 for step in f['fit_and_motion']['motion_iou_round_summary']:
  rr=[v for v in r['motion_round_rows'] if v['step']==step['round']-1];assert len(rr)==3
  for k in ['opd_loss','motion_loss','total_loss']:assert abs(step['mean_'+k]-sum(v[k] for v in rr)/3)<1e-12
  assert step['optimizer_updated_count']==sum(v['updated'] for v in rr)
  assert step['motion_pair_count_total']==sum(v['motion_valid_pair_count'] for v in rr)
 rows(pubt['requests'],t['rows'],['case_id','kind']);rows(pubt['cases'],t['cases'],['case_id']);eq(pubt['cost'],t['cost'])
 choices={(v['case_id'],v['kind']):v['selected_candidate_id'] for v in pubt['requests']}
 for k,s in pubt['condition_summary'].items():
  rr=[v for v in pubt['requests'] if v['kind']==k];assert len(rr)==8
  assert s['format_valid_count']==sum(v['format_valid'] for v in rr)
  assert s['truncated_count']==sum(v['truncated'] for v in rr)
  if s['same_geometry_count'] is not None:assert s['same_geometry_count']==sum(v['selected_candidate_id']==choices[(v['case_id'],'base')] for v in rr)
  if s['wrong_query_changed_count'] is not None:assert s['wrong_query_changed_count']==sum(v['selected_candidate_id']!=choices[(v['case_id'],'base')] for v in rr)
 byraw={alias[v['case_id']]:v for v in raw['records']};agg=defaultdict(Counter);cost=defaultdict(Counter)
 assert len(g['cases'])==8
 for case in g['cases']:
  cr=byraw[case['case_id']];cs=cr['candidate_geometry'];assert len(case['candidates'])==len(cs)
  eq(case['base_visible_id_by_candidate'],cr['base_visible_id_by_candidate']);eq(case['base_color_palette_index_by_candidate'],cr['base_color_by_candidate']);eq(case['base_draw_order'],cr['base_draw_order'])
  for c,d in zip(case['candidates'],cs):
   assert c['candidate_id']==d['candidate_id'] and c['is_Source_candidate']==d['is_Source'];group='Source' if d['is_Source'] else 'nonSource'
   eq(c['anchor_xyxy_normalized'],d['anchor_box_xyxy']);eq(c['display_support'],support(d['boxes_xyxy']));s=support(d['geometry_boxes_xyxy']);eq(c['tracked_geometry_support'],s)
   for target in [group,'all_candidates']:agg[target].update(s);agg[target]['candidate_count']+=1
   for j,slot in enumerate(c['observation_slots']):
    assert slot['observation_slot']==j+1;eq(slot['display_xyxy_normalized'],d['boxes_xyxy'][j]);eq(slot['tracked_geometry_xyxy_normalized'],d['geometry_boxes_xyxy'][j]);eq(slot['rho'],d['rho_by_observation'][j])
   tr=d['tracklet'];assert len(c['tracklet_trajectory'])==len(tr['trajectory']);assert len(c['transition_diagnostics'])==len(tr['transitions'])
   for j,(x,y) in enumerate(zip(c['tracklet_trajectory'],tr['trajectory'])):
    assert x['dense_step']==j
    for k in ['status','reason']:eq(x[k],y.get(k))
    eq(x['box_xyxy_normalized'],y.get('box_xyxy_norm'));counts['trajectory_rows']+=1
   for j,(x,y) in enumerate(zip(c['transition_diagnostics'],tr['transitions'])):
    assert x['transition_step']==j+1
    for k in x:
     if k=='transition_step':continue
     eq(x[k],y.get('accepted_box_xyxy_norm' if k=='accepted_box_xyxy_normalized' else k))
    counts['transition_rows']+=1
   eq(c['track_cost_total'],tr['cost']['total']);eq(c['track_cost_by_direction'],tr['cost']['directions'])
   for target in [group,'all_candidates']:cost[target].update(c['track_cost_total'])
 eq(g['support_summary_by_group'],dict(agg));eq(g['tracking_cost_summary_by_group'],dict(cost))
 old=read(ROOT/'artifacts/stvg_motion_guided_instance_opd_v1/CANDIDATE_INPUT_BARRIER.json');oldsel={alias[x['case_id']]:x for x in old['records'] if x['case_id'] in alias};oldagg=defaultdict(Counter)
 for case in oldsel.values():
  for c in case['candidates']:
   s=support(c['boxes_xyxy']);tr=c.get('tracklet') or {};dc=tr.get('cost',{}).get('directions') or tr.get('directions',{})
   for k in ['attempted_transitions','accepted_transitions']:s[k]=sum(int(dc.get(d,{}).get(k,0) or 0) for d in ['backward','forward'])
   s['rejected_transitions']=max(0,s['attempted_transitions']-s['accepted_transitions'])
   for group in ['all_candidates','Source' if c['is_Source'] else 'nonSource']:oldagg[group].update(s);oldagg[group]['candidate_count']+=1
 eq(g['historical_lk_comparison']['support_summary_by_group'],{'all_old_candidates':dict(oldagg['all_candidates']),'Source_display':dict(oldagg['Source']),'nonSource_LK_track':dict(oldagg['nonSource'])})
 assert g['preparation_elapsed_cpu_seconds']==raw['elapsed_cpu_seconds']
 # Exhaustive leaf scan of anonymous outputs: paths/private text/GT coordinates absent.
 def scan(x):
  if isinstance(x,dict):
   for k,v in x.items():
    assert k.lower() not in {'query','prompt','raw_output','raw_response','gt_boxes','gt_box','gt_span','truth','truths','weights','state_dict','parameters','gradient','physical_frame_id','physical_frame_ids','positions'},k
    scan(v)
  elif isinstance(x,list):
   for v in x:scan(v)
  elif isinstance(x,str):assert not any(v in x for v in ['/home/wwww','/attachments/','hc2_00','vidstg_00'])
 for p in A.glob('*.json'):scan(read(p))
 out=dict(status='pass',scope='all retained public-source leaves and independently recomputed aggregates; no GPU or new GT scoring',metric_rows=72,candidate_geometry_rows=43,candidate_display_rows=43,teacher_calls=48,actual_fits=9,optimizer_steps=45,counts=dict(counts),anonymous_files={p.name:sha(p) for p in sorted(A.glob('*.json'))},GT_coordinates_and_private_text_excluded=True,all_positive_negative_neutral_unknown_retained=True)
 for name in ['ROOT_ANONYMOUS_SCALAR_READBACK.json','PUBLIC_SCALAR_AUDIT.json']:(B/name).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='anonymous_files'}))
if __name__=='__main__':main()
