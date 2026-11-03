"""Read-only audit of saved expert-anchor outputs and fixed-support scoring.

No predictor, model, Source, expert or method change. All old artifacts remain
immutable. GT is used only to verify the already reported frame-IoU estimand.
"""
from cpu import *
A=D/'observed_frame_audit_v1';B=P/'observed_frame_audit_v1'
def prepare():
 assert not(A/'plan.json').exists()
 ff=[Path(__file__),R/'stvg/spatial.py',R/'stvg/current_shared_roi.py',R/'tools/full_score.py',S/'cpu.py',R/'studies/pure_expert_interpolation_v1/run.py',D/'cpu_plan.json']
 for ds in COUNTS:ff += [D/'pure_full'/ds/'sealed.json',D/'pure_full'/ds/'metrics.json',D/'pure_full'/ds/'reporting_receipt.json',RUN/ds/'sealed.json']
 p=dict(scope='Read-only full-cohort observation-box/index/GT-support/inside-outside-event audit. No new method/predictor/model/Source/expert/optimizer/media. Saved predictions only; optional first-three saved-geometry DP traces per domain reproduce the same existing output, never overwrite predictions.',
  support='Original8positions mapped to physical frame IDs; same GT-supported sampled observations for both methods. Split original metric all annotated observations into common predicted interval inside/outside, half-open. Missing support remains unscorable not0. Exact/within1e-12 coordinate checks, E/expert candidate provenance; equal-query means and10000parentCI.',
  files_sha256={str(f.relative_to(R)):sha(f)for f in ff});p['fingerprint']=digest(p);atomic(A/'plan.json',p);atomic(B/'protocol.json',p)
def validate_audit():
 p=read(A/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for f,h in p['files_sha256'].items():assert sha(R/f)==h,f
 return p
def run():
 plan=validate_audit();torch.set_num_threads(1);sys.addaudithook(parent.deny_gpu_media);start=time.perf_counter();reports={}
 from stvg.metrics_base import xyxy
 from stvg.metrics import box_iou
 from stvg.spatial import SpatialGeometry,evidence_dp
 for ds in COUNTS:
  truth=SEL.labels(ds);seal=read(D/'pure_full'/ds/'sealed.json');original=fixed(RUN/ds/'sealed.json');metrics={r['id']:r for r in read(D/'pure_full'/ds/'metrics.json')['private_rows']};rr=[];traces=[]
  counts={k:0 for k in ['queries','observations','pure_not_exact_expert','Current_exact_expert','Current_exact_E','Current_neither_E_nor_expert','inside_event_observations','inside_event_Current_expert','inside_event_Current_E_not_expert','outside_event_observations','outside_event_Current_not_E','outside_event_Current_not_expert','GT_supported_observations','GT_supported_inside','GT_supported_outside','GT_inside_Current_E_not_expert','GT_inside_Current_expert','GT_outside_Current_not_expert']};maxerr=0.
  for row in rows(ds):
   parent.resources(start);key=row['id'];e=seal['files'][key];cap_e=original['files'][key];cp=RUN/ds/cap_e['capture_path'];assert sha(cp)==cap_e['capture_sha256']and sha(D/e['path'])==e['sha256'];cap=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);gg=cap['geometry'];expert=np.asarray(gg['native_boxes'],float);E=np.asarray(cap['E'],float);g=truth[key];gt={int(f):b for f,b in g['boxes'].items()};counts['queries']+=1
   with np.load(D/e['path'])as z:
    ids=z['frame_ids'];pos=z['positions8'];current=z['Current'];pure=z['PureMatched'];iv=z['Current_interval'];assert np.array_equal(iv,z['PureMatched_interval'])and list(ids)==cap['frame_ids']and list(pos)==cap['positions8'];assert expert.shape==(8,4)
    flags=dict(expert=np.all(current[pos]==expert,axis=1),E=np.all(current[pos]==E[pos],axis=1),pure_expert=np.all(pure[pos]==expert,axis=1));inside=(ids[pos]>=iv[0])&(ids[pos]<iv[1]);different_inside=inside&~flags['expert'];support=np.array([int(ids[t])in gt for t in pos]);counts['observations']+=8;counts['pure_not_exact_expert']+=int((~flags['pure_expert']).sum());counts['Current_exact_expert']+=int(flags['expert'].sum());counts['Current_exact_E']+=int(flags['E'].sum());counts['Current_neither_E_nor_expert']+=int((~flags['expert']&~flags['E']).sum());counts['inside_event_observations']+=int(inside.sum());counts['inside_event_Current_expert']+=int((inside&flags['expert']).sum());counts['inside_event_Current_E_not_expert']+=int(different_inside.sum());counts['outside_event_observations']+=int((~inside).sum());counts['outside_event_Current_not_E']+=int((~inside&~flags['E']).sum());counts['outside_event_Current_not_expert']+=int((~inside&~flags['expert']).sum());counts['GT_supported_observations']+=int(support.sum());counts['GT_supported_inside']+=int((support&inside).sum());counts['GT_supported_outside']+=int((support&~inside).sum());counts['GT_inside_Current_E_not_expert']+=int((support&different_inside).sum());counts['GT_inside_Current_expert']+=int((support&inside&flags['expert']).sum());counts['GT_outside_Current_not_expert']+=int((support&~inside&~flags['expert']).sum())
    geom_boxes=np.asarray(gg['boxes']);assert np.array_equal(geom_boxes[pos,0],E[pos])and np.array_equal(geom_boxes[pos,1],expert)and np.array_equal(geom_boxes[pos,2],expert)
    assert flags['pure_expert'].all()and(flags['expert']|flags['E']).all()and flags['E'][~inside].all()
    pixels={a:xyxy(b,row['metadata']['width'],row['metadata']['height'])for a,b in [('Current',current),('Pure',pure)]}
    if ds=='hc2':pixels={a:np.maximum(b,0)for a,b in pixels.items()}
    frame_values={a:np.array([float(box_iou(bb[int(t)],gt[int(ids[int(t)])]))if support[j]else np.nan for j,t in enumerate(pos)])for a,bb in pixels.items()};all_difference=frame_values['Current']-frame_values['Pure']
    for a,b in [('Current','Current'),('Pure','PureMatched')]:
     val=metrics[key]['values'][b]['observed_actual_sIoU'];v=float(np.nanmean(frame_values[a]))if support.any()else None;assert(val is None and v is None)or abs(v-val)<1e-12
     if v is not None:maxerr=max(maxerr,abs(v-val))
    # Exact additive contribution to the original query-equal all-observed delta.
    n=int(support.sum());inside_contribution=100*float(np.nansum(all_difference[inside]))/n if n else None;outside_contribution=100*float(np.nansum(all_difference[~inside]))/n if n else None
    if n:assert abs(inside_contribution+outside_contribution-100*float(np.nanmean(all_difference)))<1e-12
    for mode,mask in [('all_observed',support),('inside_common_event',support&inside),('outside_common_event',support&~inside)]:
     number=int(mask.sum());qq=dict(id=key,parent=str(g['parent']),mode=mode,frames=number,current_IoU_percent=100*float(np.mean(frame_values['Current'][mask]))if number else None,pure_IoU_percent=100*float(np.mean(frame_values['Pure'][mask]))if number else None,delta_IoU_pp=100*float(np.mean(all_difference[mask]))if number else None,inside_contribution_pp=inside_contribution if mode=='all_observed'else None,outside_contribution_pp=outside_contribution if mode=='all_observed'else None);rr.append(qq)
    if different_inside.any()and len(traces)<3:
     geometry=SpatialGeometry.from_cached_geometry(gg);bank=geometry.for_interval(iv);out=evidence_dp(bank);assert np.array_equal(out['final'].numpy(),current)
     jj=int(np.where(different_inside)[0][0]);t=int(pos[jj]);assert int(out['actions'][t])==0
     traces.append(dict(private_query_id=key,position=t,physical_frame=int(ids[t]),Current_action=0,expert_action=1,E_and_expert_both_valid=bool(bank.valid[t,0]and bank.valid[t,1]),unary_E=float(bank.score_unary[t,0]),unary_expert=float(bank.score_unary[t,1]),same_complete_DP_saved_output_exact=True))
   if counts['queries']%100==0:atomic(A/'progress.json',dict(dataset=ds,queries=counts['queries'],total=COUNTS[ds],read_only=True,worker=identity()))
  groups={}
  for mode in ['all_observed','inside_common_event','outside_common_event']:
   qq=[r for r in rr if r['mode']==mode];fields=['current_IoU_percent','pure_IoU_percent','delta_IoU_pp']+(['inside_contribution_pp','outside_contribution_pp']if mode=='all_observed'else[]);groups[mode]=dict(annotated_frames=sum(r['frames']for r in qq),missing_support_queries=sum(r['frames']==0 for r in qq),**stats(qq,fields))
  rep=dict(dataset=ds,counts=counts,groups=groups,max_reproduction_error=maxerr,all_physical_positions_and_common_intervals_exact=True,pure_all_raw_expert_boxes_exact=True,Current_every_observed_output_E_or_expert_exact=True,Current_outside_event_forced_E=True,three_inside_event_DP_traces_reproduce_existing_predictions=True,root_inference=False,new_Source=0,new_expert=0,new_model=0,new_optimizer=0,method_changed=False,plan_fingerprint=plan['fingerprint']);reports[ds]=rep;atomic(A/ds/'private_rows.json',rr);atomic(A/ds/'private_DP_traces.json',traces);atomic(B/ds/'report.json',rep)
  with(B/ds/'all_query_observation_scalars.tsv').open('w')as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','group','annotated_frames','Current_IoU_percent','Pure_IoU_percent','delta_IoU_pp','inside_contribution_pp','outside_contribution_pp']);keys={r['id']:f'Q{i:05d}'for i,r in enumerate(rows(ds),1)};w.writerows([[keys[r['id']],r['mode'],r['frames'],*[r[k]for k in ['current_IoU_percent','pure_IoU_percent','delta_IoU_pp','inside_contribution_pp','outside_contribution_pp']]]for r in rr])
  print(json.dumps(rep),flush=True)
 atomic(A/'completion.json',dict(status='read_only_both_full_observed_output_and_metric_scope_audited',reports_sha256={ds:sha(B/ds/'report.json')for ds in COUNTS},worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_method_prediction=0,method_changed=False));assert not torch.cuda.is_initialized()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);globals()[ap.parse_args().stage]()
