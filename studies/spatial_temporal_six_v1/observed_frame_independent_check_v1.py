"""Second read-only verification: original prediction NPZ, no geometry PT.

Recompute observed frame IoUs in normalized coordinates, independently of the
first audit's pixel-coordinate helpers. No prediction generation/method change.
"""
from cpu import *
A=D/'observed_frame_audit_v1';B=P/'observed_frame_audit_v1'
def run():
 torch.set_num_threads(1);sys.addaudithook(parent.deny_gpu_media);start=time.perf_counter();proof={}
 for ds in COUNTS:
  old=fixed(RUN/ds/'sealed.json');new=fixed(D/'pure_full'/ds/'sealed.json');truth=SEL.labels(ds);reported=read(B/ds/'report.json');private=read(A/ds/'private_rows.json');lookup={(r['id'],r['mode']):r for r in private};cnt={k:0 for k in ['observations','inside_event_observations','inside_event_Current_expert','inside_event_Current_E_not_expert','outside_event_observations','outside_event_Current_not_E','GT_supported_observations','GT_supported_inside','GT_supported_outside','GT_inside_Current_E_not_expert','GT_inside_Current_expert']};checked=0;err=0.
  for row in rows(ds):
   key=row['id'];e=old['files'][key];n=new['files'][key];assert sha(RUN/ds/e['path'])==e['sha256']and sha(D/n['path'])==n['sha256']
   with np.load(RUN/ds/e['path'])as src,np.load(D/n['path'])as cur:
    ids=src['frame_ids'];pos=src['positions8'];assert np.array_equal(ids,cur['frame_ids'])and np.array_equal(pos,cur['positions8']);expert=src['native_boxes'];E=src['SharedInit'][pos];ours=cur['Current'][pos];pure=cur['PureMatched'][pos];assert np.array_equal(pure,expert);inside=(ids[pos]>=cur['Current_interval'][0])&(ids[pos]<cur['Current_interval'][1]);ex=np.max(np.abs(ours-expert),axis=1)==0;eqE=np.max(np.abs(ours-E),axis=1)==0;assert(ex|eqE).all()and eqE[~inside].all()
    gt={int(f):np.asarray(b,float)/np.array([row['metadata']['width'],row['metadata']['height']]*2)for f,b in truth[key]['boxes'].items()};support=np.array([int(f)in gt for f in ids[pos]]);nobs=int(support.sum());cnt['observations']+=len(pos);cnt['inside_event_observations']+=int(inside.sum());cnt['inside_event_Current_expert']+=int((inside&ex).sum());cnt['inside_event_Current_E_not_expert']+=int((inside&~ex).sum());cnt['outside_event_observations']+=int((~inside).sum());cnt['outside_event_Current_not_E']+=int((~inside&~eqE).sum());cnt['GT_supported_observations']+=nobs;cnt['GT_supported_inside']+=int((support&inside).sum());cnt['GT_supported_outside']+=int((support&~inside).sum());cnt['GT_inside_Current_E_not_expert']+=int((support&inside&~ex).sum());cnt['GT_inside_Current_expert']+=int((support&inside&ex).sum());vv={}
    for name,bb in [('Current',ours),('Pure',pure)]:
     corner=np.column_stack((bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2));corner=np.maximum(corner,0)if ds=='hc2'else corner;v=np.full(8,np.nan)
     for j,f in enumerate(ids[pos]):
      if not support[j]:continue
      a=corner[j];b=gt[int(f)];w=max(0.,min(a[2],b[2])-max(a[0],b[0]));h=max(0.,min(a[3],b[3])-max(a[1],b[1]));inter=w*h;areaA=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);areaB=max(0.,b[2]-b[0])*max(0.,b[3]-b[1]);union=areaA+areaB-inter;v[j]=inter/union if union>0 else 0.;checked+=1
     vv[name]=v
    for group,mask in [('all_observed',support),('inside_common_event',support&inside),('outside_common_event',support&~inside)]:
     expected=lookup[key,group];assert expected['frames']==int(mask.sum())
     for arm,field in [('Current','current_IoU_percent'),('Pure','pure_IoU_percent')]:
      val=100*float(vv[arm][mask].mean())if mask.any()else None;ref=expected[field]
      if val is None:assert ref is None
      else:err=max(err,abs(val-ref));assert abs(val-ref)<1e-10
    if nobs:
     dd=100*(vv['Current']-vv['Pure']);exp=lookup[key,'all_observed'];assert abs(float(np.nansum(dd[inside]))/nobs-exp['inside_contribution_pp'])<1e-10 and abs(float(np.nansum(dd[~inside]))/nobs-exp['outside_contribution_pp'])<1e-10
   if len(lookup)%100==0:parent.resources(start)
  for key,val in cnt.items():assert val==reported['counts'][key],(ds,key,val,reported['counts'][key])
  proof[ds]=dict(queries=COUNTS[ds],all_count_categories_reproduced=True,GT_supported_observation_arm_checks=checked,max_IoU_percent_error=err,original_NPZ_expert_and_shared_E_used_instead_of_geometry_PT=True,report_sha256=sha(B/ds/'report.json'))
 ack=dict(status='independent_original_NPZ_anchor_identity_and_normalized_IoU_support_decomposition_verified',datasets=proof,worker=identity(),code_sha256=sha(Path(__file__)),first_audit_plan_sha256=sha(A/'plan.json'),method_changed=False,new_Source=0,new_expert=0,new_official_tube_scores=0,elapsed_wall_seconds=time.perf_counter()-start);atomic(A/'independent_check.json',ack);atomic(B/'independent_check.json',ack);print(json.dumps(ack),flush=True);assert not torch.cuda.is_initialized()
if __name__=='__main__':run()
