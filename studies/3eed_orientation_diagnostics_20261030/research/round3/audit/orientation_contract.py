"""Synthetic yaw/scene/L-shape contracts only. Never opens dataset/caches or imports models.
The fixture 'target' arrays are analytic boxes declared here, not dataset targets.
"""
import json,math,pathlib,hashlib
import numpy as np
R=pathlib.Path(__file__).resolve().parent

def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])
def corners(box):
 b=np.asarray(box,dtype=np.float64);theta=float(b[6]);c,s=math.cos(theta),math.sin(theta)
 xy=np.array([[-b[3]/2,-b[4]/2],[b[3]/2,-b[4]/2],[b[3]/2,b[4]/2],[-b[3]/2,b[4]/2]])
 return xy@np.array([[c,-s],[s,c]]).T+b[:2]
def clipped_area(polygon,against):
 p=polygon.copy()
 for a,b in zip(against,np.roll(against,-1,axis=0)):
  if not len(p):return 0.
  new=[];prev=p[-1];before=cross(b-a,prev-a)
  for current in p:
   after=cross(b-a,current-a);inside0=before>=-1e-12;inside1=after>=-1e-12
   if inside0!=inside1:new.append(prev+(current-prev)*(before/(before-after)))
   if inside1:new.append(current)
   prev=current;before=after
  p=np.asarray(new,dtype=np.float64).reshape(-1,2)
 if len(p)<3:return 0.
 p=p-p.mean(0);return abs(sum(cross(a,b) for a,b in zip(p,np.roll(p,-1,axis=0))))/2

def iou(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 assert a.shape==b.shape==(7,) and np.isfinite(a).all() and np.isfinite(b).all()
 if min(a[3:6])<=0 or min(b[3:6])<=0:return 0.
 z=max(0.,min(a[2]+a[5]/2,b[2]+b[5]/2)-max(a[2]-a[5]/2,b[2]-b[5]/2))
 inter=clipped_area(corners(a),corners(b))*z
 return float(inter/(np.prod(a[3:6])+np.prod(b[3:6])-inter))

def yaw_only(source,yaw):
 result=np.asarray(source,dtype=np.float64).copy();assert result.shape==(7,)
 result[6]=float(yaw);assert np.array_equal(result[:6],np.asarray(source)[:6]);return result

def assert_fixed_query_yaw_only(source,output,source_query,output_query):
 assert source_query==output_query,'Orientation diagnostic changed selected source query'
 assert np.array_equal(np.asarray(source)[:6],np.asarray(output)[:6]),'Orientation-only arm changed center or extents'

def volume_height_upper_bound(a,b):
 """Certified, usually loose upper bound for *all yaw* with fixed center/size.
 Only arithmetic volume/z-overlap restrictions. Not a grid/refinement certificate.
 """
 a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
 assert min(a[3:6])>0 and min(b[3:6])>0
 z=max(0.,min(a[2]+a[5]/2,b[2]+b[5]/2)-max(a[2]-a[5]/2,b[2]-b[5]/2))
 inter=z*min(a[3]*a[4],b[3]*b[4]);va=np.prod(a[3:6]);vb=np.prod(b[3:6])
 return float(inter/(va+vb-inter))

def paper_distances(c):
 choices=np.stack([np.max(c)-c,c-np.min(c)])
 return choices[np.argmin(np.sum(choices*choices,axis=1))] # explicit first-vector tie convention

def lshape_scores(c1,c2,variant):
 if variant=='paper_literal':d1,d2=paper_distances(c1),paper_distances(c2)
 elif variant=='common_implementation':d1=np.minimum(np.max(c1)-c1,c1-np.min(c1));d2=np.minimum(np.max(c2)-c2,c2-np.min(c2))
 else:raise ValueError(variant)
 e1=d1[d1<d2];e2=d2[d2<d1] if variant=='paper_literal' else d2[d1>=d2]
 return {'closeness':float(np.sum(1/np.maximum(np.minimum(d1,d2),.01))),'variance':float(-(np.var(e1) if len(e1) else 0)-(np.var(e2) if len(e2) else 0)),'E1_n':len(e1),'E2_n':len(e2),'distance1':d1.tolist(),'distance2':d2.tolist()}

def stratified_scene_bootstrap(records,platform_weights,replicates=100,seed=1):
 """Synthetic-contract helper: fixed platform weights, equal-scene estimand within platform.
 No eligibility/gate fitting. Empty platform invalid rather than silently renormalized.
 """
 assert abs(sum(platform_weights.values())-1)<1e-12
 grouped={p:{} for p in platform_weights}
 for p,scene,value in records:grouped[p].setdefault(scene,[]).append(value)
 assert all(grouped[p] for p in grouped)
 rng=np.random.default_rng(seed);draws=[]
 for _ in range(replicates):
  value=0.
  for p,weight in platform_weights.items():
   ids=sorted(grouped[p]);chosen=rng.choice(ids,len(ids),replace=True)
   value+=weight*np.mean([np.mean(grouped[p][s]) for s in chosen])
  draws.append(value)
 return np.asarray(draws)

def main():
 unit=np.array([0.,0.,0.,2.,2.,2.,0.]);assert abs(iou(unit,unit)-1)<1e-12
 shifted=unit.copy();shifted[0]=1;assert abs(iou(shifted,unit)-1/3)<1e-12
 fixtures=[]
 # Correct center/dimensions: supplied target yaw can fully recover orientation.
 target=np.array([0.,0.,0.,4.,1.,1.,math.pi/6]);source=yaw_only(target,0)
 assert abs(iou(yaw_only(source,target[6]),target)-1)<1e-12
 fixtures.append({'name':'correct_center_size_yaw_recovery','source_iou':iou(source,target),'GTyaw_injected_iou':1.})
 # Wrong extent assignment: source yaw0 already gives the identical physical box.
 target=np.array([0.,0.,0.,4.,1.,1.,math.pi/2]);source=np.array([0.,0.,0.,1.,4.,1.,0.])
 start=iou(source,target);injected=iou(yaw_only(source,target[6]),target)
 assert abs(start-1)<1e-12 and abs(injected-1/7)<1e-12
 fixtures.append({'name':'GTyaw_can_harm_wrong_extents','source_iou':start,'GTyaw_injected_iou':injected,'interpretation':'GTyaw is not a guaranteed-improvement or max-IoU oracle under fixed wrong dimensions.'})
 # Wrong center, correct dimensions: GTyaw has zero overlap while another yaw overlaps.
 target=np.array([0.,0.,0.,4.,1.,1.,0.]);source=np.array([0.,1.1,0.,4.,1.,1.,0.]);atGT=iou(source,target);other=iou(yaw_only(source,math.pi/2),target)
 assert atGT==0 and abs(other-1/7)<1e-12
 fixtures.append({'name':'GTyaw_not_max_with_wrong_center','GTyaw_injected_iou':atGT,'other_feasible_yaw_iou':other})
 # Finite-grid maximum is only a lower bound; missed continuous optimum here is known exactly.
 target=np.array([0.,0.,0.,4.,1.,1.,math.pi/12]);source=yaw_only(target,0);grid=[0,math.pi/2]
 lower=max(iou(yaw_only(source,a),target) for a in grid);known=iou(yaw_only(source,math.pi/12),target)
 assert lower<known-1e-3 and abs(known-1)<1e-12
 fixtures.append({'name':'finite_grid_lower_not_upper','grid_best':lower,'known_continuous_feasible_iou':known,'certified_volume_height_upper':volume_height_upper_bound(source,target)})
 target=np.array([0.,0.,3.,4.,4.,2.,.3]);source=np.array([0.,0.,0.,2.,2.,1.,0.]);assert volume_height_upper_bound(source,target)==0
 fixtures.append({'name':'vertical_nonoverlap_certifies_no_yaw_headroom','certified_upper':0.})
 target=np.array([0.,0.,0.,4.,4.,2.,0.]);source=np.array([0.,0.,0.,2.,2.,1.,0.]);assert abs(volume_height_upper_bound(source,target)-.125)<1e-12 and abs(iou(source,target)-.125)<1e-12
 fixtures.append({'name':'contained_volume_upper_tight','certified_upper':.125})
 assert_fixed_query_yaw_only(source,yaw_only(source,.5),4,4)
 rejected={}
 for name,out,q in [('query_switch',yaw_only(source,.5),3),('center_change',source+np.array([.1,0,0,0,0,0,0]),4)]:
  try:assert_fixed_query_yaw_only(source,out,4,q)
  except AssertionError:rejected[name]=True
 assert len(rejected)==2
 c1=np.array([0.,.2,1.,2.,4.]);c2=np.array([0.,1.,.1,2.,1.]);paper=lshape_scores(c1,c2,'paper_literal');common=lshape_scores(c1,c2,'common_implementation');assert abs(paper['closeness']-116.5)<1e-12 and abs(common['closeness']-315)<1e-12
 common_angles=np.rad2deg(np.arange(0.,np.pi/2-np.pi/180,np.pi/180));assert len(common_angles)==89 and abs(common_angles[-1]-88)<1e-10
 synthetic=[('waymo','a',1.),('waymo','b',-1.),('drone','c',.2),('quad','d',.4),('quad','e',-.4)]
 w={'waymo':1/3,'drone':1/3,'quad':1/3};draws=stratified_scene_bootstrap(synthetic,w);assert np.isfinite(draws).all()
 singleton_draws=stratified_scene_bootstrap([('drone','c',.2)],{'drone':1.});assert np.all(singleton_draws==.2)
 result={'kind':'synthetic_CPU_contract_not_task_evidence','fixtures':fixtures,'fixed_query_center_extents_guards_reject_mutations':rejected,'Lshape_primary_vs_reference_differ':{'paper_literal':paper,'common_implementation':common,'common_grid_count':len(common_angles),'common_last_degrees':float(common_angles[-1]),'paper_grid_convention':'delta1degree, inclusive0..89,90angles','conventions':'paper norm ties first upper-vector, population variance ddof0, empty sets zero, d0.01 declared from reference default; none proves author-MATLAB bitexactness'},'platform_scene_bootstrap_contract':{'fixed_platform_weights':w,'single_scene_bootstrap_degenerate_not_population_certificate':True,'estimand':'equal platform then equal scene, not frame weighted'},'target_dataset_files_read':0,'model_calls':0,'task_GT_scores':0,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
 (R/'CPU_CONTRACT_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
