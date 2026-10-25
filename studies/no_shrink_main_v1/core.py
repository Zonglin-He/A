"""Exact cache consumers; only kappa is changed, every interval re-solves DP."""
from ns_common import *
import numpy as np,torch,textwrap
from stvg.prototype import one_step
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.current import map_interval

def solve(g,iv):return evidence_dp(g.for_interval(iv))['final'].numpy()
def derive(task,ds,item):
 row=item['row'];key=row['id']
 if task=='components':
  c,parent=suite().load_parent(ds,row);g=SpatialGeometry.from_cached_geometry(c['geometry']);q,_=one_step(c['content_features'],c['native_prior'],kappa=0);iv=map_interval(c['frame_ids'],q);e=fixed(R/'outputs/map_components_v1'/f'components_{ds}_sealed.json')['files'][key];root=R/'outputs/map_components_v1';op=root/e['path'];rp=root/e['receipt_path'];assert sha(op)==e['sha256'] and sha(rp)==e['receipt_sha256']
  with np.load(op)as z:a={k:z[k].copy()for k in z.files}
  from supports import greedy
  a.update(FullMAP=solve(g,iv),FullMAP_interval=np.asarray(iv),FinalIndependent=greedy(g.for_interval(iv)),FinalIndependent_interval=np.asarray(iv));proof=verify_main(ds,key,a);return a,dict(parent=parent,no_shrink=proof,unchanged_component_prediction_sha256=e['sha256'],unchanged_component_receipt_sha256=e['receipt_sha256'])
 if task=='CG':
  root=CG;e=fixed(root/f'{ds}_sealed.json')['files'][key];pp=root/e['prediction_path'];rp=root/e['receipt_path'];cp=root/e['compact_path'];assert sha(pp)==e['prediction_sha256'] and sha(rp)==e['receipt_sha256'] and sha(cp)==e['compact_sha256'];c=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);assert not c['GT_read'];g=SpatialGeometry.from_cached_geometry(c['geometry']);profile={**c['profile'],'kappa':0};q,_=one_step(c['pooled_features'],c['q0'],kappa=0,beta=profile['beta']);iv=map_interval(c['frame_ids'],q)
  from studies.cgstvg_full_v1.core import solve as cg_solve
  from cgstvg_runtime_v9 import clip_native_boxes
  mapped=clip_native_boxes(cg_solve(g,iv,profile));a=dict(Source=c['Source'].copy(),Source_interval=np.asarray(c['native_map']),SourceJointMAP=c['Source'].copy(),SourceJointMAP_interval=np.asarray(map_interval(c['frame_ids'],c['q0'])),FullMAP=mapped,FullMAP_interval=np.asarray(iv),frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']))
  if ds=='hc2':u=np.ones(len(mapped),bool);u[c['positions8']]=False;assert np.array_equal(mapped[u],c['Source'][u])
  return a,dict(parent=e,parent_compact=str(cp),parent_compact_sha256=e['compact_sha256'],profile=profile)
 if task=='budget':
  old=components();src=textwrap.dedent(inspect.getsource(old.budget_arrays));needle='iv=map_core.map_interval(ids,q);';assert src.count(needle)==1;src=src.replace(needle,'q,_=one_step(f,q0,kappa=0);iv=map_core.map_interval(ids,q);');needle="z['FullMAP']";src=src.replace(needle,"z['NoShrink']").replace("z['FullMAP_interval']","z['NoShrink_interval']")
  # The original numerical runtime used Torch4. Preserve its exact upstream
  # reconstruction and hidden legacy-output check before the kappa0 update.
  class ParentMap:
   def __truediv__(self,name):
    for domain in COUNTS:
     if str(name)==f'TA_{domain}_sealed.json':
      original=fixed(MAP/f'TA_{domain}_sealed.json');files={}
      for query in original['files']:
       entry=fixed(PREV/f'temporal_v2_{domain}_sealed.json')['files'][query];files[query]=dict(path=str(PREV/entry['path']),sha256=entry['sha256'])
      return ('no_shrink_seal',domain,files)
    raise AssertionError(name)
  def cache_read(p):
   if isinstance(p,tuple):return dict(files=p[2])
   return fixed(p)
  class WrappedMap(ParentMap):
   def __truediv__(self,name):
    if isinstance(name,str) and name.startswith('/'):return Path(name)
    return super().__truediv__(name)
  scope={**vars(old),'MAP':WrappedMap(),'fixed':cache_read};exec(compile(src,'<ownK_kappa0_exact_runtime>','exec'),scope);torch.set_num_threads(4)
  try:a,proof=scope['budget_arrays'](ds,row)
  finally:torch.set_num_threads(1)
  for name,e in proof['configs'].items():
   receipt=read(e['teacher']['receipt']);a[name+'_positions']=np.asarray(receipt['context']['positions'])
  return a,proof
 if task=='corruption':
  e=fixed(PREV/'corruptions/sealed.json')['files'][ds][key][item['view']];c,proof=checked_payload(PREV,e);q,_=one_step(c['content_features'],c['native_prior'],kappa=0);iv=map_interval(c['frame_ids'],q);g=SpatialGeometry.from_cached_geometry(c['geometry']);a=dict(Source=np.asarray(c['Source']).copy(),Source_interval=np.asarray(c['Source_interval']),FullMAP=solve(g,iv),FullMAP_interval=np.asarray(iv),frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']));proof.update(original_acquisition=e,view=item['view']);return a,proof
 if task=='spatial':
  root=PREV/'spatial';rp=root/ds/'receipts'/(key+'.json');assert rp.exists(),'Only committed collector receipts may be consumed';e=read(rp);assert e['fingerprint']==fixed(PREV/'spatial_plan.json')['fingerprint'];e=dict(e,receipt_path=str(rp.relative_to(root)),receipt_sha256=sha(rp));c,proof=checked_payload(root,e);assert e['input_binding']==binding(row,fixed(RUN/'plan.json')['provider_runtime_fingerprint']);assert c['parent_capture_sha256']==fixed(RUN/ds/'sealed.json')['files'][key]['capture_sha256'];old=spatial();legacy=old.replay(c)
  with np.load(root/e['path'])as z:assert all(np.array_equal(legacy[k],z[k])for k in legacy)
  f=types.FunctionType(old.replay.__code__,{**old.replay.__globals__,'one_step':partial(one_step,kappa=0)},'no_shrink_replay');a=f(c);proof.update(no_shrink=verify_main(ds,key,a),original_acquisition=e);return a,proof
 raise ValueError(task)
