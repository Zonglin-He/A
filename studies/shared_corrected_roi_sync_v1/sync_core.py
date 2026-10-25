"""Direct shared-box pooling controls; preserved old numerical checks precede changes."""
from sync_common import *
import numpy as np,torch,textwrap
from stvg.prototype import one_step
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.current import map_interval

def solve(g,iv):return evidence_dp(g.for_interval(iv))['final'].numpy()
def derive(task,ds,item):
 row=item['row'];key=item['key']
 if task=='spatial':
  root=PREV/'spatial';e=fixed(root/f'{ds}_sealed.json')['files'][key];c,proof=checked_payload(root,e)
  assert e['input_binding']==binding(row,fixed(RUN/'plan.json')['provider_runtime_fingerprint'])
  assert c['parent_capture_sha256']==fixed(RUN/ds/'sealed.json')['files'][key]['capture_sha256']
  old=fixed(parent.D/f'spatial_{ds}_sealed.json')['files'][key];assert sha(parent.D/old['path'])==old['sha256']and sha(parent.D/old['receipt_path'])==old['receipt_sha256']
  ids=c['ids'];q0=c['q0'];gs={k:SpatialGeometry.from_cached_geometry(v)for k,v in c['geometries'].items()};g=gs['main'];iv0=map_interval(ids,q0)
  a=dict(Source=c['Source'].copy(),Source_interval=np.asarray(c['original_interval']),SourceJointMAP=c['Source'].copy(),SourceJointMAP_interval=np.asarray(iv0),frame_ids=np.asarray(ids),positions8=np.asarray(c['pos']))
  qe,_=one_step(c['features']['AuxE_ROI'],q0,kappa=0,beta=1);ive=map_interval(ids,qe)
  for name in ['FullMAP','NoSharedCorrection','PastOnly','OneWayAffinity','AuxIndependent','AuxSourceROI','AuxE_ROI']:
   # Removing shared correction changes the pooling boxes to Source. Past/oneway
   # change final geometry but retain the same shared boxes and direct features.
   f=c['features']['AuxSourceROI'if name in ['NoSharedCorrection','AuxSourceROI']else'AuxE_ROI'];q,_=one_step(f,q0,kappa=0,beta=1);iv=map_interval(ids,q);geo=gs[dict(NoSharedCorrection='uncorrected',PastOnly='past',OneWayAffinity='oneway').get(name,'main')]
   a[name]=solve(geo,iv);a[name+'_interval']=np.asarray(iv)
  a.update(NoSpatialTrajectory=c['E'].copy(),NoSpatialTrajectory_interval=np.asarray(ive),NoTemporalInference=solve(g,c['original_interval']),NoTemporalInference_interval=np.asarray(c['original_interval']),S0T0=c['E'].copy(),S0T0_interval=np.asarray(iv0),S1T0=solve(g,iv0),S1T0_interval=np.asarray(iv0),S0T1=c['E'].copy(),S0T1_interval=np.asarray(ive),S1T1=a['FullMAP'].copy(),S1T1_interval=a['FullMAP_interval'].copy())
  with np.load(parent.D/old['path'])as z:assert np.array_equal(a['FullMAP'],z['AuxE_ROI'])and np.array_equal(a['FullMAP_interval'],z['AuxE_ROI_interval'])
  assert np.array_equal(a['FullMAP'],a['AuxIndependent'])and np.array_equal(a['FullMAP'],a['AuxE_ROI'])
  proof.update(selected_method_first_aux_DP=False,retained_E_ROI_features=True,final_geometry_and_posterior_recomputed=True,old_structural_prediction_sha256=old['sha256'],old_structural_receipt_sha256=old['receipt_sha256'],AuxIndependent_is_redundant_identity_control=True)
  return a,proof
 if task=='budget':
  old=components();src=textwrap.dedent(inspect.getsource(old.budget_arrays))
  needle='iv=map_core.map_interval(ids,q);';assert src.count(needle)==1
  src=src.replace(needle,"f,_=extract(dict(frame_ids=ids,native_local_grid=s['native_local_grid']),E);q,_=one_step(f,q0,kappa=0,beta=1);iv=map_core.map_interval(ids,q);")
  src=src.replace("z['FullMAP']","z['AuxE_ROI']").replace("z['FullMAP_interval']","z['AuxE_ROI_interval']")
  class SelectedMap:
   def __truediv__(self,name):
    for domain in COUNTS:
     if name==f'TA_{domain}_sealed.json':
      entries=fixed(parent.D/f'spatial_{domain}_sealed.json')['files'];return ('selected_auxE',entries)
    return Path(name)
  def read_selected(path):return dict(files={k:dict(path=str(parent.D/e['path']),sha256=e['sha256'])for k,e in path[1].items()})if isinstance(path,tuple)else fixed(path)
  scope={**vars(old),'MAP':SelectedMap(),'fixed':read_selected};exec(compile(src,'<shared_ROI_ownK_runtime_v1>','exec'),scope);torch.set_num_threads(4)
  try:a,proof=scope['budget_arrays'](ds,row)
  finally:torch.set_num_threads(1)
  for name,e in proof['configs'].items():
   a[name+'_positions']=np.asarray(read(e['teacher']['receipt'])['context']['positions']);e.update(selected_method_first_aux_DP=False,compatibility_only_old_aux_DP=True,new_E_ROI_pool=True,new_kappa0_posterior_and_interval_final_DP=True)
  return a,proof
 raise ValueError(task)
