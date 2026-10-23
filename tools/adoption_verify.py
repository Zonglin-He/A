"""CPU-only exact adoption checks against frozen modular experiments, no GT."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.evidence import SourceEvidence
from stvg.prototype import one_step
from stvg.prior import from_capture
from stvg.temporal import final_mbr
OLD=ROOT.parent/'stvg-native2d-nativeprior'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);start=time.perf_counter();checks={}
 s=read(OLD/'outputs/algorithm_simplification_v1/plan.json');ps=read(OLD/'outputs/algorithm_pruning_v2/sealed.json')
 for n,i in enumerate(s['items'],1):
  assert sha(i['capture_path'])==i['capture_sha256'] and sha(i['raw_path'])==i['raw_sha256'] and sha(i['mask_path'])==i['mask_sha256']
  cap=torch.load(i['capture_path'],weights_only=False);raw=torch.load(i['raw_path'],weights_only=False);mask=torch.load(i['mask_path'],weights_only=False);j=int(mask['selected_query']);ev=SourceEvidence.from_capture(raw['spatial_inputs'],cap['frame_ids'],cap['positions8'],mask['raw_out']['pred_masks'][0,:,j].float(),mask['input_shape'][-2:],selected_query=j)
  g=SpatialGeometry(ev,cap['E'],cap['geometry']['native_boxes'],collect_mask_diagnostics=True);cached=SpatialGeometry.from_cached_geometry(cap['geometry'])
  for f in ['boxes','source_anchors','side_available','side_reference_boxes','seed_features','features','association_scores','valid','u','link','supported','side_maps']:
   assert torch.equal(getattr(g,f),getattr(cached,f)),(n,f,float((getattr(g,f).double()-getattr(cached,f).double()).abs().max()))
  e=ps['files']['S_'+i['source_key']];p=OLD/'outputs/algorithm_pruning_v2'/e['path'];assert sha(p)==e['sha256']
  with np.load(p,allow_pickle=False)as z:
   for impl in [g,cached]:assert np.array_equal(evidence_dp(impl.for_interval(z['SimplePath_interval'].tolist()))['final'].numpy(),z['SimplePath'])
  if n%16==0:print('spatial',n,flush=True)
 checks['spatial_direct_cached_and_sealed_SimplePath']=n
 kp=read(OLD/'outputs/native_prior_kappa_compare_v1/plan.json')
 for n,i in enumerate(kp['items'],1):
  assert sha(i['path'])==i['sha256'] and sha(i['feature_path'])==i['feature_sha256']
  cap=torch.load(i['path'],weights_only=False);tr=torch.load(i['feature_path'],weights_only=False);features=tr['features']['Native2D'] if isinstance(tr['features'],dict)else tr['features'];q0=from_capture(cap);q,stats=one_step(features,q0)
  # Literal independent implementation frozen in the former experiment.
  import importlib.util
  spec=importlib.util.spec_from_file_location("frozen_NoSplit",OLD/"studies/algorithm_pruning/simple_time.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);reference=module.one_step
  assert np.array_equal(q,reference(features,q0))
  key='T_'+i['dataset']+'_'+i['alias'];e=ps['files'][key];path=OLD/'outputs/algorithm_pruning_v2'/e['path'];assert sha(path)==e['sha256']
  with np.load(path,allow_pickle=False)as z:assert np.array_equal(final_mbr(cap['frame_ids'],q)['interval'],z['NoSplit_interval'])
  assert stats['prototypes']['prototype_pairs']==1 and stats['rounds']==1
 checks['temporal_all_video_exact_saved_NoSplit']=n
 out=dict(status='passed',checks=checks,GT_read=False,new_model=0,new_expert=0,seconds=time.perf_counter()-start,scope='Modular adoption only; changed auxiliary path requires a separate actual-grid bridge.')
 dest=ROOT/'outputs/adoption_simplepath_20261023/cpu_verification.json';dest.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
