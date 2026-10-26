"""Extra exact intermediate checks before the original real-paid Source root ack.

No scientific function/output changes, no model/Source/expert/GT calls. The original
frozen bridge still runs after these stronger checks, with raw scratch intact.
"""
from subset_common import *
import argparse
def equal(a,b):
 import numpy as np,torch
 if torch.is_tensor(a):assert torch.is_tensor(b)and a.dtype==b.dtype and torch.equal(a,b)
 elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray)and a.dtype==b.dtype and np.array_equal(a,b)
 elif isinstance(a,dict):
  assert isinstance(b,dict)and set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert isinstance(b,(list,tuple))and len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
def bridge(ds,name):
 import numpy as np,torch
 from stvg.runtime import guard
 from stvg.decode import decode
 from stvg.readout import prediction
 from stvg.temporal import probabilities,mass_table
 from stvg.positions import observation_positions
 from stvg.evidence import SourceEvidence
 from stvg.spatial import build_geometry
 from stvg.representation import extract
 assert not name.startswith('legacy_');sys.addaudithook(guard);torch.set_num_threads(4);validate();root=unit(ds,name);wait=read(root/'bridge_waiting.json');assert wait['paid_count']==2 and not(root/'root_actual_bridge.json').exists();proof=[];started=time.perf_counter()
 for key,e in wait['files'].items():
  sc=root/'active'/key;assert sha(sc/'full_Source_checkpoint.pt')==e['full_Source_checkpoint_sha256']and sha(sc/'processed_raw.pt')==e['processed_raw_sha256'];whole=torch.load(sc/'full_Source_checkpoint.pt',map_location='cpu',weights_only=False,mmap=True);raw=torch.load(sc/'processed_raw.pt',map_location='cpu',weights_only=False,mmap=True);c=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);equal(whole['zero']['logits'],raw['native_logits']);equal(whole['records'],raw['records']);equal(whole['spatial_inputs'],raw['spatial_inputs']);equal(whole['native_local_grid'],raw['native_local_grid']);ids,q=mass_table(probabilities(whole['records'],whole['zero']['logits']));equal(ids,raw['frame_ids']);equal(q,raw['posterior']['Native']);equal(q,c['q0']);native=prediction(whole['zero']['logits'],whole['zero']['boxes'],whole['records'],ids);equal(native['boxes'].numpy(),raw['Source']);equal(raw['Source'],c['Source']);equal(native['physical_interval'],list(c['Source_interval']));pos=observation_positions(ids,native['indices']);equal(pos,raw['planned_positions']);equal(pos,c['positions8']);te=e['teacher'];assert sha(te['path'])==te['sha256'];teacher=torch.load(te['path'],map_location='cpu',weights_only=False);cap=checkpoint_class()(ds).initialize_cached({**raw,'teacher_native':teacher});ev=SourceEvidence.from_capture(raw['spatial_inputs'],ids,pos,teacher['native_mask_logits'],teacher['input_shape'][-2:],selected_query=teacher['selected_query']);geometry=build_geometry(ev,cap['E'],teacher['boxes'],collect_mask_diagnostics=True);equal(geometry.state_dict(),c['geometry']);features,_=extract(raw,cap['E']);equal(features,c['E_ROI_features']);features2,_=extract(raw,cap['E']);equal(features,features2)
  proof.append(dict(full_Source_sha256=e['full_Source_checkpoint_sha256'],processed_raw_sha256=e['processed_raw_sha256'],native_logits_records_spatial_inputs_grids_exact=True,native_Source_q0_positions_exact=True,all_geometry_state_arrays_exact=True,actual_E_ROI_features_double_repeat_and_compact_exact=True));del whole,raw,c,teacher,cap,ev,geometry,features,features2;gc.collect()
 assert len(proof)==2 and not torch.cuda.is_initialized();atomic(root/'root_exact_intermediates_pre_ack_v2.json',dict(status='passed_complete_Source_native_geometry_Efeatures_exact_before_original_root_ack',cases=proof,bridge_waiting_sha256=sha(root/'bridge_waiting.json'),code_sha256=sha(Path(__file__)),worker=identity(),seconds=time.perf_counter()-started,new_model_Source_expert_GT=0))
 spec=importlib.util.spec_from_file_location('_subset_original_root_paid_bridge',S/'report.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bridge(ds,name)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);p.add_argument('--condition',required=True);a=p.parse_args();bridge(a.dataset,a.condition)
