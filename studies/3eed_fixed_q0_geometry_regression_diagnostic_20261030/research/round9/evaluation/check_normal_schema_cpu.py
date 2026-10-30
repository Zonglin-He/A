"""Actual runner serializer to independent scorer synthetic CPU bridge; no data."""
import sys,tempfile, pathlib, hashlib,json,dataclasses
import numpy as np
P=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(P/'crossplatform/round9'))
from run_diagnostic import serialize_npz,save_source_cache,ARM_NAMES
from score_normal import prediction,direction_diagnostic,cancellation_summaries
from geometry_metrics import ARMS,aggregate
assert tuple(ARM_NAMES)==ARMS
source=np.tile(np.array([0,0,0,2,2,2],np.float32),(256,1));source[:,0]=np.arange(256)*.1;scores=np.linspace(1,0,256,dtype=np.float32);q=0
G=[];arms=[source[q].copy()];statuses=['source_q0']
for mi,mode in enumerate(('rgb_fulltext','pc_only')):
 p=source[q].copy();p[0]+=.1*(mi+1);g={'original_source_q':q,'associated_q':q,'pseudo6':p,'raw_consensus_center':p[:3].astype(float),'r_global':1.,'w6':np.array([1,0,0,0,0,0],np.float32)};G.append(g);arms.extend([p.copy() for _ in range(5)]);statuses.extend(['direct_G','effective_pseudo_target_requested','effective_pseudo_target_requested','effective_pseudo_target_requested','effective_pseudo_target_requested'])
z=serialize_npz(sample_id='synthetic',source=source,scores=scores,top=np.arange(10),q0=q,G_results=G,selected_arms=arms,statuses=statuses,prepared=None,cosines=np.empty((0,2)),point_sha='a'*64,index_sha='b'*64,image_sha='c'*64)
with tempfile.TemporaryDirectory() as td:
 f=pathlib.Path(td)/'synthetic.npz';np.savez_compressed(f,**z);a=prediction(f)
 bad={k:v.copy() for k,v in z.items()};bad['arm_query_index'][1]=1;np.savez_compressed(f,**bad)
 try:prediction(f)
 except AssertionError:pass
 else:raise AssertionError('Scorer accepted changed query')
target=np.array([1,0,0,2,2,2,0],float);g=direction_diagnostic(a,0,target,{'G_diagnostics':{'reason':'synthetic'}});assert g['raw_consensus_cosine_GT_correction']==1. and g['direct_center_error_delta_m']<0
record={'state':'synthetic_CPU_actual_runner_serializer_independent_scorer_bridge_pass','normal11_exact_names_and_shapes':True,'query_reselection_rejected':True,'postseal_direction_alignment_formula_pass':True,'no_GPU_model_checkpoint_data_GT_annotation_file_access':True,'runner_sha256':hashlib.sha256((P/'crossplatform/round9/run_diagnostic.py').read_bytes()).hexdigest()}
f=pathlib.Path(__file__).with_name('CPU_NORMAL_SCHEMA_FIXTURE.json');assert not f.exists();f.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
