"""Synthetic no-GT check of actual R11 producer reader and cohort selections."""
from pathlib import Path
import tempfile,json,hashlib
import numpy as np
from score_reliability import prediction,validate_cohort,sha,CONTROLS
from metrics import ARMS,eval_boxes,TIE_SALT
from discrimination import deterministic_select
D=Path(__file__).resolve().parent

def main():
 rows=[{'id':str(i),'platform':'drone' if i<2 else 'quad'} for i in range(4)]
 controls={'selection':{k:{'selected_ids':[]} for k in CONTROLS},'uniform_means_perplatform':{p:{'face':{'n_defined_active':2,'mean':.5,'defined_zero_count':0},'RGB':{'n_defined_active':2,'mean':.5,'defined_zero_count':0}} for p in ('drone','quad')}}
 selections={}
 for platform in ('drone','quad'):
  ids=[r['id'] for r in rows if r['platform']==platform]
  for j,k in enumerate(CONTROLS):
   score={sid:0 if k=='hash_random' else .5 for sid in ids};sel=deterministic_select(ids,.5,score,dict.fromkeys(ids,True),salt=TIE_SALT,platform=platform)['selected'];selections[platform,k]=set(sel);controls['selection'][k]['selected_ids']+=sel
 preds={};trace=[]
 with tempfile.TemporaryDirectory(prefix='R11_synthetic_reader_',dir=D) as temp:
  temp=Path(temp)
  for row in rows:
   sid=row['id'];pool=np.tile(np.array([0,0,0,2,2,2],np.float32),(256,1));scores=-np.arange(256,dtype=np.float32);src=pool[0];full=src.copy();full[0]=.1
   c={'boxes_raw6':pool[None],'text_scores':scores[None],'query_index':np.array(0),'point_sample_sha256':np.zeros(32,np.uint8),'raw_indices_sha256':np.ones(32,np.uint8)};cp=temp/(sid+'_cache.npz');np.savez(cp,**c)
   gates=np.array([sid in selections[row['platform'],k] for k in CONTROLS],bool);boxes=np.stack([src,full,full,full,full]+[full if b else src for b in gates]);atomic=np.tile(src,(6,1));atomic[np.arange(6),np.arange(6)]=full
   z={'arm_names':np.array(ARMS),'control_score_names':np.array(CONTROLS),'source_boxes_raw6':pool,'source_text_scores256':scores,'q_source':np.array(0),'arm_query_index':np.zeros(11,np.int64),'arm_boxes_raw6':boxes,'arm_boxes_eval6':eval_boxes(boxes),'atomic_boxes_raw6':atomic,'atomic_boxes_eval6':eval_boxes(atomic),'atomic_active6':full!=src,'base_acceptance_mask6':full!=src,'coordinate_reliability6':np.array([.5,0,0,0,0,0]),'coordinate_defined6':np.array([1,0,0,0,0,0],bool),'RGB_reliability':np.array(.5),'RGB_defined':np.array(True),'case_reliability':np.array(.5),'case_defined':np.array(True),'control_scores':np.array([.5,0,.5,.5,.5,.5]),'control_defined':np.ones(6,bool),'gate_selection':gates,'association_status':np.array('association_supported'),'face_weights':np.full(6,.5),'point_sample_sha256':c['point_sample_sha256'],'raw_indices_sha256':c['raw_indices_sha256'],'source_cache_sha256':np.frombuffer(bytes.fromhex(sha(cp)),np.uint8),'RGB_mask2d_private':np.zeros((2,2),bool)}
   pp=temp/(sid+'.npz');np.savez(pp,**z);preds[sid]=prediction(pp,cp)
   trace.append({'id':sid,'base_actual_change':True,'reliability':{'association_status':'association_supported','active_face_keys':['0:1'],'axis_reliability6':z['coordinate_reliability6'].tolist(),'axis_reliability_defined6':z['coordinate_defined6'].tolist(),'face_weights':{'0:1':.5},'features':{'faces':{'0:1':{'defined':True}}},'RGB_weight_defined':True,'RGB_weight':.5}})
  assert validate_cohort(preds,rows,controls,trace)==controls['uniform_means_perplatform']
 out={'state':'PASS_CPU_SYNTHETIC_READER_CONTRACT','literal_producer11arm_atomic6_shapes_and_sourcecache_bindings':True,'exact_K_NAlast_producer_tie_selection':True,'separate_perplatform_face_RGB_means_recomputed':True,'no_task_data_metadata_GT_model_GPU':True,'code_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('score_reliability.py','metrics.py','discrimination.py','check_reader_cpu.py')}}
 (D/'CPU_READER_FIXTURE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
