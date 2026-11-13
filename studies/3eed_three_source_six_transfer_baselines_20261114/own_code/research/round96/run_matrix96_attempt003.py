"""Six fixed cross-platform frozen evaluations after both genuine source100 runs complete."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import sys,json,hashlib,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';sys.path.insert(0,str(B));from factory import configure,model,dataset,order_contract,loader,state_fingerprint,torch,sha
from source_validation import predict_source_val
rd=lambda p:json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def eval_configure(plan):
 configure(plan)
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_tf32=False
 torch.use_deterministic_algorithms(True)
def main():
 plan=rd(H/'experiment/MATRIX_REPAIR_LOCK_attempt003.json');assert sha(H/'experiment/MATRIX_REPAIR_LOCK_attempt003.json')==rd(H/'experiment/MATRIX_REPAIR_RELEASE_attempt003.json')['plan_sha256']
 for f,v in plan['code_sha256'].items():assert sha(R/f)==v,f
 ck={'waymo':R/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth'}
 for pl in ['drone','quad']:
  c=rd(B/(pl+'_source100')/'TRAINING_COMPLETE.json');assert c['primary_epoch']==100 and c['trained_platforms']==[pl];ck[pl]=Path(c['primary_checkpoint']);assert sha(ck[pl])==c['primary_checkpoint_sha256']
 for src,path in ck.items():assert path.is_absolute() and path.is_file(),(src,str(path))
 assert sha(ck['waymo'])==rd(R/'crossplatform/round6/SOURCE_CHECKPOINT.json')['checkpoint_sha256']
 eval_configure({'seed':0,'torch_num_threads':4});core=model();core.requires_grad_(False).cuda().eval();results=[]
 for source in ['waymo','drone','quad']:
  payload=torch.load(ck[source],map_location='cpu',weights_only=True);assert payload['trained_platforms']==[source] and payload['epoch']==100;core.load_state_dict(payload['model'],strict=True);del payload;fp=state_fingerprint(core)
  for target in ['waymo','drone','quad']:
   if source==target:continue
   eval_configure({'seed':0,'torch_num_threads':4});ds=dataset(target,'val');contract=order_contract(ds,target,'val');assert contract==plan['target_contracts'][target];g=torch.Generator().manual_seed(0);dl=loader(ds,'val',g,plan);out=B/'frozen_matrix_attempt003'/(source+'_to_'+target);se=predict_source_val(core,dl,out,100,ck[source],contract);assert state_fingerprint(core)==fp;results.append({'source':source,'target':target,'n':se['n'],'prediction_directory':str(out),'prediction_seal_sha256':sha(out/'sealed.json'),'checkpoint_sha256':sha(ck[source]),'no_parameter_update':True});print('MATRIX96',source,target,se['n'],flush=True)
 out=B/'frozen_matrix_attempt003';dump(out/'MATRIX_SEALED.json',{'state':'SIX_SOURCE100_FROZEN_DIRECTIONS_SEALED_BEFORE_TASK_SCORES','directions':results,'primary':'last_text_logodds','GT_assisted_readouts_secondary_only':True,'protected_frames_model_input':False,'TTA_performed':False,'target_GT_for_model_selection':False})
if __name__=='__main__':main()
