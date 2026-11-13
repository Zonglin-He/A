"""Six fixed cross-platform frozen evaluations after both genuine source100 runs complete."""
from pathlib import Path
import sys,json,hashlib,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';sys.path.insert(0,str(B));from factory import *
from source_validation import predict_source_val
rd=lambda p:json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 plan=rd(H/'experiment/MATRIX_LOCK.json');assert sha(H/'experiment/MATRIX_LOCK.json')==rd(H/'experiment/MATRIX_RELEASE.json')['plan_sha256']
 ck={'waymo':R/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth'}
 for pl in ['drone','quad']:
  c=rd(B/(pl+'_source100')/'TRAINING_COMPLETE.json');assert c['primary_epoch']==100 and c['trained_platforms']==[pl];ck[pl]=Path(c['primary_checkpoint']);assert sha(ck[pl])==c['primary_checkpoint_sha256']
 configure({'seed':0,'torch_num_threads':4});core=model();core.requires_grad_(False).cuda().eval();results=[]
 for source in ['waymo','drone','quad']:
  payload=torch.load(ck[source],map_location='cpu',weights_only=True);assert payload['trained_platforms']==[source] and payload['epoch']==100;core.load_state_dict(payload['model'],strict=True);del payload;fp=state_fingerprint(core)
  for target in ['waymo','drone','quad']:
   if source==target:continue
   configure({'seed':0,'torch_num_threads':4});ds=dataset(target,'val');contract=order_contract(ds,target,'val');assert contract==plan['target_contracts'][target];g=torch.Generator().manual_seed(0);dl=loader(ds,'val',g,plan);out=B/'frozen_matrix'/(source+'_to_'+target);se=predict_source_val(core,dl,out,100,ck[source],contract);assert state_fingerprint(core)==fp;results.append({'source':source,'target':target,'n':se['n'],'prediction_directory':str(out),'prediction_seal_sha256':sha(out/'sealed.json'),'checkpoint_sha256':sha(ck[source]),'no_parameter_update':True});print('MATRIX96',source,target,se['n'],flush=True)
 out=B/'frozen_matrix';dump(out/'MATRIX_SEALED.json',{'state':'SIX_SOURCE100_FROZEN_DIRECTIONS_SEALED_BEFORE_TASK_SCORES','directions':results,'primary':'last_text_logodds','GT_assisted_readouts_secondary_only':True,'protected_frames_model_input':False,'TTA_performed':False,'target_GT_for_model_selection':False})
if __name__=='__main__':main()
