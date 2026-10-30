"""CPU-only full-plan template; parent must supply formal GPU authorization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json
import torch
from factory import R,P,sha,configure,dataset,order_contract

def main():
 old=json.loads((R/'TIMING_PLAN_TEMPLATE.json').read_text());configure(old);val=dataset('val');val_contract=order_contract(val,'val');cfg={**old,'mode':'waymo_source_train100','GPU_execution_authorized':False,'max_epochs':100,'source_validation_every':10,'save_model_every':10,'source_val_contract':val_contract,'code_sha256':{f:sha(R/f) for f in ['factory.py','checkpointing.py','source_validation.py','train_source.py']},'initialization':'generic localRoBERTa only, originalrandomvisual/grounding, no jointtaskcheckpoint','primary_selection':'fixed epoch100;sourcevalidation doesnotchooseprimary','source_val_prediction_contract':'last/proposal full256 yaw0; raw/evalclamp boxes; originalGTclassspan BBF/BBS and textonlyfirstnoun on samefulltext forward, B24 dynamicpadding, separateCPUscore aftersourcepredictionseal','checkpoint_policy':'immutable model snapshots every10 beforeVAL; rolling resume_latest atomically aftereverycompletedepoch inclscheduledVAL; finalepoch100 modelplusfullresumeimmutable; incompleteepochdiscarded onresume','resume_state':'model/optimizer/scheduler/Python+NumPy+torchCPU+allCUDA RNG/sharedloadergenerator','save_storage_estimate_GiB':7,'native_effective_milestones_optimizer_steps':[3978,4131],'scheduler_new_lr_first_used_epochs':[27,28],'source_loader_order_not_sorted':True,'timing_doesnot_initialize_formal_training':True};cfg.pop('steps');cfg.pop('warmup_steps');(R/'TRAIN_PLAN_TEMPLATE.json').write_text(json.dumps(cfg,indent=2)+'\n');assert not torch.cuda.is_initialized();print(json.dumps({'CODE_READY':True,'code_sha256':cfg['code_sha256'],'val_contract':val_contract,'template_sha256':sha(R/'TRAIN_PLAN_TEMPLATE.json')}))
if __name__=='__main__':main()
