"""Independent full training ledger, provenance, code, resume and source-validation seal audit."""
from pathlib import Path
import sys,json,hashlib,math
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';sys.path.insert(0,str(B));from factory import torch,sha,state_fingerprint,model,configure
rd=lambda p:json.loads(p.read_text())
def main():
 out=[]
 for pl in ['drone','quad']:
  d=B/(pl+'_source100');reg=rd(H/'experiment/ACTIVE_TRAINING_REGISTRATION.json')[pl]['train'];p=rd(Path(reg['path']));c=rd(d/'TRAINING_COMPLETE.json');n=p['steps_per_epoch'];assert c['primary_epoch']==100 and c['completed_optimizer_steps']==100*n and c['trained_platforms']==[pl] and c['no_joint_checkpoint_initialization'];assert c['plan_sha256']==reg['sha256']
  steps=[]
  for a in sorted((d/'attempts').iterdir()):
   sp=a/'steps.jsonl'
   if sp.exists():steps.extend(json.loads(s) for s in sp.read_text().splitlines())
  # A completed resumed epoch can supersede an interrupted prefix, never silently delete the original.
  by={}
  for s in steps:by[(s['epoch'],s['batch'])]=s
  assert len(by)==100*n and all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm_before_clip']) for x in by.values())
  for ep in range(1,101):
   assert all(by[ep,b]['global_step']==(ep-1)*n+b+1 for b in range(n))
  ck=Path(c['primary_checkpoint']);assert sha(ck)==c['primary_checkpoint_sha256'];a=torch.load(ck,map_location='cpu',weights_only=True);assert a['trained_platforms']==[pl] and a['epoch']==100 and a['source_provenance']['initial_model_sha256']==p['expected_model_initial_sha256']
  resume=torch.load(Path(c['final_resume_checkpoint']),map_location='cpu',weights_only=True);assert resume['trained_platforms']==[pl] and resume['global_step']==100*n and resume['next_epoch']==101 and resume['source_provenance']['plan_sha256']==c['plan_sha256'];assert all(torch.equal(a['model'][k],resume['model'][k]) for k in a['model']);del a,resume
  milestones=[json.loads(s) for s in (d/'MILESTONES.jsonl').read_text().splitlines()];eps=sorted(set(x['epoch'] for x in milestones));assert eps==list(range(p['source_validation_every'],101,p['source_validation_every']))
  for x in milestones:
   se=rd(Path(x['val_prediction_dir'])/'sealed.json');assert sha(Path(x['val_prediction_dir'])/'sealed.json')==x['val_prediction_seal_sha256'] and se['n']==p['source_val_contract']['n'] and se['parameters_buffers_unchanged'] and se['checkpoint_sha256']==x['checkpoint_sha256']
  out.append({'source':pl,'epochs':100,'optimizer_updates':100*n,'checkpoint_sha256':c['primary_checkpoint_sha256'],'source_val_sealed_epochs':eps,'source_only_provenance':True,'no_target_checkpoint_selection':True,'safe_full_resume_matches_final_model':True})
 z={'state':'PASS_R96_BOTH_SOURCE100_LEDGERS_CHECKPOINTS_AND_SOURCE_VAL_SEALS','sources':out,'FINAL_access':False};(H/'TRAINING_AUDIT.json').write_text(json.dumps(z,indent=2)+'\n');print(z)
if __name__=='__main__':main()
