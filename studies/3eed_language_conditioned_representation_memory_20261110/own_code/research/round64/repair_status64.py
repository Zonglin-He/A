"""Metadata-only correction: preserve every prediction byte; infer source-state flags."""
import shutil,os,sys
from common64 import *
sys.path.insert(0,str(ROOT/'research/round58'));import independent58 as independent
H=ROOT/'research/round64';p=read(H/'experiment/EXPERIMENT_SPEC_LOCK.json');pin(p);src=ROOT/p['prediction_root'];ss=read(src/'SEALED.json');out=ROOT/'crossplatform/round64/memory336_attempt002';assert not out.exists();out.mkdir(parents=True)
for rel,h in ss['files'].items():
 assert sha(src/rel)==h
 if rel=='READOUT_STATUS_PRIVATE.json':continue
 dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);os.link(src/rel,dest)
initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');sourcehash=independent.digest(independent.params(initial));oldroot=ROOT/p['r58_root'];oldseal=read(oldroot/'SEALED.json');oldfile=oldroot/'READOUT_STATUS_PRIVATE.json';assert sha(oldfile)==oldseal['files'][oldfile.name];old={r['id']:r['arm_statuses'] for r in read(oldfile)};rr=read(src/'READOUT_STATUS_PRIVATE.json');changed=0
for r in rr:
 sid=r['id'];cases=read(src/'cases_PRIVATE'/(sid+'.json'))
 for arm,st in r['arm_statuses'].items():
  o,a=arm.split('__')
  if not a.startswith('r58_'):continue
  if o=='original':correct=old[sid]['backtrack_online_ema_'+a[4:]]
  else:
   _,who,when=a.split('_');h=cases[o]['R58_step'][who+('_before_hash' if when=='pre' else '_after_hash')];unchanged=h==sourcehash;correct={'unknown':False,'fallback':unchanged,'no_parameter_update':unchanged}
  changed+=int(st!=correct);r['arm_statuses'][arm]=correct
receipt={'state':'R64_METADATA_ONLY_R58_SOURCE_STATE_FLAGS_REPAIRED','changed_statuses':changed,'prediction_arrays_and_cases_byte_identical':True,'producer_predictions_changed':False,'before_GT':True,'prior_attempt_preserved':True,'original_seal_sha256':sha(src/'SEALED.json'),'original_R58_status_sha256':sha(oldfile),'source_head_sha256':sourcehash};dump(out/'READOUT_STATUS_PRIVATE.json',rr);dump(out/'METADATA_REPAIR.json',receipt);seal(out,'R64_MEMORY_AND_MATCHED_ORDER_BASELINES_SEALED_METADATA_REPAIRED_BEFORE_SCORE',n=336,arm_names=ss['arm_names']);q=dict(p);q['prediction_root']=out.relative_to(ROOT).as_posix();q['independent_audit']='research/round64/independent_audit_attempt001/AUDIT_RESULT.json';q['score_root']='research/round64/score_dev336_attempt001';q['scalar_audit_root']='research/round64/independent_scalar_audit_attempt001';q['arm_names']=ss['arm_names'];q['reference_arms']=['original__source','original__frozen_ABC','original__r59_direct_all'];q['mode']='memory';q['inputs_sha256']=dict(p['inputs_sha256']);q['inputs_sha256'][str((src/'SEALED.json').relative_to(ROOT))]=sha(src/'SEALED.json');q['inputs_sha256'][str((out/'SEALED.json').relative_to(ROOT))]=sha(out/'SEALED.json');q['inputs_sha256'][str((ROOT/p['prepared_root']/'SEALED.json').relative_to(ROOT))]=sha(ROOT/p['prepared_root']/'SEALED.json');q['programs_sha256']=dict(p['programs_sha256'])
for f in ['audit64.py','score64.py','audit_scalar64.py','repair_status64.py','analyze64.py']:q['programs_sha256']['research/round64/'+f]=sha(H/f)
dump(H/'experiment/FORMAL_AUDIT_AND_SCORE_PLAN_LOCK.json',q);print(receipt)
