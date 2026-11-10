"""Freeze downstream stages and explicit exposed-DEV site/head selection."""
import argparse
from common63 import *
H=ROOT/'research/round63';ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','score','scalar','select_site','select_head']);ap.add_argument('--mode',choices=['screen','heads','edit'],default='screen');a=ap.parse_args();p=read(H/'experiment/FORMAL_EXECUTION_PLAN_LOCK.json')
if a.stage.startswith('select'):
 mode='screen' if a.stage=='select_site' else 'heads';assert read(H/('scalar_'+mode+'_audit_attempt001/AUDIT_RESULT.json'))['state'].startswith('PASS_');summary=read(H/('score_'+mode+'336_attempt001/TASK_SUMMARY.json'))['ALL336']['balanced_mIoU_percent'];baseline=summary['frozen_ABC']
 if mode=='screen':
  candidates=[(i,k) for i in range(6) for k in ['drop_reference','drop_subject'] if summary[f'L{i}_{k}']>baseline+1e-10];best=max(candidates,key=lambda x:(summary[f'L{x[0]}_{x[1]}'],-x[0],x[1]=='drop_reference')) if candidates else None;result={'selected_layer':best[0] if best else None,'selected_kind':best[1] if best else None};out=H/'experiment/SITE_SELECTION_LOCK.json'
 else:
  selected=read(H/'experiment/SITE_SELECTION_LOCK.json');candidates=[i for i in range(8) if summary['H'+str(i)]>baseline+1e-10];best=max(candidates,key=lambda i:(summary['H'+str(i)],-i)) if candidates else None;result={**selected,'selected_head':best};out=H/'experiment/HEAD_SELECTION_LOCK.json'
 result.update(state='R63_EXPOSED_DEV_SELECTION_CONTINUE' if best is not None else 'R63_NO_POSITIVE_CAUSAL_EFFECT_STOP_BEFORE_NEXT_STAGE',selection_uses_development_task_labels=True,not_independent_confirmation=True,all_arm_balanced_mIoU_percent=summary);assert not out.exists();dump(out,result)
 runtime={k:result[k] for k in ['selected_layer','selected_kind']+(['selected_head'] if mode=='heads' else [])};runtime['selection_provenance_sha256']=sha(out);runtime['selected_on_exposed_development']=True;dump(out.with_name(('SITE' if mode=='screen' else 'HEAD')+'_RUNTIME_CONFIG_LOCK.json'),runtime);print(result);raise SystemExit(0)
mode=a.mode
if mode=='heads':
 p=read(H/'experiment/HEAD_EXECUTION_PLAN_LOCK.json')
 p.update(read(H/'experiment/SITE_RUNTIME_CONFIG_LOCK.json'));assert p['selected_layer'] is not None;p['heads_arms']=p['reference_arms']+['H'+str(i) for i in range(8)]
if mode=='edit':p=read(H/'experiment/EDIT_FORMAL_EXECUTION_PLAN_LOCK.json')
p.update(mode=mode,prediction_root=p[mode+'_root'],audit_root='research/round63/'+mode+'_audit_attempt001',score_root='research/round63/score_'+mode+'336_attempt001',scalar_audit_root='research/round63/scalar_'+mode+'_audit_attempt001',arm_names=p[mode+'_arms'])
if a.stage=='audit':
 p['programs_sha256']={f:sha(ROOT/f) for f in ['research/round63/common63.py','research/round63/audit63.py']};p['inputs_sha256'][p['prediction_root']+'/SEALED.json']=sha(ROOT/p['prediction_root']/'SEALED.json')
 if mode=='edit':p['programs_sha256']={f:sha(ROOT/f) for f in ['research/round63/common63.py','research/round63/audit_value_edit63.py','research/round62/audit62.py']}
elif a.stage=='score':
 p['independent_audit']=p['audit_root']+'/AUDIT_RESULT.json';p['programs_sha256']={f:sha(ROOT/f) for f in ['research/round63/common63.py','research/round63/score63.py','research/round55/score55.py']};p['inputs_sha256'][p['independent_audit']]=sha(ROOT/p['independent_audit']);p['inputs_sha256'][p['prediction_root']+'/SEALED.json']=sha(ROOT/p['prediction_root']/'SEALED.json')
else:
 p['programs_sha256']={f:sha(ROOT/f) for f in ['research/round63/common63.py','research/round63/audit_scalar63.py','research/round55/audit_saved_scalars55.py']};p['inputs_sha256'][p['score_root']+'/SEALED.json']=sha(ROOT/p['score_root']/'SEALED.json')
if a.stage in ['score','scalar']:p['inputs_sha256']['research/round55/score_dev336_attempt001/SEALED.json']=sha(ROOT/'research/round55/score_dev336_attempt001/SEALED.json')
out=H/'experiment'/(mode.upper()+'_'+a.stage.upper()+'_PLAN_LOCK.json');assert not out.exists();dump(out,p);print({'stage':a.stage,'mode':mode,'plan_sha256':sha(out)})
