"""Freeze one development-selected site as a label-free runtime config."""
from common62 import *
e=ROOT/'research/round62/experiment';s=read(e/'SEMANTIC_SITE_SELECTION_LOCK.json');assert s['selected_layer'] is not None
runtime=e/'SEMANTIC_RUNTIME_CONFIG_LOCK.json';assert not runtime.exists();dump(runtime,{'selected_layer':s['selected_layer'],'gain':1.5,'selected_on_exposed_development':True,'selection_provenance_sha256':sha(e/'SEMANTIC_SITE_SELECTION_LOCK.json'),'runtime_does_not_read_task_metrics_or_per_sample_labels':True})
p=read(e/'EXPERIMENT_SPEC_LOCK.json');p['selection']=runtime.relative_to(ROOT).as_posix()
for rel in ['research/round62/semantics62.py','research/round62/common62.py','research/round62/compute62.py']:p['programs_sha256'][rel]=sha(ROOT/rel)
for rel in [p['selection'],p['capture_root']+'/SEALED.json',p['layers_root']+'/SEALED.json']:p['inputs_sha256'][rel]=sha(ROOT/rel)
out=e/'SEMANTIC_EXECUTION_PLAN_LOCK.json';assert not out.exists();dump(out,p);print({'selected_layer':s['selected_layer'],'runtime_config_sha256':sha(runtime),'execution_plan_sha256':sha(out)})
