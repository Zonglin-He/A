"""Freeze one DEV-selected pair and its direct mechanism controls before extension."""
from core81 import *
s=read(H81/'DEV_SELECTION.json');assert s['eligible'];cfg=(s['scope'],s['signal']);configs=[cfg]
if cfg[1]!='D' and ('all',cfg[1]) not in configs:configs.append(('all',cfg[1]))
if ('all','ALL') not in configs:configs.append(('all','ALL'))
lock={'state':'R81_FIXED_SINGLE_DEV_SELECTION_EXTENSION','selected':s['selected'],'selected_scope':cfg[0],'selected_signal':cfg[1],'selected_parameters':s['parameters'],'correct_configs':configs,'arms':REFS81+[a+'__'+b for a,b in configs]+[s['selected']+'__shuffle'],'depth_all_scope_is_exact_alias_of_image_adapter':cfg[1]=='D','same_source_masks_step_rule_and_coefficient_as_DEV':True,'configuration_selection_used_only_DEV_labels':True,'extension_labels_used_for_configuration_or_step':False,'input_roster':'research/round69/input/ALL2304_PRIVATE.json','cohorts':['VAL1280','TRAIN1024'],'no_extra_model_or_source_training':True,'FINAL_access':False,'inputs_sha256':{}}
for f in ['research/round81/DEV_SELECTION.json','crossplatform/round81/dev336_attempt001/SEALED.json','research/round69/input/ALL2304_PRIVATE.json','crossplatform/round80/prediction2640_attempt001/SEALED.json']:lock['inputs_sha256'][f]=sha(ROOT/f)
path=H81/'experiment/EXTENSION_LOCK.json';assert not path.exists();dump(path,lock);print(lock)
