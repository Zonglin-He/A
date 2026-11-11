"""Only hashes/roles of registered inputs and programs; no data/weight content."""
from common71 import *
p=read(H/'experiment/DESIGN_LOCK.json');plans={}
for name in ['DESIGN_LOCK','CLIP_PLAN_LOCK','PREDICT_PLAN_LOCK','AUDIT_PLAN_LOCK','SCORE_PLAN_LOCK']:
 f=H/'experiment'/(name+'.json');assert f.exists();plans[name+'_sha256']=sha(f)
inputs=[{'file':name,'sha256':digest} for name,digest in sorted(p['inputs_sha256'].items())];programs=[{'file':name,'sha256':digest} for name,digest in sorted(p['programs_sha256'].items())]
for x in inputs+programs:assert sha(ROOT/x['file'])==x['sha256']
rec={'state':'R71_REGISTERED_INPUT_PROGRAM_AND_PREDICTION_HASH_PROVENANCE','registered_input_hashes':inputs,'registered_program_hashes':programs,'stage_plan_digests':plans,'prediction_seal_sha256':sha(ROOT/p['prediction_root']/'SEALED.json'),'preparation_seal_sha256':sha(ROOT/p['prepared_root']/'SEALED.json'),'CLIP_evidence_seal_sha256':sha(ROOT/p['clip_root']/'SEALED.json'),'all_inputs_are_already_registered_R69_or_R70':True,'no_raw_input_feature_weight_or_identity_content_exported':True,'no_new_source_forward_or_parameter_update':True};dump(H/'PUBLIC_PROVENANCE.json',rec);print({'provenance':True,'registered_input_hashes':len(inputs),'registered_program_hashes':len(programs)})
