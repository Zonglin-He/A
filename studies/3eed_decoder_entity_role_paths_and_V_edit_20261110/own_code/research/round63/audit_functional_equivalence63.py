"""Saved representation equality implies identical deterministic R59 readout."""
from common63 import *
H=ROOT/'research/round63';p=read(H/'experiment/EDIT_FORMAL_EXECUTION_PLAN_LOCK.json');root=ROOT/p['edit_root'];ss=read(root/'SEALED.json');mx=0.;n=0
for r in read(ROOT/p['manifest']):
 rel='states_PRIVATE/'+r['id']+'.npz';assert sha(root/rel)==ss['files'][rel];z=load(root/rel,[k+'_'+v+'_h' for k in ['physical','functional'] for v in 'ABC'])
 for v in 'ABC':mx=max(mx,float(abs(z['physical_'+v+'_h']-z['functional_'+v+'_h']).max()));n+=1
assert mx==0.
out=H/'functional_equivalence_audit_attempt001';assert not out.exists();out.mkdir();dump(out/'AUDIT_RESULT.json',{'state':'PASS_R63_PHYSICAL_FUNCTIONAL_ALL_HIDDEN_VALUES_BITEXACT','n':336,'paired_sample_views':n,'max_hidden_error':mx,'R59_same_deterministic_readout_of_identical_representations_equivalent':True,'R59_functional_alias_inferred_not_an_extra_scored_arm':True,'prediction_seal_sha256':sha(root/'SEALED.json'),'no_GT_or_model_or_new_scoring':True,'program_sha256':sha(__file__)});print({'n':336,'views':n,'max_hidden_error':mx,'R59_equivalence_established_from_identical_inputs':True})
