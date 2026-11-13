from target_core89 import *
P=B89/'engineering_target_attempt001';out=[]
for r in [next(r for r in read(H89/'experiment/DEV336_PRIVATE.json') if r['platform']==p) for p in ['drone','quad']]:
 a=load(P/'samples_PRIVATE'/(r['id']+'.npz'));pp=dict(zip(a['arm_names'],a['scores']));rec=read(P/'records_PRIVATE'/(r['id']+'.json'));assert np.array_equal(pp['no_residual_frozen'],pp['frozen_bi']);assert rec['candidate'][0]['gradient_norm_raw']>0
 for tr in rec.values():
  for t in tr:assert not t['accepted'] or t['after']<=t['before']+1e-10
 out.append({'platform':r['platform'],'candidate_vs_R76_probability_change':float(abs(pp['candidate_frozen']-pp['frozen_bi']).max()),'candidate_vs_query_only_probability_change':float(abs(pp['candidate_frozen']-pp['query_frozen']).max()),'wrong_space_probability_change':float(abs(pp['candidate_frozen']-pp['wrong_space_frozen']).max()),'first_step_parameter_L2':rec['candidate'][0]['parameter_L2'],'first_step_accepted':rec['candidate'][0]['accepted'],'task_GT_read':False})
assert any(x['first_step_parameter_L2']>0 for x in out);dump(H89/'ENGINEERING_TARGET_AUDIT.json',{'state':'PASS_R89_TWO_REAL_PLATFORM_FIXED_BOX_READOUT_AND_NONZERO_TTA','n':2,'rows':out});print(out)
