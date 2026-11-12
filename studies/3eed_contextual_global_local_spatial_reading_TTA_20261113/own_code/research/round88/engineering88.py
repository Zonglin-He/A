from core88 import *
P=B88/'engineering_prediction_attempt001';cases=[];accepted=0
for r in [next(r for r in read(H88/'experiment/DEV336_PRIVATE.json') if r['platform']==p) for p in ['drone','quad']]:
 a=load(P/'samples_PRIVATE'/(r['id']+'.npz'));probs=dict(zip(a['arm_names'],a['scores']));rec=read(P/'records_PRIVATE'/(r['id']+'.json'));assert np.array_equal(probs['no_local_direct'],probs['frozen_bi']);diff=float(abs(probs['full_direct']-probs['frozen_bi']).max());assert diff>1e-8
 for t in rec['full']:assert t['gradient_norm_clipped']>0 and t['parameter_L2']>0;assert not t['accepted'] or t['after']<=t['before']+1e-10
 accepted+=sum(x['accepted'] for tr in rec.values() for x in tr)
 cases.append({'platform':r['platform'],'zero_route_native_parity':True,'local_route_max_probability_change':diff,'correct_vs_wrong_space_max_probability_change':float(abs(probs['full_direct']-probs['wrong_space_direct']).max()),'correct_vs_role_swap_max_probability_change':float(abs(probs['full_direct']-probs['role_swap_direct']).max()),'relation_routing_max_probability_change':float(abs(probs['full_direct']-probs['no_relation_direct']).max()),'first_step_parameter_L2':rec['full'][0]['parameter_L2']})
assert accepted>0;dump(H88/'ENGINEERING_AUDIT.json',{'state':'PASS_R88_REAL_TWO_PLATFORM_NATIVE_PARITY_SPATIAL_INTERVENTION_AND_NONZERO_TTA','n':2,'accepted_steps':accepted,'cases':cases,'task_GT_read':False});print(cases)
