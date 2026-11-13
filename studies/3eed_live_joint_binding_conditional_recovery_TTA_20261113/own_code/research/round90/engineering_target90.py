"""Real target input contract only, no task labels."""
from target90 import *
P=B90/'engineering_target_attempt001';out=[]
for r in [next(r for r in read(H90/'experiment/DEV336_PRIVATE.json') if r['platform']==p) for p in ['drone','quad']]:
 a=load(P/'samples_PRIVATE'/(r['id']+'.npz'));assert len(a['arm_names'])==len(ARMS90);assert np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(-1),1.,atol=1e-12);tr=read(P/'records_PRIVATE'/(r['id']+'.json'));full=tr['candidate_joint'][0];bypass=tr['candidate_joint_bypass'][0];assert full['gradient_eta_router']>1e-8 and bypass['gradient_eta_router']<1e-6
 for t in [v for rr in tr.values() for v in rr]:assert not t['accepted'] or t['after']<=t['before']+1e-10
 out.append({'platform':r['platform'],'primary_gradient_eta_router':full['gradient_eta_router'],'bypass_gradient_eta_router':bypass['gradient_eta_router'],'primary_parameter_L2':full['parameter_L2'],'primary_step_accepted':full['accepted'],'task_GT_read':False})
dump(H90/'ENGINEERING_TARGET_AUDIT.json',{'state':'PASS_R90_REAL_TARGET_LEARNED_ROUTER_UPDATE_AND_ALL_READOUTS','n':2,'rows':out});print(out)
