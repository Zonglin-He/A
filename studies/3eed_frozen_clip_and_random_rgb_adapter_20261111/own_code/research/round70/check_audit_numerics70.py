from common70 import *
from scipy.special import softmax
p=read(H/'experiment/DESIGN_LOCK.json');maximum={'independent':0.,'same_float32_matmul':0.,'float64':0.};cases=[]
for r in rows():
 rel='samples_PRIVATE/'+r['id']+'.npz';z=prefix(r);a=load(ROOT/p['prepared_root']/rel);c=load(ROOT/p['clip_root']/rel);ap=load(ROOT/p['A_root']/rel);am=dict(zip(ap['arm_names'],ap['scores']));base=np.log(np.maximum(z['scores'].astype(float),1e-300));v=c['visual'];qg=a['query_group']
 for arm,kind,joint,relia,shuffle in [('A_subject_tight','subject',False,True,False),('A_subject_joint','subject',True,True,False),('A_full_joint','full',True,True,False),('A_projectonly_joint','subject',True,False,False),('A_mismatch_joint','subject',True,True,True)]:
  vv=v.mean(1) if joint else v[:,0];tt=c['subject_text' if kind=='subject' else 'full_text'];rr=a['reliability'] if relia else a['project_reliability'];got={}
  for name,sim in [('independent',np.einsum('ij,j->i',vv,tt)),('same_float32_matmul',vv@tt),('float64',vv.astype(float)@tt.astype(float))]:
   if shuffle:sim=sim[perm(len(sim),'R70Avisualbinding')]
   center=np.dot(rr,sim)/rr.sum() if rr.sum()>0 else 0.;delta=rr*(sim-center)*float(c['logit_scale']);out=softmax(base+delta[qg]);error=float(np.max(abs(out-am[arm])));maximum[name]=max(maximum[name],error)
   if name=='same_float32_matmul':assert error<1e-12,(arm,error)
  cases.append({'arm':arm,'row':r['id']})
dump(H/'AUDIT_FLOAT_REDUCTION_DIAGNOSTIC.json',{'state':'PASS_EXACT_FLOAT32_MATMUL_PRODUCER_REPLAY_ALL336','maximum_absolute_probability_errors':maximum,'no_task_GT_used':True});print(maximum)
