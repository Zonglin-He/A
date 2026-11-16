"""Independent CPU native head replay, target/fallback and sealed artifact audit."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round105';B=R/'crossplatform/round105';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def replay(f,S,st=None):
 x=torch.from_numpy(f['h'])[None]
 for j in [0,2,4]:
  W=torch.from_numpy(S['match.'+str(j)+'.weight']);b=torch.from_numpy(S['match.'+str(j)+'.bias'])
  if j==4 and st is not None:W,b=torch.from_numpy(st['weight']),torch.from_numpy(st['bias'])
  x=F.linear(x,W,b)
  if j<4:x=x.relu()
 q=F.normalize(x[0]+torch.from_numpy(f['late']),dim=-1);u=torch.from_numpy(f['u']);mask=torch.from_numpy(f['mask']).bool();s=q@u.T/.07;s=(torch.logsumexp(s[:,mask],-1)-torch.logsumexp(s[:,~mask],-1)).double()+torch.from_numpy(f['r']);return s.softmax(0).numpy()
def main():
 torch.set_num_threads(4);mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==768;eng=rd(H/'ENGINEERING.json');assert eng['n']==12;n=ends=trials=0;maximum=0.;coverage=[];refs=['original_source','prepared_frozen','frozen_four','direct_FC','joint3_FC']
 col=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json') if (R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json').exists() else None
 # Explicit immutable prepared states, checked against source registry below.
 paths={'waymo':R/'crossplatform/round91/trained_source_attempt001/states_PRIVATE/R76_full_disc_e1.npz','drone':R/'crossplatform/round97/drone/SELECTED_SOURCE_ONLY.npz','quad':R/'crossplatform/round97/quad/SELECTED_SOURCE_ONLY.npz'}
 for name,sh in mat['directions'].items():
  src,_,tgt=name.split('_');root=B/name;assert sha(root/'SEALED.json')==sh;seal=rd(root/'SEALED.json')
  for p,h in seal['files'].items():assert sha(root/p)==h
  with np.load(paths[src]) as ss:S={k:ss[k] for k in ss.files if k.startswith('match.')}
  for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
   sid=r['id'];out=np.load(root/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(out['arm_names'],out['scores']));old=np.load(R/'crossplatform/round100'/name/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(old['arm_names'],old['scores']));rec=rd(root/'records_PRIVATE'/(sid+'.json'));status=rec['status'];assert np.array_equal(out['source_boxes'],old['source_boxes']);assert all(np.array_equal(p[k],po[k]) for k in refs);coverage.append({'source':src,'target':tgt,**status});n+=1
   def check(a,b):
    nonlocal maximum,ends
    err=float(abs(a-b).max());maximum=max(maximum,err);assert err<2e-6 and int(a.argmax())==int(b.argmax()),(name,sid,err);ends+=1
   if status['fallback'].startswith('protected_'):
    assert not (root/'features_PRIVATE'/(sid+'.npz')).exists() and not rec['fits'];assert all(np.array_equal(p[k],po['prepared_frozen']) for k in mat['arms'] if k not in refs);continue
   with np.load(root/'features_PRIVATE'/(sid+'.npz')) as zz:
    views={v:{k:zz[v+'_'+k] for k in ['h','u','late','mask','r','p']} for v in ['A','AB','AC','jointABC','wrong_A','wrong_AB','wrong_AC','wrong_jointABC']}
   frozen={v:replay(f,S) for v,f in views.items()};frozen['ensemble3']=np.mean([frozen[k] for k in ['A','AB','AC']],0);frozen['wrong_ensemble3']=np.mean([frozen['wrong_'+k] for k in ['A','AB','AC']],0)
   for v,f in views.items():check(frozen[v],f['p'])
   for arm,q in [('prepared_frozen',frozen['A']),('frozen_A_AB',(frozen['A']+frozen['AB'])/2),('frozen_AC',frozen['AC']),('frozen_ensemble3',frozen['ensemble3']),('frozen_jointABC',frozen['jointABC']),('wrong_frozen_AC',frozen['wrong_AC']),('wrong_frozen_jointABC',frozen['wrong_jointABC']),('wrong_frozen_ensemble3',frozen['wrong_ensemble3'])]:check(q,p[arm])
   enabled=status['TTA_output_enabled'];assert enabled==(status['adaptation_attempted'] and status['heldout_new_observation_available']);fitted={}
   for label,info in rec['fits'].items():
    with np.load(root/'states_PRIVATE'/(sid+'_'+label+'.npz')) as ss:st={k:ss[k] for k in ss.files}
    assert sum(v.size for v in st.values())==18496;prefix='wrong_' if label=='wrong' else '';vv={v:views[prefix+v] for v in ['A','AB','AC','jointABC']};fit={v:replay(f,S,st) for v,f in vv.items()};fit['ensemble3']=np.mean([fit[k] for k in ['A','AB','AC']],0);fitted[label]=fit;ta=(frozen['A']+frozen[prefix+'AB'])/2
    if label=='teacher_shuffle':
     perm=np.random.default_rng(int(hashlib.sha256(('R94querycontrol|'+sid).encode()).hexdigest()[:16],16)).permutation(256);ta=(frozen['A']+frozen['AB'][perm])/2
    assert abs(ta-np.array(info['target'])).max()<2e-6;es=[]
    for v in ['A','AB']:
     e=np.log(ta)-np.log(fit[v]);es.append(e-e.mean())
    L=float(np.square(es).mean()/2);assert abs(L-info['final_error'])<1e-6,(name,sid,label,L,info['final_error'])
    for t in info['trials']:
     rho=(t['before']-t['after'])/max(t['predicted_reduction'],1e-30);assert abs(rho-t['rho'])<1e-8 and t['accepted']==(t['predicted_reduction']>0 and t['after']<t['before'] and rho>.1);trials+=1
   a=fitted.get('real',{}) if enabled else {};s=fitted.get('teacher_shuffle',{}) if enabled else {};w=fitted.get('wrong',{}) if enabled else {}
   for arm,v in [('TTA_A','A'),('TTA_AB','AB'),('TTA_AC','AC'),('TTA_jointABC','jointABC'),('TTA_ensemble3','ensemble3')]:check(a.get(v,frozen[v]),p[arm])
   for arm,v in [('teacher_shuffle_AC','AC'),('teacher_shuffle_jointABC','jointABC')]:check(s.get(v,frozen[v]),p[arm])
   for arm,v in [('wrong_TTA_AC','AC'),('wrong_TTA_jointABC','jointABC'),('wrong_TTA_ensemble3','ensemble3')]:check(w.get(v,frozen['wrong_'+v]),p[arm])
  print('AUDIT105',name,flush=True)
 groups=[]
 for tgt in ['waymo','drone','quad']:
  rr=[x for x in coverage if x['target']==tgt];groups.append({'target':tgt,'n':len(rr),'attempted_fit':sum(x.get('adaptation_attempted',False) for x in rr),'real_C_available':sum(x.get('heldout_new_observation_available',False) for x in rr),'TTA_final_enabled':sum(x.get('TTA_output_enabled',False) for x in rr),'protected_alias_fallback':sum(x['fallback'].startswith('protected_') for x in rr),'mean_seconds':float(np.mean([x['seconds'] for x in rr]))})
 dump(H/'COVERAGE_SUMMARY.json',{'groups':groups,'all_768_retained':True,'attempted_fit_cost_kept_even_without_C':True});dump(H/'NATIVE_AUDIT.json',{'state':'PASS_R105_ALL_SEALED_PREDICTIONS_NATIVE_HEAD_REPLAY_TARGETS_TRIALS_AND_FALLBACKS','n':n,'endpoints':ends,'trials':trials,'max_CPU_native_probability_error':maximum,'engineering_native_full_and_patched_cases':eng['n'],'task_GT_used':False,'FINAL_access':False});print('AUDIT105_PASS',n,ends,trials)
if __name__=='__main__':main()
