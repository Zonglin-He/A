"""Independent CPU audit of sealed R107 targets, state edits and all endpoints."""
from pathlib import Path
import json,sys,hashlib,numpy as np,torch
R=Path(__file__).resolve().parents[2];H=R/'research/round107';B=R/'crossplatform/round107';sys.path.insert(0,str(R/'research/round105'));from audit105 import replay
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 torch.set_num_threads(4);matrix=rd(B/'MATRIX_SEALED.json');assert matrix['n']==768 and len(matrix['arms'])==55 and rd(H/'ENGINEERING.json')['n']==12
 paths={'waymo':R/'crossplatform/round91/trained_source_attempt001/states_PRIVATE/R76_full_disc_e1.npz','drone':R/'crossplatform/round97/drone/SELECTED_SOURCE_ONLY.npz','quad':R/'crossplatform/round97/quad/SELECTED_SOURCE_ONLY.npz'};n=ends=trials=fits=aliases=enabled=0;mx=0.;norms=[]
 for name,hs in matrix['directions'].items():
  src,_,tgt=name.split('_');O=B/name;old=R/'crossplatform/round106'/name;assert sha(O/'SEALED.json')==hs
  for p,h in rd(O/'SEALED.json')['files'].items():assert sha(O/p)==h
  with np.load(paths[src]) as z:S={k:z[k] for k in z.files if k.startswith('match.')}
  head={'weight':S['match.4.weight'],'bias':S['match.4.bias']}
  for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
   sid=r['id'];out=np.load(O/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(out['arm_names'],out['scores']));op=np.load(old/'samples_PRIVATE'/(sid+'.npz'));opp=dict(zip(op['arm_names'],op['scores']));rec=rd(O/'records_PRIVATE'/(sid+'.json'));orec=rd(old/'records_PRIVATE'/(sid+'.json'));n+=1
   assert np.array_equal(out['source_boxes'],op['source_boxes']);assert np.array_equal(out['queries'],out['scores'].argmax(1));assert np.isfinite(out['scores']).all() and np.allclose(out['scores'].sum(1),1,atol=1e-12)
   for a in set(ps)&set(opp):assert np.array_equal(ps[a],opp[a])
   su=np.load(R/'crossplatform/round106/direct_teacher_supplement_PRIVATE'/name/(sid+'.npz'))
   for a,p in zip(su['arm_names'],su['scores']):assert np.array_equal(ps['R106_'+a],p)
   def ck(a,b):
    nonlocal ends,mx
    e=float(abs(a-b).max());mx=max(mx,e);assert e<3e-6 and int(a.argmax())==int(b.argmax()),(name,sid,e,int(a.argmax()),int(b.argmax()));ends+=1
   if rec['protected_alias_fallback']:
    aliases+=1;assert not rec['fits'] and not (O/'features_PRIVATE'/(sid+'.npz')).exists();assert all(np.array_equal(p,opp['prepared_frozen']) for a,p in ps.items() if a not in opp and not a.startswith('R106_'));continue
   with np.load(old/'features_PRIVATE'/(sid+'.npz')) as z:vv={k:{v:{f:z[k+'_'+v+'_'+f] for f in ['h','u','late','mask','r','p']} for v in ['A','B','C','fusedABC']} for k in ['real','randomA','wrong']}
   with np.load(O/'features_PRIVATE'/(sid+'.npz')) as z:
    for k in vv:vv[k]['AB']={f:z[k+'_AB_'+f] for f in ['h','u','late','mask','r','p']}
   frozen={k:{v:replay(f,S) for v,f in fs.items()} for k,fs in vv.items()}
   for k in vv:
    for v,f in vv[k].items():ck(frozen[k][v],f['p'])
    ck(frozen[k]['AB'],ps[k+'_frozen_AB']);ck((frozen[k]['AB']+frozen[k]['C'])/2,ps[k+'_frozen_AB_C']);ck(np.mean([frozen[k][v] for v in ['A','B','AB','C']],0),ps[k+'_frozen_ensemble4'])
   enabled+=rec['updated_output_enabled'];assert rec['updated_output_enabled']==orec['families']['real']['updated_output_enabled'];ev=rec['events'];ic=ev.index('cache_C_and_fusedABC');assert all(j<ic for j,x in enumerate(ev) if x.startswith('fit_') or x.startswith('norm_'))
   sts={};fitted={}
   for label,info in rec['fits'].items():
    fits+=1;k='randomA' if label.startswith('randomA') else 'wrong' if label.startswith('wrong') else 'real';st=dict(np.load(O/'states_PRIVATE'/(sid+'_'+label+'.npz')));sts[label]=st;assert sum(x.size for x in st.values())==18496
    pa,pb,pab=[vv[k][v]['p'] for v in ['A','B','AB']]
    if label=='teacher_shuffle_keep':pb=pb[np.random.default_rng(int(hashlib.sha256(('R94querycontrol|'+sid).encode()).hexdigest()[:16],16)).permutation(256)]
    q=pab if '_native' in label else (pa+pb)/2;targets=[q,q]+([pab] if label.endswith('keep') else []);tt=np.log(np.stack(targets));assert np.max(abs(tt-np.array(info['target_log'])))<1e-12
    views=['A','B']+(['AB'] if label.endswith('keep') else []);pp={v:replay(f,S,st) for v,f in vv[k].items()};fitted[label]=pp;err=np.stack([t-np.log(pp[v]) for t,v in zip(tt,views)]);err-=err.mean(1,keepdims=True);assert abs(float(np.square(err).mean()/2)-info['final_error'])<2e-6
    for t in info['trials']:
     rho=(t['before']-t['after'])/max(t['predicted_reduction'],1e-30);assert abs(rho-t['rho'])<1e-8 and t['accepted']==(t['predicted_reduction']>0 and t['after']<t['before'] and rho>.1);trials+=1
   if sts:
    ost=dict(np.load(old/'states_PRIVATE'/(sid+'_real.npz')));new=sts['real_mean_keep'];ns=dict(np.load(O/'states_PRIVATE'/(sid+'_norm_old_direction.npz')));od={k:ost[k].astype(float)-head[k] for k in head};nd={k:new[k].astype(float)-head[k] for k in head};on=np.sqrt(sum((v*v).sum() for v in od.values()));nn=np.sqrt(sum((v*v).sum() for v in nd.values()));scale=nn/on if on else 0.;assert all(np.array_equal(ns[k],(head[k].astype(float)+scale*od[k]).astype(np.float32)) for k in head);assert abs(scale-rec['norm_control']['scale_old_direction'])<1e-12;norms.append(rec['norm_control']);fitted['norm_old_direction']={v:replay(f,S,ns) for v,f in vv['real'].items()}
    for k in ['real','randomA']:
     pf=rec['B_proof'][k];prior=orec['families'][k]['support']['B'];assert all(pf[a]==prior[a] for a in ['point_sha256','sample_sha256','registration_sha256','aligned_seed_n'])
   for qn,q in [('mean',(vv['real']['A']['p']+vv['real']['B']['p'])/2),('native',vv['real']['AB']['p'])]:
    on=rec['updated_output_enabled'];ck(q,ps['real_'+qn+'_direct_all']);ck(q if on else frozen['real']['fusedABC'],ps['real_'+qn+'_direct_matched']);ck((q+vv['real']['C']['p'])/2 if on else (frozen['real']['AB']+frozen['real']['C'])/2,ps['real_'+qn+'_direct_matched_C'])
   for label in ['real_mean_keep','real_native2','real_native_keep','randomA_mean_keep','wrong_mean_keep','teacher_shuffle_keep','norm_old_direction']:
    k='randomA' if label.startswith('randomA') else 'wrong' if label.startswith('wrong') else 'real';ff=fitted[label] if orec['families'][k]['updated_output_enabled'] else frozen[k]
    for v in ['C','fusedABC','AB_C']:ck((ff['AB']+ff['C'])/2 if v=='AB_C' else ff[v],ps[label+'_'+v])
  print('AUDIT107',name,flush=True)
 dump(H/'NATIVE_AUDIT.json',{'state':'PASS_SEALS_BOXES_REFERENCES_TARGETS_LASTHEAD_STATES_TRIALS_ALL_READOUTS_NORM_CONTROL_FALLBACKS','n':n,'arms':55,'fits':fits,'trials':trials,'endpoints':ends,'max_CPU_native_probability_error':mx,'protected_alias_fallback':aliases,'updated_output_enabled':enabled,'full_native_engineering':12,'C_used_in_updates':False,'GT_used':False,'FINAL_access':False});dump(H/'NORM_CONTROL_SUMMARY.json',{'n':len(norms),'old_zero_direction':sum(x['old_zero_direction'] for x in norms),'mean':{k:float(np.mean([x[k] for x in norms])) for k in ['old_parameter_L2','protected_parameter_L2','normmatched_parameter_L2','scale_old_direction','direction_cosine']}});print('AUDIT107_PASS',n,ends,fits,trials)
if __name__=='__main__':main()
