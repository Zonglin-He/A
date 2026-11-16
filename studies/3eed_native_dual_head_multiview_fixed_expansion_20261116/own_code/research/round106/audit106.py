"""Independent CPU head replay, exact frozen fallback, target and trial audit."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
R=Path(__file__).resolve().parents[2];H=R/'research/round106';B=R/'crossplatform/round106';sys.path.insert(0,str(R/'research/round105'));from audit105 import replay
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,z):p.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def main():
 torch.set_num_threads(4);matrix=rd(B/'MATRIX_SEALED.json');assert matrix['n']==768 and rd(H/'ENGINEERING.json')['n']==12;paths={'waymo':R/'crossplatform/round91/trained_source_attempt001/states_PRIVATE/R76_full_disc_e1.npz','drone':R/'crossplatform/round97/drone/SELECTED_SOURCE_ONLY.npz','quad':R/'crossplatform/round97/quad/SELECTED_SOURCE_ONLY.npz'};n=ends=trials=0;maximum=0.;summary=[]
 for name,h in matrix['directions'].items():
  src,_,tgt=name.split('_');root=B/name;assert sha(root/'SEALED.json')==h
  for p,x in rd(root/'SEALED.json')['files'].items():assert sha(root/p)==x
  with np.load(paths[src]) as x:S={k:x[k] for k in x.files if k.startswith('match.')}
  rows=rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');counts={k:{'B_available':0,'C_available':0,'updated_output_enabled':0,'B_aligned_seed_mean':[],'C_aligned_seed_mean':[]} for k in ['real','randomA','wrong']};alias=0
  for r in rows:
   sid=r['id'];out=np.load(root/'samples_PRIVATE'/(sid+'.npz'));scores=dict(zip(out['arm_names'],out['scores']));old=np.load(R/'crossplatform/round100'/name/'samples_PRIVATE'/(sid+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']));rec=rd(root/'records_PRIVATE'/(sid+'.json'));n+=1;assert np.array_equal(out['source_boxes'],old['source_boxes']);refs=['original_source','prepared_frozen','frozen_four','direct_FC','joint3_FC'];assert all(np.array_equal(scores[k],oldp[k]) for k in refs)
   def check(a,b):
    nonlocal maximum,ends
    e=float(abs(a-b).max());maximum=max(maximum,e);assert e<3e-6 and int(a.argmax())==int(b.argmax()),(name,sid,e);ends+=1
   if rec['protected_alias_fallback']:
    alias+=1;assert not rec['fits'] and not (root/'features_PRIVATE'/(sid+'.npz')).exists();assert all(np.array_equal(p,oldp['prepared_frozen']) for a,p in scores.items() if a not in refs);continue
   with np.load(root/'features_PRIVATE'/(sid+'.npz')) as x:views={k:{v:{f:x[k+'_'+v+'_'+f] for f in ['h','u','late','mask','r','p']} for v in ['A','B','C','fusedABC']} for k in ['real','randomA','wrong']}
   frozen={}
   for k,vv in views.items():
    frozen[k]={v:replay(f,S) for v,f in vv.items()};frozen[k]['ensemble']=np.mean([frozen[k][v] for v in ['A','B','C']],0)
    for v,f in vv.items():check(frozen[k][v],f['p'])
    for v in ['B','C','ensemble','fusedABC']:check(frozen[k][v],scores[k+'_frozen_'+v])
    for f in ['B_available','C_available','updated_output_enabled']:counts[k][f]+=rec['families'][k][f]
    for v in ['B','C']:
     if k!='wrong' and v in rec['families'][k]['support']:counts[k][v+'_aligned_seed_mean'].append(rec['families'][k]['support'][v].get('aligned_seed_n',0))
   check(frozen['real']['A'],oldp['prepared_frozen']);fitted={}
   for label,info in rec['fits'].items():
    k='real' if label=='teacher_shuffle' else label
    with np.load(root/'states_PRIVATE'/(sid+'_'+label+'.npz')) as x:st={a:x[a] for a in x.files}
    assert sum(a.size for a in st.values())==18496;fit={v:replay(f,S,st) for v,f in views[k].items()};fit['ensemble']=np.mean([fit[v] for v in ['A','B','C']],0);fitted[label]=fit;pb=frozen[k]['B']
    if label=='teacher_shuffle':perm=np.random.default_rng(int(hashlib.sha256(('R94querycontrol|'+sid).encode()).hexdigest()[:16],16)).permutation(256);pb=pb[perm]
    target=(frozen[k]['A']+pb)/2;assert abs(target-np.array(info['target'])).max()<3e-6;errors=[]
    for v in ['A','B']:e=np.log(target)-np.log(fit[v]);errors.append(e-e.mean())
    assert abs(float(np.square(errors).mean()/2)-info['final_error'])<2e-6
    for t in info['trials']:
     rho=(t['before']-t['after'])/max(t['predicted_reduction'],1e-30);assert abs(rho-t['rho'])<1e-8 and t['accepted']==(t['predicted_reduction']>0 and t['after']<t['before'] and rho>.1);trials+=1
   for k in ['real','randomA','wrong']:
    f=rec['families'][k];assert f['updated_output_enabled']==(f['B_available'] and f['C_available']);ps=fitted.get(k,frozen[k]) if f['updated_output_enabled'] else frozen[k]
    for v in ['A','B','C','ensemble','fusedABC']:check(ps[v],scores[k+'_TTA_'+v])
   ps=fitted.get('teacher_shuffle',frozen['real']) if rec['families']['real']['updated_output_enabled'] else frozen['real']
   for v in ['C','fusedABC']:check(ps[v],scores['teacher_shuffle_'+v])
  summary.append({'source':src,'target':tgt,'n':len(rows),'protected_alias_fallback':alias,'families':{k:{a:(float(np.mean(v)) if v else 0.) if isinstance(v,list) else v for a,v in z.items()} for k,z in counts.items()}});print('AUDIT106',name,flush=True)
 dump(H/'NATIVE_AUDIT.json',{'state':'PASS_R106_ALL_SEALED_NATIVE_HEAD_ENDPOINTS_TARGETS_TRIALS_FALLBACKS_AND_SOURCE_REFERENCES','n':n,'endpoints':ends,'trials':trials,'max_CPU_native_probability_error':maximum,'engineering_full_native_cases':12,'GT_used':False,'FINAL_access':False});dump(H/'COVERAGE_MODEL_SUMMARY.json',{'groups':summary,'all_768_retained':True});print('AUDIT106_PASS',n,ends)
if __name__=='__main__':main()
