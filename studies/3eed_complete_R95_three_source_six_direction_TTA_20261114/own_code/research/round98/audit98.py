"""Full saved-evidence/head replay, source provenance and fresh native all-source sample."""
from core98 import *
def cached_independent98(m,f,r0):
 raw=m.match(f['h'][None])[0]+f['late'];q=F.normalize(raw,dim=-1);v=q@f['u'].T/.07;mask=f['mask'].bool();score=(torch.logsumexp(v[:,mask],-1)-torch.logsumexp(v[:,~mask],-1)).double()+r0;return score.softmax(0)
def main():
 matrix={};all_n=0;endpoint_n=0;native_n=0;maxp=maxe=maxst=0.;trialn=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);Shead=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   O=B98/(src+'_to_'+tgt);seal=jread(O/'SEALED.json');rows=rows97(tgt,'target');assert seal['n']==len(rows) and seal['arms']==ARMS98 and seal['source100_checkpoint_sha256']==source97(src)['checkpoint_sha256'];assert seal['same_original_A256_R97'] and not seal['FINAL_access']
   for path,s in seal['files'].items():assert sha(O/path)==s,path
   for ix,r in enumerate(rows):
    sid=r['id'];sv=load(O/'samples_PRIVATE'/(sid+'.npz'));scores=dict(zip(sv['arm_names'],sv['scores']));ev=load(O/'evidence_PRIVATE'/(sid+'.npz'));ff=load(O/'features_PRIVATE'/(sid+'.npz'));rec=jread(O/'records_PRIVATE'/(sid+'.json'));old=load(B97/(src+'_to_'+tgt)/'samples_PRIVATE'/(sid+'.npz'));assert np.array_equal(sv['source_boxes'],old['source_boxes']);oldrows=dict(zip(old['arm_names'],old['scores']))
    for k in REFERENCES98[:5]:assert np.array_equal(scores[k],oldrows[{'R97_direct_mean3':'direct_mean3','R97_direct_feature3':'direct_feature3','R97_physical_bt':'physical_bt'}.get(k,k)])
    delta=ev['correct'].mean(0)-ev['wrong'].mean((0,1));delta-=delta.mean();maxe=max(maxe,float(abs(delta-ev['delta']).max()));assert maxe<1e-10
    def soft(x):x=x-x.max();v=np.exp(x);return v/v.sum()
    q=soft(ev['correct'][0]+ev['r0']+delta);maxe=max(maxe,float(abs(q-ev['target']).max()));assert maxe<1e-10
    factual={k:torch.as_tensor(ff['F_'+k],device='cuda') for k in ['h','u','late','mask']};cfeat={k:torch.as_tensor(ff['C_'+k],device='cuda') for k in ['h','u','late','mask']};r0=torch.as_tensor(ev['r0'],device='cuda')
    with torch.no_grad():
     for name in FITS98:
      st=load(O/'states_PRIVATE'/(sid+'_'+name+'.npz'));put94(m,st);pF=cached_independent98(m,factual,r0);pC=cached_independent98(m,cfeat,r0)
      for view,p in [('F',pF),('C',pC),('FC',(pF+pC)/2)]:
       a=scores[name+'_'+view];err=float(abs(p.cpu().numpy()-a).max());maxp=max(maxp,err);assert err<1e-7 and int(p.argmax())==int(a.argmax());endpoint_n+=1
      u=rec['fits'][name];norm=float(np.sqrt(sum(np.square((st[k]-Shead[k]).astype(np.float64)).sum() for k in st)));assert abs(norm-u['parameter_L2'])<1e-10;assert u['accepted_steps']==sum(t['accepted'] for t in u['trials']);trialn+=len(u['trials'])
      for t in u['trials']:
       rho=(t['before']-t['after'])/max(t['predicted_reduction'],1e-30);assert abs(rho-t['rho'])<1e-8;assert bool(t['accepted'])==bool(t['predicted_reduction']>0 and np.isfinite(t['after']) and t['after']<t['before'] and rho>.1)
    if ix in [0,len(rows)//2,len(rows)-1]:
     restore(m,S)
     with torch.no_grad():z0,g,_,_=sample97(base,r)
     assert np.array_equal(z0['boxes'],sv['source_boxes']);rt,_=role97(base,m,r,z0,g,tensor_prefix(z0),tensor_data(g,r));ss,rr,st2,features,evidence,cost=complete_case98(base,m,S,r,z0,g,rt,old);maxp=max(maxp,float(abs(ss-sv['scores']).max()));assert np.max(abs(ss-sv['scores']))<1e-7 and np.array_equal(ss.argmax(1),sv['queries'])
     for name in FITS98:
      expected=load(O/'states_PRIVATE'/(sid+'_'+name+'.npz'));er=max(float(abs(st2[name][k]-expected[k]).max()) for k in expected);maxst=max(maxst,er);assert er<1e-7
     native_n+=1
    all_n+=1
    if (ix+1)%512==0:print('AUDIT98',src,tgt,ix+1,flush=True)
   matrix[src+'_to_'+tgt]=sha(O/'SEALED.json')
  assert fingerprint(base)==fp;del m,base;torch.cuda.empty_cache()
 assert all_n==18620 and endpoint_n==223440 and native_n==18;out={'state':'PASS_R98_ALL_EVIDENCE_FINAL_PATCHES_ENDPOINTS_AND_NATIVE_SOURCE_REPLAYS','n':all_n,'endpoints':endpoint_n,'native_cases':native_n,'LM_trial_scalar_checks':trialn,'max_probability_error':maxp,'max_evidence_error':maxe,'max_native_final_weight_error':maxst,'task_GT_used':False,'FINAL_access':False};jdump(H98/'NATIVE_AUDIT.json',out);jdump(B98/'MATRIX_SEALED.json',{'state':'R98_COMPLETE_USER_R95_SIX_DIRECTIONS_SEALED_AND_AUDITED_BEFORE_TASK_SCORING','n':all_n,'directions':matrix,'primary':'lm_FC','arms':ARMS98,'no_source_training':True,'FINAL_access':False});print(out)
if __name__=='__main__':main()
