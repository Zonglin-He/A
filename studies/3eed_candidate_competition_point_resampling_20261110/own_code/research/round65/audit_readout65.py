"""Independent samples, all source forwards, support transport and actual readouts.
No R65 sampling/capture/prediction producer import, task labels or GT access.
"""
import sys,time,torch,hashlib
from common65 import *
from scipy.special import logsumexp,softmax
from scipy.sparse import csr_matrix
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,source_capture,fingerprint

def norm(a,fallback=None):
 n=np.linalg.norm(a,axis=-1,keepdims=True);x=np.divide(a,n,out=np.zeros_like(a),where=n>1e-12);return np.where(n>1e-12,x,fallback) if fallback is not None else x

def region(x,b,g):return np.less_equal(np.abs(x-b[:3]),b[3:]*g/2).all(1)&bool((b[3:]>0).all())
def ov(a,b):
 hi=np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2);lo=np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2);i=np.clip(hi-lo,0,None).prod();den=np.clip(a[3:],0,None).prod()+np.clip(b[3:],0,None).prod()-i;return float(i/den) if den>0 else 0.
def diverse(boxes,score,count):
 used=[]
 for j in np.argsort(-score,kind='stable'):
  if (boxes[j,3:]<=0).any():continue
  if any(ov(boxes[j],boxes[k])>=.5 for k in used):continue
  used.append(int(j))
  if len(used)==count:break
 return used

def samples(x,z,mask,sid):
 boxes=z['A_boxes'].astype(float);targets=diverse(boxes,z['A_scores'],4);refs=[]
 if mask.any() and (~mask).any():
  a=z['A_query'].astype(float)@z['A_tokens'].astype(float).T/.07;ss=logsumexp(a[:,mask],axis=1)-logsumexp(a[:,~mask],axis=1);refs=diverse(boxes,ss,2)
 pools=[np.where(region(x,boxes[j],1.25))[0] for j in targets];shell=np.zeros(len(x),bool);ref=np.zeros(len(x),bool)
 for j in targets:shell|=region(x,boxes[j],2.5)&~region(x,boxes[j],1.25)
 for j in refs:ref|=region(x,boxes[j],1.5)
 shell=np.where(shell)[0];rr=np.where(ref)[0];result={}
 for v in 'BC':
  ss=int(hashlib.sha256(('R65_sampling_v1|'+sid+'|'+v).encode()).hexdigest()[:16],16);perm=np.random.default_rng(ss^0x65A17).permutation(16384)
  for mode in ['random','directed','coverage']:
   rng=np.random.default_rng(ss);whole=np.arange(len(x),dtype=np.int64)
   def pick(a,k):return rng.choice(a if len(a) else whole,k,replace=True)
   if mode=='random':ix=pick(whole,16384)
   elif mode=='coverage':ix=np.r_[rng.permutation(whole),pick(whole,16384-len(x))]
   else:
    parts=[pick(whole,8192)];k=len(pools)
    if k:
     for j,pool in enumerate(pools):parts.append(pick(pool,4096//k+int(j<4096%k)))
    else:parts.append(pick(np.array([],np.int64),4096))
    parts.extend([pick(rr,2048),pick(shell,2048)] if len(rr) else [pick(shell,4096)]);ix=np.concatenate(parts)
   result[mode+'_'+v]=ix[perm].astype(np.int64)
  inverse=np.argsort(perm);assert np.array_equal(result['random_'+v][inverse][:8192],result['directed_'+v][inverse][:8192]);assert len(np.unique(result['coverage_'+v]))==len(x)
 focus=np.unique(np.concatenate(pools)) if pools else np.array([],np.int64);context=np.unique(np.r_[shell,rr]);return result,targets,refs,focus,context

def affinity(a,b,x):
 A=np.array([region(x,v,1.) for v in a],np.float32);B=np.array([region(x,v,1.) for v in b],np.float32);intersection=(csr_matrix(A)@csr_matrix(B).T).toarray();union=A.sum(1)[:,None]+B.sum(1)[None]-intersection;j=np.divide(intersection,union,out=np.zeros_like(union),where=union>0);ll=np.maximum(a[:,None,:3]-a[:,None,3:]/2,b[None,:,:3]-b[None,:,3:]/2);hh=np.minimum(a[:,None,:3]+a[:,None,3:]/2,b[None,:,:3]+b[None,:,3:]/2);inter=np.maximum(hh-ll,0).prod(2);u=np.maximum(a[:,3:],0).prod(1)[:,None]+np.maximum(b[:,3:],0).prod(1)[None]-inter;ious=np.divide(inter,u,out=np.zeros_like(u),where=u>0);af=np.sqrt(np.maximum(j*ious,0));af[intersection<4]=0;return af.astype(np.float32)
def logits(z,v):
 a=np.einsum('ik,jk->ij',z[v+'_query'].astype(float),z[v+'_tokens'].astype(float))/.07;m=z[v+'_token_mask'];return logsumexp(a[:,m],1)-logsumexp(a[:,~m],1)
def head(h,w):
 x=torch.tensor(h)[None]
 for j in ['0','2']:x=torch.nn.functional.linear(x,torch.tensor(w[j+'.weight']),torch.tensor(w[j+'.bias'])).relu()
 y=torch.nn.functional.linear(x,torch.tensor(w['4.weight']),torch.tensor(w['4.bias']))[0].numpy();y64=np.column_stack([x[0].numpy().astype(float),np.ones(256)])@np.column_stack([w['4.weight'],w['4.bias']]).astype(float).T;return y,y64

def distribution(raw,z,g):
 out=[]
 for v in 'ABC':
  q=torch.nn.functional.normalize(torch.tensor(np.asarray(raw[v],np.float32))[None],dim=-1);l=q@torch.tensor(z[v+'_tokens'])[None].transpose(-1,-2)/.07;m=torch.tensor(z[v+'_token_mask']).bool()[None,None];d=(torch.logsumexp(l.masked_fill(~m,-torch.inf),-1)-torch.logsumexp(l.masked_fill(m,-torch.inf),-1))[0].double().softmax(-1).numpy()
  if v!='A':d=g[v+'_W']@d;d=float(g[v+'_fixed_mass'])*d/max(float(d.sum()),1e-300)+(1-float(g[v+'_fixed_mass']))*g['base']
  out.append(d)
 return sum(out)/3

def readouts(z,g,w):
 y={};y64={}
 for v in 'ABC':y[v],y64[v]=head(z[v+'_h'],w)
 qq={v:norm(y64[v]) for v in 'ABC'};W=g['B_W'];m=W.sum(1);qB=np.divide(W@qq['B'],m[:,None],out=np.zeros_like(qq['A']),where=m[:,None]>0);ta=norm(qq['A']+g['B_affinity'].max(1)[:,None]*qB,qq['A']);targets={'A':np.linalg.norm(y64['A'],axis=1)[:,None]*ta};rho={'A':g['B_affinity'].max(1)}
 for v in 'BC':
  W=g[v+'_W'];m=W.sum(0);u=np.divide(W.T@ta,m[:,None],out=np.zeros_like(qq[v]),where=m[:,None]>0);rho[v]=g[v+'_affinity'].max(0)*(np.linalg.norm(u,axis=1)>1e-12);targets[v]=np.linalg.norm(y64[v],axis=1)[:,None]*norm(u,qq[v])
 fused={v:(1-rho[v][:,None])*y[v]+rho[v][:,None]*targets[v] for v in 'ABC'};return distribution(y,z,g),distribution(fused,z,g)

def process_stats(ix,A,focus,context,xyz,boxes,targets,refs,n):
 u=np.unique(ix);sf=np.zeros(len(xyz),bool);sc=sf.copy()
 for j in targets:sf|=region(xyz,boxes[j],1.25);sc|=region(xyz,boxes[j],2.5)&~region(xyz,boxes[j],1.25)
 for j in refs:sc|=region(xyz,boxes[j],1.5)
 return {'raw_n':n,'unique_input_n':len(u),'new_unique_vs_A':len(np.setdiff1d(u,np.unique(A))),'raw_fraction_covered':len(u)/n,'focus_raw_n':len(focus),'focus_draws':int(np.isin(ix,focus).sum()),'focus_unique_n':int(np.isin(u,focus).sum()),'context_raw_n':len(context),'context_draws':int(np.isin(ix,context).sum()),'context_unique_n':int(np.isin(u,context).sum()),'seed_n':len(xyz),'focus_seed_n':int(sf.sum()),'context_seed_n':int(sc.sum())}

def main():
 H=ROOT/'research/round65';p=read(H/'experiment/READOUT_AUDIT_PLAN_LOCK_v2.json');pin(p);rows=read(ROOT/p['manifest']);pred=ROOT/p['prediction_root'];sp=read(pred/'SEALED.json');assert sp['arm_names']==ARMS and sp['n']==336;cap=ROOT/p['new_capture_root'];cs=read(cap/'SEALED.json');prep=ROOT/p['prepared_root'];ps=read(prep/'SEALED.json');roles={r['id_PRIVATE']:np.array(r['mask'],bool) for r in read(ROOT/'research/round62/role_diagnostics_attempt001/MASKS_PRIVATE.json') if r['role']=='reference'};cert=read(H/'independent_capture_audit_attempt001/AUDIT_RESULT.json');assert cert['state'].startswith('PASS_') and cert['capture_seal_sha256']==sha(cap/'SEALED.json') and cert['preparation_seal_sha256']==sha(prep/'SEALED.json');torch.set_grad_enabled(False);torch.set_num_threads(4);w=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');start=time.perf_counter();nviews=0;naf=0;maxerr=0.;allstatus=read(pred/'READOUT_STATUS_PRIVATE.json');assert len(allstatus)==336
 for rel,h in sp['files'].items():assert sha(pred/rel)==h
 for i,r in enumerate(rows):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel] and sha(prep/rel)==ps['files'][rel];zold=load(ROOT/p['capture_root']/rel);znew=load(cap/rel);prepared=load(prep/rel);rawrec=read(prep/'cases_PRIVATE'/(sid+'.json'));assert sha(r['pointpath'])==rawrec['raw_sha256'];path=Path(r['pointpath']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,np.float32).reshape(-1,4);x=np.array(raw[:,:3],np.float32,copy=True)
  if r['platform']=='drone':x[:,2]+=1.8
  generated,targets,refs,focus,context=samples(x,zold,roles[sid],sid);assert np.array_equal(prepared['target_hypothesis_indices'],targets) and np.array_equal(prepared['reference_hypothesis_indices'],refs) and np.array_equal(prepared['focus_indices'],focus) and np.array_equal(prepared['context_indices'],context);actualstats=read(cap/'cases_PRIVATE'/(sid+'.json'))
  for key,ix in generated.items():
   assert np.array_equal(ix,prepared[key+'_point_indices']) and np.array_equal(ix,znew[key+'_point_indices']);nviews+=1
   assert process_stats(ix,zold['A_point_indices'],focus,context,znew[key+'_seed_xyz'],zold['A_boxes'],targets,refs,len(x))==actualstats[key]
  saved=load(pred/rel);aa=load(pred/'association_PRIVATE'/(sid+'.npz'));previous=load(ROOT/p['r59_root']/rel);expected={a:previous['scores'][list(previous['arm_names']).index(b)] for a,b in [('source','source'),('old_ABC','frozen_ABC'),('old_R59','direct_all')]}
  for mode in ['common']+p['sampling_modes']:
   z={k:zold[k] for k in ['A_h','A_query','A_tokens','A_token_mask','A_boxes']}
   for v in 'BC':
    for k in ['h','query','tokens','token_mask','boxes']:z[v+'_'+k]=zold[v+'_'+k] if mode=='common' else znew[mode+'_'+v+'_'+k]
   g={'base':softmax(logits(z,'A'))};assert np.array_equal(g['base'],aa[mode+'_base'])
   for v in 'BC':
    af=affinity(z['A_boxes'].astype(float),z[v+'_boxes'].astype(float),x.astype(float));assert np.array_equal(af,aa[mode+'_'+v+'_affinity']);naf+=1;W=af.astype(float)/(1+af.astype(float).sum(0,keepdims=True));mass=float((W@softmax(logits(z,v))).sum());assert np.array_equal(W,aa[mode+'_'+v+'_W']) and mass==aa[mode+'_'+v+'_fixed_mass'];g[v+'_affinity']=af;g[v+'_W']=W;g[v+'_fixed_mass']=mass
   mean,fused=readouts(z,g,w);expected[mode+('_ABC' if mode=='common' else '_mean')]=mean;expected[mode+'_R59']=fused
  assert saved['arm_names'].tolist()==ARMS and np.array_equal(saved['source_boxes'],zold['A_boxes']) and np.array_equal(saved['source_scores'],zold['A_scores']);assert np.array_equal(saved['selected_original_boxes'],saved['source_boxes'][saved['queries']])
  for j,a in enumerate(ARMS):
   e=float(abs(saved['scores'][j]-expected[a]).max());maxerr=max(maxerr,e);assert e<1e-11 and int(expected[a].argmax())==saved['queries'][j],('readout',i,a,e)
  assert allstatus[i]['id']==sid and allstatus[i]['arm_statuses']=={a:{'unknown':False,'fallback':False,'no_parameter_update':True} for a in ARMS}
  if (i+1)%32==0:print('AUDIT65',i+1,len(rows),nviews,flush=True)
 assert nviews==cert['new_full_source_replay_forwards'];out=H/'independent_audit_attempt002';assert not out.exists();out.mkdir();result={'state':'PASS_R65_INDEPENDENT_RAW_SAMPLING_ALL_SOURCE_FORWARDS_ASSOCIATION_AND_READOUT','n':336,'arm_names':ARMS,'all_new_full_source_replays':nviews,'all_raw_affinity_matrices_exact':naf,'all_paired_samples_exact':True,'all_full_coverage_views_cover_every_raw_point':True,'max_probability_error':maxerr,'source_restored_exact':True,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_no_update_rows_retained':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start,'peak_GPU_bytes':cert['peak_GPU_bytes'],'independent_capture_audit_sha256':sha(H/'independent_capture_audit_attempt001/AUDIT_RESULT.json')};dump(out/'AUDIT_RESULT.json',result);print(result)
if __name__=='__main__':main()
