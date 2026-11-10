"""Independent full-source pre-fusion provenance and every region contrast, no C/GT."""
import sys,time,torch,re
from common66 import *
import independent66 as f
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,source_capture,fingerprint,firstnoun_mask,SENTINEL
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points

def main():
 H=ROOT/'research/round66';p=read(H/'experiment/EVIDENCE_AUDIT_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);cap=ROOT/p['capture_root'];ev=ROOT/p['evidence_root'];cs=read(cap/'SEALED.json');es=read(ev/'SEALED.json');model=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(model);start=time.perf_counter();calls=replays=0;maxerr=0.;zero=0
 for i,r in enumerate(rows):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel] and sha(ev/rel)==es['files'][rel];allz=load(cap/rel);e=load(ev/rel);z={v:{k[2:]:a for k,a in allz.items() if k.startswith(v+'_')} for v in 'AB'};oldcase=read(ROOT/p['AB_root']/'cases_PRIVATE'/(sid+'.json'))
  for v,seed in [('A',r['seed']),('B',oldcase['B_seed'])]:
   observed={};hooks=[]
   def encoder(_m,args,k):
    for key,name in [('vis_feats','vis0'),('pos_feats','pos0'),('text_feats','lang0'),('text_padding_mask','textpad')]:observed[name]=k[key].cpu().numpy().copy()
   def bb(_m,args,out):observed['xyz']=out['fp2_xyz'].cpu().numpy().copy();observed['backbone_features']=out['fp2_features'].cpu().numpy().copy()
   hooks.extend([model.cross_encoder.register_forward_pre_hook(encoder,with_kwargs=True),model.backbone_net.register_forward_hook(bb)])
   for j,dec in enumerate(model.decoder):
    def getpos(_m,args,k,j=j):observed['position'+str(j)]=args[3].cpu().numpy().copy()
    hooks.append(dec.register_forward_pre_hook(getpos,with_kwargs=True))
   pts,ids,_=sample_points(dict(r,seed=seed))
   try:actual,_=source_capture(model,pts,ids,r['caption']);calls+=1
   finally:
    for hook in hooks:hook.remove()
   assert np.array_equal(observed['backbone_features'].transpose(0,2,1),z[v]['vis0'])
   for k,val in observed.items():
    if k!='backbone_features':assert np.array_equal(val,z[v][k]),('prefusion_or_positions',i,v,k)
   for k in ['h','query','tokens','token_mask','boxes','scores','point_indices','query_points_sample_inds','query_anchor_xyz']:assert np.array_equal(actual[k],z[v][k]),('native_capture',i,v,k)
   zeroout=f.replay(model,z[v]);replays+=1;zero+=1
   for k in ['h','query','tokens']:assert np.array_equal(zeroout[k],z[v][k]),('native_zero',i,v,k)
  # Class cache is text-only; verify native text forward rather than external labels.
  sys.path.insert(0,str(ROOT/'crossplatform/round7'));from regression_adapter import LEXICON
  match=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',r['caption'],re.I);assert bool(match)==bool(allz['category_available'])
  if match:
   text=match.group(0)+SENTINEL;tok=model.tokenizer.batch_encode_plus([text],padding='longest',return_tensors='pt').to(next(model.parameters()).device);lang=model.text_projector(model.text_encoder(**tok).last_hidden_state).cpu().numpy();mask,_=firstnoun_mask(model,text)
   for v in 'AB':assert np.array_equal(lang,z[v]['class_lang0']) and np.array_equal(tok.attention_mask.ne(1).cpu().numpy(),z[v]['class_textpad']) and np.array_equal(mask.numpy(),z[v]['class_token_mask'])
  else:
   for v in 'AB':assert np.array_equal(z[v]['class_lang0'],z[v]['lang0']) and np.array_equal(z[v]['class_token_mask'],z[v]['token_mask'])
  pt=Path(r['pointpath']);raw=np.load(pt,allow_pickle=False) if pt.suffix=='.npy' else np.fromfile(pt,np.float32).reshape(-1,4);ix=np.unique(np.r_[z['A']['point_indices'],z['B']['point_indices']]);x=np.array(raw[ix,:3],np.float32,copy=True)
  if r['platform']=='drone':x[:,2]+=1.8
  a,_,_=f.af(z['A']['boxes'].astype(float),z['B']['boxes'].astype(float),x.astype(float));W=a.astype(float)/(1+a.astype(float).sum(0,keepdims=True));base=f.probability(z['A']['query'],z['A']['tokens'],z['A']['token_mask']);bprob=f.probability(z['B']['query'],z['B']['tokens'],z['B']['token_mask']);mass=float((W@bprob).sum());pAB=(base+f.mapped(bprob,W,mass,base))/2;assert np.array_equal(W,e['B_W']) and mass==e['B_fixed_mass'] and np.array_equal(base,e['base']);groups,pool=f.groups_and_pool(z['A']['boxes'].astype(float),x.astype(float),pAB);members=np.array([np.isin(np.arange(256),groups[j]) for j in pool]);assert np.array_equal(e['members'],members) and np.array_equal(e['group_ids'],pool);K=len(pool);baseline=np.zeros_like(e['baseline']);pert=np.zeros_like(e['perturbed']);valid=np.zeros_like(e['valid'])
  for vi,v in enumerate('AB'):
   mm=f.region_controls(z[v]['xyz'][0].astype(float),z['A']['boxes'].astype(float),groups,pool)
   for ti,category in enumerate([False,True]):
    zz=f.replay(model,z[v],category=category);replays+=1;baseline[vi,ti]=zz['p'] if v=='A' else f.mapped(zz['p'],W,mass,base)
   for j,(region,ctrl,ok) in enumerate(mm):
    assert np.array_equal(region,e[v+'_region_masks'][j]) and np.array_equal(ctrl,e[v+'_control_masks'][j]);valid[vi,j]=ok and bool(match)
    for ti,category in enumerate([False,True]):
     for ki,mask in enumerate([region,ctrl]):
      if valid[vi,j]:zz=f.replay(model,z[v],mask,category);replays+=1;pert[vi,ti,ki,j]=zz['p'] if v=='A' else f.mapped(zz['p'],W,mass,base)
      else:pert[vi,ti,ki,j]=baseline[vi,ti]
  assert np.array_equal(valid,e['valid']);maxerr=max(maxerr,float(abs(baseline-e['baseline']).max()),float(abs(pert-e['perturbed']).max()));assert np.allclose(baseline,e['baseline'],rtol=0,atol=1e-11) and np.allclose(pert,e['perturbed'],rtol=0,atol=1e-11),('contrasts',i,maxerr)
  drop=np.zeros_like(e['drops'])
  for vi in range(2):
   for ti in range(2):
    for ki in range(2):
     for j in range(K):drop[vi,ti,ki,j]=np.log(max(float(baseline[vi,ti,members[j]].sum()),1e-300))-np.log(max(float(pert[vi,ti,ki,j,members[j]].sum()),1e-300))
  full=drop[:,0,0]-drop[:,0,1];cat=drop[:,1,0]-drop[:,1,1];E=full-cat;positive=valid.all(0)&(E>p['sign_epsilon']).all(0);negative=valid.all(0)&(E< -p['sign_epsilon']).all(0);stable=positive|negative
  for name,val in [('drops',drop),('E_views',E),('E',E.mean(0)),('full_only',full.mean(0)),('category_only',cat.mean(0)),('control_only',(drop[:,0,1]-drop[:,1,1]).mean(0))]:assert np.allclose(val,e[name],rtol=0,atol=1e-10),('evidence',i,name)
  for name,val in [('positive',positive),('negative',negative),('stable',stable),('tested_query_mask',members[stable].any(0) if stable.any() else np.zeros(256,bool)),('positive_query_mask',members[positive].any(0) if positive.any() else np.zeros(256,bool))]:assert np.array_equal(val,e[name]),('set',i,name)
  assert bool(e['update_allowed'])==bool(positive.any() and negative.any())
  if (i+1)%16==0:print('AUDIT_EVIDENCE66',i+1,len(rows),calls,replays,flush=True)
 assert fingerprint(model)==fp;out=H/'independent_evidence_audit_attempt001';assert not out.exists();out.mkdir();rr={'state':'PASS_R66_INDEPENDENT_PREFUSION_PROVENANCE_FIXED_ANCHORS_MASKS_AND_ALL_EVIDENCE','n':336,'source_full_forward_replays':calls,'zero_intervention_bridges':zero,'independent_encoder_decoder_replays':replays,'max_probability_error':maxerr,'source_unchanged':True,'C_features_used':False,'task_GT_used':False,'capture_seal_sha256':sha(cap/'SEALED.json'),'evidence_seal_sha256':sha(ev/'SEALED.json'),'program_sha256':sha(__file__),'helper_sha256':sha(H/'independent66.py'),'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'AUDIT_RESULT.json',rr);print(rr)
if __name__=='__main__':main()
