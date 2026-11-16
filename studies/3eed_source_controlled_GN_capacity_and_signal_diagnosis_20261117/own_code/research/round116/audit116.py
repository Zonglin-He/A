"""CPU-only native readout, GN update and measured-depth recovery audit before taskGT."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round116';B=R/'crossplatform/round116';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112
sys.path.insert(0,str(R/'research/round91'));from core91 import Fusion91
sys.path.insert(0,str(R/'research/round76'));from model76 import tensor_data,tensor_prefix,masks

def cpudepth(m,z,d,im,ids,target):
 vv=[]
 for mask in im:
  mm=mask.reshape(1,1,32,48).repeat_interleave(8,2).repeat_interleave(8,3);x=d['image'].masked_fill(mm,.5);I=m.image_adapter(m.rgb((x-.5)*2).flatten(2).transpose(1,2)[0]);p=m.pred_depth(I).squeeze(-1);vv.append(F.smooth_l1_loss(p[ids],target[ids]) if len(ids) else p.sum()*0)
 return float(torch.stack(vv).mean())
def split(r,d):
 ids=torch.nonzero(d['depth_valid'],as_tuple=True)[0];rng=np.random.default_rng(int(hashlib.sha256(('R116_physical_holdout|'+r['id']).encode()).hexdigest()[:16],16));order=rng.permutation(len(ids));n=len(ids)//4;hold=ids[order[:n]];train=ids[order[n:]];p=rng.permutation(len(train))
 while len(train)>1 and np.any(p==np.arange(len(train))):p=rng.permutation(len(train))
 wrong=d['depth_target'].clone();wrong[train]=wrong[train[p]];return train,hold,wrong

def main():
 torch.set_num_threads(4);mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==768 and len(mat['new_arms'])==21;assert rd(H/'ENGINEERING.json')['n']==12;n=heads=aliases=ties=states=0;maxp=maxdepth=maxhold=maxdelta=maxwrite=0.;reg=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules']
 for src in ['waymo','drone','quad']:
  stpath=R/reg[src]['state_path'];assert sha(stpath)==reg[src]['state_sha256']
  with np.load(stpath) as zz:S={k:zz[k] for k in zz.files}
  head={k[6:]:v for k,v in S.items() if k.startswith('match.')};model=Fusion91(nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64))).cpu().eval().requires_grad_(False);model.load_state_dict({k:torch.from_numpy(np.asarray(v).copy()) for k,v in S.items()},strict=True);names=[('rgb.'+path+'.'+name,p) for path,mod in model.rgb.named_modules() if isinstance(mod,nn.GroupNorm) for name,p in mod.named_parameters(recurse=False)];assert sum(p.numel() for _,p in names)==224;flat=torch.cat([torch.from_numpy(S[k]).flatten() for k,_ in names])
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=B/'formal'/name;assert sha(D/'SEALED.json')==mat['directions'][name]
   for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
   for ix,r in enumerate(rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')):
    sid=r['id'];meta=rd(D/'records_PRIVATE'/(sid+'.json'));pred=np.load(D/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(pred['arm_names'],pred['scores']));old=np.load(R/'crossplatform/round114/formal'/name/'samples_PRIVATE'/(sid+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(pred['source_boxes'],old['source_boxes']) and np.array_equal(pred['queries'],pred['scores'].argmax(1));n+=1
    for arm in po:
     if arm not in mat['new_arms']:assert np.array_equal(po[arm],oldp[arm])
    if meta['protected_alias_fallback']:aliases+=1;assert all(np.array_equal(po[k],oldp['prepared_frozen']) for k in mat['new_arms']);continue
    with np.load(R/'crossplatform/round113'/name/'features_PRIVATE'/(sid+'.npz')) as aa,np.load(D/'features_PRIVATE'/(sid+'.npz')) as ff:
     cache={v:{k:aa['frozen_anchor_'+v+'_'+k][0] if k in ['h','u'] else aa['frozen_anchor_'+v+'_'+k] for k in ['h','u','mask','r']} for v in ['A','C','fusedABC']}
     for arm in mat['new_arms']:
      a=cache[arm.split('_')[-1]];pc=cpu112(a['h'],a['u'],a['mask'],ff[arm+'_late'],a['r'],head);err=float(abs(pc-po[arm]).max());maxp=max(maxp,err);assert err<5e-6
      if int(pc.argmax())!=int(po[arm].argmax()):assert po[arm].max()-po[arm][int(pc.argmax())]<=2*err+1e-12;ties+=1
      heads+=1
    with np.load(R/'crossplatform/round111'/name/'prefix_PRIVATE'/(sid+'.npz')) as zf:z0={k:zf[k] for k in ['xyz','point_indices']}
    key=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest();gf=R/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+key[:20]+'.npz')
    with np.load(gf) as zz:geom={k:zz[k] for k in zz.files}
    d=tensor_data(geom,r,device='cpu');train,hold,wrong=split(r,d);ims=[masks(r,d,t)[1] for t in [76001,76002]];fits=meta['fits'];actual={x['setting']+'_'+x['state']:x for x in meta['evaluation']['writeback']}
    with np.load(D/'states_PRIVATE'/(sid+'.npz')) as zz:state_arrays={k:zz[k] for k in zz.files}
    g=torch.from_numpy(state_arrays['gradient']);gw=torch.from_numpy(state_arrays['wrong_gradient']);assert g.numel()==224 and torch.isfinite(g).all() and gw.numel()==224;radius=.01*float(flat.double().norm());gnorm=float(g.norm());gnatural=-.5*torch.clamp(1/(g.norm()+1e-6),max=1)*g;gtrust=-radius*g/g.norm() if gnorm>0 else g*0;gwrong=-radius*gw/gw.norm() if float(gw.norm()) else gw*0
    for key in ['depth_trust_raw','depth_natural_raw','wrong_depth_trust_raw']:
     expected={'depth_trust_raw':gtrust,'depth_natural_raw':gnatural,'wrong_depth_trust_raw':gwrong}[key];er=float(abs(torch.from_numpy(state_arrays[key])-expected).max());maxdelta=max(maxdelta,er);assert er<1e-6
    for key,act in actual.items():
     delta=torch.from_numpy(state_arrays[key]);setting=act['setting'];tag=act['state'];off=0
     with torch.no_grad():
      for k,p in names:nn0=p.numel();p.copy_(torch.from_numpy(S[k])+delta[off:off+nn0].reshape_as(p));off+=nn0
      write=float(torch.sqrt(sum((p-torch.from_numpy(S[k])).double().square().sum() for k,p in names)));maxwrite=max(maxwrite,abs(write-act['actual_writeback_L2']));assert abs(write-act['actual_writeback_L2'])<1e-10
      target=wrong if setting=='wrong_depth_trust' else d['depth_target'];fitloss=cpudepth(model,None,d,ims,train,target);rec=fits[setting];tr=rec['trials'][0] if tag=='raw' else (rec['trials'][-1] if rec['accepted'] else None)
      if tr:err=abs(fitloss-tr['depth_fit']);maxdepth=max(maxdepth,err);assert err<5e-6,(key,fitloss,tr['depth_fit'])
      hl=cpudepth(model,None,d,ims,hold,d['depth_target']);err=abs(hl-meta['evaluation']['physical_holdout_loss'][key]);maxhold=max(maxhold,err);assert err<5e-6
     states+=1
    if (ix+1)%32==0:print('AUDIT116',name,ix+1,flush=True)
 assert n==768 and aliases==44 and heads==15204 and states==5068;result={'state':'PASS_ALL21_NEW_CPU_READOUTS_GN224_PROPOSALS_ACTUAL_WRITEBACK_FIT_DEPTH_AND_HOLDOUT','n':n,'new_CPU_readouts':heads,'actual_states':states,'aliases':aliases,'CPU_probability_max_error':maxp,'CPU_depth_fit_max_error':maxdepth,'CPU_depth_holdout_max_error':maxhold,'CPU_proposal_max_error':maxdelta,'actual_writeback_norm_error':maxwrite,'CPU_near_ties':ties,'GT_used':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':main()
