"""Strictly point-independent text for the MM-TTA RGB branch, source-only fit.

Native projected text *after* cross-encoding can contain LiDAR information.
Use cached RoBERTa/text-projector output BEFORE the cross encoder, then the
frozen source text projection. Main MM-TTA therefore has separate modalities.
"""
from pathlib import Path
import sys,json,hashlib,time,numpy as np,torch,gc
from torch.nn import functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round128';B=R/'crossplatform/round128';sys.path.insert(0,str(H));from models128 import PairHead,RGBHead
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as x:return {k:torch.as_tensor(x[k].copy()) for k in x.files}
def text(prefix,W,b):
 with np.load(prefix,allow_pickle=False) as z:L=torch.from_numpy(z['lang0'][0]);mask=torch.from_numpy(z['token_mask']).bool();u=L
 for j in [0,2,4]:
  u=F.linear(u,W[j],b[j])
  if j<4:u=F.relu(u)
 u=F.normalize(u,dim=-1);return F.normalize(u[mask].mean(0),dim=0)
def binary(log,y):
 ls=[]
 if (y>=.25).any():ls.append(F.cross_entropy(log[y>=.25],torch.ones(int((y>=.25).sum()),dtype=torch.long)))
 if (y<.05).any():ls.append(F.cross_entropy(log[y<.05],torch.zeros(int((y<.05).sum()),dtype=torch.long)))
 return torch.stack(ls).mean()
def main():
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);reg=rd(H/'REGION_REGISTRY_PRIVATE.json');sources=rd(R/'research/round97/experiment/DESIGN_LOCK.json')['sources'];hist=[]
 for src in ['waymo','drone','quad']:
  entry=next(e for e in sources if e['source']==src);p=Path(entry['checkpoint_path']);assert sha(p)==entry['checkpoint_sha256'];ck=torch.load(p,map_location='cpu',weights_only=True);W={j:ck['model'][f'contrastive_align_projection_text.{j}.weight'].clone() for j in [0,2,4]};b={j:ck['model'][f'contrastive_align_projection_text.{j}.bias'].clone() for j in [0,2,4]};del ck;gc.collect();data={}
  for split in ['TRAIN512','VAL128']:
   rows=[]
   for e in reg:
    if not(e['role']=='source' and e['source']==src and e['split']==split):continue
    sid=Path(e['path']).stem;pp=R/(f'crossplatform/round76/source_prefix640_attempt001/samples_PRIVATE/{sid}.npz' if src=='waymo' else f'crossplatform/round97/{src}/samples_PRIVATE/{sid}_prefix.npz');a=load(R/e['path']);a['text']=text(pp,W,b);rows.append(a)
   data[split]=rows
  torch.manual_seed(128);PairHead();m=RGBHead();op=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001);best=-1;root=B/'prepared_PRIVATE'/src;choice=None;tick=time.perf_counter()
  for epoch in range(1,7):
   m.train();losses=[]
   for ix in np.random.default_rng(128000+epoch).permutation(len(data['TRAIN512'])):
    a=data['TRAIN512'][int(ix)];v=a['visible'];log=m(a['images'].permute(0,3,1,2).float()/255,a['text']);L=binary(log[v],a['source_IoU'][v]) if v.any() else log.sum()*0;op.zero_grad(set_to_none=True);L.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);op.step();losses.append(float(L.detach()))
   m.eval();vals=[]
   with torch.no_grad():
    for a in data['VAL128']:
     log=m(a['images'].permute(0,3,1,2).float()/255,a['text']);s=torch.where(a['visible'],log[:,1]-log[:,0],a['native_scores']);vals.append(float(a['source_IoU'][s.argmax()]))
   val=float(np.mean(vals)*100);hist.append({'source':src,'epoch':epoch,'loss':float(np.mean(losses)),'source_val_mIoU_percent':val});print('R128_INDEPENDENT_RGB_SOURCE',src,epoch,val,flush=True)
   if val>best+1e-12:
    best=val;pp=root/'rgb_independent_uncalibrated.pt';torch.save(m.state_dict(),pp);choice={'epoch':epoch,'source_val_mIoU_percent':val}
  m.load_state_dict(torch.load(root/'rgb_independent_uncalibrated.pt',map_location='cpu',weights_only=True));m.eval();x=[];y=[]
  with torch.no_grad():
   for a in data['VAL128']:
    log=m(a['images'].permute(0,3,1,2).float()/255,a['text']);v=a['visible']&((a['source_IoU']>=.25)|(a['source_IoU']<.05));x.extend((log[:,1]-log[:,0])[v].numpy());y.extend((a['source_IoU'][v]>=.25).numpy())
  x=np.array(x);y=np.array(y);order=np.argsort(x,kind='stable');xx=x[order];yy=y[order];sc=.5*((y.sum()-np.r_[0,np.cumsum(yy)])/y.sum()+np.r_[0,np.cumsum(~yy)]/(~y).sum());js=np.flatnonzero(sc==sc.max());j=int(js[len(js)//2]);th=(xx[j-1]+xx[j])/2 if 0<j<len(xx) else (xx[0]-1e-5 if j==0 else xx[-1]+1e-5)
  with torch.no_grad():m.match[-1].bias[1]-=float(th/2);m.match[-1].bias[0]+=float(th/2)
  pp=root/'rgb_independent.pt';torch.save(m.state_dict(),pp);dump(root/'MMTTA_INDEPENDENT_RGB_SELECTED.json',dict(choice,path=str(pp.relative_to(R)),sha256=sha(pp),source_val_role_threshold=float(th),source_train_n=192,source_val_n=64,epochs=6,seconds=time.perf_counter()-tick,text_point_independent=True,target_GT_used=False))
  for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json'):
   if e['cohort']!='DEV' or e['source']!=src or e['protected_alias_fallback']:continue
   tp=B/f"independent_text_PRIVATE/{src}_to_{e['target']}/{e['row']['id']}.npy";tp.parent.mkdir(parents=True,exist_ok=True);np.save(tp,text(R/e['prefix'],W,b).numpy(),allow_pickle=False)
  del data,m;gc.collect()
 dump(H/'MMTTA_INDEPENDENT_RGB_SOURCE_HISTORY.json',hist);dump(H/'MMTTA_INDEPENDENT_TEXT_AUDIT.json',{'source_fits':3,'RGB_text_provider':'cached lang0 before any LiDAR cross-encoding plus frozen source text projection','source_only_selection':True,'target_GT_read':False,'protected_alias_text_read':False})
if __name__=='__main__':main()
