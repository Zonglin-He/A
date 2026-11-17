"""Finite baseline-specific source preparation, disjoint source train/val only."""
from core128 import *
from models128 import PairHead,RGBHead,tensor,image_tensor
# Preserve the exact initial source-preparation attempt; main MATE is separately
# retrained with Chamfer by train128_mate_chamfer.py before target inference.
from models128_L2_source_attempt001 import MaskedRegion
import copy
def loss_binary(logits,y):
 pos=y>=.25;neg=y<.05;terms=[]
 if pos.any():terms.append(F.cross_entropy(logits[pos],torch.ones(int(pos.sum()),device=logits.device,dtype=torch.long)))
 if neg.any():terms.append(F.cross_entropy(logits[neg],torch.zeros(int(neg.sum()),device=logits.device,dtype=torch.long)))
 return torch.stack(terms).mean()
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=ap.parse_args();dev=args.device
 strict97();reg=jread(H/'REGION_REGISTRY_PRIVATE.json');history=[]
 for src in ['waymo','drone','quad']:
  root=B/'prepared_PRIVATE'/src;root.mkdir(parents=True,exist_ok=True)
  if (root/'SELECTED.json').exists():continue
  rows={split:[load(R/e['path']) for e in reg if e['role']=='source' and e['source']==src and e['split']==split] for split in ['TRAIN512','VAL128']};torch.manual_seed(128);models={'pair':PairHead().to(dev),'rgb':RGBHead().to(dev),'mate':MaskedRegion().to(dev)};opts={k:torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001) for k,m in models.items()};best={k:-1 for k in models};selected={};steps=0;t0=time.perf_counter()
  for epoch in range(1,7):
   means={k:[] for k in models}
   for m in models.values():m.train()
   for ix in np.random.default_rng(128000+epoch).permutation(len(rows['TRAIN512'])):
    a=tensor(rows['TRAIN512'][int(ix)],dev);y=a['source_IoU'];joint=F.normalize(a['joint'].float(),dim=-1)
    for k,m in models.items():
     opts[k].zero_grad(set_to_none=True)
     if k=='pair':log=m(joint);L=loss_binary(log,y)
     elif k=='rgb':
      log=m(image_tensor(a),a['text']);vis=a['visible'];L=loss_binary(log[vis],y[vis]) if vis.any() else log.sum()*0
     else:log,_=m(a['crops'],joint);_,aux=m(a['crops'],joint,True);L=loss_binary(log,y)+aux
     assert torch.isfinite(L);L.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(gn);opts[k].step();means[k].append(float(L.detach()));steps+=1
   for k,m in models.items():
    m.eval();vals=[];source_zero=[]
    with torch.no_grad():
     for item in rows['VAL128']:
      a=tensor(item,dev);joint=F.normalize(a['joint'].float(),dim=-1)
      if k=='pair':log=m(joint)
      elif k=='rgb':log=m(image_tensor(a),a['text'])
      else:log,_=m(a['crops'],joint)
      ss=log[:,1]-log[:,0]
      if k=='rgb':ss=torch.where(a['visible'],ss,a['native_scores'])
      vals.append(float(a['source_IoU'][ss.argmax()]));source_zero.append(float(a['source_IoU'][a['native_scores'].argmax()]))
    val=float(np.mean(vals)*100);row={'source':src,'method':k,'epoch':epoch,'train_inputs':len(rows['TRAIN512']),'source_val_inputs':len(vals),'mean_training_loss':float(np.mean(means[k])),'source_val_mIoU_percent':val,'source_zero_mIoU_percent':float(np.mean(source_zero)*100)};history.append(row);print('R128_SOURCE',row,flush=True)
    if val>best[k]+1e-12:
     best[k]=val;p=root/(k+'.pt');torch.save({n:v.detach().cpu() for n,v in m.state_dict().items()},p);selected[k]={'epoch':epoch,'source_val_mIoU_percent':val,'sha256':sha(p),'path':str(p.relative_to(R)),'parameters':sum(p.numel() for p in m.parameters())}
   jdump(H/'SOURCE_HISTORY.json',history)
  jdump(root/'SELECTED.json',{'source':src,'source_train_n':len(rows['TRAIN512']),'source_val_n':len(rows['VAL128']),'new_optimizer_steps':steps,'selection':'earliest best source validation mIoU per branch; no target examples or labels loaded','models':selected,'seconds':time.perf_counter()-t0,'device':dev,'original_source_checkpoint_edited':False,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30 if dev=='cuda' else 0.})
if __name__=='__main__':main()
