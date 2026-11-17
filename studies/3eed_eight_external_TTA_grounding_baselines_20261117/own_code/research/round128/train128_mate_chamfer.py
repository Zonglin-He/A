"""Source-only correction to the original MATE unordered Chamfer objective.

The initial L2 source-preparation attempt is preserved; it is not a target-data
result and is not the main MATE port. No target inference/scoring preceded this.
"""
from core128 import *
from models128 import MaskedRegion,tensor
from train128_prepared import loss_binary
def main():
 strict97();reg=jread(H/'REGION_REGISTRY_PRIVATE.json');hist=[]
 for src in ['waymo','drone','quad']:
  root=B/'prepared_PRIVATE'/src;sel=jread(root/'SELECTED.json');jdump(root/'INITIAL_L2_SOURCE_SELECTION.json',sel);torch.manual_seed(128);m=MaskedRegion().cpu();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001);rr={sp:[load(R/e['path']) for e in reg if e['role']=='source' and e['source']==src and e['split']==sp] for sp in ['TRAIN512','VAL128']};best=-1;t=time.perf_counter();choice=None
  for epoch in range(1,7):
   m.train();losses=[]
   for ix in np.random.default_rng(128000+epoch).permutation(len(rr['TRAIN512'])):
    a=tensor(rr['TRAIN512'][int(ix)],'cpu');joint=F.normalize(a['joint'].float(),dim=-1);opt.zero_grad(set_to_none=True);log,_=m(a['crops'],joint);_,aux=m(a['crops'],joint,True);L=loss_binary(log,a['source_IoU'])+aux;assert torch.isfinite(L);L.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(L.detach()))
   m.eval();ys=[]
   with torch.no_grad():
    for row in rr['VAL128']:
     a=tensor(row,'cpu');log,_=m(a['crops'],F.normalize(a['joint'].float(),dim=-1));ys.append(float(a['source_IoU'][(log[:,1]-log[:,0]).argmax()]))
   val=float(np.mean(ys)*100);rec={'source':src,'epoch':epoch,'loss_mean':float(np.mean(losses)),'source_val_mIoU_percent':val,'auxiliary':'bidirectional squared Chamfer masked patch offsets'};hist.append(rec);print('R128_MATE_CHAMFER',rec,flush=True)
   if val>best+1e-12:
    best=val;p=root/'mate_chamfer.pt';torch.save({n:v.detach().cpu() for n,v in m.state_dict().items()},p);choice={'epoch':epoch,'source_val_mIoU_percent':val,'sha256':sha(p),'path':str(p.relative_to(R)),'parameters':sum(p.numel() for p in m.parameters()),'source_reconstruction':'unordered Chamfer-L2,87.5% mask','CPU_source_train_steps':epoch*len(rr['TRAIN512'])}
  sel['models']['mate']=choice;sel['MATE_Chamfer_correction']={'initial_L2_attempt_preserved':True,'new_training_steps':6*len(rr['TRAIN512']),'device':'CPU','seconds':time.perf_counter()-t,'no_target_labels_or_inference_read':True};jdump(root/'SELECTED.json',sel);jdump(H/'MATE_CHAMFER_SOURCE_HISTORY.json',hist)
if __name__=='__main__':main()
