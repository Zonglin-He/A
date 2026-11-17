"""Source-only binary role calibration; target predictions/labels not loaded.

3DVG pair labels are highly imbalanced unlike the original fixed-class tasks.
A single global source-VAL threshold per head maximizes balanced accuracy.
Only output biases change; within-input standalone candidate ranking is invariant.
"""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
R=Path(__file__).resolve().parents[2];H=R/'research/round128';B=R/'crossplatform/round128';sys.path.insert(0,str(H));from models128 import PairHead,RGBHead
rd=lambda p:json.loads(Path(p).read_text());torch.set_num_threads(4)
def main():
 reg=rd(H/'REGION_REGISTRY_PRIVATE.json');report=[]
 for src in ['waymo','drone','quad']:
  root=B/'prepared_PRIVATE'/src;sel=rd(root/'SELECTED.json');assert 'MATE_Chamfer_correction' in sel
  (root/'UNCALIBRATED_SELECTION.json').write_text(json.dumps(sel,indent=2)+'\n')
  for k,cls in [('pair',PairHead),('rgb',RGBHead)]:
   m=cls().eval();m.load_state_dict(torch.load(R/sel['models'][k]['path'],weights_only=True,map_location='cpu'));logits=[];labels=[]
   with torch.no_grad():
    for e in reg:
     if not(e['role']=='source' and e['source']==src and e['split']=='VAL128'):continue
     with np.load(R/e['path']) as a:
      y=a['source_IoU'];valid=(y>=.25)|(y<.05)
      if k=='rgb':valid&=a['visible']
      f=torch.nn.functional.normalize(torch.from_numpy(a['joint']),dim=1);l=m(f) if k=='pair' else m(torch.from_numpy(a['images']).permute(0,3,1,2).float()/255,torch.from_numpy(a['text']));logits.extend((l[:,1]-l[:,0]).numpy()[valid]);labels.extend((y[valid]>=.25).tolist())
   x=np.array(logits);y=np.array(labels);order=np.argsort(x,kind='stable');xx=x[order];yy=y[order];P=y.sum();N=(~y).sum();assert P>0 and N>0;tp=P-np.r_[0,np.cumsum(yy)];tn=np.r_[0,np.cumsum(~yy)];score=.5*(tp/P+tn/N);best=np.flatnonzero(score==score.max());j=int(best[len(best)//2]);th=(xx[j-1]+xx[j])/2 if 0<j<len(xx) else (xx[0]-1e-5 if j==0 else xx[-1]+1e-5);last=m.net[-1] if k=='pair' else m.match[-1]
   with torch.no_grad():last.bias[1]-=float(th/2);last.bias[0]+=float(th/2)
   p=root/(k+'_calibrated.pt');torch.save(m.state_dict(),p);sel['models'][k]=dict(sel['models'][k],path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_only_logodds_threshold=float(th));report.append({'source':src,'head':k,'source_positive_pairs':int(P),'source_negative_pairs':int(N),'threshold':float(th),'balanced_accuracy_before':float(.5*((x[y]>0).mean()+(x[~y]<=0).mean())),'balanced_accuracy_after':float(score.max()),'standalone_within_input_ranking_unchanged':True})
  sel['binary_role_calibration']='one analytic source-VAL balanced-accuracy threshold per pair/RGB head; no target access';(root/'SELECTED.json').write_text(json.dumps(sel,indent=2)+'\n')
 (H/'SOURCE_ROLE_CALIBRATION.json').write_text(json.dumps({'checks':report,'target_data_or_labels_loaded':False},indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
