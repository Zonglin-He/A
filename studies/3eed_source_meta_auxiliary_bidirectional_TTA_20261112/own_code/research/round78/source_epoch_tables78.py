"""Expose every saved source-validation endpoint, including unused epochs and harms."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'research/round76'));from common76 import read,dump,load
H=R/'research/round78';TR=R/'crossplatform/round78/trained_source_attempt001';rows=[];neg=[]
for b in ['joint','meta']:
 for epoch in range(1,5):
  z=load(TR/'val_PRIVATE'/(b+'_e'+str(epoch)+'.npz'));assert z['IoU'].shape==(128,2)
  for i,v in enumerate(z['IoU']):
   delta=float(v[1]-v[0]);row=f'SOURCE128_{i+1:05d}'
   for j in [0,1]:rows.append({'row':row,'branch':b,'epoch':epoch,'step':j,'IoU':float(v[j]),'one_vs_zero_delta_pp':delta*100})
   if delta<-1e-10:neg.append({'row':row,'branch':b,'epoch':epoch,'one_vs_zero_delta_pp':delta*100})
assert len(rows)==2048;dump(H/'SOURCE_ALL_EPOCH_TASK_ROWS.json',rows);dump(H/'SOURCE_ALL_EPOCH_ADAPTATION_NEGATIVES.json',neg);print({'source_epoch_task_values':len(rows),'source_epoch_adaptation_negatives':len(neg)})
