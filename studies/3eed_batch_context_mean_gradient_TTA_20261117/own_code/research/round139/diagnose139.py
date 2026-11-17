"""Postseal GT-surrogate diagnostic of leave-frame-out task-gradient transfer.

No gradients computed anew, no model updates, no GT vector used as a method.
"""
from pathlib import Path
import json,hashlib,datetime,numpy as np,collections
R=Path(__file__).resolve().parents[2];H=R/'research/round139'
def rd(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d>1e-20 else None
def main():
 plan=rd(H/'experiment/DESIGN_LOCK.json');reg=rd(R/'research/round136/INPUT_REGISTRY_PRIVATE.json');prior={(r['source'],r['target'],r['id_PRIVATE']):r for r in rd(R/'research/round136/GRADIENT_ROWS_PRIVATE.json')};items=[]
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;ev=np.load(R/'crossplatform/round136/evidence_PRIVATE'/name/(sid+'.npz'))
  if not bool(ev['eligible']):continue
  g=np.load(R/'crossplatform/round136/method'/name/'states_PRIVATE'/(sid+'.npz'))['C_B_A_gradient1'].astype(float);gt=np.load(R/'crossplatform/round136/oracle'/name/'states_PRIVATE'/(sid+'.npz'))['gradient1'].astype(float);items.append({'source':src,'target':tgt,'id_PRIVATE':sid,'frame_PRIVATE':e['frame_PRIVATE'],'content_group_PRIVATE':e['content_group_PRIVATE'],'recoverable':prior[(src,tgt,sid)]['old_A_recoverable_error'],'g':g,'gt':gt})
 allrows=[];partitions=[];summaries=[]
 for order in [0,1,2]:
  for src,tgt in sorted({(r['source'],r['target']) for r in items}):
   rr=[r for r in items if (r['source'],r['target'])==(src,tgt)];blocks=collections.defaultdict(list)
   for r in rr:blocks[r['content_group_PRIVATE']].append(r)
   keys=sorted(blocks,key=lambda k:hashlib.sha256(('R139batch8|'+str(order)+'|'+src+'|'+tgt+'|'+k).encode()).hexdigest());batches=[];cur=[]
   for k in keys:
    block=sorted(blocks[k],key=lambda r:r['id_PRIVATE'])
    if cur and len(cur)+len(block)>8:batches.append(cur);cur=[]
    cur+=block
   if cur:batches.append(cur)
   for j,batch in enumerate(batches):
    bid=src+'_to_'+tgt+'_'+str(order)+'_'+str(j);partitions.append({'batch_PRIVATE':bid,'order':order,'source':src,'target':tgt,'members_PRIVATE':[r['id_PRIVATE'] for r in batch],'n':len(batch)})
    for r in batch:
     peers=[x for x in batch if x['frame_PRIVATE']!=r['frame_PRIVATE'] and x['content_group_PRIVATE']!=r['content_group_PRIVATE']];own=cos(r['gt'],r['g']);cross=cos(r['gt'],np.mean([x['g'] for x in peers],axis=0)) if peers else None;allrows.append({k:r[k] for k in ['source','target','id_PRIVATE','frame_PRIVATE','content_group_PRIVATE','recoverable']}|{'order':order,'batch_PRIVATE':bid,'peers':len(peers),'own_cosine':own,'cross_cosine':cross})
  for subset,pred in [('all',lambda r:True),('recoverable',lambda r:r['recoverable'])]:
   rr=[r for r in allrows if r['order']==order and pred(r) and r['own_cosine'] is not None and r['cross_cosine'] is not None];summaries.append({'order':order,'subset':subset,'n':len(rr),'own_mean':float(np.mean([r['own_cosine'] for r in rr])),'cross_mean':float(np.mean([r['cross_cosine'] for r in rr])),'cross_median':float(np.median([r['cross_cosine'] for r in rr])),'cross_positive_fraction':float(np.mean([r['cross_cosine']>0 for r in rr]))})
 rr=[r for r in allrows if r['recoverable'] and r['own_cosine'] is not None and r['cross_cosine'] is not None];a=float(np.mean([r['own_cosine'] for r in rr]));b=float(np.mean([r['cross_cosine'] for r in rr]));pos=float(np.mean([r['cross_cosine']>0 for r in rr]));passed=bool(b>0 and b>a and pos>.5);decision={'prerequisite_passed':passed,'own_mean':a,'leave_frame_out_peer_mean':b,'peer_positive_fraction':pos,'decision':'lock_batch8_same_loss_3order_correct_wrong_TTA' if passed else 'stop_this_batch8_shared_update_hypothesis_without_GPU','GT_only_diagnostic':True,'GT_deltas_not_deployed':True,'new_model_or_scoring':False,'FINAL_access':False,'target_development_labels_used_for_research_gate_not_runtime':True};dump(H/'GRADIENT_TRANSFER_ROWS_PRIVATE.json',allrows);dump(H/'BATCH_PARTITIONS_PRIVATE.json',partitions);dump(H/'SUMMARY_ROWS.json',summaries);dump(H/'DECISION.json',decision);print(decision,flush=True)
if __name__=='__main__':main()
