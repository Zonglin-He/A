"""Independent lineage, optimizer, physical mapping and scalar metric audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_referdino_condition_hc128_v1 import D,P,verify

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 from vg_tta.stvg_hc_anchor_content16_v1 import local_cost
 torch.set_num_threads(4);p=verify(True);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==128;start=time.perf_counter();pub=read(P/'RESULTS_PUBLIC.json');qs=[r['query_ordinal']for r in p['records']];gt,span,_=truths('hc2',dict(parents=qs));plans=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];count=dict(roster=0,lineages=0,selections=0,real_updated_reference=0,shared_S1=0,Adam_steps=0,fit_costs=0,metrics=0,outside8=0,bootstraps=0);errors=dict(Adam=0.,metrics=0.,loss=0.)
 assert len(set(r['parent']for r in p['records']))==128 and not {r['parent']for r in p['records']}&{r['parent']for r in p['confirmation_reserved']};count['roster']=128
 for r in p['records']:
  a=r['alias'];z=load(D/'student2'/(a+'.pt'));s=load(D/'student1'/(a+'.pt'));inp=load(D/'inputs'/(a+'.pt'));old=load(D/'teacher0'/(a+'.pt'));new=load(D/'teacher1'/(a+'.pt'));rec=next(x for x in pub['rows']if x['alias']==a);pos=r['positions8'];q=r['query_ordinal'];row=plans[q]
  assert np.array_equal(old['student_reference'],inp['E'][pos]);assert np.array_equal(new['student_reference'],s['fit']['final'][pos].numpy());assert old['slots']==new['slots']==old['native']['trace']['selected_slots_native'];assert old['rgb_sha256']==new['rgb_sha256'];count['real_updated_reference']+=1
  for n in [old['native'],old['conditional'],new['conditional']]:
   tr=n['trace'];prunes=tr['pruning_lineages'];assert[len(x)for x in prunes]==[450,225,113,57,29,15];raw=prunes[-1].T;matches=tr['matching_permutations'];tracked=torch.stack([raw[0]]+[raw[t+1,torch.tensor(m[0])]for t,m in enumerate(matches)]);assert torch.equal(tracked,tr['tracked_lineages']);logits=tr['all_logits'];scores=logits.sigmoid().mean(0).max(-1).values;idx=int(scores.argmax());assert idx==n['selected_query']and torch.equal(n['boxes'],tr['all_boxes'][:,idx]);assert torch.equal(tracked[:,idx],tr['selected_lineages']);count['lineages']+=1;count['selections']+=1
   if'injected_slots'in tr:
    wanted=tr['injected_slots'];assert tr['selected_is_intervened']==(tracked[:,idx]==wanted).tolist();assert tr['survives_per_layer']==[(x==wanted[None]).any(0).tolist()for x in prunes];assert tr['other_references_bitexact']
  assert torch.equal(z['fixed']['path'][0]['state'],s['fit']['final_state'])and torch.equal(z['refresh']['path'][0]['state'],s['fit']['final_state']);assert torch.equal(z['fixed']['path'][0]['boxes'],z['refresh']['path'][0]['boxes']);assert np.array_equal(z['outputs']['E_MBR'],inp['E'])and z['interval']==inp['interval'];count['shared_S1']+=1
  for fit,target in [(s['fit'],old['conditional']['boxes']),(z['fixed'],old['conditional']['boxes']),(z['refresh'],new['conditional']['boxes'])]:
   state=fit['path'][0]['state'].clone().requires_grad_(True);opt=torch.optim.Adam([state],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0.)
   for k,update in enumerate(fit['updates']):
    opt.zero_grad(set_to_none=True);state.grad=update['gradient'].clone();opt.step();err=float((state-fit['path'][k+1]['state']).abs().max());errors['Adam']=max(errors['Adam'],err);assert err<3e-6;count['Adam_steps']+=1
   for path in fit['path']:
    cost=float(local_cost(path['boxes'][pos],target.double()).mean());err=abs(cost-path['loss']);errors['loss']=max(errors['loss'],err);assert err<1e-10;count['fit_costs']+=1
  for arm,boxes in z['outputs'].items():
   m=DenseTube(boxes,row,gt[q],span[q],clip=True).score(z['interval'])
   for ep in ['v','s','t']:
    err=abs(m[ep]-rec['values'][arm][ep]);errors['metrics']=max(errors['metrics'],err);assert err<2e-10;count['metrics']+=1
   vv=[iou_gt(boxes[i],gt[q][int(fid)],row,'hc2')for i,fid in enumerate(z['frame_ids'])if i not in pos and int(fid)in gt[q]];v=float(np.mean(vv))if vv else None;expected=rec['values'][arm]['unobserved_actual_s'];assert v is None and expected is None or v is not None and abs(v-expected)<1e-12;count['outside8']+=1
 for pair,eps in pub['groups']['HC128']['paired'].items():
  a,b=pair.split('_minus_')
  for ep,st in eps.items():
   v=np.asarray([100*(r['values'][a][ep]-r['values'][b][ep])for r in pub['rows']if r['values'][a][ep]is not None and r['values'][b][ep]is not None]);idx=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)));assert abs(float(v.mean())-st['mean'])<1e-12 and np.allclose(np.quantile(v[idx].mean(1),[.025,.975]),st['CI95'],rtol=0,atol=1e-12);count['bootstraps']+=1
 verify(True);assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=count,max_errors=errors,CPU_seconds=time.perf_counter()-start,scope='saved trajectory/lineage/optimizer/physical metrics; no second GPU run',reserved32_not_scored=True));print(count,errors,flush=True)
if __name__=='__main__':main()
