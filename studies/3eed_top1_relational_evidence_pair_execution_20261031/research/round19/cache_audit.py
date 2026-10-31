"""Replay cached R18 projection episodes; old task scores only diagnose history."""
import sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round19'))
from common import *
def main():
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);src=np.load(ROOT/'crossplatform/round17/features96_attempt001/projection_source_PRIVATE.npz');source={'weight':torch.tensor(src['4.weight']),'bias':torch.tensor(src['4.bias'])};oldscore={r['id_PRIVATE']:r for r in read(ROOT/'research/round18/experiment/scored96_attempt001/ROWS_PRIVATE.json') if r['split']=='A64'};records=[]
 for r in read(EXP.parent/'input/A64_PRIVATE.json'):
  z=feature(r);pref=read(ROOT/'crossplatform/round17/G_selected64_attempt001/cases_PRIVATE'/(r['id']+'.json'))['preference'];q0=int(z['scores'].argmax());p=ROOT/'crossplatform/round18/formal96_attempt001/predictions_PRIVATE'/(r['id']+'.npz')
  with np.load(p) as a:n=a['arm_names'].tolist();actual={k:int(a['query_indices'][n.index(k)]) for k in ('source','own_G_direct_no_grounder_update','own_G_episode_projection32')}
  rec={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'status':pref['status'],'actual_query_PRIVATE':actual}
  if pref['status']=='usable':
   layer=torch.nn.Linear(288,64);layer.load_state_dict(source);base={n:p.detach().clone() for n,p in layer.named_parameters()};x=torch.tensor(z['x']);t=torch.tensor(z['tokens']);mask=torch.tensor(z['token_mask']);pos=pref['positive_queries'];neg=pref['negative_queries'];s0=torch.tensor(z['scores']);opt=torch.optim.AdamW(layer.parameters(),lr=1e-4,weight_decay=0.)
   def score():return ((torch.nn.functional.normalize(layer(x),dim=-1)@t.T/.07).softmax(-1)*mask).sum(-1)
   before=score().detach().numpy();steps=[]
   for i in range(32):
    opt.zero_grad();s=score().clamp_min(1e-12);gap=s[pos].log().mean()-s[neg].log().mean();p0=(s0/s0.sum()).clamp_min(1e-12);pp=(s/s.sum()).clamp_min(1e-12);rank=torch.nn.functional.softplus(1-gap);kl=(p0*(p0.log()-pp.log())).sum();loss=rank+.1*kl+sum((p-base[n]).square().mean() for n,p in layer.named_parameters());loss.backward();opt.step();steps.append(float(gap.detach()))
   after=score().detach().numpy();other=[j for j in range(256) if j not in pos and j not in neg];qp=max(pos,key=lambda q:(float(z['scores'][q]),-q));qn=max(neg,key=lambda q:(float(z['scores'][q]),-q))
   def state(ss):
    return {'source_Top1_score':float(ss[q0]),'fixed_positive_representative_score':float(ss[qp]),'fixed_negative_representative_score':float(ss[qn]),'positive_max':float(ss[pos].max()),'negative_max':float(ss[neg].max()),'other_max':float(ss[other].max()) if other else None,'mean_log_pair_gap':float(np.log(np.maximum(ss[pos],1e-12)).mean()-np.log(np.maximum(ss[neg],1e-12)).mean()),'pair_max_satisfied':bool(ss[pos].max()>ss[neg].max()),'positive_beats_every_outside_query':bool(ss[pos].max()>np.max(np.delete(ss,pos))),'replayed_Top1_PRIVATE':int(ss.argmax())}
   rec.update(before=state(before),after=state(after),q0_role='positive' if q0 in pos else 'negative' if q0 in neg else 'unknown',positive_n=len(pos),negative_n=len(neg),CPU_replay_Top1_equals_actual=bool(after.argmax()==actual['own_G_episode_projection32']),loss_gap_trace=steps)
   np.savez_compressed(EXP/'cache_audit_PRIVATE'/(r['id']+'.npz'),before=before,after=after)
  io=oldscore[r['id']]['ious'];rec.update(direct_gain_pp=100*(io['own_G_direct_no_grounder_update']-io['source']),TTA_gain_pp=100*(io['own_G_episode_projection32']-io['source']));records.append(rec)
 dump(EXP/'CACHE_AUDIT_PRIVATE.json',records);accepted=[r for r in records if r['status']=='usable'];summary={'state':'R19_existing_cache_diagnosis_completed_no_new_VLM_or_grounder_forward','n':64,'accepted':len(accepted),'q0_roles':dict(collections.Counter(r['q0_role'] for r in accepted)),'CPU_replay_Top1_matches_actual':sum(r['CPU_replay_Top1_equals_actual'] for r in accepted),'actual_R18_selected_query_is_authority':True,'replay_is_numerical_diagnostic_not_replacement':True,'direct_improves_but_TTA_not':[r['alias'] for r in records if r['direct_gain_pp']>0 and r['TTA_gain_pp']<=0],'TTA_harms':[r['alias'] for r in records if r['TTA_gain_pp']<0]};dump(EXP/'CACHE_AUDIT_SUMMARY.json',summary);print(summary)
if __name__=='__main__':(EXP/'cache_audit_PRIVATE').mkdir(exist_ok=False);main()
