"""One concrete q0/competitor pair, actual native score hinge, frozen features."""
import time
import numpy as np
from common import *
def optimize(z,qa,qb,spec,source,device):
 import torch
 layer=torch.nn.Linear(288,64,device=device);layer.load_state_dict(source);base={k:v.detach().clone() for k,v in layer.named_parameters()};x=torch.tensor(z['x'],device=device)[None];tokens=torch.tensor(z['tokens'],device=device);mask=torch.tensor(z['token_mask'],device=device);opt=torch.optim.AdamW(layer.parameters(),lr=spec['lr'],weight_decay=0.)
 def score():return ((torch.nn.functional.normalize(layer(x),dim=-1)[0]@tokens.T/.07).softmax(-1)*mask).sum(-1)
 def state(s):
  other=[i for i in range(len(s)) if i not in (qa,qb)];p=s/s.sum().clamp_min(1e-30)
  return {'sA':float(s[qa]),'sB':float(s[qb]),'other_max':float(s[other].max()),'pair_gap':float(s[qb]-s[qa]),'pair_margin_satisfied':bool(s[qb]-s[qa]>=spec['margin']),'B_above_A':bool(s[qb]>s[qa]),'B_global_first':bool(s.argmax()==qb),'Top1_PRIVATE':int(s.argmax()),'conditional_candidate_entropy':float(-(p*p.clamp_min(1e-30).log()).sum())}
 before=score().detach();trace=[];grad=0.;start=time.perf_counter()
 for i in range(spec['max_steps']):
  s=score();gap=s[qb]-s[qa]
  if gap.detach()>=spec['margin']:break
  opt.zero_grad();loss=(spec['margin']-gap).clamp_min(0);loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in layer.parameters())));opt.step()
  with torch.no_grad():cur=score()
  trace.append({'step':i+1,'loss_before':float(loss.detach()),'gradient_L2':grad,**state(cur)})
 after=score().detach();delta=float(torch.sqrt(sum((p-base[n]).square().sum() for n,p in layer.named_parameters())));others=[j for j in range(256) if j not in (qa,qb)];rank0=np.argsort(-before.cpu().numpy()[others],kind='stable');rank1=np.argsort(-after.cpu().numpy()[others],kind='stable')
 d={'steps':len(trace),'selected_parameters':sum(p.numel() for p in layer.parameters()),'parameter_delta_L2':delta,'gradient_L2_last':grad,'before':state(before),'after':state(after),'trace':trace,'other_candidate_order_positions_changed':int((rank0!=rank1).sum()),'unverified_C_top1':bool(after.argmax() not in (qa,qb)),'seconds':time.perf_counter()-start,'no_C_automatically_added_as_negative':True};return {k:v.detach().cpu().numpy() for k,v in layer.state_dict().items()},before.cpu().numpy(),after.cpu().numpy(),d
def main():
 import torch
 spec=locked('FIT_LOCK.json');verify(ROOT/'crossplatform/round19/G96_attempt001');out=ROOT/'crossplatform/round19/pair_fit96_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();(out/'weights_PRIVATE').mkdir();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 with np.load(ROOT/'crossplatform/round17/features96_attempt001/projection_source_PRIVATE.npz') as z:source={'weight':torch.tensor(z['4.weight'],device='cuda'),'bias':torch.tensor(z['4.bias'],device='cuda')}
 traces=[]
 for i,r in enumerate(read(EXP.parent/'input/ALL96_PRIVATE.json')):
  z=feature(r);pref=read(ROOT/'crossplatform/round19/G96_attempt001/cases_PRIVATE'/(r['id']+'.json'))['preference'];qa=int(z['scores'].argmax());qb=qa;qt=qa;d={'steps':0,'parameter_delta_L2':0.}
  if pref['status']=='usable':
   assert pref['qA_PRIVATE']==qa;qb=pref['qB_PRIVATE'];w,before,after,d=optimize(z,qa,qb,spec,source,'cuda');qt=int(after.argmax());np.savez_compressed(out/'weights_PRIVATE'/(r['id']+'.npz'),**w);np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(['source','direct_pair_no_update','pair_projection_TTA']),query_indices=np.array([qa,qb,qt]),arm_boxes_raw6=z['boxes'][[qa,qb,qt]],before_scores=before,after_scores=after)
  else:np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(['source','direct_pair_no_update','pair_projection_TTA']),query_indices=np.array([qa,qb,qt]),arm_boxes_raw6=z['boxes'][[qa,qb,qt]])
  traces.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'stage':r['stage'],'feedback_accepted':pref['status']=='usable','diagnostics':d});dump(out/'TRACE_PRIVATE.json',traces)
 seal(out,'R19_pairwise_actual_projection_updates_sealed_no_GT',n=96,accepted=sum(r['feedback_accepted'] for r in traces),steps=sum(r['diagnostics']['steps'] for r in traces),fixed_boxes=True)
 # Historical q0-negative cases only, selected without task GT. Not new-G success.
 hd=ROOT/'crossplatform/round19/historical_pair_interface_attempt001';hd.mkdir(exist_ok=False);hist=[]
 for r in read(EXP.parent/'input/A64_PRIVATE.json'):
  z=feature(r);p=read(ROOT/'crossplatform/round17/G_selected64_attempt001/cases_PRIVATE'/(r['id']+'.json'))['preference'];qa=int(z['scores'].argmax())
  if p['status']!='usable' or qa not in p['negative_queries']:continue
  qb=max(p['positive_queries'],key=lambda q:(float(z['scores'][q]),-q));_,before,after,d=optimize(z,qa,qb,spec,source,'cuda');hist.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'diagnostics':d});np.savez_compressed(hd/(r['id']+'.npz'),before=before,after=after,boxes=z['boxes'][[qa,qb,int(after.argmax())]])
 dump(hd/'TRACE_PRIVATE.json',hist);seal(hd,'R19_historical_q0negative_pair_execution_diagnostic_not_newG_result',n=len(hist));print('PAIR_FIT_COMPLETE',sum(r['feedback_accepted'] for r in traces),len(hist),flush=True)
if __name__=='__main__':main()
