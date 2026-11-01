"""Fixed-denominator leave-one-feedback-observation-out, no task labels."""
import time,numpy as np,collections
from common31 import *
import torch,fit24
fit24.feature=feature
from fit24 import module,batch,scores
from fit29 import logodds_scores,group_readout
# Importing legacy fit29 mutates fit24.feature; restore full S/R/Q-aware reader.
fit24.feature=feature
F=torch.nn.functional

def main():
 verify(ROOT/'crossplatform/round30/screen_attempt001');verify(ROOT/'crossplatform/round29/direct208_attempt001');out=BASE/'masscontrol_attempt001';out.mkdir(exist_ok=False);(out/'states_PRIVATE').mkdir();(out/'predictions_PRIVATE').mkdir();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.perf_counter();feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');allfids={f['id_PRIVATE'] for f in feedback};receipts=[];models=[]
 for pl in ['drone','quad']:
  A=[r for r in rows() if r['platform']==pl];rr=[r for split in ['A208','DEV64','R64','S64'] for r in rows(split) if r['platform']==pl];a=batch(A);b=batch(rr);lookup={r['id']:i for i,r in enumerate(A)};ff=[f for f in feedback if f['platform']==pl];N=len(ff);assert N=={'drone':10,'quad':12}[pl];U=[i for i,r in enumerate(A) if r['id'] not in allfids];assert len(U)=={'drone':96,'quad':93}[pl];q0=a['scores'].argmax(1);omits=sorted({sum(f['id_PRIVATE']==r['id'] for f in ff) for r in A if r['id'] in allfids})
  for omitted in omits:
   removed_count=omitted;omitted=None;zero=False;alias='mass_'+str(removed_count);name=pl+'_'+alias;m=module('query_mlp');initial={n:p.detach().clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=0.);weight=torch.full((N,),(N-removed_count)/N,device='cuda');trace=[]
   with torch.no_grad():z0=logodds_scores(m,a);d0=z0-z0.gather(1,q0[:,None])
   for step in range(1,17):
    opt.zero_grad();z=logodds_scores(m,a);gap=torch.stack([z[lookup[f['id_PRIVATE']],f['positive_group_PRIVATE']].max()-z[lookup[f['id_PRIVATE']],f['negative_group_PRIVATE']].max() for f in ff]);pref=(F.softplus(-gap/.1)*weight).sum()/N;dist=z-z.gather(1,q0[:,None]);keep=(dist[U]-d0[U]).square().mean();loss=pref+keep;assert torch.isfinite(loss);loss.backward();opt.step()
    if step in [1,4,8,16]:trace.append({'step':step,'loss':float(loss.detach()),'pref':float(pref.detach()),'keep':float(keep.detach())})
   with torch.no_grad():ss=scores(m,'query_mlp',b).cpu().numpy();za=logodds_scores(m,a);gaps=torch.stack([za[lookup[f['id_PRIVATE']],f['positive_group_PRIVATE']].max()-za[lookup[f['id_PRIVATE']],f['negative_group_PRIVATE']].max() for f in ff]).cpu().numpy()
   state={n:p.detach().cpu().numpy() for n,p in m.state_dict().items()};np.savez_compressed(out/'states_PRIVATE'/(name+'.npz'),**state);qs=np.array([group_readout(r,s) if r['stage'] not in ['Q','R','S'] else int(s.argmax()) for r,s in zip(rr,ss)]);np.savez_compressed(out/'predictions_PRIVATE'/(name+'.npz'),row_ids_PRIVATE=np.array([r['id'] for r in rr]),aliases=np.array([r['alias'] for r in rr]),queries=qs,scores=ss)
   baseline_error=None
   if False:
    old=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));baseline_error=max(float(abs(state[n]-old[n]).max()) for n in state);assert baseline_error<1e-6,(pl,baseline_error)
   delta=float(torch.sqrt(sum((p-initial[n]).square().sum() for n,p in m.named_parameters())))
   if zero:assert delta==0.,(pl,delta)
   spec={'name':name,'platform':pl,'kind':'uniform_mass_control','removed_count_matched':removed_count,'uniform_weight':(N-removed_count)/N,'omitted_alias':None if omitted is None or zero else omitted['alias'],'omitted_id_PRIVATE':None if omitted is None or zero else omitted['id'],'original_N':N,'active_preferences':int(weight.sum()),'keep_n':len(U),'removed_added_to_keep':False};models.append(spec);receipts.append(dict(spec,steps=16,source_initialization=True,parameter_delta_L2=delta,baseline_R30_state_maxerror=baseline_error,all_original_pair_gaps=gaps.tolist(),trace=trace));print('LOO31',name,'active',int(weight.sum()),'N',N,'seconds',round(time.perf_counter()-start,1),flush=True)
 dump(out/'MODEL_SPECS_PRIVATE.json',models);dump(out/'FIT_RECEIPTS_PRIVATE.json',receipts);seal(out,'R31_allfeedback_uniform_nominalmass_control_sealed',models=4,steps=64,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),new_Qwen_calls=0,task_labels_read=False);print('MASS31_SEALED')
if __name__=='__main__':main()
