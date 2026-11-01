"""One keep-mask intervention, plus exact old-R30 control. No task labels."""
import time,numpy as np
from common35 import *
from mask35 import make_masks
import torch,fit24
from fit29 import logodds_scores,group_readout
# Legacy imports modify fit24.feature; restore the approved A/DEV reader last.
fit24.feature=feature
from fit24 import module,batch,scores
F=torch.nn.functional

def main():
 plan=read(EXP/'PLAN_LOCK.json');assert plan['configurations']==['R30_U_rows','U_rows_matched_denominator','partial_output_keep']
 assert plan['lr']==.0005 and plan['steps']==16 and plan['beta']==1 and plan['tau']==.1
 for program,digest in plan['implementation_sha256'].items():assert sha(BASE/program)==digest,('locked_program_changed',program)
 assert sha(EXP/'MASK_COUNTS_LOCK.json')==plan['mask_counts_lock_sha256']
 assert sha(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz')==plan['source_projection_sha256']
 assert sha(ROOT/'crossplatform/round30/screen_attempt001/SEALED.json')==plan['R30_reference_seal_sha256']
 feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json')
 assert len(feedback)==22 and sha(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json')==plan['feedback_sha256']
 verify(ROOT/'crossplatform/round30/screen_attempt001')
 out=BASE/'partial_keep_attempt001';out.mkdir(exist_ok=False);(out/'states_PRIVATE').mkdir();(out/'predictions_PRIVATE').mkdir();(out/'masks_PRIVATE').mkdir()
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 started=time.perf_counter();receipts=[];maskrecords=[]
 for pl in ['drone','quad']:
  A=[r for r in rows() if r['platform']==pl];D=[r for r in rows('DEV64') if r['platform']==pl];ff=[f for f in feedback if f['platform']==pl]
  U,old,partial,count,detail=make_masks(A,ff);expected=read(EXP/'MASK_COUNTS_LOCK.json')[pl]
  assert count==expected and len(ff)=={'drone':10,'quad':12}[pl] and len(U)=={'drone':96,'quad':93}[pl]
  np.savez_compressed(out/'masks_PRIVATE'/(pl+'.npz'),old_keep=old,partial_keep=partial,row_ids_PRIVATE=np.array([r['id'] for r in A]))
  maskrecords.append({'platform':pl,'count':count,'rows_PRIVATE':detail})
  a=batch(A);d=batch(D);lookup={r['id']:i for i,r in enumerate(A)};q0=a['scores'].argmax(1);mask=torch.tensor(partial,device='cuda')
  for name in plan['configurations']:
   m=module('query_mlp');initial={n:p.detach().clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=0.)
   with torch.no_grad():
    z0=logodds_scores(m,a);d0=z0-z0.gather(1,q0[:,None]);err=float(abs(scores(m,'query_mlp',a)-a['scores']).max());assert err<2e-5
   def gaps(z):return torch.stack([z[lookup[f['id_PRIVATE']],f['positive_group_PRIVATE']].max()-z[lookup[f['id_PRIVATE']],f['negative_group_PRIVATE']].max() for f in ff])
   def keepterm(z):
    relative=z-z.gather(1,q0[:,None]);square=(relative-d0).square()
    # Preserve R30 reduction order for the reproduction control.
    if name=='R30_U_rows':return square[U].mean()
    if name=='U_rows_matched_denominator':return square[U].sum()/int(partial.sum())
    return square[mask].mean()
   trace=[]
   for step in range(1,17):
    opt.zero_grad();z=logodds_scores(m,a);gap=gaps(z);pref=F.softplus(-gap/.1).mean();keep=keepterm(z);loss=pref+keep
    assert torch.isfinite(loss);loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())));opt.step()
    if step in [1,4,8,16]:trace.append({'step':step,'timing':'before_this_optimizer_step','loss':float(loss.detach()),'preference':float(pref.detach()),'keep':float(keep.detach()),'gradient_L2':grad})
   with torch.no_grad():
    pa=scores(m,'query_mlp',a).cpu().numpy();pd=scores(m,'query_mlp',d).cpu().numpy();zg=logodds_scores(m,a);g=gaps(zg).cpu().numpy();lastkeep=float(keepterm(zg));lastpref=float(F.softplus(-gaps(zg)/.1).mean())
   state={n:p.detach().cpu().numpy() for n,p in m.state_dict().items()};np.savez_compressed(out/'states_PRIVATE'/(pl+'_'+name+'.npz'),**state)
   qa=np.array([group_readout(r,s) for r,s in zip(A,pa)]);qd=pd.argmax(1)
   for split,rr,ss,qq in [('A208',A,pa,qa),('DEV64',D,pd,qd)]:
    np.savez_compressed(out/'predictions_PRIVATE'/(pl+'_'+name+'_'+split+'.npz'),row_ids_PRIVATE=np.array([r['id'] for r in rr]),queries=qq,scores=ss)
   repro=None
   if name=='R30_U_rows':
    r30=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));repro=max(float(abs(state[n]-r30[n]).max()) for n in state)
    assert repro<1e-6,('R30_control_failure',pl,repro)
   rec={'platform':pl,'name':name,'steps':16,'lr':.0005,'beta':1,'params':sum(p.numel() for p in m.parameters()),'original_preference_denominator':len(ff),'all_feedback_observations':count['feedback_observations'],'original_U_observations':len(U),'keep_mask_coordinate_count':int(partial.sum() if name=='partial_output_keep' else old.sum()),'keep_normalization_denominator':int(old.sum() if name=='R30_U_rows' else partial.sum()),'derived_original_U_strength_ratio':float(1. if name=='R30_U_rows' else old.sum()/partial.sum()),'original_U_overlap_coordinates':int(old.sum()),'source_q0_fixed':True,'initial_source_score_maxerror':err,'R30_state_replay_maxerror':repro,'pairs_satisfied':int((g>0).sum()),'B_pairs_satisfied':sum(g[j]>0 for j,f in enumerate(ff) if f['label']=='B'),'B_final_positive':sum(qa[lookup[f['id_PRIVATE']]] in f['positive_group_PRIVATE'] for f in ff if f['label']=='B'),'final_preference_loss':lastpref,'final_keep_MSE':lastkeep,'all_original_pair_gaps':g.tolist(),'parameter_delta_L2':float(torch.sqrt(sum((p-initial[n]).square().sum() for n,p in m.named_parameters()))),'trace':trace}
   receipts.append(rec);print('PARTIAL_KEEP35',pl,name,'B',rec['B_pairs_satisfied'],rec['B_final_positive'],'K',rec['keep_mask_coordinate_count'],'denominator',rec['keep_normalization_denominator'],flush=True)
 dump(out/'FIT_RECEIPTS.json',receipts);dump(out/'MASK_RECORDS_PRIVATE.json',maskrecords)
 seal(out,'R35_partial_keep_with_derived_strength_control_ADEV_before_selection',models=6,steps=96,no_task_labels_fit=True,new_G_or_Qwen_calls=0,seconds=time.perf_counter()-started,peak_bytes=torch.cuda.max_memory_allocated(),plan_lock_sha256=sha(EXP/'PLAN_LOCK.json'),program_sha256=sha(__file__),mask_program_sha256=sha(BASE/'mask35.py'))
 print('PARTIAL_KEEP35_SEALED',flush=True)
if __name__=='__main__':main()
