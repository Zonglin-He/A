"""Postfit complete/masked/held readouts, same candidate geometry and fixed U0."""
from common75 import *
from numerics75 import rep,matrix,cand,KL,head
import torch,time

def main():
 p=read(H/'experiment/DESIGN_LOCK.json');pin(p);out=ROOT/'crossplatform/round75/predicted336_attempt001';assert not out.exists();out.mkdir(parents=True)
 for f in ['samples_PRIVATE','diagnostics_PRIVATE']:(out/f).mkdir()
 fit=ROOT/'crossplatform/round75/fit336_attempt001';fs=read(fit/'SEALED.json');held=ROOT/'crossplatform/round75/held336_attempt001';hs=read(held/'SEALED.json');masked=ROOT/'crossplatform/round75/masked336_attempt001';ms=read(masked/'SEALED.json');initial=load(ROOT/'crossplatform/round50/episodic336_capture_attempt001/SOURCE_MATCHING_MLP_PRIVATE.npz');torch.set_grad_enabled(False);start=time.perf_counter();statuses=[]
 for i,r in enumerate(rows()):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(held/rel)==hs['files'][rel] and sha(masked/rel)==ms['files'][rel];z=original(r);mm=load(masked/rel);hh=load(held/rel);u=torch.tensor(z['tokens']);mask=torch.tensor(z['token_mask']).bool();source=head(initial);feats={'A':torch.tensor(z['h']),'M1':torch.tensor(mm['M1_h']),'M2':torch.tensor(mm['M2_h']),'M3':torch.tensor(hh['M3_h'])};Q={v:rep(source,h) for v,h in feats.items()};S={v:matrix(q,u) for v,q in Q.items()};pp={v:cand(s,mask).numpy() for v,s in S.items()};base=z['scores'].astype(float);base/=base.sum();assert int(base.argmax())==int(pp['A'].argmax()),'source_native_argmax_bridge';probs={'source':base,'frozen_mask_probability':np.mean([pp[v] for v in ['A','M1','M2']],axis=0),'frozen_mask_token':cand(sum(S[v] for v in ['A','M1','M2'])/3,mask).numpy(),'frozen_mask_representation':cand(matrix(torch.nn.functional.normalize(sum(Q[v] for v in ['A','M1','M2'])/3,dim=-1),u),mask).numpy(),'held_mask_source':pp['M3']};diagnostic={};arrays={};records={};R=S['A'].log_softmax(1);C=S['A'].log_softmax(0)
  for kind in ['source']+ADAPT:
   if kind=='source':m=source
   else:
    f=fit/'weights_PRIVATE'/(sid+'_'+kind+'.npz');assert sha(f)==fs['files'][f.relative_to(fit).as_posix()];m=head(load(f));records[kind]=read(fit/'cases_PRIVATE'/(sid+'_'+kind+'.json'))
   qs={v:rep(m,h) for v,h in feats.items()};ss={v:matrix(q,u) for v,q in qs.items()};pv={v:cand(s,mask).numpy() for v,s in ss.items()};diagnostic[kind]={'held_row_KL':float(KL(R,ss['M3'].log_softmax(1),1).mean()),'held_column_KL':float(KL(C,ss['M3'].log_softmax(0),0).mean()),'held_row_top1_agreement':float((ss['M3'].argmax(1)==S['A'].argmax(1)).double().mean()),'held_column_top1_agreement':float((ss['M3'].argmax(0)==S['A'].argmax(0)).double().mean()),'full_matrix_L2':float((ss['A']-S['A']).square().sum().sqrt()),'complete_text_reference_fixed':True};arrays[kind+'_S_full']=ss['A'].numpy();arrays[kind+'_S_held']=ss['M3'].numpy()
   if kind!='source':
    probs[kind+'_full']=pv['A'];probs[kind+'_ensemble']=np.mean([pv[v] for v in ['A','M1','M2']],axis=0);probs['held_mask_'+kind]=pv['M3']
  historic=load(ROOT/'crossplatform/round66/predicted336_attempt001'/rel,['arm_names','scores','source_boxes']);hd=dict(zip(historic['arm_names'],historic['scores']));assert np.array_equal(z['boxes'],historic['source_boxes']);probs.update(frozen_ABC=hd['frozen_ABC'],R59=hd['R59']);score=np.stack([probs[k] for k in ARMS]);qq=score.argmax(1);assert np.isfinite(score).all();np.savez_compressed(out/rel,arm_names=np.array(ARMS),scores=score,queries=qq,source_boxes=z['boxes'],selected_original_boxes=z['boxes'][qq]);np.savez_compressed(out/'diagnostics_PRIVATE'/(sid+'.npz'),**arrays);dump(out/'diagnostics_PRIVATE'/(sid+'.json'),diagnostic)
  st={}
  for k in ARMS:
   kind=next((a for a in ADAPT if k in [a+'_full',a+'_ensemble','held_mask_'+a]),None);st[k]={'unknown':False,'fallback':not records[kind]['accepted'] if kind else False,'no_parameter_update':not records[kind]['accepted'] if kind else True}
  statuses.append({'id':sid,'arm_statuses':st})
 dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COUNTS.json',{'n':336,'arms':len(ARMS),'seconds':time.perf_counter()-start,'M3_in_fitting_or_acceptance':False,'task_GT_read':False,'source_native_winner_bridge_all_inputs':True});seal(out,'R75_ALL_TASK_AND_HELD_SEMANTIC_READOUTS_SEALED_BEFORE_GT',n=336,arm_names=ARMS);print(read(out/'COUNTS.json'))
if __name__=='__main__':main()
