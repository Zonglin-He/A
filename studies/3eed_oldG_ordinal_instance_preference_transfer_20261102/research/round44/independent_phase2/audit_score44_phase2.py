"""Independent post-seal R44 head/loss/full CPU32 NN audit, then task scoring."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,json,hashlib,time,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;EXP=ROOT/'research/round44/phase2';BASE=ROOT/'crossplatform/round44/fit272_attempt001';DIRECT=ROOT/'crossplatform/round44/direct208_attempt001';GEOM=ROOT/'crossplatform/round37/prepared208_attempt001'
sys.path.insert(0,str(ROOT/'research/round43/independent_audit'))
# Reuse only reviewer-owned CPU primitives, never the author's train/loss code.
from audit_score43_independent import read,sha,dump,load,verify,feature,teacher,head,key,close,allrows
def group_readout(s,p0,groups):
 used=set();out=[]
 for g in groups.values():
  if not g:continue
  assert not used.intersection(g);used.update(g);rep=max(g,key=lambda q:(float(p0[q]),-q));out.append((float(np.max(s[g])),-rep,rep))
 out.extend((float(s[q]),-q,q) for q in range(256) if q not in used);return int(max(out)[2])
def gaps(z,prefs):return np.array([float(z[f['row_index'],f['positive_queries']].max()-z[f['row_index'],f['negative_queries']].max()) for f in prefs])
def terms(z,z0,q0,prefs,U):
 g=gaps(z.astype(float),prefs);d=z.astype(float)-z[np.arange(len(z)),q0].astype(float)[:,None];d0=z0.astype(float)-z0[np.arange(len(z0)),q0].astype(float)[:,None];pref=float(np.logaddexp(0,-g/.1).mean());keep=float(((d[U]-d0[U])**2).mean());return {'total':pref+keep,'preference':pref,'relative_z_keep':keep}
def audit():
 started=time.perf_counter();HERE.mkdir(exist_ok=True);plan=read(EXP/'PLAN_LOCK.json');auth=read(EXP/'ROOT_EXECUTION_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_FINITE_PHASE2_MATCHING_TTA';assert plan['root_authorization_sha256']==sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json')
 for p,h in dict(plan['program_sha256'],**plan['input_sha256']).items():assert sha(ROOT/p)==h,p
 impl=read(EXP/'IMPLEMENTATION_LOCK.json');assert impl['plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 seal=verify(BASE);assert seal['states']==2 and seal['optimizer_steps']==32 and seal['n']==272 and seal['plan_sha256']==sha(EXP/'PLAN_LOCK.json');verify(DIRECT)
 rr=allrows();zs={r['id']:feature(r) for r in rr};pred={r['id']:load(BASE/'predictions_PRIVATE'/(r['id']+'.npz')) for r in rr};rawsrc=load(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz');source={k.removeprefix('image.'):v for k,v in rawsrc.items() if k.startswith('image.')};assert sum(v.size for v in source.values())==184960
 cs=read(ROOT/'crossplatform/round24/capture336_attempt001/SEALED.json');bs=read(ROOT/'crossplatform/round23/bank272_attempt001/SEALED.json');gs=read(GEOM/'SEALED.json');assert sha(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz')==cs['files']['projection_source_PRIVATE.npz']
 for r in rr:
  rel='samples_PRIVATE/'+r['id']+'.npz';assert sha(ROOT/'crossplatform/round24/capture336_attempt001'/rel)==cs['files'][rel];assert sha(ROOT/'crossplatform/round23/bank272_attempt001'/rel)==bs['files'][rel]
 receipts=read(BASE/'FIT_RECEIPTS.json');assert len(receipts)==2;headrows=[];lossrows=[];pairrows=[];qdiff=[];memrows=[];maxp=maxz=0.;nncells=0;topkties=0;nn_index_diff=0
 import torch
 torch.set_num_threads(4);assert not torch.cuda.is_initialized()
 for pl,N in [('drone',14),('quad',18)]:
  A=[r for r in rr if r['split']=='D208' and r['platform']==pl];V=[r for r in rr if r['split']=='V64' and r['platform']==pl];inp=load(BASE/'input_contract_PRIVATE'/(pl+'.npz'));pf=read(BASE/'input_contract_PRIVATE'/(pl+'_preferences.json'));prefs=pf['preferences'];assert len(A)==104 and len(V)==32;assert inp['A_ids'].tolist()==[r['id'] for r in A] and inp['V_ids'].tolist()==[r['id'] for r in V]
  expectedprefs=[];groups=[]
  for j,r in enumerate(A):
   rel='cases_PRIVATE/'+r['id']+'.json';assert sha(GEOM/rel)==gs['files'][rel];g=read(GEOM/rel)['full_groups'];groups.append(g);c=read(DIRECT/rel);f=c['decision']['accepted_preference']
   case=read(BASE/rel);assert case['full_groups']==g and case['phase1_decision']==c['decision']
   if f is not None:expectedprefs.append(dict(f,row_index=j,alias=r['alias'],id_PRIVATE=r['id']))
  assert prefs==expectedprefs and len(prefs)==N;E=[f['row_index'] for f in prefs];U=[j for j in range(104) if j not in E];assert len(set(E))==N and len(U)==104-N and pf['U_indices']==U and np.array_equal(inp['E_indices'],E) and np.array_equal(inp['U_indices'],U)
  sA=np.stack([zs[r['id']]['scores'] for r in A]);sV=np.stack([zs[r['id']]['scores'] for r in V]);assert np.array_equal(inp['source_A_scores'],sA) and np.array_equal(inp['source_V_scores'],sV);q0=sA.argmax(-1);assert np.array_equal(inp['source_q0'],q0)
  for tag,rs in [('A',A),('V',V)]:
   T=max(len(zs[r['id']]['tokens']) for r in rs);mask=np.zeros((len(rs),T),bool);pad=np.ones_like(mask)
   for j,r in enumerate(rs):n=len(zs[r['id']]['tokens']);mask[j,:n]=zs[r['id']]['token_mask'];pad[j,:n]=False
   assert np.array_equal(inp[tag+'_mask'],mask) and np.array_equal(inp[tag+'_padding'],pad)
  state=load(BASE/'states_PRIVATE'/(pl+'_query_mlp.npz'));assert state.keys()==source.keys() and sum(v.size for v in state.values())==184960;rec=next(r for r in receipts if r['platform']==pl);assert rec['E']==N and rec['U']==104-N and rec['steps']==16 and rec['z0_no_grad']
  freshA=[];savedA=[]
  for j,r in enumerate(A+V):
   z=zs[r['id']];p=pred[r['id']];srcp,srcz=head(z,source);pp,zz=head(z,state);pe=float(abs(pp-p['adapted_native_probability']).max());ze=float(abs(zz-p['adapted_token_logodds']).max());maxp=max(maxp,pe);maxz=max(maxz,ze);assert pe<2e-5 and ze<5e-4;assert abs(srcp-p['source_replayed_native_probability']).max()<2e-5 and abs(srcz-p['source_token_logodds']).max()<5e-4
   assert np.array_equal(p['source_native_probability'],z['scores']);tag='A' if r['split']=='D208' else 'V';idx=j if tag=='A' else j-len(A);assert np.array_equal(p['source_token_logodds'],inp['source_'+tag+'_z'][idx]) and np.array_equal(p['source_replayed_native_probability'],inp['source_'+tag+'_replayed'][idx])
   headrows.append({'alias':r['alias'],'platform':pl,'split':r['split'],'native_maxerror':pe,'z_maxerror':ze})
   if int(pp.argmax())!=int(p['adapted_native_probability'].argmax()):qdiff.append({'alias':r['alias'],'CPU64_query':int(pp.argmax()),'GPU_query':int(p['adapted_native_probability'].argmax()),'not_replaced':True})
   FG=teacher(r,z);assert np.array_equal(p['oldG_F64'],FG) and np.array_equal(p['deltaG_F64'],FG-z['scores'].astype(float));names=p['arm_names'].tolist();ss=p['scores'];qq=p['queries'];assert np.array_equal(p['boxes'],z['boxes'][qq])
   if tag=='A':
    direct=read(DIRECT/'cases_PRIVATE'/(r['id']+'.json'));expect=['source','eligible_G_direct','oldG_unrestricted','TTA_group_fixed_source_rep','TTA_raw_query'];eq=[int(z['scores'].argmax()),direct['decision']['query'],int(FG.argmax()),group_readout(p['adapted_native_probability'],z['scores'],groups[idx]),int(p['adapted_native_probability'].argmax())];assert names==expect and np.array_equal(qq,eq);assert group_readout(z['scores'],z['scores'],groups[idx])==int(q0[idx]);assert np.array_equal(ss[0],z['scores']) and np.array_equal(ss[1],FG) and np.array_equal(ss[2],FG) and np.array_equal(ss[3],p['adapted_native_probability']) and np.array_equal(ss[4],p['adapted_native_probability']);freshA.append(zz);savedA.append(p['adapted_token_logodds'])
   else:
    assert names==['source','TTA','memory_E_only','memory_allA_zeroU','oldG_direct_current','TTA_plus_current_G'] and np.array_equal(qq,ss.argmax(-1));assert np.array_equal(ss[0],z['scores']) and np.array_equal(ss[1],p['adapted_native_probability']) and np.array_equal(ss[4],FG) and np.array_equal(ss[5],p['adapted_native_probability'].astype(float)+p['deltaG_F64'])
  final=terms(np.stack(savedA),inp['source_A_z'],q0,prefs,U);fresh=terms(np.stack(freshA),inp['source_A_z'],q0,prefs,U);initial=terms(inp['source_A_z'],inp['source_A_z'],q0,prefs,U)
  for k in final:close(final[k],rec['final_terms_after_step16'][k],5e-4);close(fresh[k],final[k],1e-3);close(initial[k],rec['initial_terms'][k],5e-4)
  delta=np.sqrt(sum(float(((state[k].astype(float)-source[k].astype(float))**2).sum()) for k in state));close(delta,rec['parameter_delta_L2'],2e-5);lossrows.append({'platform':pl,'E':N,'U':len(U),'final_CPU_from_GPU_z':final,'fresh_CPU64_head_terms':fresh,'initial_terms':initial,'parameter_delta_L2':delta})
  g0=gaps(inp['source_A_z'],prefs);g1=gaps(np.stack(savedA),prefs)
  for f,b0,b1,oldrec in zip(prefs,g0,g1,rec['preferences']):
   j=f['row_index'];p=pred[A[j]['id']];raw=int(p['queries'][4]);q=int(p['queries'][3]);P=f['positive_queries'];NN=f['negative_queries'];assert oldrec['alias']==f['alias'] and oldrec['raw_query']==raw and oldrec['fixed_source_group_query']==q;close(oldrec['initial_gap_z'],b0,5e-4);close(oldrec['final_gap_z'],b1,5e-4);assert oldrec['pair_satisfied_z']==bool(b1>0) and oldrec['winner_in_positive']==(q in P) and oldrec['winner_in_negative']==(q in NN) and oldrec['third_unjudged_winner']==(q not in P+NN);pairrows.append({'alias':f['alias'],'platform':pl,'initial_gap':float(b0),'final_gap':float(b1),'pair_satisfied':bool(b1>0),'final_in_positive':q in P,'final_in_negative':q in NN,'third_unjudged':q not in P+NN})
  # Independently reconstruct all keys/values and full CPU32 search space.
  ak=np.stack([key(zs[r['id']]) for r in A]);ak/=np.maximum(np.linalg.norm(ak,axis=-1,keepdims=True),1e-12);vk=np.stack([key(zs[r['id']]) for r in V]);vk/=np.maximum(np.linalg.norm(vk,axis=-1,keepdims=True),1e-12);values=np.zeros_like(sA,np.float32);counts=np.zeros_like(values)
  for f in prefs:
   j=f['row_index'];P=f['positive_queries'];NN=f['negative_queries'];d=max(.1-float(sA[j,P].max()-sA[j,NN].max()),0)/2;values[j,P]+=d;values[j,NN]-=d;counts[j,P]+=1;counts[j,NN]+=1
  values/=np.maximum(counts,1);library=load(BASE/'memory_PRIVATE'/(pl+'_library.npz'));assert np.array_equal(library['A_keys'],ak) and np.array_equal(library['V_keys'],vk) and np.array_equal(library['values_probability_units'],values) and np.count_nonzero(values[U])==0
  for name,lib in [('memory_E_only',E),('memory_allA_zeroU',list(range(104)))]:
   K=torch.tensor(ak[lib],device='cpu');value=torch.tensor(values,device='cpu');lookup={a:i for i,a in enumerate(lib)}
   for j,r in enumerate(V):
    nn=load(BASE/'memory_PRIVATE'/(r['id']+'_'+name+'.npz'));obs=nn['A_row_indices'];pos=nn['A_query_indices'];assert obs.shape==pos.shape==(256,4) and all(int(i) in lookup for i in obs.ravel()) and np.all((pos>=0)&(pos<256));assert all(len(set(a))==4 for a in obs.tolist());local=torch.tensor(np.vectorize(lookup.__getitem__)(obs),dtype=torch.long);posT=torch.tensor(pos,dtype=torch.long);sim=torch.einsum('qc,anc->qan',torch.tensor(vk[j],device='cpu'),K);near,which=sim.max(-1);nncells+=sim.numel();ar=torch.arange(256)[:,None];sel=near[ar,local];assert torch.equal(which[ar,local],posT);assert torch.equal(sel,sim[ar,local,posT]);assert np.array_equal(sel.numpy(),nn['cosine']);assert torch.all(sel[:,:-1]>=sel[:,1:]);unselected=near.clone();unselected.scatter_(1,local,-torch.inf);assert torch.all(unselected.max(-1).values<=sel[:,-1]);topkties+=int(((near==sel[:,-1,None]).sum(-1)>(sel==sel[:,-1,None]).sum(-1)).sum());_,originalobs=near.topk(4,-1);nn_index_diff+=int((originalobs!=local).any(-1).sum());weights=((sel-1)/.1).exp();assert np.array_equal(weights.numpy(),nn['weights']);corr=(value[torch.tensor(obs),posT]*weights).sum(-1)/weights.sum(-1).clamp_min(1e-12);assert np.array_equal(corr.numpy(),nn['correction_probability_units']);score=zs[r['id']]['scores']+corr.numpy();ix=pred[r['id']]['arm_names'].tolist().index(name);assert np.array_equal(pred[r['id']]['scores'][ix],score);assert np.array_equal(zs[r['id']]['scores']+np.zeros(256,np.float32),zs[r['id']]['scores']);memrows.append({'alias':r['alias'],'platform':pl,'library':name,'library_size':len(lib),'all256x4_actual_neighbors_legal':True,'source_query_argmax_contract_exact':True,'top4_order_and_unselected_boundary_valid':True,'weights_values_correction_and_score_exact':True})
 assert not torch.cuda.is_initialized();dump(HERE/'HEAD_REPLAY_ROWS.json',headrows);dump(HERE/'LOSS_REPLAY.json',lossrows);dump(HERE/'ALL32_FEEDBACK_EXECUTION.json',pairrows);dump(HERE/'ALL128_MEMORY_REPLAY_ROWS.json',memrows)
 result={'state':'PASS_R44_PHASE2_INDEPENDENT_ARRAY_LOSS_FULL_NN_AUDIT','states':2,'optimizer_steps_reviewed_not_rerun':32,'head_vectors':272,'feedback':32,'keep':176,'native_maxerror':maxp,'z_maxerror':maxz,'CPU64_argmax_differences_not_replaced':qdiff,'memory_retrievals':128,'full_CPU32_similarity_cells_checked':nncells,'top4_boundary_tied_queries':topkties,'actual_vs_recomputed_CPU32_topk_index_difference_rows':nn_index_diff,'all_actual_neighbors_legal_max_top4_and_weights':True,'zero_U_values_and_native_probability_units_exact':True,'all1424_selected_queries_original_boxes_exact':True,'no_task_GT_GPU_optimizer_FINAL_U_T':True,'prediction_seal_sha256':sha(BASE/'SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-started};dump(HERE/'AUDIT_RESULT.json',result);print(json.dumps(result))
def score():
 ar=read(HERE/'AUDIT_RESULT.json');assert ar['state']=='PASS_R44_PHASE2_INDEPENDENT_ARRAY_LOSS_FULL_NN_AUDIT' and ar['prediction_seal_sha256']==sha(BASE/'SEALED.json') and ar['program_sha256']==sha(__file__);verify(BASE);sys.path.insert(0,str(ROOT/'research/round5/audit'));from independent_core_v2 import candidate_ious,transform_gt
 index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};out=[]
 for r in allrows():
  p=load(BASE/'predictions_PRIVATE'/(r['id']+'.npz'));ix=index[r['id']];path=ix['meta_path'];assert sha(path)==ix['metadata_container_sha256']
  if path not in containers:containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));bb=p['boxes'].copy();bb[:,3:]=np.maximum(bb[:,3:],1e-6);v=candidate_ious(bb,gt)[0];names=p['arm_names'].tolist();out.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'split':r['split'],'scene_PRIVATE':r['scene'],'queries':dict(zip(names,map(int,p['queries']))),'IoU':dict(zip(names,map(float,v)))})
 old={r['id_PRIVATE']:r for r in read(ROOT/'research/round44/independent_phase1/ALL208_TASK_ROWS_PRIVATE.json')};nold=0
 for r in out:
  if r['split']=='D208':
   for a,b in [('source','source'),('eligible_G_direct','eligible_fixed_source_rep'),('oldG_unrestricted','unrestricted_oldG')]:close(r['IoU'][a],old[r['id_PRIVATE']]['IoU'][b],1e-12);nold+=1
 summary={};neg=[];scenes=[]
 for split in ['D208','V64']:
  group=[r for r in out if r['split']==split];names=list(group[0]['IoU']);s={}
  for pl in ['drone','quad']:
   part=[r for r in group if r['platform']==pl];s[pl]={}
   for a in names:
    v=np.array([r['IoU'][a] for r in part]);d=v-np.array([r['IoU']['source'] for r in part]);s[pl][a]={'n':len(part),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][a]!=r['queries']['source'] for r in part),'scene_clusters':len({r['scene_PRIVATE'] for r in part})}
   for j,scene in enumerate(sorted({r['scene_PRIVATE'] for r in part})):
    ss=[r for r in part if r['scene_PRIVATE']==scene];scenes.append({'platform':pl,'split':split,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in ss])*100) for a in names}})
  s['balanced_mIoU_percent']={a:float(np.mean([s[p][a]['mIoU_percent'] for p in ['drone','quad']])) for a in names};summary[split]=s
  for r in group:
   for a in names:
    for b in names:
     if a!=b and r['IoU'][a]<r['IoU'][b]-1e-10:neg.append({'alias':r['alias'],'platform':r['platform'],'split':split,'arm':a,'reference':b,'delta_pp':100*(r['IoU'][a]-r['IoU'][b])})
 dump(EXP/'ALL_ROWS_PRIVATE.json',out);dump(EXP/'TASK_SUMMARY.json',summary);dump(EXP/'ALL_NEGATIVES.json',neg);dump(EXP/'SCENE_ROWS.json',scenes);receipt={'state':'PASS_R44_PHASE2_INDEPENDENT_POSTSEAL_TASK_SCORE','rows':272,'task_values':sum(len(r['IoU']) for r in out),'negative_comparisons':len(neg),'phase1_taskvalues_exact':nold,'prediction_seal_sha256':sha(BASE/'SEALED.json'),'independent_array_audit_sha256':sha(HERE/'AUDIT_RESULT.json'),'no_GT_feedback_filter_or_new_GPU_optimizer_FINAL_U_T':True};dump(HERE/'SCORING_RECEIPT.json',receipt);print(json.dumps(receipt));print(json.dumps({k:v['balanced_mIoU_percent'] for k,v in summary.items()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','score']);a=p.parse_args();audit() if a.phase=='audit' else score()
