"""Independent R43 CPU algebra/memory audit; task labels only after audit PASS."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,json,time,hashlib,argparse,datetime
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;EXP=ROOT/'research/round43/experiment';BASE=ROOT/'crossplatform/round43/fit272_attempt001';CAP=ROOT/'crossplatform/round24/capture336_attempt001';BANK=ROOT/'crossplatform/round23/bank272_attempt001'
CONFIGS=['full_probability','full_token_logodds','null_probability','null_token_logodds'];ARMS=['source']+CONFIGS+['direct_G','full_probability_plus_current_G','full_token_logodds_plus_current_G','zero_delta_memory']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(d):
 x=read(d/'SEALED.json')
 for p,h in x['files'].items():assert sha(d/p)==h,p
 return x
def allrows():return [dict(r,split=split) for split in ['D208','V64'] for r in read(ROOT/'research/round24/input'/(split+'_PRIVATE.json'))]
def feature(r):return load(CAP/'samples_PRIVATE'/(r['id']+'.npz'))
def teacher(r,z):
 b=load(BANK/'samples_PRIVATE'/(r['id']+'.npz'));return z['scores'].astype(float)+2*b['vision']+2*b['position']+2*b['relation']
def head(z,state):
 h=z['h'].astype(float)
 for layer in [0,2,4]:
  h=h@state[str(layer)+'.weight'].astype(float).T+state[str(layer)+'.bias'].astype(float)
  if layer<4:h=np.maximum(h,0)
 q=h/np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-12);v=q@z['tokens'].astype(float).T/.07;mask=z['token_mask'];p=np.exp(v-logsumexp(v,axis=-1,keepdims=True));return p[:,mask].sum(-1),logsumexp(v[:,mask],axis=-1)-logsumexp(v[:,~mask],axis=-1)
def lp(x):return x-logsumexp(x,axis=-1,keepdims=True)
def kl(target,log_student):
 p=target.astype(float);term=np.zeros_like(p);positive=p>0;term[positive]=p[positive]*(np.log(p[positive])-log_student[positive]);return float(term.sum(-1).mean())
def objective(ps,zs,pT,p0,target,state,source,usez):
 native=lp(ps.astype(float)/.25);a=kl(pT,lp((zs if usez else ps).astype(float)/.25));b=kl(p0,native);other=ps.astype(float).copy();other[np.arange(len(other)),target]=-np.inf;c=float(np.maximum(.05+other.max(-1)-ps[np.arange(len(ps)),target],0).mean());d=sum(float(np.mean((state[k].astype(float)-source[k].astype(float))**2)) for k in state);return {'teacher_KL':a,'source_KL_native':b,'hinge_native':c,'parameter_anchor':d,'total':a+.2*b+.5*c+d}
def close(a,b,tol):assert abs(float(a)-float(b))<=tol,(a,b,tol)
def key(z):
 t=z['tokens'][z['token_mask']].mean(0);t=t/max(float(np.linalg.norm(t)),1e-12);return z['query']*t
def audit():
 start=time.perf_counter();HERE.mkdir(exist_ok=True);plan=read(EXP/'PLAN_LOCK.json');auth=read(EXP/'ROOT_EXECUTION_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_PAIRED_DENSE_COORDINATE_DIAGNOSTIC';assert plan['root_authorization_sha256']==sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json')
 for p,h in dict(plan['program_sha256'],**plan['input_sha256']).items():assert sha(ROOT/p)==h,p
 seal=verify(BASE);assert seal['states']==8 and seal['optimizer_steps']==128 and seal['n']==272;assert seal['plan_sha256']==sha(EXP/'PLAN_LOCK.json');rr=allrows();assert len(rr)==272 and sum(r['split']=='D208' for r in rr)==208
 cs=read(CAP/'SEALED.json');bs=read(BANK/'SEALED.json');zs={};preds={}
 for r in rr:
  f='samples_PRIVATE/'+r['id']+'.npz';assert sha(CAP/f)==cs['files'][f] and sha(BANK/f)==bs['files'][f];zs[r['id']]=feature(r);preds[r['id']]=load(BASE/'predictions_PRIVATE'/(r['id']+'.npz'))
 assert sha(CAP/'projection_source_PRIVATE.npz')==cs['files']['projection_source_PRIVATE.npz'];source0=load(CAP/'projection_source_PRIVATE.npz');source={k.removeprefix('image.'):v for k,v in source0.items() if k.startswith('image.')};assert sum(v.size for v in source.values())==184960
 fit=read(BASE/'FIT_RECEIPTS.json');assert len(fit)==8;control=read(BASE/'MATCHED_CONTROL_REPLAY.json');assert control['state']=='PASSED_BOTH_PLATFORMS_BEFORE_OTHER_STATES';assert [r['configuration'] for r in fit[:2]]==['full_probability']*2
 maxp=maxz=0.;replay=[];lossrows=[];cpudiffs=[];initial={};lossmax=0.;controls=[]
 for pl in ['drone','quad']:
  A=[r for r in rr if r['split']=='D208' and r['platform']==pl];V=[r for r in rr if r['split']=='V64' and r['platform']==pl];assert (len(A),len(V))==(104,32);inp=load(BASE/'input_contract_PRIVATE'/(pl+'.npz'));assert inp['A_ids'].tolist()==[r['id'] for r in A] and inp['V_ids'].tolist()==[r['id'] for r in V]
  for label,rs in [('A',A),('V',V)]:
   T=max(len(zs[r['id']]['tokens']) for r in rs);pad=np.ones((len(rs),T),bool);mask=np.zeros_like(pad)
   for i,r in enumerate(rs):n=len(zs[r['id']]['tokens']);pad[i,:n]=False;mask[i,:n]=zs[r['id']]['token_mask']
   assert np.array_equal(pad,inp[label+'_padding']) and np.array_equal(mask,inp[label+'_mask'])
  F32=np.stack([teacher(r,zs[r['id']]) for r in A]).astype(np.float32);assert np.array_equal(F32,inp['teacher_F32']);s0=np.stack([zs[r['id']]['scores'] for r in A]);assert np.array_equal(s0,inp['source_scores']);pT=np.exp(lp(F32.astype(float)/.25));p0=np.exp(lp(s0.astype(float)/.25));assert abs(pT-inp['teacher_pT']).max()<2e-6 and abs(p0-inp['source_p0']).max()<2e-6
  ip=[];iz=[]
  for r in A:
   p,z=head(zs[r['id']],source);ip.append(p);iz.append(z)
  initial[pl]=(np.stack(ip),np.stack(iz))
  for j,name in enumerate(CONFIGS):
   state=load(BASE/'states_PRIVATE'/(pl+'_'+name+'.npz'));assert state.keys()==source.keys() and all(state[k].shape==source[k].shape for k in state);assert sum(v.size for v in state.values())==184960;rec=next(t for t in fit if t['platform']==pl and t['configuration']==name);assert rec['steps']==16 and rec['params']==184960 and rec['parameter_delta_L2']>0
   AP=[];AZ=[];GP=[];GZ=[]
   for r in A+V:
    z=zs[r['id']];pred=preds[r['id']];p,ell=head(z,state);saved=pred['scores'][ARMS.index(name)];sz=pred['trained_token_logodds'][j];pe=float(abs(p-saved).max());ze=float(abs(ell-sz).max());maxp=max(maxp,pe);maxz=max(maxz,ze);assert pe<2e-5 and ze<5e-4,(r['alias'],name,pe,ze)
    replay.append({'alias':r['alias'],'platform':pl,'split':r['split'],'configuration':name,'CPU64_probability_maxerror':pe,'CPU64_z_maxerror':ze})
    if int(p.argmax())!=int(saved.argmax()):cpudiffs.append({'alias':r['alias'],'configuration':name,'CPU64_query':int(p.argmax()),'sealed_GPU_query':int(saved.argmax()),'probability_maxerror':pe,'not_replaced':True})
    if r['split']=='D208':AP.append(p);AZ.append(ell);GP.append(saved);GZ.append(sz)
   target=(s0 if name.startswith('null_') else F32).argmax(-1);tgt=inp['source_p0'] if name.startswith('null_') else inp['teacher_pT'];usez=name.endswith('token_logodds');final=objective(np.stack(GP),np.stack(GZ),tgt,inp['source_p0'],target,state,source,usez);fresh=objective(np.stack(AP),np.stack(AZ),tgt,inp['source_p0'],target,state,source,usez)
   first=objective(*initial[pl],tgt,inp['source_p0'],target,source,source,usez)
   for k in final:
    err=abs(final[k]-rec['final_terms_after_step16'][k]);lossmax=max(lossmax,err);assert err<5e-4,(pl,name,k,final[k],rec['final_terms_after_step16'][k]);close(fresh[k],final[k],1e-3);close(first[k],rec['trace'][0][k],1e-3)
   delta=np.sqrt(sum(float(((state[k].astype(float)-source[k].astype(float))**2).sum()) for k in state));close(delta,rec['parameter_delta_L2'],2e-5);lossrows.append({'platform':pl,'configuration':name,'CPU_from_saved_GPU_vectors':final,'CPU64_fresh_head':fresh,'CPU64_source_initial':first,'author_GPU_poststep16':rec['final_terms_after_step16']})
   if name=='full_probability':
    old=load(ROOT/'crossplatform/round24/screen_attempt001/states_PRIVATE'/(pl+'_query_mlp_lr1_s16.npz'));err=max(float(abs(old[k]-state[k]).max()) for k in state);assert err<=1e-6;matches=0
    for r in A+V:
     op=load(ROOT/'crossplatform/round24/screen_attempt001'/(r['split']+'_predictions/predictions_PRIVATE')/(r['id']+'.npz'));ix=op['arm_names'].tolist().index('query_mlp_lr1_s16');assert int(preds[r['id']]['queries'][1])==int(op['query_indices'][ix]);matches+=1
    controls.append({'platform':pl,'parameter_maxerror':err,'parameter_bitexact':all(np.array_equal(old[k],state[k]) for k in state),'A_V_exactquerymatches':matches})
 for r in rr:
  z=zs[r['id']];p=preds[r['id']];g=teacher(r,z);assert p['arm_names'].tolist()==ARMS and p['scores'].shape==(9,256);assert np.array_equal(p['queries'],p['scores'].argmax(-1));assert np.array_equal(p['boxes'],z['boxes'][p['queries']]);assert np.array_equal(p['teacher_F64'],g) and np.array_equal(p['teacher_F32'],g.astype(np.float32));assert np.array_equal(p['deltaG'],g-z['scores'].astype(float));assert np.array_equal(p['scores'][0],z['scores']) and np.array_equal(p['scores'][5],g) and np.array_equal(p['scores'][8],z['scores']);assert np.array_equal(p['scores'][6],p['scores'][1]+p['deltaG']) and np.array_equal(p['scores'][7],p['scores'][2]+p['deltaG'])
 # Independent literal CPU32 old memory arithmetic; no new feedback/parameters.
 import torch
 torch.set_num_threads(4);assert not torch.cuda.is_initialized();memory=[];md=HERE/'MEMORY_REPLAY_PRIVATE';md.mkdir(exist_ok=True)
 for pl in ['drone','quad']:
  A=[r for r in rr if r['split']=='D208' and r['platform']==pl];V=[r for r in rr if r['split']=='V64' and r['platform']==pl];keys=np.stack([key(zs[r['id']]) for r in A]);keys/=np.maximum(np.linalg.norm(keys,axis=-1,keepdims=True),1e-12);values=np.stack([teacher(r,zs[r['id']])-zs[r['id']]['scores'] for r in A]).astype(np.float32);K=torch.tensor(keys);val=torch.tensor(values)
  for r in V:
   q=key(zs[r['id']]);q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12);sim=torch.einsum('qc,anc->qan',torch.tensor(q),K);near,which=sim.max(-1);best,obs=near.topk(4,dim=-1);pos=which.gather(1,obs);w=((best-1)/.1).exp();corr=(val[obs,pos]*w).sum(-1)/w.sum(-1).clamp_min(1e-12);scores=zs[r['id']]['scores']+corr.numpy();chosen=int(scores.argmax());p=preds[r['id']];op=ROOT/str(p['legacy_fullG_memory_original_file'].item());assert sha(op)==str(p['legacy_fullG_memory_original_sha256'].item());old=load(op);i=old['arm_names'].tolist().index('memory_a1.0');assert chosen==int(p['legacy_fullG_memory_query'])==int(old['query_indices'][i]),('memoryquery',r['alias'],chosen,int(p['legacy_fullG_memory_query']));assert np.array_equal(p['legacy_fullG_memory_box'],zs[r['id']]['boxes'][chosen]);np.savez_compressed(md/(r['id']+'.npz'),scores=scores,query=chosen);memory.append({'alias':r['alias'],'platform':pl,'query':chosen,'same_historical_query':True,'alpha':1.,'CPU32_original_algorithm_replayed':True})
 assert not torch.cuda.is_initialized();dump(HERE/'HEAD_REPLAY_ROWS.json',replay);dump(HERE/'LOSS_REPLAY.json',lossrows);dump(HERE/'MEMORY_REPLAY_ROWS.json',memory);dump(HERE/'CONTROL_REPLAY.json',controls)
 result={'state':'PASS_INDEPENDENT_R43_ARRAY_LOSS_CONTROL_MEMORY_AUDIT','states':8,'optimizer_steps_reviewed_not_rerun':128,'parameters_per_state':184960,'A208_V64':272,'trained_full_head_vectors_replayed':1088,'all_original_feature_bank_input_bytes_verified':True,'native_probability_maxerror':maxp,'token_logodds_maxerror':maxz,'poststep16_loss_from_GPU_vectors_maxerror':lossmax,'CPU64_argmax_differences_not_replaced':cpudiffs,'matched_R24_control':controls,'all_native_queries_frozenboxes_exact':2448,'sameinfo_memory_V64_exactqueries':len(memory),'all_zero_delta_memory_exactsource':True,'no_GT_container_GPU_model_optimizer_FINAL_U_T_reads_or_calls':True,'prediction_seal_sha256':sha(BASE/'SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(HERE/'AUDIT_RESULT.json',result);print(json.dumps(result))
def score():
 audit=read(HERE/'AUDIT_RESULT.json');assert audit['state']=='PASS_INDEPENDENT_R43_ARRAY_LOSS_CONTROL_MEMORY_AUDIT' and audit['prediction_seal_sha256']==sha(BASE/'SEALED.json') and audit['program_sha256']==sha(__file__);verify(BASE)
 sys.path.insert(0,str(ROOT/'research/round5/audit'));from independent_core_v2 import candidate_ious,transform_gt
 rr=allrows();index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};task=[]
 for r in rr:
  p=load(BASE/'predictions_PRIVATE'/(r['id']+'.npz'));z=feature(r);names=ARMS.copy();q=p['queries'].tolist()
  if r['split']=='V64':names+=['legacy_fullG_memory'];q.append(int(p['legacy_fullG_memory_query']))
  ix=index[r['id']];path=ix['meta_path'];assert sha(path)==ix['metadata_container_sha256']
  if path not in containers:containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));boxes=z['boxes'][q].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);values=candidate_ious(boxes,gt)[0];task.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'split':r['split'],'scene_PRIVATE':r['scene'],'queries':dict(zip(names,map(int,q))),'IoU':dict(zip(names,map(float,values)))})
 summary={};neg=[];scenes=[];contrasts={}
 for split in ['D208','V64']:
  group=[r for r in task if r['split']==split];names=list(group[0]['IoU']);ss={}
  for pl in ['drone','quad']:
   part=[r for r in group if r['platform']==pl];ss[pl]={}
   for arm in names:
    v=np.array([r['IoU'][arm] for r in part]);d=v-np.array([r['IoU']['source'] for r in part]);ss[pl][arm]={'n':len(part),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in part),'scene_clusters':len({r['scene_PRIVATE'] for r in part})}
   for j,scene in enumerate(sorted({r['scene_PRIVATE'] for r in part})):
    sr=[r for r in part if r['scene_PRIVATE']==scene];scenes.append({'split':split,'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(sr),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in sr])*100) for a in names}})
  ss['balanced_mIoU_percent']={a:float(np.mean([ss[p][a]['mIoU_percent'] for p in ['drone','quad']])) for a in names};summary[split]=ss
  cc={}
  for pl in ['drone','quad','balanced']:
   m=ss['balanced_mIoU_percent'] if pl=='balanced' else {a:ss[pl][a]['mIoU_percent'] for a in names};cc[pl]={'z_minus_probability_full_pp':m['full_token_logodds']-m['full_probability'],'z_minus_probability_null_pp':m['null_token_logodds']-m['null_probability'],'coordinate_by_G_interaction_pp':(m['full_token_logodds']-m['full_probability'])-(m['null_token_logodds']-m['null_probability']),'G_full_minus_null_probability_pp':m['full_probability']-m['null_probability'],'G_full_minus_null_z_pp':m['full_token_logodds']-m['null_token_logodds'],'zfull_plus_currentG_vs_directG_pp':m['full_token_logodds_plus_current_G']-m['direct_G']}
  contrasts[split]=cc
  refs=['source','full_probability','full_token_logodds','null_probability','null_token_logodds','direct_G']+(['legacy_fullG_memory'] if split=='V64' else [])
  for r in group:
   for a in names:
    for ref in refs:
     d=r['IoU'][a]-r['IoU'][ref]
     if a!=ref and d< -1e-10:neg.append({'alias':r['alias'],'platform':r['platform'],'split':split,'arm':a,'reference':ref,'delta_pp':float(d*100)})
 # Original R24 positive control has independent original sealed task rows.
 comparisons=0
 for split,filename in [('D208','D208_LAYER_ROWS_PRIVATE.json'),('V64','V64_LAYER_ROWS_PRIVATE.json')]:
  old={r['id_PRIVATE']:r for r in read(ROOT/'research/round24/experiment'/filename)}
  for r in [x for x in task if x['split']==split]:close(r['IoU']['full_probability'],old[r['id_PRIVATE']]['ious']['query_mlp_lr1_s16'],1e-12);comparisons+=1
 dump(EXP/'ALL_ROWS_PRIVATE.json',task);dump(EXP/'TASK_SUMMARY.json',summary);dump(EXP/'FACTOR_CONTRASTS.json',contrasts);dump(EXP/'ALL_NEGATIVES.json',neg);dump(EXP/'SCENE_ROWS.json',scenes)
 result={'state':'PASS_INDEPENDENT_R43_POSTSEAL_TASK_SCORING','rows':len(task),'taskvalues':sum(len(r['IoU']) for r in task),'all_negative_comparisons':len(neg),'original_R24_control_taskvalues_exact':comparisons,'prediction_seal_sha256':sha(BASE/'SEALED.json'),'CPU_audit_sha256':sha(HERE/'AUDIT_RESULT.json'),'new_G_or_update_from_labels':False,'FINAL_U_T_untouched':True,'fullgrounderclosure_not_performed':True};dump(HERE/'TASK_AUDIT_RESULT.json',result);print(json.dumps(result));print(json.dumps(contrasts))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','score']);a=p.parse_args();audit() if a.phase=='audit' else score()
