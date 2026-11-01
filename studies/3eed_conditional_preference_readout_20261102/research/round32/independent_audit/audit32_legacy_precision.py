"""CPU-only literal legacy replay and fixed float64 numerical control.
All posthoc; never alters primary or original sealed GPU predictions.
"""
import sys,json,numpy as np,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'crossplatform/round32'),str(ROOT/'research/round5/audit')]
from common32 import read,dump,sha,seal,verify,load_npz,rows,feature
from independent_core_v2 import candidate_ious,transform_gt
import torch
OUT=ROOT/'research/round32/independent_audit';torch.set_num_threads(4)
def interact(z,dtype):
 t=z['tokens'][z['token_mask']].astype(dtype).mean(0);t=t/max(float(np.linalg.norm(t)),1e-12);return z['query'].astype(dtype)*t
def make_lib(A,feedback,dtype):
 keys=np.stack([interact(feature(r),dtype) for r in A]);keys/=np.maximum(np.linalg.norm(keys,axis=-1,keepdims=True),1e-12);values=np.zeros((len(A),256),dtype);counts=np.zeros_like(values);lookup={r['id']:i for i,r in enumerate(A)};roles=collections.defaultdict(list)
 for f in feedback:
  if f['id_PRIVATE'] not in lookup:continue
  j=lookup[f['id_PRIVATE']];s=feature(A[j])['scores'].astype(dtype);P=f['positive_group_PRIVATE'];N=f['negative_group_PRIVATE'];delta=max(.1-float(s[P].max()-s[N].max()),0)/2;values[j,P]+=delta;values[j,N]-=delta;counts[j,P]+=1;counts[j,N]+=1
  for q in P:roles[j,q].append(f['pair_alias']+':P')
  for q in N:roles[j,q].append(f['pair_alias']+':N')
 values/=np.maximum(counts,1);active=np.array(sorted({lookup[f['id_PRIVATE']] for f in feedback if f['id_PRIVATE'] in lookup}));return keys,values,active,roles

def replay(z,lib,ix,kind):
 keys,values,active,roles=lib;q=interact(z,keys.dtype);q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
 if kind=='torchCPU32':
  with torch.inference_mode():
   sim=torch.einsum('qc,anc->qan',torch.tensor(q),torch.tensor(keys[ix]));near,which=sim.max(-1);best,obs=near.topk(4,dim=-1);pos=which.gather(1,obs);w=((best-1)/.1).exp();v=torch.tensor(values[ix])[obs,pos];correction=((v*w).sum(-1)/w.sum(-1).clamp_min(1e-12)).numpy();near=near.numpy();which=which.numpy();obs=obs.numpy();best=best.numpy();pos=pos.numpy();sim=sim.numpy()
 else:
  sim=(q@keys[ix].reshape(-1,64).T).reshape(256,len(ix),256);which=sim.argmax(-1);near=np.take_along_axis(sim,which[:,:,None],-1)[:,:,0];obs=np.argsort(-near,axis=-1,kind='stable')[:,:4];best=np.take_along_axis(near,obs,-1);pos=np.take_along_axis(which,obs,-1);w=np.exp((best-1)/.1);v=values[ix][obs,pos];correction=(v*w).sum(-1)/np.maximum(w.sum(-1),1e-12)
 score=z['scores'].astype(keys.dtype)+correction;return {'scores':score,'q':int(score.argmax()),'sim':sim,'near':near,'which':which,'obs':ix[obs],'local_obs':obs,'pos':pos,'best':best}

def desc(result,query,A,roles):
 near=result['near'][query];ordered=np.sort(near)[::-1];out=[]
 for ai,pos,s in zip(result['obs'][query],result['pos'][query],result['best'][query]):out.append({'A_alias':A[int(ai)]['alias'],'nearest_A_query':int(pos),'cosine':float(s),'accepted_endpoint_roles':roles[int(ai),int(pos)]})
 sgaps=[]
 for loc in result['local_obs'][query]:
  values=np.sort(result['sim'][query,int(loc)])[::-1];sgaps.append(float(values[0]-values[1]))
 return {'nearest4':out,'first_second_observation_gap':float(ordered[0]-ordered[1]),'fourth_fifth_observation_gap':float(ordered[3]-ordered[4]) if len(ordered)>4 else None,'selected_observation_best_second_query_gaps':sgaps}

def main():
 start=time.perf_counter();verify(ROOT/'crossplatform/round32/selectedT128_attempt001');feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');out=OUT/'legacy_precision_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();diagnostics=[];allrows=[];mismatch=collections.Counter();maxgaps={};libs={};allpred={};T=rows('T128')
 for pl in ['drone','quad']:
  A=[r for r in rows() if r['platform']==pl];l32=make_lib(A,feedback,np.float32);l64=make_lib(A,feedback,np.float64)
  for r in [r for r in T if r['platform']==pl]:
   z=feature(r);original=load_npz(ROOT/'crossplatform/round32/selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));qdict={};scores64={}
   for arm,ix in [('R30_feedbackA_memory',l32[2]),('R30_allA_zeroU_memory',np.arange(len(A)))]:
    qgpu=int(original['queries'][original['arm_names'].tolist().index(arm)]);n32=replay(z,l32,ix,'numpy32');t32=replay(z,l32,ix,'torchCPU32');n64=replay(z,l64,ix,'numpy64');qdict[arm+'_float64']=n64['q'];scores64[arm+'_float64']=n64['scores'];rr={'alias':r['alias'],'platform':pl,'arm':arm,'original_GPU_q_PRIVATE':qgpu,'numpy32_q_PRIVATE':n32['q'],'torchCPU32_q_PRIVATE':t32['q'],'float64_q_PRIVATE':n64['q'],'numpy32_max_minus_GPUchosen':float(n32['scores'].max()-n32['scores'][qgpu]),'torchCPU32_max_minus_GPUchosen':float(t32['scores'].max()-t32['scores'][qgpu]),'float64_max_minus_GPUchosen':float(n64['scores'].max()-n64['scores'][qgpu])};allrows.append(rr)
    if n32['q']!=qgpu:mismatch['numpy32_vs_original_GPU']+=1
    if t32['q']!=qgpu:mismatch['literal_torchCPU32_vs_original_GPU']+=1
    if n64['q']!=qgpu:mismatch['fixed_float64_vs_original_GPU']+=1
    if n32['q']!=qgpu or t32['q']!=qgpu:
     roles=l32[3];d=dict(rr);d['neighbor_details']={name:{'GPUchosen_query_under_CPU_neighbors':desc(res,qgpu,A,roles),'CPUchosen_query':desc(res,res['q'],A,roles)} for name,res in [('numpy32',n32),('literal_torchCPU32',t32)]};cross=[]
     for qq in set([qgpu,n32['q'],t32['q']]):
      nr={int(a):tuple(roles[int(a),int(q)]) for a,q in zip(n32['obs'][qq],n32['pos'][qq])};tr={int(a):tuple(roles[int(a),int(q)]) for a,q in zip(t32['obs'][qq],t32['pos'][qq])};cross.append({'query_PRIVATE':int(qq),'CPU_numpy_Torch_nearest_observation_set_differs':set(nr)!=set(tr),'shared_A_endpoint_roles_differ':any(nr[a]!=tr[a] for a in nr.keys()&tr.keys()),'numpy_role_sets':list(nr.values()),'torch_role_sets':list(tr.values())})
     d['CPU_method_role_boundary_checks']=cross;d['actual_GPU_neighbor_indices_not_saved_cannot_certify_GPU_role_switch']=True;diagnostics.append(d)
   np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(qdict)),queries=np.array(list(qdict.values())),**{k+'_scores':v for k,v in scores64.items()});allpred[r['id']]=qdict
   if len(allpred)%32==0:print('LEGACY_PRECISION32',len(allpred),dict(mismatch),flush=True)
 dump(out/'PRECISION_COMPARISON_PRIVATE.json',allrows);dump(out/'NEIGHBOR_DIFFERENCES_PRIVATE.json',diagnostics);seal(out,'R32_posthoc_CPU_numerical_legacy_controls_before_current_CPU_taskscore',n=128,float64_family_fixed_no_selection=True,no_primary_or_sealed_GPU_outputs_changed=True,no_GPU_or_optimizer=True)
 # Postseal diagnostic scores for original and bothCPUprecision paths; no hidden selection.
 idx={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};scored=[];byalias={r['alias']:r for r in T};dlookup={(r['alias'],r['arm']):r for r in allrows}
 for r in T:
  i=idx[r['id']];p=i['meta_path']
  if p not in containers:assert sha(p)==i['metadata_container_sha256'];containers[p]=read(p)
  gt=transform_gt(containers[p]['ground_info'][i['object_index']]['bbox_3d'],r['platform'],i.get('waymo_view_suffix'));z=feature(r)
  for arm in ['R30_feedbackA_memory','R30_allA_zeroU_memory']:
   rr=dlookup[r['alias'],arm];labels=['original_GPU','numpy32','torchCPU32','float64'];qs=[rr[k+'_q_PRIVATE'] for k in labels];b=z['boxes'][qs].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);v=candidate_ious(b,gt)[0];scored.append({'alias':r['alias'],'platform':r['platform'],'arm':arm,'queries_PRIVATE':dict(zip(labels,qs)),'IoUs':dict(zip(labels,map(float,v)))})
 dump(OUT/'LEGACY_PRECISION_TASK_ROWS_PRIVATE.json',scored);detailed=[]
 for d in diagnostics:
  s=next(x for x in scored if x['alias']==d['alias'] and x['arm']==d['arm']);detailed.append(dict(d,IoUs=s['IoUs']))
 dump(OUT/'LEGACY_NEIGHBOR_PRECISION_DIAGNOSTICS_PRIVATE.json',detailed);summary={}
 for arm in ['R30_feedbackA_memory','R30_allA_zeroU_memory']:
  summary[arm]={}
  for pl in ['drone','quad']:
   rr=[x for x in scored if x['arm']==arm and x['platform']==pl];summary[arm][pl]={}
   for label in ['original_GPU','numpy32','torchCPU32','float64']:
    v=np.array([x['IoUs'][label] for x in rr]);orig=np.array([x['IoUs']['original_GPU'] for x in rr]);summary[arm][pl][label]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'task_changed_vs_originalGPU':int((abs(v-orig)>1e-12).sum()),'query_changed_vs_originalGPU':sum(x['queries_PRIVATE'][label]!=x['queries_PRIVATE']['original_GPU'] for x in rr),'gain_vs_originalGPU_pp':float((v-orig).mean()*100)}
 result={'state':'LEGACY_NUMERICAL_REPLAY_WARNING_POSTHOC_CONTROLS_SEALED','mismatches':dict(mismatch),'posthoc_task_summary':summary,'raw_GPU_neighbor_indices_not_logged':True,'no_primary_or_original_control_replaced':True,'float64control_prespecified_precision_path_not_perf_selected':True,'numeric_scope':'literal original numpykeyconstruction; NumPyBLAS32/TorchCPU32 differ in cosine and weightedfloat32rounding; floating ties in candidate scores can change selectedquery','seconds':time.perf_counter()-start};dump(OUT/'LEGACY_PRECISION_AUDIT_RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
