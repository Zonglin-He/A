"""Independent R34 recorded-neighbor replay and postseal geometry. CPU only."""
import sys,time,json,hashlib,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'crossplatform/round32'),str(ROOT/'research/round32/independent_audit'),str(ROOT/'research/round5/audit')]
from common32 import read,dump,sha,verify,load_npz,rows,feature
from audit32_independent import independent_qp,qr
from independent_core_v2 import transform_gt,candidate_ious
OUT=Path(__file__).resolve().parent;BASE=ROOT/'crossplatform/round34';EXP=ROOT/'research/round34/experiment';PRED=BASE/'scope_diagnostic_attempt001'
ARMS=['source','R30_ungated','h_scoped_R30','noun_scoped_R30','h_samegate_QP','noun_samegate_QP','h_global_mean','h_global_norm','noun_global_mean','noun_global_norm']
KEYS=['h288','noun64']
def close(a,b,t=1e-10):assert abs(float(a)-float(b))<t,(a,b,t)
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def key(z,k):
 a=z['h'] if k=='h288' else z['query']*norm(z['tokens'][z['token_mask']].mean(0))
 return norm(a).astype(np.float32)
def relative(p,q0):
 p=p.astype(float);l=np.log(p+1e-6)-np.log(1+1e-6-p);return l-l[q0]

def main():
 start=time.perf_counter();verify(PRED);plan=read(EXP/'PLAN_LOCK.json');amend=read(EXP/'EXECUTION_SCOPE_AMENDMENT_LOCK.json');assert amend['original_plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 prog=dict(plan['program_sha256']);prog.update(amend['new_program_sha256'])
 for p,s in {**prog,**plan['immutable_input_sha256']}.items():assert sha(ROOT/p)==s,p
 rr=[dict(r,split=s) for s in ['A208','DEV64','R64','S64','T128'] for r in rows(s)];assert len(rr)==528 and all(r['stage'] not in ['U','F'] for r in rr);lookup={r['id']:r for r in rr};zmap={r['id']:feature(r) for r in rr}
 fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');assert len(fb)==22 and len({f['id_PRIVATE'] for f in fb})==19;libs={};qp=[]
 for pl in ['drone','quad']:
  A=[r for r in rr if r['platform']==pl and r['split']=='A208'];assert len(A)==104;ep=np.zeros((104,256),np.float32);raw=np.zeros((104,256),float)
  for j,r in enumerate(A):
   ff=[f for f in fb if f['id_PRIVATE']==r['id']]
   for f in ff:ep[j,f['positive_group_PRIVATE']+f['negative_group_PRIVATE']]=1
   if ff:
    p=zmap[r['id']]['scores'];s=relative(p,int(p.argmax()));v,obj,d,C,gs,dual=independent_qp(s,ff,p);raw[j]=v
    qp.append({'alias':r['alias'],'platform':pl,'constraints':len(ff),'objective':obj,'KKT_stationarity':float(abs(d-C.T@dual).max()),'KKT_min_dual':float(dual.min()),'KKT_min_slack':float((C@(gs+d)-.1).min()),'KKT_complementarity':float(abs(dual*(C@(gs+d)-.1)).max())})
  for k in KEYS:
   lib=load_npz(PRED/'libraries_PRIVATE'/(pl+'_'+k+'.npz'));libs[pl,k]=lib;assert lib['A_ids'].tolist()==[r['id'] for r in A] and np.array_equal(ep,lib['endpoint']);assert np.array_equal(np.stack([key(zmap[r['id']],k) for r in A]),lib['keys']);assert abs(raw-lib['raw']).max()<1e-8
 assert len(qp)==19
 count=0;scoreerr=coserr=memoryerr=0.;preds={};taskrows=[];oldexact=0;oldendpointq=0;oldvec=0;oldcomp=[];gate_stats=[]
 parent={}
 for pl in ['drone','quad']:
  p=load_npz(ROOT/'crossplatform/round31/loo19_attempt002/predictions_PRIVATE'/(pl+'_all.npz'));parent.update({rid:p['scores'][j] for j,rid in enumerate(p['row_ids_PRIVATE'].tolist())})
 idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};original={x['id_PRIVATE']:x for x in read(EXP/'diagnostic_ALL_ROWS_PRIVATE.json')}
 for r in rr:
  z=zmap[r['id']];rec=load_npz(PRED/'predictions_PRIVATE'/(r['id']+'.npz'));preds[r['id']]=rec;assert rec['arm_names'].tolist()==ARMS;assert np.array_equal(rec['source_probability'],z['scores']);p0=rec['source_probability'];pT=rec['R30_probability'];q0=int(p0.argmax());assert int(rec['source_q0'])==q0
  if r['stage']=='T':old=load_npz(ROOT/'crossplatform/round32/selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(pT,old['adapted_probability'])
  else:assert np.array_equal(pT,parent[r['id']])
  d0=relative(p0,q0);dT=relative(pT,q0);delta=dT-d0;vectors={'source':d0,'R30_ungated':dT}
  for k in KEYS:
   lib=libs[r['platform'],k];o=rec[k+'_obs'];p=rec[k+'_pos'];sim=rec[k+'_similarity'];w=rec[k+'_weights'];assert o.shape==p.shape==sim.shape==w.shape==(256,4) and np.all(np.diff(sim,axis=1)<=0) and all(len(set(x))==4 for x in o)
   assert o.min()>=0 and o.max()<104 and p.min()>=0 and p.max()<256
   cpu=np.einsum('qc,qkc->qk',key(z,k).astype(float),lib['keys'][o,p].astype(float));err=float(abs(cpu-sim).max());coserr=max(coserr,err);assert err<3e-6
   # The stored float32 GPU topk is the inference contract; do not substitute a CPU NN ordering.
   assert np.all(w>0) and np.allclose(w,np.exp((sim.astype(float)-1)/.1),rtol=2e-6,atol=1e-14)
   close(float(rec[k+'_nearest_A_row_margin'].min()),float((sim[:,0]-sim[:,1]).min()),1e-7)
   oo=o[:,:1];pp=p[:,:1];ww=w[:,:1];g=lib['endpoint'][oo,pp].max(1).astype(float);raw=(lib['raw'][oo,pp]*ww).sum(1)/np.maximum(ww.sum(1),1e-12);md=raw-raw[q0];local=d0+g*delta;local[g==0]=d0[g==0];local[g==1]=dT[g==1]
   assert np.array_equal(g,rec[k+'_gate']);memoryerr=max(memoryerr,float(abs(raw-rec[k+'_raw_retrieved']).max()),float(abs(md-rec[k+'_relative_memory']).max()));assert memoryerr<1e-12
   prefix='h_' if k=='h288' else 'noun_';vectors.update({prefix+'scoped_R30':local,prefix+'samegate_QP':d0+g*md,prefix+'global_mean':d0+float(g.mean())*delta,prefix+'global_norm':d0+float(np.linalg.norm(g*delta)/max(np.linalg.norm(delta),1e-12))*delta})
   gate_stats.append({'alias':r['alias'],'platform':r['platform'],'split':r['split'],'key':k,'gate_fraction':float(g.mean()),'q0_gate':float(g[q0]),'NN_exact_tie_queries':int((rec[k+'_nearest_A_row_exact_tie_count']>1).sum())})
  mat=np.stack([vectors[a] for a in ARMS]);scoreerr=max(scoreerr,float(abs(mat-rec['complete_scores']).max()));assert scoreerr<1e-12
  qs=np.array([qr(r,s,z) for s in mat]);assert np.array_equal(qs,rec['queries']);assert np.array_equal(rec['boxes'],z['boxes'][qs]);count+=10
  oldpath=ROOT/'crossplatform/round32/scope400_endpoint_attempt001/predictions_PRIVATE'/(r['id']+'_noun_interaction.npz') if r['stage']!='T' else ROOT/'crossplatform/round32/selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz');old=load_npz(oldpath);names=old['arm_names'].tolist()
  mappings={'source':'source','R30_ungated':'R30_ungated','noun_scoped_R30':'scoped_TTA','noun_samegate_QP':'same_gate_QP_memory','noun_global_mean':'global_mean_matched','noun_global_norm':'global_norm_matched'} if r['stage']=='T' else {'source':'source','R30_ungated':'R30_ungated','noun_scoped_R30':'k1_scope_TTA','noun_samegate_QP':'k1_scope_memory','noun_global_mean':'k1_global_mean_TTA','noun_global_norm':'k1_global_norm_TTA'}
  for new,oldname in mappings.items():
   a=ARMS.index(new);j=names.index(oldname);assert int(qs[a])==int(old['queries'][j]),(r['alias'],new,int(qs[a]),int(old['queries'][j]));assert np.array_equal(rec['boxes'][a],old['boxes'][j]);oldendpointq+=1
  oldexact+=1
  if r['stage']=='T':
   assert np.array_equal(vectors['noun_scoped_R30'],old['primary_scores']) and np.array_equal(vectors['noun_samegate_QP'],old['memory_scores']);oldvec+=2
  ix=idx[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));b=rec['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);values=candidate_ious(b,gt)[0];ref=original[r['id']]
  for a,q,v in zip(ARMS,qs,values):assert int(q)==ref['queries'][a];close(v,ref['IoU'][a],1e-12)
  taskrows.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'split':r['split'],'queries':dict(zip(ARMS,map(int,qs))),'IoU':dict(zip(ARMS,map(float,values)))})
 summaries=read(EXP/'diagnostic_SUMMARY.json');aggs=[];neg=[];scenes=[]
 for split in ['A208','DEV64','R64','S64','T128']:
  rs=[r for r in taskrows if r['split']==split]
  for pl in ['drone','quad']:
   part=[r for r in rs if r['platform']==pl]
   for arm in ARMS:
    v=np.array([r['IoU'][arm] for r in part]);d=v-np.array([r['IoU']['source'] for r in part]);agg={'n':len(part),'mIoU_percent':100*float(v.mean()),'Acc25_percent':100*float((v>.25).mean()),'Acc50_percent':100*float((v>.5).mean()),'delta_vs_source_pp':100*float(d.mean()),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in part),'scene_clusters':len({r['scene_PRIVATE'] for r in part})}
    for k,vv in agg.items():close(vv,summaries[split][pl][arm][k],1e-10)
    aggs.append({'split':split,'platform':pl,'arm':arm,**agg})
   for r in part:
    for arm in ARMS:
     for reference in ['source','R30_ungated','h_samegate_QP','noun_samegate_QP']:
      d=r['IoU'][arm]-r['IoU'][reference]
      if d< -1e-10:neg.append({'alias':r['alias'],'platform':pl,'split':split,'arm':arm,'reference':reference,'delta_pp':100*d})
   for j,scene in enumerate(sorted({r['scene_PRIVATE'] for r in part})):
    ss=[r for r in part if r['scene_PRIVATE']==scene];scenes.append({'split':split,'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{arm:100*float(np.mean([r['IoU'][arm] for r in ss])) for arm in ARMS}})
  for arm in ARMS:close(summaries[split]['balanced_mIoU_percent'][arm],np.mean([x['mIoU_percent'] for x in aggs if x['split']==split and x['arm']==arm]),1e-10)
 assert neg==read(EXP/'diagnostic_ALL_NEGATIVES.json');assert scenes==read(EXP/'diagnostic_SCENE_ROWS.json')
 # Feedback is only partial; evaluate its direction and final group separately.
 pref=[];feas=[]
 for arm in ARMS:
  for f in fb:
   r=lookup[f['id_PRIVATE']];p=preds[r['id']];s=p['complete_scores'][ARMS.index(arm)];gap=float(s[f['positive_group_PRIVATE']].max()-s[f['negative_group_PRIVATE']].max());q=int(p['queries'][ARMS.index(arm)]);pref.append({'alias':r['alias'],'pair_alias':f['pair_alias'],'platform':r['platform'],'arm':arm,'label':f['label'],'gap':gap,'direction':gap>0,'final_positive':q in f['positive_group_PRIVATE']})
  pp=[r for r in pref if r['arm']==arm and r['label']=='B'];directions=sum(r['direction'] for r in pp);final=sum(r['final_positive'] for r in pp);feas.append({'arm':arm,'B_directions':directions,'B_final_positive':final,'B_total':len(pp),'feasible':directions>=5 and final>=4})
 # Review closure artifact consistency; GPU forwards are not rerun here.
 closure=BASE/'closure_diagnostic_attempt001';verify(closure);cr=read(closure/'CLOSURE_AUDIT.json');assert cr['state']=='PASSED' and cr['observations']==10 and cr['full_model_forwards']==20 and cr['prediction_seal_sha256']==sha(PRED/'SEALED.json')
 selected=[]
 for split in ['A208','DEV64','R64','S64','T128']:
  for pl in ['drone','quad']:
   pool=[r for r in rr if r['split']==split and r['platform']==pl]
   ih=lambda r:hashlib.sha256(json.dumps({k:v for k,v in r.items() if k not in ('alias','frame_alias','stage','split')},sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
   r=min(pool,key=lambda r:(ih(r),r['id']));selected.append(r['alias']);c=load_npz(closure/'observations_PRIVATE'/(r['id']+'.npz'));p=preds[r['id']];assert np.array_equal(c['queries'],p['queries']) and np.array_equal(c['boxes'],p['boxes'])
 assert set(selected)=={r['alias'] for r in cr['rows']} and cr['source_restored_bitexact'] and cr['nonmatching_parameters_unchanged'] and cr['all_GPU_neighbors_and_final_queries_reproduced']
 dump(OUT/'QP19_INDEPENDENT.json',qp);dump(OUT/'ALL528_TASK_ROWS_PRIVATE.json',taskrows);dump(OUT/'ALL_NEGATIVES.json',neg);dump(OUT/'AGGREGATES.json',aggs);dump(OUT/'ALL220_PREFERENCE_EXECUTION.json',pref);dump(OUT/'FEASIBILITY.json',feas);dump(OUT/'GATE_STATS.json',gate_stats)
 result={'state':'PASS_RECORDED_GPU_NEIGHBOR_CONTRACT','observations':528,'recomputed_score_vectors':count,'selected_task_values':count,'independent_QP_libraries':19,'source_keys_and_endpoint_flags_exact':True,'saved_neighbor_selected_cosine_maxerror':coserr,'score_vector_maxerror':scoreerr,'retrieved_memory_maxerror':memoryerr,'original_R32_noun_all528_choices_and_boxes_exact':True,'original_R32_noun_reference_choices_verified':oldendpointq,'T128_original_noun_complete_vectors_bitexact':oldvec,'all_platform_arm_metrics_and_scene_rows_exact':True,'negative_comparisons_retained':len(neg),'feedback_execution_rows':len(pref),'closure_hash_selected_rows':len(selected),'closure_forward_receipt':20,'U_FINAL_not_read':True,'no_GPU_training_or_autograd':True,'limitations':['Recorded GPU NN indices are validated against source key cosine and replayed, not replaced by a new CPU NN topk. Full topk optimality is supported by reviewed10-rowGPUclosure, not independently rerun on528rows.','GPU full-model closure covers10 prespecified hash-selected observations; all528 cached score/readout/task records audited.','R/S/T are exposed historical diagnostics; no newU confirmation in this audit, and no final adoption decision.','The one-sceneDrone diagnostics do not support betweenDrone-scene generalization or a conventional scene-bootstrap significance claim.'],'seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
