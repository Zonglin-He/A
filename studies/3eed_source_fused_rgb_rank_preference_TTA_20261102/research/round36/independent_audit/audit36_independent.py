"""Independent R36 sealed native-score, QP, head and task audit. CPU only."""
import os
for n in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[n]='4'
import sys,json,time,hashlib,collections,itertools
from pathlib import Path
import numpy as np
from scipy.special import erf
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'crossplatform/round32'),str(ROOT/'research/round32/independent_audit'),str(ROOT/'research/round5/audit')]
from common32 import read,dump,sha,verify,load_npz,rows,feature
from audit32_independent import independent_qp
from independent_core_v2 import candidate_ious,transform_gt
OUT=Path(__file__).resolve().parent;BASE=ROOT/'crossplatform/round36';EXP=ROOT/'research/round36/experiment';MODES=['rgb','no_rgb'];ARMS=['fused_frozen','rank_TTA','sameinfo_memory','XYZ_source','XYZ_R30']
def close(a,b,t=1e-9):assert abs(float(a)-float(b))<t,(a,b,t)
def norm(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def rank(x,s):
 h=x.astype(float)@s['0.weight'].astype(float).T+s['0.bias'];h=.5*h*(1+erf(h/np.sqrt(2)));return (h@s['2.weight'].astype(float).T+s['2.bias']).squeeze(-1)
def grouped(r,s,s0):
 if r['split']=='DEV64':return int(np.argmax(s))
 gg=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];used=set();options=[]
 for g in gg.values():
  rep=max(g,key=lambda q:(float(s0[q]),-q));used.update(g);options.append((float(s[g].max()),-rep,rep))
 options.extend((float(s[q]),-q,q) for q in range(256) if q not in used);return int(max(options)[2])
def main():
 start=time.perf_counter();plan=read(EXP/'PLAN_LOCK.json')
 for p,v in {**plan['program_sha256'],**plan['input_sha256']}.items():assert sha(ROOT/p)==v,p
 for folder in ['rgbDEV64_attempt001','capture272_attempt001','fit272_attempt001','closure4_attempt001']:verify(BASE/folder)
 selection=read(ROOT/'research/round25/experiment/FINAL_SOURCE_SELECTED_LOCK.json');assert selection['family']=='residual_rank' and selection['selection_source']=='WaymoVAL512 only' and not selection['target_scores_or_labels_used'];assert selection['primary_seed']==0
 rr=[dict(r,split=sp) for sp in ['A208','DEV64'] for r in rows(sp)];assert len(rr)==272 and all(r['stage'] not in ['U','F'] for r in rr)
 lookup={r['id']:r for r in rr};features={r['id']:feature(r) for r in rr};fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');assert len(fb)==22 and len({f['id_PRIVATE'] for f in fb})==19
 references={}
 for pl in ['drone','quad']:
  z=load_npz(ROOT/'crossplatform/round31/loo19_attempt002/predictions_PRIVATE'/(pl+'_all.npz'));references.update({rid:z['scores'][j] for j,rid in enumerate(z['row_ids_PRIVATE'].tolist())})
 index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};original={(r['id_PRIVATE'],r['mode']):r for r in read(EXP/'ALL_ROWS_PRIVATE.json')};receipts=read(BASE/'fit272_attempt001/FIT_RECEIPTS.json');assert len(receipts)==4
 taskrows=[];qp_rows=[];headrows=[];pairs=[];coserr=rankerr=sourceerr=0.;oldcount=0;oldmax=0.;queries64=[];rawgroups=[];sourcehash=[]
 for mode in MODES:
  config=next(c for c in selection['selected'] if c['seed']==0 and c['variant']==mode);assert config['epoch']==1
  path=ROOT/'crossplatform/round25'/config['training_folder']/'states_PRIVATE'/config['state_name'];assert sha(path)==config['state_sha256'];src={k.removeprefix('rank.'):v for k,v in load_npz(path).items() if k.startswith('rank.')};assert sum(v.size for v in src.values())==12929;sourcehash.append({'mode':mode,'source_state_sha256':sha(path),'selected_epoch':1,'selection':'source WaymoVAL512 only'})
  for pl in ['drone','quad']:
   A=[r for r in rr if r['split']=='A208' and r['platform']==pl];rs=[r for r in rr if r['platform']==pl];ff=[f for f in fb if f['platform']==pl];U=[j for j,r in enumerate(A) if not any(f['id_PRIVATE']==r['id'] for f in ff)];assert len(U)=={'drone':96,'quad':93}[pl] and len(ff)=={'drone':10,'quad':12}[pl]
   receipt=next(x for x in receipts if x['mode']==mode and x['platform']==pl);assert receipt['params']==12929 and receipt['steps']==16 and receipt['U']==len(U) and receipt['pref_denominator']==len(ff)
   state=load_npz(BASE/'fit272_attempt001/states_PRIVATE'/(pl+'_'+mode+'.npz'));assert {k:v.shape for k,v in state.items()}=={k:v.shape for k,v in src.items()};assert sum(v.size for v in state.values())==12929
   lib=load_npz(BASE/'fit272_attempt001/libraries_PRIVATE'/(pl+'_'+mode+'.npz'));assert lib['A_ids_PRIVATE'].tolist()==[r['id'] for r in A]
   caps={r['id']:load_npz(BASE/'capture272_attempt001/samples_PRIVATE'/(r['id']+'_'+mode+'.npz')) for r in rs};raw=np.zeros((104,256),float)
   assert np.array_equal(lib['keys'],np.stack([norm(caps[r['id']]['rankinput']).astype(np.float32) for r in A]))
   for j,r in enumerate(A):
    fff=[f for f in ff if f['id_PRIVATE']==r['id']]
    if fff:
     c=caps[r['id']];raw[j],obj,dd,C,gs,dual=independent_qp(c['native_scores'].astype(float),fff,c['native_scores']);qp_rows.append({'alias':r['alias'],'mode':mode,'platform':pl,'objective':obj,'constraints':len(fff),'group_count':len(dd),'max_complementarity':float(abs(dual*(C@(gs+dd)-.1)).max()),'minimum_slack':float((C@(gs+dd)-.1).min())})
   assert abs(raw-lib['raw']).max()<1e-9
   aa={};aorig={};aq={}
   for r in rs:
    z=features[r['id']];c=caps[r['id']];p=load_npz(BASE/'fit272_attempt001/predictions_PRIVATE'/(r['id']+'_'+mode+'.npz'));s0=c['native_scores'].astype(float);q0=int(s0.argmax());assert p['source_q0']==q0 and c['source_q0']==q0 and np.array_equal(p['native_scores'][0],s0)
    sr=rank(c['rankinput'],src);sourceerr=max(sourceerr,float(abs(sr-p['source_rank0_used']).max()));assert sourceerr<2e-5
    st=s0+c['gate']*(rank(c['rankinput'],state)-p['source_rank0_used']);err=float(abs(st-p['native_scores'][1]).max());rankerr=max(rankerr,err);assert err<2e-5
    if r['split']=='A208':
     j=next(j for j,x in enumerate(A) if x['id']==r['id']);memory=s0+raw[j];aa[r['id']]=p['native_scores'][1];aorig[r['id']]=s0;aq[r['id']]=int(p['queries'][1]);old=load_npz(ROOT/'crossplatform/round25/target400_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));nm='fusion_rgb_seed0' if mode=='rgb' else 'noRGB_source_train_seed0';oi=old['arm_names'].tolist().index(nm);e=float(abs(s0-old['scores'][oi]).max());oldmax=max(oldmax,e);assert e<2e-5 and q0==old['query_indices'][oi] and np.array_equal(c['boxes'][q0],old['boxes'][oi]);oldcount+=1
    else:
     o=p['GPU_obs'];pos=p['GPU_pos'];w=p['GPU_weights'];sim=p['GPU_similarity'];assert o.shape==pos.shape==w.shape==sim.shape==(256,4) and o.min()>=0 and o.max()<104 and pos.min()>=0 and pos.max()<256 and all(len(set(x))==4 for x in o) and np.all(np.diff(sim,axis=1)<=0)
     actual=np.einsum('qc,qkc->qk',norm(c['rankinput']).astype(float),lib['keys'][o,pos].astype(float));coserr=max(coserr,float(abs(actual-sim).max()));assert coserr<3e-6;assert np.allclose(w,np.exp((sim.astype(float)-1)/.1),rtol=2e-6,atol=1e-14)
     retrieved=(raw[o[:,:1],pos[:,:1]]*w[:,:1]).sum(-1)/np.maximum(w[:,:1].sum(-1),1e-12);assert abs(retrieved-p['retrieved_raw']).max()<1e-12;memory=s0+retrieved-retrieved[q0];assert np.allclose(p['A_row_margin'],sim[:,0]-sim[:,1],rtol=0,atol=1e-7)
    assert abs(memory-p['native_scores'][2]).max()<1e-12
    official=np.array([grouped(r,s,s0) for s in p['native_scores']]);assert np.array_equal(official,p['queries']) and np.array_equal(p['boxes'],z['boxes'][official]);cq=grouped(r,st,s0)
    if cq!=official[1]:queries64.append({'alias':r['alias'],'mode':mode,'CPU64':cq,'officialGPU':int(official[1])})
    rawq=int(p['native_scores'][1].argmax())
    if rawq!=official[1]:rawgroups.append({'alias':r['alias'],'mode':mode,'raw_query':rawq,'group_representative_query':int(official[1]),'raw_and_grouped_boxes_equal':bool(np.array_equal(z['boxes'][rawq],z['boxes'][official[1]]))})
    q=dict(zip(ARMS[:3],map(int,official)));q['XYZ_source']=int(z['scores'].argmax());q['XYZ_R30']=grouped(r,references[r['id']],z['scores']);ref=original[r['id'],mode];assert q==ref['queries']
    ix=index[r['id']];path=ix['meta_path']
    if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
    gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));boxes=z['boxes'][list(q.values())].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);vals=candidate_ious(boxes,gt)[0]
    for arm,v in zip(ARMS,vals):close(v,ref['IoU'][arm],1e-12)
    taskrows.append({'alias':r['alias'],'id_PRIVATE':r['id'],'mode':mode,'platform':pl,'split':r['split'],'scene_PRIVATE':r['scene'],'queries':q,'IoU':dict(zip(ARMS,map(float,vals)))})
    headrows.append({'alias':r['alias'],'mode':mode,'platform':pl,'split':r['split'],'native_rank_replay_maxerror':err,'source_q0':q0,'memory_arithmetic_exact':True})
   pref=[]
   for f in ff:
    s=aa[f['id_PRIVATE']];gap=float(s[f['positive_group_PRIVATE']].max()-s[f['negative_group_PRIVATE']].max());record={'alias':lookup[f['id_PRIVATE']]['alias'],'pair_alias':f['pair_alias'],'mode':mode,'platform':pl,'legacy_label':f['label'],'gap':gap,'direction':gap>0,'final_positive':aq[f['id_PRIVATE']] in f['positive_group_PRIVATE']};pairs.append(record);pref.append(record)
   assert sum(x['direction'] for x in pref)==receipt['all_pairs_satisfied'];assert sum(x['direction'] for x in pref if x['legacy_label']=='B')==receipt['legacy_B_pairs_satisfied'];assert sum(x['final_positive'] for x in pref if x['legacy_label']=='B')==receipt['legacy_B_final_positive']
 summary=read(EXP/'SUMMARY.json');aggs=[];neg=[];scenes=[]
 for split in ['A208','DEV64']:
  for mode in MODES:
   for pl in ['drone','quad']:
    part=[r for r in taskrows if (r['split'],r['mode'],r['platform'])==(split,mode,pl)]
    for arm in ARMS:
     v=np.array([r['IoU'][arm] for r in part]);delta=v-np.array([r['IoU']['fused_frozen'] for r in part]);a={'n':len(part),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'delta_vs_own_fused_frozen_pp':float(delta.mean()*100),'improved':int((delta>1e-10).sum()),'worsened':int((delta< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['fused_frozen'] for r in part)}
     for k,vv in a.items():close(vv,summary[split][mode][pl][arm][k]);aggs.append({'split':split,'mode':mode,'platform':pl,'arm':arm,**a})
    for r in part:
     for arm in ARMS:
      for ref in ['fused_frozen','sameinfo_memory','XYZ_source','XYZ_R30']:
       if arm!=ref and r['IoU'][arm]<r['IoU'][ref]-1e-10:neg.append({'alias':r['alias'],'mode':mode,'platform':pl,'split':split,'arm':arm,'reference':ref,'delta_pp':100*(r['IoU'][arm]-r['IoU'][ref])})
   for arm in ARMS:close(summary[split][mode]['balanced_mIoU_percent'][arm],np.mean([summary[split][mode][pl][arm]['mIoU_percent'] for pl in ['drone','quad']]))
 oldnegative=read(EXP/'ALL_NEGATIVES.json');key=lambda r:(r['split'],r['mode'],r['platform'],r['alias'],r['arm'],r['reference']);assert sorted(neg,key=key)==sorted(oldnegative,key=key)
 oldscene=read(EXP/'SCENE_ROWS_PRIVATE.json')
 for sr in oldscene:
  ss=[r for r in taskrows if all(r[k]==sr[k] for k in ['split','mode','platform','scene_PRIVATE'])];assert len(ss)==sr['n']
  for arm in ARMS:close(np.mean([r['IoU'][arm] for r in ss])*100,sr['mIoU_percent'][arm])
 selected=read(EXP/'CLOSURE_INPUTS_PRIVATE.json');closure=read(BASE/'closure4_attempt001/CLOSURE_AUDIT.json');assert len(selected)==4 and closure['state']=='PASSED' and closure['actual_full_grounding_RGB_fusion_forwards']==16 and len(closure['rows'])==8
 assert {(r['alias'],mode) for r in selected for mode in MODES}=={(r['alias'],r['mode']) for r in closure['rows']}
 assert all(r['boxes_bitexact'] and r['frozen_query_same'] and r['adapted_query_same'] and max(r['source_native_maxerror'],r['adapted_native_maxerror'],r['rankinput_maxerror'])<2e-5 for r in closure['rows']);assert closure['nonrank_and_backbone_frozen']
 mechanism=read(EXP/'PAIRED_MECHANISM.json')
 for split in ['A208','DEV64']:
  for pl in ['drone','quad']:
   f=lambda mode:summary[split][mode][pl]['rank_TTA']['mIoU_percent']-summary[split][mode][pl]['fused_frozen']['mIoU_percent'];close(f('rgb')-f('no_rgb'),mechanism[split][pl]['paired_delta_TTA_RGB_minus_delta_TTA_noRGB_pp'])
  close(np.mean([mechanism[split][pl]['paired_delta_TTA_RGB_minus_delta_TTA_noRGB_pp'] for pl in ['drone','quad']]),mechanism[split]['balanced_paired_interaction_pp'])
 dev=summary['DEV64']['rgb']['balanced_mIoU_percent'];gate=dev['rank_TTA']>dev['fused_frozen'] and dev['rank_TTA']>dev['sameinfo_memory'];assert gate and read(EXP/'DECISION.json')['fixed_DEV_gate_pass'];assert len(taskrows)==544 and len(qp_rows)==38 and len(pairs)==44 and not queries64
 dump(OUT/'ALL544_TASK_ROWS_PRIVATE.json',taskrows);dump(OUT/'ALL_NEGATIVES.json',neg);dump(OUT/'QP38_INDEPENDENT.json',qp_rows);dump(OUT/'HEAD_REPLAY_ROWS.json',headrows);dump(OUT/'ALL44_PAIR_EXECUTION.json',pairs);dump(OUT/'RAW_VS_GROUPED_READOUT.json',rawgroups);dump(OUT/'SOURCE_PROVENANCE.json',sourcehash)
 result={'state':'PASS','sample_mode_rows':544,'selected_task_values':2720,'rank_states':4,'parameters_per_state':12929,'optimizer_steps_reviewed_not_rerun':64,'source_selection':'R25 residual_rank seed0 e01 by WaymoVAL512 only','source_own_native_q0_verified':True,'U_keep_counts':[96,93],'feedback_count':22,'feedback_observations':19,'independent_QPs':38,'native_rank_replay_maxerror':rankerr,'source_rank_replay_maxerror':sourceerr,'CPU64_query_differences':queries64,'memory_arithmetic_exact':True,'selected_GPU_cosine_maxerror':coserr,'GPU_NN_indices_replayed_not_CPU_reselected':True,'old_R25_A_source_score_replays':oldcount,'old_R25_A_score_maxerror':oldmax,'negative_comparisons_reproduced':len(neg),'scene_tables_reproduced':len(oldscene),'source_frozen_raw_q0_exact':True,'raw_vs_grouped_TTA_differences':len(rawgroups),'fullclosure_observations':4,'fullclosure_receipt_forwards':16,'fullclosure_receipts_reviewed_not_GPU_rerun':True,'fixed_DEV_gate_pass':bool(gate),'primary_DEV_interaction_pp':mechanism['DEV64']['balanced_paired_interaction_pp'],'resource_stop_does_not_change_gate':True,'no_U_FINAL_manifest_media_GT_access':True,'no_GPU_optimizer_training_autograd':True,'limitations':['No independently regenerated full GPU topk; recorded indices checked by source cosines and arithmetic.','Only four registered observations have full-model closure; all544 cached sample-mode heads/tasks audited.','A/DEV already exposed development data, no independent confirmation.','RGB/noRGB source weights differ, so interaction is not pure input causality.'],'prediction_seal_sha256':sha(BASE/'fit272_attempt001/SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
