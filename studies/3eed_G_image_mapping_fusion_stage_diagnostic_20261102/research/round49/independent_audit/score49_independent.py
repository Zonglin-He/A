"""Root-authorized one-shot old336 fixed-box 16-arm task score; no adaptation."""
from audit49_independent import ROOT,HERE,EXP,P,ARMS,read,sha,dump,load,verify,code_check,old_rows
import sys,time,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import candidate_ious,transform_gt

def old_index(rows):
 import ijson
 fp=ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json';prep=read(ROOT/'crossplatform/round5/baseline_engineering/PREPARATION_RECEIPT.json');assert sha(fp)==prep['scoring_index_sha256'];wanted={r['id']:r for r in rows};out={}
 with fp.open('rb') as f:
  for x in ijson.items(f,'item'):
   if x['id'] not in wanted:continue
   r=wanted[x['id']];assert x['platform']==r['platform'] and x['scene']==r['scene'] and Path(x['meta_path']).resolve()==Path(r['metapath']).resolve() and type(x['object_index']) is int and x['object_index']>=0
   out[x['id']]={k:x[k] for k in ['id','platform','scene','meta_path','object_index','metadata_container_sha256']}
 assert set(out)==set(wanted);return out

def contrasts(m):
 out={}
 for k,name in enumerate(['V','P','R']):
  others=[i for i in range(3) if i!=k];d=[]
  for x,y in itertools.product([0,1],repeat=2):
   a=[0,0,0];a[others[0]]=x;a[others[1]]=y;b=a.copy();b[k]=1;key=lambda z:f'legacy_V{z[0]}P{z[1]}R{z[2]}';d.append(m[key(b)]-m[key(a)])
  out[name+'_average_conditional_task_effect_pp']=float(np.mean(d));out[name+'_four_conditional_effects_pp']=d
 out['full_minus_source_pp']=m[ARMS[7]]-m[ARMS[0]];out['fullmapped_minus_full_pp']=m[ARMS[10]]-m[ARMS[7]];out['unclipV_minus_full_pp']=m[ARMS[11]]-m[ARMS[7]];out['no_gray_minus_full_pp']=m[ARMS[12]]-m[ARMS[7]];out['joint_same_operator_minus_full_pp']=m[ARMS[13]]-m[ARMS[7]];out['joint_literal_minus_full_pp']=m[ARMS[14]]-m[ARMS[7]];out['postclip_cue_minus_rawcue_pp']=m[ARMS[9]]-m[ARMS[8]];out['not_additive_identity_error_attribution']=True;return out

def main():
 start=time.perf_counter();code_check();auth=read(EXP/'ROOT_SINGLE_TASK_SCORING_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_R49_SINGLE_POSTSEAL_OLD336_TASK_SCORE';assert auth['scoring_program_sha256']==sha(__file__) and auth['independent_code_lock_sha256']==sha(HERE/'INDEPENDENT_CODE_LOCK.json');assert auth['stage_plan_sha256']==sha(EXP/'STAGE_PLAN_LOCK.json') and auth['prediction_seal_sha256']==sha(P/'SEALED.json') and auth['independent_audit_sha256']==sha(HERE/'AUDIT_RESULT.json');audit=read(HERE/'AUDIT_RESULT.json');assert audit['state']=='PASS_R49_INDEPENDENT_CACHED_STAGE_AND_READOUT_AUDIT_NO_GT';verify(P);rows,split=old_rows();index=old_index(rows);flags={r['id_PRIVATE']:r for r in read(HERE/'ALL336_STAGE_AUDIT_FLAGS_PRIVATE.json')};containers={};tasks=[];sourceold={r['id_PRIVATE']:r for r in read(ROOT/'research/round48/independent_audit/ALL336_TASK_ROWS_PRIVATE.json')}
 for r in rows:
  ix=index[r['id']];mp=ix['meta_path']
  if mp not in containers:assert sha(mp)==ix['metadata_container_sha256'];containers[mp]=read(mp)
  obj=containers[mp]['ground_info'][ix['object_index']];assert ' '.join(obj['caption'].replace(',',' ,').split())==r['caption'];gt=transform_gt(obj['bbox_3d'],r['platform']);pred=load(P/'samples_PRIVATE'/(r['id']+'.npz'));boxes=pred['selected_boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);values=candidate_ious(boxes,gt)[0];assert abs(float(values[0])-sourceold[r['id']]['IoU']['cached_native_current_BRIDGE'])<1e-12
  tasks.append({**flags[r['id']],'queries':dict(zip(ARMS,map(int,pred['queries']))),'IoU':dict(zip(ARMS,map(float,values)))})
 summary={};scenes=[];effects={};neg=[]
 for part in ['ALL336','D208','V64','M64']:
  rr=tasks if part=='ALL336' else [r for r in tasks if r['split']==part];ss={}
  for pl in ['drone','quad']:
   pr=[r for r in rr if r['platform']==pl];ss[pl]={}
   for arm in ARMS:
    v=np.array([r['IoU'][arm] for r in pr]);base=np.array([r['IoU'][ARMS[0]] for r in pr]);d=v-base;ss[pl][arm]={'n':len(pr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries'][ARMS[0]] for r in pr),'scene_clusters':len({r['scene_PRIVATE'] for r in pr})}
   for scene in sorted({r['scene_PRIVATE'] for r in pr}):
    sr=[r for r in pr if r['scene_PRIVATE']==scene];scenes.append({'part':part,'platform':pl,'scene_PRIVATE':scene,'n':len(sr),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in sr])*100) for a in ARMS}})
  ss['balanced_mIoU_percent']={a:float(np.mean([ss[pl][a]['mIoU_percent'] for pl in ['drone','quad']])) for a in ARMS};summary[part]=ss;effects[part]={pl:contrasts(ss['balanced_mIoU_percent'] if pl=='balanced' else {a:ss[pl][a]['mIoU_percent'] for a in ARMS}) for pl in ['drone','quad','balanced']}
 for r in tasks:
  for a in ARMS:
   for ref in ARMS:
    d=r['IoU'][a]-r['IoU'][ref]
    if a!=ref and d< -1e-10:neg.append({'alias':r['alias'],'split':r['split'],'platform':r['platform'],'arm':a,'reference':ref,'delta_pp':float(d*100)})
 dump(HERE/'ALL336_TASK_ROWS_PRIVATE.json',tasks);dump(HERE/'TASK_SUMMARY.json',summary);dump(HERE/'ALL_NEGATIVES.json',neg);dump(HERE/'ALL_SCENE_SUMMARIES_PRIVATE.json',scenes);dump(HERE/'FACTORIAL_AND_INTERFACE_CONTRASTS.json',effects);receipt={'state':'PASS_R49_SINGLE_OLD336_FIXED_BOX_TASK_SCORE','n':336,'arms':16,'task_values':5376,'negative_ordered_comparisons':len(neg),'original_R48_native_source_IoU_reproduced':336,'strict_thresholds':[.25,.5],'same_original_6D_yaw0_boxes_vs_P5_transformed_7D_GT':True,'exact_object_index_and_metadata_sha_join':True,'no_feedback_rule_selection_or_adaptation_after_GT':True,'blind_identity_does_not_come_from_IoU':True,'no_FINAL_access':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'independent_audit_sha256':sha(HERE/'AUDIT_RESULT.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(HERE/'SCORING_RECEIPT.json',receipt);print(json.dumps({'receipt':receipt,'ALL336_balanced':summary['ALL336']['balanced_mIoU_percent'],'effects_ALL336':effects['ALL336']},ensure_ascii=False))
if __name__=='__main__':main()
