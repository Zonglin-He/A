"""Joint authorized U task audit: R34 recorded neighbors + R35 saved heads. CPU."""
import sys,json,time,datetime,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(OUT),str(ROOT/'research/round31/independent_audit')]
from audit34_independent import key,relative,qr,read,dump,sha,verify,load_npz,rows,feature,ARMS,KEYS
from audit31_independent import head
from independent_core_v2 import transform_gt,candidate_ious
B34=ROOT/'crossplatform/round34/scope_U128_attempt001';B35=ROOT/'crossplatform/round35/predictionsU128_attempt001'
E34=ROOT/'research/round34/experiment';E35=ROOT/'research/round35/experiment';O35=ROOT/'research/round35/independent_audit'
C35=['R30_U_rows','U_rows_matched_denominator','partial_output_keep'];A35=['source']+C35
LOCK='4f3823de10bf40b3d6c8c275f3a0bceaed9f76949df5a19c41c551005843f5b4'
def close(a,b,t=1e-10):assert abs(float(a)-float(b))<t,(a,b,t)

def aggregate34(records):
 summary=read(E34/'U128_SUMMARY.json')['U128'];neg=[];scenes=[]
 for pl in ['drone','quad']:
  rr=[r for r in records if r['platform']==pl]
  for arm in ARMS:
   v=np.array([r['IoU'][arm] for r in rr]);d=v-np.array([r['IoU']['source'] for r in rr]);s={'n':len(rr),'mIoU_percent':100*float(v.mean()),'Acc25_percent':100*float((v>.25).mean()),'Acc50_percent':100*float((v>.5).mean()),'delta_vs_source_pp':100*float(d.mean()),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in rr),'scene_clusters':len({r['scene_PRIVATE'] for r in rr})}
   for k,v in s.items():close(v,summary[pl][arm][k])
  for r in rr:
   for arm in ARMS:
    for ref in ['source','R30_ungated','h_samegate_QP','noun_samegate_QP']:
     d=r['IoU'][arm]-r['IoU'][ref]
     if d< -1e-10:neg.append({'alias':r['alias'],'platform':pl,'split':'U128','arm':arm,'reference':ref,'delta_pp':100*d})
  for j,scene in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
   ss=[r for r in rr if r['scene_PRIVATE']==scene];scenes.append({'split':'U128','platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{arm:100*float(np.mean([r['IoU'][arm] for r in ss])) for arm in ARMS}})
 assert neg==read(E34/'U128_ALL_NEGATIVES.json') and scenes==read(E34/'U128_SCENE_ROWS.json')
 for arm in ARMS:close(summary['balanced_mIoU_percent'][arm],np.mean([summary[pl][arm]['mIoU_percent'] for pl in ['drone','quad']]))
 for p in read(E34/'U128_PAIRED_COMPARISONS.json'):
  for pl in ['drone','quad']:close(p['platform_gain_pp'][pl],100*np.mean([r['IoU'][p['arm']]-r['IoU'][p['reference']] for r in records if r['platform']==pl]))
  assert p['clusters']=={'drone':1,'quad':3} and p['descriptive_scene_bootstrap95_pp'] is None and p['between_scene_generalization_not_estimated_if_a_platform_has_one_scene']
 return neg,summary

def aggregate35(records):
 summary=read(E35/'U128_SUMMARY.json');neg=[];scenes=[];arms=list(records[0]['queries'])
 for pl in ['drone','quad']:
  rr=[r for r in records if r['platform']==pl]
  for arm in arms:
   v=np.array([r['IoU'][arm] for r in rr]);d=v-np.array([r['IoU']['source'] for r in rr]);s={'n':len(rr),'mIoU_percent':100*float(v.mean()),'Acc25_percent':100*float((v>.25).mean()),'Acc50_percent':100*float((v>.5).mean()),'delta_vs_source_pp':100*float(d.mean()),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in rr),'scene_clusters':len({r['scene_PRIVATE'] for r in rr})}
   for k,v in s.items():close(v,summary[pl][arm][k])
  for j,scene in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
   ss=[r for r in rr if r['scene_PRIVATE']==scene];scenes.append({'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{arm:100*float(np.mean([r['IoU'][arm] for r in ss])) for arm in arms}})
 for r in records:
  for arm in arms:
   for ref in ['source','R30_U_rows','U_rows_matched_denominator','R34_h_samegate_QP','R34_noun_samegate_QP']:
    d=r['IoU'][arm]-r['IoU'][ref]
    if arm!=ref and d< -1e-10:neg.append({'alias':r['alias'],'platform':r['platform'],'configuration':arm,'reference':ref,'delta_pp':100*d})
 assert neg==read(E35/'U128_ALL_NEGATIVES.json') and scenes==read(E35/'U128_SCENE_ROWS.json')
 for arm in arms:close(summary['balanced_mIoU_percent'][arm],np.mean([summary[pl][arm]['mIoU_percent'] for pl in ['drone','quad']]))
 for p in read(E35/'U128_PAIRED_COMPARISONS.json'):
  for pl in ['drone','quad']:close(p['platform_delta_pp'][pl],100*np.mean([r['IoU'][p['arm']]-r['IoU'][p['reference']] for r in records if r['platform']==pl]))
  assert not p['U_used_for_method_selection']
 return neg,summary

def drift(states,source):
 fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');refs={(r['platform'],r['configuration']):r for r in read(E35/'A_PRESERVATION_DRIFT_DECOMPOSITION.json')};out=[]
 for pl in ['drone','quad']:
  A=[r for r in rows('A208') if r['platform']==pl];N=len(A);ep=np.zeros((N,256),bool);U=np.ones(N,bool)
  for j,r in enumerate(A):
   ff=[f for f in fb if f['id_PRIVATE']==r['id']]
   if ff:U[j]=False
   for f in ff:ep[j,f['positive_group_PRIVATE']+f['negative_group_PRIVATE']]=True
  old=np.repeat(U[:,None],256,axis=1);added=~ep&~old
  zz=[feature(r) for r in A];q0=np.array([z['scores'].argmax() for z in zz]);lo0=np.stack([head(source,z)[0] for z in zz]);d0=lo0-lo0[np.arange(N),q0,None]
  for name in C35:
   lo=np.stack([head(states[pl,name],z)[0] for z in zz]);d=lo-lo[np.arange(N),q0,None];sq=(d-d0)**2;ref=refs[pl,name];v={'platform':pl,'configuration':name,'original_U_MSE':float(sq[old].mean()),'new_feedback_unjudged_MSE':float(sq[added].mean()),'accepted_endpoint_MSE':float(sq[ep].mean()),'all_A_coordinate_MSE':float(sq.mean()),'old_U_coordinate_count':int(old.sum()),'new_feedback_unjudged_coordinate_count':int(added.sum()),'freed_endpoint_count':int(ep.sum())}
   for k in v:
    if k not in ['platform','configuration']:close(v[k],ref[k],1e-9)
   out.append(v)
 return out

def main():
 started=time.perf_counter();O35.mkdir(exist_ok=True);rr=read(ROOT/'research/NIGHT_20261101/input/U128_PRIVATE.json');assert len(rr)==128 and all(r['stage']=='U' for r in rr)
 lockfile=ROOT/'research/NIGHT_20261101/SELECTED_NIGHT_CANDIDATE_BEFORE_U_LOCK.json';assert sha(lockfile)==LOCK;lock=read(lockfile);assert lock['selected_primary']=='h_scoped_R30' and lock['U_not_for_selection'] and lock['matched_strength_control_not_eligible_for_selection']
 # Recorded guard chronology verifies the fixed registration preceded U work.
 registered=datetime.datetime.fromisoformat(lock['registered_local']);jobs=[]
 for p in (ROOT/'research/NIGHT_20261101/job_receipts').glob('*.json'):
  j=read(p);cmd=j['command'];s=' '.join(cmd)
  if any(x in s for x in ['captureU34.py','predictU35.py']) or ('round34/keyscope34.py' in s and 'U128' in cmd):
   assert datetime.datetime.fromisoformat(j['started_utc'])>registered and j['returncode']==0;jobs.append({'job_receipt':p.name,'started_utc':j['started_utc'],'command':cmd})
 assert len(jobs)==3
 for b in [B34,B35,ROOT/'crossplatform/round34/closure_U128_attempt001',ROOT/'crossplatform/round35/closureU128_attempt001',ROOT/'crossplatform/round35/partial_keep_attempt001']:verify(b)
 for path in ['R34_plan_sha256','R35_plan_sha256']:
  exp=E34 if path.startswith('R34') else E35;assert sha(exp/'PLAN_LOCK.json')==lock[path]
 assert sha(ROOT/'crossplatform/round35/partial_keep_attempt001/SEALED.json')==lock['R35_training_seal_sha256']
 s35=read(B35/'SEALED.json');assert s35['root_primary_lock_sha256']==LOCK;content=read(OUT/'U_CONTENT_AUDIT_RESULT.json');assert content['state']=='PASS' and content['U_primary_preselection_lock_sha256']==LOCK
 source=load_npz(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz');source={k[6:]:v for k,v in source.items() if k.startswith('image.')};states={(pl,n):load_npz(ROOT/'crossplatform/round35/partial_keep_attempt001/states_PRIVATE'/(pl+'_'+n+'.npz')) for pl in ['drone','quad'] for n in C35}
 for pl in ['drone','quad']:
  r30=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));assert all(np.array_equal(states[pl,'R30_U_rows'][n],r30[n]) for n in r30)
 libs={}
 for pl in ['drone','quad']:
  for k in KEYS:
   lib=load_npz(B34/'libraries_PRIVATE'/(pl+'_'+k+'.npz'));old=load_npz(ROOT/'crossplatform/round34/scope_diagnostic_attempt001/libraries_PRIVATE'/(pl+'_'+k+'.npz'));assert lib.keys()==old.keys() and all(np.array_equal(lib[n],old[n]) for n in lib);libs[pl,k]=lib
 ref34={r['alias']:r for r in read(E34/'U128_ALL_ROWS_PRIVATE.json')};ref35={r['alias']:r for r in read(E35/'U128_ALL_ROWS_PRIVATE.json')};closure_receipts={r['alias']:r for r in read(ROOT/'crossplatform/round34/closure_U128_attempt001/CLOSURE_AUDIT.json')['rows']};rec34=[];rec35=[];maxscore=maxcos=maxprob=maxlo=maxclosure=0.;cpuqdiff=[];closure_n=0;rawcount34=rawcount35=0;partialquerydiff=0;taskexact=0
 idx={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={}
 for r in rr:
  z=load_npz(ROOT/'crossplatform/round33/sourceU128_attempt001/samples_PRIVATE'/(r['id']+'.npz'));p34=load_npz(B34/'predictions_PRIVATE'/(r['id']+'.npz'));p35=load_npz(B35/'predictions_PRIVATE'/(r['id']+'.npz'));assert p34['arm_names'].tolist()==ARMS and p35['arm_names'].tolist()==A35
  p0=p34['source_probability'];pT=p34['R30_probability'];q0=int(p0.argmax());assert np.array_equal(p0,z['scores']) and np.array_equal(p35['probability_scores'][0],p0) and np.array_equal(p35['probability_scores'][1],pT)
  d0=relative(p0,q0);dT=relative(pT,q0);delta=dT-d0;v={'source':d0,'R30_ungated':dT}
  for k in KEYS:
   lib=libs[r['platform'],k];obs=p34[k+'_obs'];pos=p34[k+'_pos'];sim=p34[k+'_similarity'];w=p34[k+'_weights'];assert obs.shape==pos.shape==sim.shape==w.shape==(256,4) and np.all(np.diff(sim,axis=1)<=0) and all(len(set(x))==4 for x in obs)
   cos=np.einsum('qc,qkc->qk',key(z,k).astype(float),lib['keys'][obs,pos].astype(float));maxcos=max(maxcos,float(abs(cos-sim).max()));assert maxcos<3e-6
   o=obs[:,:1];p=pos[:,:1];ww=w[:,:1];g=lib['endpoint'][o,p].max(1).astype(float);raw=(lib['raw'][o,p]*ww).sum(1)/np.maximum(ww.sum(1),1e-12);md=raw-raw[q0];local=d0+g*delta;local[g==0]=d0[g==0];local[g==1]=dT[g==1]
   assert np.array_equal(g,p34[k+'_gate']) and np.array_equal(raw,p34[k+'_raw_retrieved']) and np.array_equal(md,p34[k+'_relative_memory']);pr='h_' if k=='h288' else 'noun_';v.update({pr+'scoped_R30':local,pr+'samegate_QP':d0+g*md,pr+'global_mean':d0+float(g.mean())*delta,pr+'global_norm':d0+float(np.linalg.norm(g*delta)/max(np.linalg.norm(delta),1e-12))*delta})
  mat=np.stack([v[n] for n in ARMS]);maxscore=max(maxscore,float(abs(mat-p34['complete_scores']).max()));assert maxscore==0;qs34=np.array([qr(r,s,z) for s in mat]);assert np.array_equal(qs34,p34['queries']) and np.array_equal(p34['boxes'],z['boxes'][qs34]);rawcount34+=10
  for j,name in enumerate(A35):
   s=source if name=='source' else states[r['platform'],name];lo,prob=head(s,z);maxprob=max(maxprob,float(abs(prob-p35['probability_scores'][j]).max()));maxlo=max(maxlo,float(abs(lo-p35['exact_token_logodds'][j]).max()));assert maxprob<2e-5 and maxlo<1e-4
   if int(prob.argmax())!=int(p35['queries'][j]):cpuqdiff.append({'alias':r['alias'],'arm':name,'CPU':int(prob.argmax()),'GPU':int(p35['queries'][j])})
  assert not cpuqdiff;qs35=p35['probability_scores'].argmax(1);assert np.array_equal(qs35,p35['queries']) and np.array_equal(p35['boxes'],z['boxes'][qs35]);rawcount35+=4
  assert qs35[0]==qs34[0] and qs35[1]==qs34[1];partialquerydiff+=int(qs35[1]!=qs35[3])
  ix=idx[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));qdict=dict(zip(A35,map(int,qs35)));qdict.update({'R34_'+n:int(q) for n,q in zip(ARMS,qs34) if n not in ['source','R30_ungated']});boxes=z['boxes'][list(qdict.values())].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);values=candidate_ious(boxes,gt)[0];valdict=dict(zip(qdict,map(float,values)))
  val34={n:valdict['source' if n=='source' else 'R30_U_rows' if n=='R30_ungated' else 'R34_'+n] for n in ARMS}
  a={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'split':'U128','queries':dict(zip(ARMS,map(int,qs34))),'IoU':val34};b={'alias':r['alias'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':qdict,'IoU':valdict}
  for computed,reference in [(a,ref34[r['alias']]),(b,ref35[r['alias']])]:
   assert computed['queries']==reference['queries']
   for n in computed['IoU']:close(computed['IoU'][n],reference['IoU'][n],1e-12)
  rec34.append(a);rec35.append(b);taskexact+=int(valdict['partial_output_keep']==valdict['R30_U_rows'])
  c34=load_npz(ROOT/'crossplatform/round34/closure_U128_attempt001/observations_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(c34['queries'],qs34) and np.array_equal(c34['boxes'],p34['boxes']);ce=float(abs(c34['complete_scores']-mat).max());maxclosure=max(maxclosure,ce);close(ce,closure_receipts[r['alias']]['complete_coordinate_score_maxerror'],1e-12)
  perr=max(float(abs(c34['source_probability']-p0).max()),float(abs(c34['R30_probability']-pT).max()));close(perr,closure_receipts[r['alias']]['source_adapted_probability_maxerror'],1e-12);assert perr<2e-5
  for n in ['U_rows_matched_denominator','partial_output_keep']:
   c35=load_npz(ROOT/'crossplatform/round35/closureU128_attempt001/observations_PRIVATE'/(r['id']+'_'+n+'.npz'));j=A35.index(n);assert int(c35['query'])==int(qs35[j]) and np.array_equal(c35['box'],p35['boxes'][j]);assert abs(c35['probability_scores']-p35['probability_scores'][j]).max()<2e-5
  closure_n+=1
 neg34,summary34=aggregate34(rec34);neg35,summary35=aggregate35(rec35);dr=drift(states,source)
 c34=read(ROOT/'crossplatform/round34/closure_U128_attempt001/CLOSURE_AUDIT.json');c35=read(ROOT/'crossplatform/round35/closureU128_attempt001/CLOSURE_AUDIT.json')
 assert c34['state']==c35['state']=='PASSED' and c34['observations']==c35['n']==128 and c34['full_model_forwards']==c35['full_model_forwards']==256
 assert c34['source_restored_bitexact'] and c35['source_restored_bitexact'] and c34['nonmatching_parameters_unchanged'] and c35['nonmatching_parameters_unchanged'];assert taskexact==128
 primary=summary34['balanced_mIoU_percent']['h_scoped_R30'];assert lock['selected_primary']=='h_scoped_R30' and sha(lockfile)==LOCK
 common={'root_primary_lock_sha256':LOCK,'fixed_primary':'h_scoped_R30','primary_unchanged_despite_U_best_R30':True,'registration_precedes_all_three_U_capture_prediction_jobs':jobs,'U128_scenes':{'drone':1,'quad':3},'FINAL_manifest_media_GT_not_read':True,'no_GPU_optimizer_training_autograd':True,'content_audit_pass':True,'no_between_Drone_scene_CI':True,'all_negative_cases_retained':True,'CPU_argmax_differences':cpuqdiff,'CPU_head_probability_maxerror':maxprob,'CPU_head_exactlogodds_maxerror':maxlo,'limitations':['U content is distinct from historical observations but allU scenes overlap history; no unseen-scene claim.','Recorded GPU nearest indices are used for arithmetic; no CPU neighbor substitution. GPU fullclosure receipts cover all128, reviewed here without rerunningGPU.','Source/R30 fullclosure256forwards plus two newR35heads256forwards total512 unique fullmodel calls; do not doublecount R30 twice.','Partialkeep reduces unjudged A drift but has no IoU improvement overR30 onU and lowerDEV; not evidence generally betterTTA.']}
 result34={'state':'PASS',**common,'R34_score_vectors_and_task_values':rawcount34,'R35_probability_and_logodds_vectors':rawcount35,'R34_fullscore_replay_maxerror':maxscore,'R34_selected_cosine_maxerror':maxcos,'R34_R35_source_and_R30_probabilities_bitexact':True,'fullmodel_closure_coordinate_scores_maxerror':maxclosure,'fullmodel_closure_score_errors_match_original_receipts':True,'R34_negative_comparisons':len(neg34),'R35_combined_control_negative_comparisons':len(neg35),'all128_fullclosure_selected_queries_and_boxes_checked':closure_n,'balanced_U_mIoU_percent':summary34['balanced_mIoU_percent'],'primary_balanced_gain_vs_source_pp':primary-summary34['balanced_mIoU_percent']['source'],'auditor_strict_closure_bitexact_assumption_corrected':'Initial audit asserted complete GPU closure scores bitexact; actual original receipt permits probability tolerance and records coordinate drift. Failure log retained; corrected audit checks actual errors vs original receipts plus exact queries/boxes, no predictions changed.','seconds':time.perf_counter()-started}
 result35={'state':'PASS',**common,'R35_saved_heads':6,'source_projection_also_CPU_replayed':True,'U_probability_and_exactlogodds_vectors':rawcount35,'R35_own_task_values_checked':rawcount35,'merged_R34_controls_task_values_checked':128*12,'partial_vs_R30_per_observation_IoU_exact':taskexact,'partial_vs_R30_different_queries':partialquerydiff,'A_drift_decomposition_states_recomputed':len(dr),'balanced_U_mIoU_percent':summary35['balanced_mIoU_percent'],'new_R35_fullclosure_forward_receipt':256,'R30_source_closure_reused_from_R34':True}
 dump(OUT/'U_AUDIT_RESULT.json',result34);dump(O35/'U_AUDIT_RESULT.json',result35);dump(OUT/'U_ALL128_TASK_ROWS_PRIVATE.json',rec34);dump(O35/'U_ALL128_COMBINED_TASK_ROWS_PRIVATE.json',rec35);dump(O35/'A_DRIFT_DECOMPOSITION_INDEPENDENT.json',dr);dump(OUT/'U_ALL_NEGATIVES.json',neg34);dump(O35/'U_ALL_NEGATIVES.json',neg35);print(json.dumps({'R34':result34,'R35':result35},indent=2))
if __name__=='__main__':main()
