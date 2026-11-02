"""R51 sole repair six-arm comparison assembly, no inference/task labels."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ARMS=['source','external_G','initial_replace_all_c05','no_instance_association_c05','grouped_replace_all_c05','grouped_no_relation_c05']
OLD=['source','external_G','replace_all_c05','no_instance_association_c05']
NEW=['source','grouped_replace_all_c05','grouped_no_relation_c05']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def path(s):
 p=Path(s);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve();assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p);return p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,label):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args();plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_GROUPED_REPAIR_SIX_ARM_FRAMES_NO_GT' and auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(args.plan)
 assert plan['n']==336 and plan['arm_names']==ARMS and plan['fixed_appearance_factor']==.5 and plan['single_registered_repair'] is True
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():assert sha(path(rel))==h,('pin',rel)
 roots={k:path(plan[k]) for k in ('initial_prediction_root','grouped_root')};seals={k:read(p/'SEALED.json') for k,p in roots.items()}
 for k,p in roots.items():assert plan['inputs_sha256'][(p/'SEALED.json').relative_to(ROOT).as_posix()]==sha(p/'SEALED.json')
 audit=read(path(plan['grouped_audit']));assert plan['grouped_audit'] in plan['inputs_sha256'] and auth['grouped_audit_sha256']==sha(path(plan['grouped_audit']))
 assert audit['state']=='PASS_R51_INDEPENDENT_GROUP_MEDOID_G_NUMERIC_AND_HASH2_RGB_XYZ_AUDIT_NO_GT' and audit['prediction_seal_sha256']==sha(roots['grouped_root']/'SEALED.json')
 def bound(k,rel):
  p=roots[k]/rel;assert sha(p)==seals[k]['files'][rel];return p
 rows=read(path(plan['manifest']));assert plan['manifest']=='research/round24/input/ALL336_PRIVATE.json' and len(rows)==336
 old_status=read(bound('initial_prediction_root','READOUT_STATUS_PRIVATE.json'));status_byid={r['id']:r['arm_statuses'] for r in old_status};assert set(status_byid)=={r['id'] for r in rows}
 out=path(plan['output_root']);assert out.parent==ROOT/'crossplatform/round51' and not out.exists();out.mkdir(parents=True)
 for name in ('samples_PRIVATE','cases_PRIVATE'):(out/name).mkdir()
 done=[];statuses=[];start=time.perf_counter()
 try:
  for row in rows:
   sid=row['id'];old=load(bound('initial_prediction_root','samples_PRIVATE/'+sid+'.npz'));new=load(bound('grouped_root','samples_PRIVATE/'+sid+'.npz'))
   assert new['arm_names'].tolist()==NEW;indices=[old['arm_names'].tolist().index(a) for a in OLD]
   exact(new['source_boxes'],old['source_boxes'],'fixed original256 boxes');exact(new['source_scores'],old['source_scores'],'fixed original native scores');exact(new['scores'][0],old['scores'][indices[0]],'source F bridge')
   scores=np.stack([*[old['scores'][i] for i in indices],new['scores'][1],new['scores'][2]]);q=scores.argmax(1).astype(np.int64);selected=old['source_boxes'][q]
   assert np.array_equal(q[:4],old['queries'][indices]) and np.array_equal(q[4:],new['queries'][1:]);exact(selected[:4],old['selected_original_boxes'][indices],'old four endpoint boxes');exact(selected[4:],new['selected_original_boxes'][1:],'new two endpoint boxes')
   np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.asarray(ARMS),scores=scores,queries=q,selected_original_boxes=selected,source_boxes=old['source_boxes'],source_scores=old['source_scores'])
   nc=read(bound('grouped_root','cases_PRIVATE/'+sid+'.json'));ss={a:status_byid[sid][b] for a,b in zip(ARMS[:4],OLD)};ss.update({a:nc['arm_statuses'][a] for a in NEW[1:]})
   statuses.append({'id':sid,'arm_statuses':ss});done.append(sid)
   dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'old_four_and_new_two_F_queries_originalboxes_copied_exact':True,'no_new_calibration_or_TTA':True,'fixed_appearance_factor':.5,'initial_primary_16arm_results_not_replaced':True})
  dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COSTS.json',{'seconds':time.perf_counter()-start,'new_models_GT_or_optimizer':False})
  dump(out/'SEALED.json',{'state':'R51_GROUPED_REPAIR_SIX_ARM_COMPARISON_BEFORE_NEW_SCORE','n':len(done),'arm_names':ARMS,'plan_sha256':sha(args.plan),'grouped_audit_sha256':sha(path(plan['grouped_audit'])),'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_n':len(done),'no_automatic_retry_or_GT':True});raise
if __name__=='__main__':main()
