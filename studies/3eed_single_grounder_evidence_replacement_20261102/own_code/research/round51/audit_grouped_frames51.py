"""Independent six-frame endpoint copy audit; saved arrays only, no NN/GT."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ARMS=['source','external_G','initial_replace_all_c05','no_instance_association_c05','grouped_replace_all_c05','grouped_no_relation_c05']
OLD=['source','external_G','replace_all_c05','no_instance_association_c05']
NEW=['grouped_replace_all_c05','grouped_no_relation_c05']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def path(s):
 p=Path(s);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve();assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p);return p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,name):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args();plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_GROUPED_SIX_FRAMES_ENDPOINT_AUDIT_NO_GT' and auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(args.plan)
 for g in ('programs_sha256','inputs_sha256'):
  for f,h in plan[g].items():assert sha(path(f))==h,('pin',f)
 roots={k:path(plan[k]) for k in ('frames_root','initial_prediction_root','grouped_root')};seals={k:read(p/'SEALED.json') for k,p in roots.items()}
 for k,p in roots.items():assert plan['inputs_sha256'][(p/'SEALED.json').relative_to(ROOT).as_posix()]==sha(p/'SEALED.json')
 assert auth['prediction_seal_sha256']==sha(roots['frames_root']/'SEALED.json') and seals['frames_root']['arm_names']==ARMS
 for rootkey,auditkey in (('initial_prediction_root','initial_independent_audit'),('grouped_root','grouped_independent_audit')):
  p=path(plan[auditkey]);assert plan[auditkey] in plan['inputs_sha256'];audit=read(p)
  assert audit['state'].startswith('PASS_') and audit['n']==336 and audit['prediction_seal_sha256']==sha(roots[rootkey]/'SEALED.json') and audit['canonical_source_256_boxes_and_readout_verified'] is True
 def bound(k,rel):
  p=roots[k]/rel;assert sha(p)==seals[k]['files'][rel];return p
 rows=read(path(plan['manifest']));assert plan['manifest']=='research/round24/input/ALL336_PRIVATE.json' and len(rows)==plan['n']==336
 out=path(plan['output_directory']);assert not out.exists();out.mkdir(parents=True);done=[];start=time.perf_counter()
 try:
  ss=read(bound('frames_root','READOUT_STATUS_PRIVATE.json'));assert len(ss)==336;statuses={r['id']:r['arm_statuses'] for r in ss}
  oo=read(bound('initial_prediction_root','READOUT_STATUS_PRIVATE.json'));oldstatus={r['id']:r['arm_statuses'] for r in oo};assert set(statuses)==set(oldstatus)=={r['id'] for r in rows}
  for row in rows:
   sid=row['id'];p=load(bound('frames_root','samples_PRIVATE/'+sid+'.npz'));old=load(bound('initial_prediction_root','samples_PRIVATE/'+sid+'.npz'));new=load(bound('grouped_root','samples_PRIVATE/'+sid+'.npz'))
   assert p['arm_names'].tolist()==ARMS;ix=[old['arm_names'].tolist().index(a) for a in OLD];ni=[new['arm_names'].tolist().index(a) for a in NEW]
   exact(p['source_boxes'],old['source_boxes'],'original boxes');exact(p['source_scores'],old['source_scores'],'source native');exact(new['source_boxes'],old['source_boxes'],'new original boxes');exact(new['source_scores'],old['source_scores'],'new source')
   expected=np.concatenate([old['scores'][ix],new['scores'][ni]],0);exact(p['scores'],expected,'four old two new F exact')
   q=expected.argmax(1).astype(np.int64);exact(p['queries'],q,'six argmax');exact(p['selected_original_boxes'],old['source_boxes'][q],'six original selected boxes')
   nc=read(bound('grouped_root','cases_PRIVATE/'+sid+'.json'));expect={a:oldstatus[sid][b] for a,b in zip(ARMS[:4],OLD)};expect.update({a:nc['arm_statuses'][a] for a in NEW});assert statuses[sid]==expect
   for v in expect.values():assert set(v)=={'unknown','fallback','no_parameter_update'} and all(type(x) is bool for x in v.values())
   done.append(sid)
  result={'state':'PASS_R51_GROUPED_REPAIR_SIX_FRAMES_COPY_AUDIT_NO_GT','n':336,'arm_names':ARMS,'prediction_seal_sha256':sha(roots['frames_root']/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'four_old_two_new_score_query_box_status_exact':True,'source_numeric_audits_both_PASS':True,'no_GT_NN_or_geometry_recomputation':True,'program_sha256':sha(__file__),'plan_sha256':sha(args.plan),'seconds':time.perf_counter()-start}
  dump(out/'AUDIT_RESULT.json',result);dump(out/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}});print(json.dumps(result),flush=True)
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_n':len(done),'no_GT_NN_or_retry':True});raise
if __name__=='__main__':main()
