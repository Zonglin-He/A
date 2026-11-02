"""R51 conditional seven-arm TTA readout assembler. UNEXECUTED, no GT/model.

Canonical AB0 uses captured source A/B, fixed proposal groups and frozen VPR.
Both interfaces' fit0 must select the same original A query/box. Updates are
copied from sealed episodes; rejection/UNKNOWN/zero-update rows are retained.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ARMS=['source','single_G','sameinfo_consensus0','mlp_s1','mlp_s3','lora_qv_s1','lora_qv_s3']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def path(s):
 p=Path(s);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve();assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p);return p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,name):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name

def zero_readout(a,b,aid,bid,cues):
 consensus=np.asarray(a,float).copy();b=np.asarray(b,float)
 for group in np.unique(aid[aid>=0]):
  aq=np.flatnonzero(aid==group);bq=np.flatnonzero(bid==group)
  if len(bq):consensus[aq]=(np.asarray(a,float)[aq]+b[bq].max())/2
 return consensus+2*cues[0]+2*cues[1]+2*cues[2]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_CONDITIONAL_SEVEN_ARM_TTA_ASSEMBLY_NO_GT' and auth['plan_sha256']==sha(args.plan) and auth['program_sha256']==sha(__file__)
 assert plan['n']==336 and plan['arm_names']==ARMS and plan['no_new_G_models_optimizer_or_GT'] is True
 assert plan['direct_G_gate_PASS'] is True and plan['selected_G_arm'] in ('replace_all_c05','replace_all_c1','replace_all_c2')
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():assert sha(path(rel))==h,('pin',rel)
 roots={k:path(plan[k]) for k in ('capture_A_root','capture_B_root','structure_root','fit_root','DEV_prediction_root')};seals={k:read(v/'SEALED.json') for k,v in roots.items()}
 for k,p in roots.items():assert plan['inputs_sha256'][(p/'SEALED.json').relative_to(ROOT).as_posix()]==sha(p/'SEALED.json')
 assert seals['fit_root']['state']=='R51_DEV336_PROPOSAL_REAL_EPISODIC_FITS_SAME_INFORMATION_NO_GT' and seals['fit_root']['n']==336
 def bound(k,rel):
  p=roots[k]/rel;assert sha(p)==seals[k]['files'][rel];return p
 rows=read(path(plan['manifest']));assert plan['manifest']=='research/round24/input/ALL336_PRIVATE.json' and len(rows)==336
 out=path(plan['output_root']);assert out.parent==ROOT/'crossplatform/round51' and not out.exists();out.mkdir(parents=True)
 for name in ('samples_PRIVATE','cases_PRIVATE'):(out/name).mkdir()
 dev_status_rows=read(bound('DEV_prediction_root','READOUT_STATUS_PRIVATE.json'));assert len(dev_status_rows)==336
 dev_status={r['id']:r['arm_statuses'] for r in dev_status_rows};assert set(dev_status)=={r['id'] for r in rows}
 statuses=[];done=[];start=time.perf_counter()
 try:
  for row in rows:
   sid=row['id'];ac=read(bound('capture_A_root','cases_PRIVATE/'+sid+'.json'));bc=read(bound('capture_B_root','cases_PRIVATE/'+sid+'.json'))
   A=load(bound('capture_A_root',ac['subject']['file']));B=load(bound('capture_B_root',bc['B_file']))
   info=read(bound('structure_root','cases_PRIVATE/'+sid+'.json'));st=load(bound('structure_root','arrays_PRIVATE/'+sid+'.npz'))
   dev=load(bound('DEV_prediction_root','samples_PRIVATE/'+sid+'.npz'));ga=dev['arm_names'].tolist().index(plan['selected_G_arm']);sa=dev['arm_names'].tolist().index('source')
   assert info['selected_G_arm']==plan['selected_G_arm'] and info['selected_appearance_factor']==plan['selected_factor']
   exact(A['boxes'],dev['source_boxes'],'canonical A boxes');exact(A['scores'],dev['source_scores'],'canonical A native scores')
   exact(st['literal_G_scores_A'],dev['scores'][ga],'literal singleG F exact')
   base=zero_readout(A['scores'],B['scores'],st['A_proposal_group_ids'],st['B_A_proposal_group_ids'],st['fixed_G_components_A'])
   q0=int(base.argmax());fits={};receipts={}
   for interface in ('mlp','lora_qv'):
    tag=sid+'__'+interface;z=load(bound('fit_root','episodes_PRIVATE/'+tag+'.npz'));r=read(bound('fit_root','cases_PRIVATE/'+tag+'.json'))
    assert r['accepted']==info['accepted'] and r['interface']==interface and r['GT_read'] is False
    assert int(z['s0_canonical_A_query'])==q0 and int(z['s0_scores'].argmax())==q0
    exact(z['s0_selected_original_A_box'],A['boxes'][q0],'both fit0 same canonical AB0 box')
    fits[interface]=z;receipts[interface]=r
   fs=[dev['scores'][sa],dev['scores'][ga],base,fits['mlp']['s1_scores'],fits['mlp']['s3_scores'],fits['lora_qv']['s1_scores'],fits['lora_qv']['s3_scores']]
   scores=np.stack(fs);q=scores.argmax(1).astype(np.int64);selected=A['boxes'][q]
   assert q[0]==dev['queries'][sa] and q[1]==dev['queries'][ga]
   for j,(interface,step) in enumerate((('mlp',1),('mlp',3),('lora_qv',1),('lora_qv',3)),3):
    z=fits[interface];assert q[j]==int(z[f's{step}_canonical_A_query']);exact(selected[j],z[f's{step}_selected_original_A_box'],'copied adapted endpoint')
   np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.asarray(ARMS),scores=scores,queries=q,selected_original_boxes=selected,source_boxes=A['boxes'],source_scores=A['scores'])
   status={arm:{'unknown':not info['accepted'],'fallback':False,'no_parameter_update':True} for arm in ARMS}
   status['source']=dev_status[sid]['source'];status['single_G']=dev_status[sid][plan['selected_G_arm']]
   for interface in ('mlp','lora_qv'):
    for step in (1,3):status[interface+f'_s{step}']['no_parameter_update']=not receipts[interface]['nonzero_parameter_updates'][str(step)]
   statuses.append({'id':sid,'arm_statuses':status});done.append(sid)
   dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'accepted':info['accepted'],'rejection_reasons':info['rejection_reasons'],'both_fit0_match_canonical_AB0_query_box':True,'two_fit0_scores_max_difference':float(np.max(np.abs(fits['mlp']['s0_scores']-fits['lora_qv']['s0_scores']))),'fit0_native_replay_vs_captured_AB0_max_difference':{k:float(np.max(np.abs(v['s0_scores']-base))) for k,v in fits.items()},'all_A_queries_selectable':True,'frozen_G_not_refreshed':True,'source_and_singleG_exact_DEV_endpoint':True})
  dump(out/'READOUT_STATUS_PRIVATE.json',statuses)
  dump(out/'COSTS.json',{'seconds':time.perf_counter()-start,'new_model_forwards':0,'optimizer_steps':0,'all_fit_capture_installed_model_costs_belong_to_upstream_receipts':True})
  dump(out/'SEALED.json',{'state':'R51_CONDITIONAL_DEV336_SEVEN_TTA_READOUTS_NO_GT','n':len(done),'arm_names':ARMS,'selected_G_arm':plan['selected_G_arm'],'selected_factor':plan['selected_factor'],'plan_sha256':sha(args.plan),'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()},'no_new_G_models_optimizer_or_GT':True})
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_n':len(done),'no_automatic_retry_or_score':True});raise
if __name__=='__main__':main()
