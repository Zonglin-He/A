"""Seal/support before exact16 source labels; native gate and conditional wrong controls."""
import argparse,sys,time,traceback,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_external_policy_native import D,TEACHER,OUT,read,write,sha,load,check_pins
from vg_tta.external_qualification_io import verify_seal

def main(stage):
 import numpy as np,torch
 from vg_tta.desta3d_v3_policy_gate_views import build_views
 from vg_tta.exact_frame_decode_audit_v2 import decode
 from scripts.desta3d_v3_a0_fast_screen import labels_for
 from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,summarize_arm,paired_parent_bootstrap
 from vg_tta.external_evidence_metrics import tensor_metrics,external_prediction,scalar_external
 start=time.monotonic();run=OUT/('extgate_'+stage);dest=D/('evaluation' if stage=='policy001' else 'wrong_evaluation');assert not dest.exists()
 done,sealed=verify_seal(run);check_pins(read(run/'LOCK.json')['pins']);rows=read(run/'INPUTS.json');arms=read(run/'CONFIG.json')['arms'];verify_seal(TEACHER)
 assert read(TEACHER/'ROOT_EVIDENCE_READBACK.json')['status']=='passed';assert len(rows)==len({r['source'] for r in rows})==16
 preds={a:[] for a in arms};checks=0
 for i,row in enumerate(rows):
  ep=run/'episodes'/f'{i:02}';frames,ids=decode(row['input']);ev=read(TEACHER/'episodes'/f'{i:02}'/'EVIDENCE.json');views,meta=build_views(frames,ids,ev)
  saved=read(ep/'INPUT.json');assert saved['physical_pixel_sha']==hashlib.sha256(frames.tobytes()).hexdigest();assert saved['view_metadata']==json.loads(json.dumps(meta))
  assert saved['view_pixel_sha']=={a:hashlib.sha256(v.tobytes()).hexdigest() for a,v in views.items()}
  if 'B1' in arms:assert all(read(ep/'BASELINE_REPLAY.json').values())
  fixed=None
  for a in arms:
   p=load(ep/(a+'.pt'));assert p['key']==row['key'] and p['source']==row['source'] and p['frame_ids']==ids and not p['GT_read'] and not p['target_read'] and p['optimizer_steps']==0
   if fixed is None:fixed=p['physical_support']
   else:assert fixed==p['physical_support']
   preds[a].append(p);checks+=1
 write(dest/'PRE_SCORE_AUDIT.json',dict(status='passed',physical_native_checks=checks,seal_sha=sha(run/'PREDICTIONS_SEAL.json'),source_labels_scope='only the fixed16 exposed source queries, after new inference sealed',GT_worker=False,target_read=False))
 labels=labels_for(rows,'dev');results={a:[] for a in arms};err=0.
 if stage=='policy001':results['external']=[]
 for i,row in enumerate(rows):
  for a in results:
   if a=='external':
    ev=read(TEACHER/'episodes'/f'{i:02}'/'EVIDENCE.json');p=external_prediction(ev,labels[row['key']]['frame_ids']);m=scalar_external(p,labels[row['key']]);alt=tensor_metrics(p,labels[row['key']],external=True);interval=p['interval_physical'];invalid=None
   else:
    p=preds[a][i];m=score_tube_independently(p,labels[row['key']]);alt=tensor_metrics(p,labels[row['key']]);interval=p['interval'];invalid=int((~p['geometry_valid']).sum())
   err=max(err,max(abs(m[k]-alt[k]) for k in ['tIoU','sIoU','vIoU']));assert err<1e-6
   results[a].append(dict(key=row['key'],source=row['source'],metrics=m,interval=interval,invalid_geometry=invalid))
 if stage!='policy001':
  old=read(D/'evaluation/REPORT.json');results={**{a:x['rows'] for a,x in old['arms'].items()},**results}
 parents={a:summarize_parents(r) for a,r in results.items()};metrics=['tIoU','sIoU','vIoU'];comps={a:{k:paired_parent_bootstrap(parents[a],parents['B1'],k) for k in metrics} for a in results if a!='B1'}
 tails={};ret={}
 for a in comps:
  tails[a]={};ret[a]={}
  for k in metrics:
   b=np.array([r['metrics'][k] for r in results['B1']]);v=np.array([r['metrics'][k] for r in results[a]]);diff=100*(v-b)
   tails[a][k]=dict(query_harm_gt5pp=int((diff< -5).sum()),positive=int((diff>0).sum()),negative=int((diff<0).sum()),zero=int((diff==0).sum()),worst_pp=float(diff.min()),best_pp=float(diff.max()))
   ret[a][k]=dict(eligible=int((b>.5).sum()),retained=int(((b>.5)&(v>.5)).sum()))
 collapse=any(comps['TS'][k]['mean_delta_pp']<=-1 and comps['TS'][k]['negative_parents']>=12 for k in ['tIoU','sIoU'])
 positives=[a for a,k in [('T','tIoU'),('S','sIoU')] if comps[a][k]['mean_delta_pp']>0]
 preliminary=comps['TS']['vIoU']['mean_delta_pp']>0 and bool(positives) and not collapse
 wrong={a:{k:paired_parent_bootstrap(parents[a],parents[a+'_wrong'],k) for k in metrics} for a in ['T','S','TS'] if a+'_wrong' in results}
 passed=None if stage=='policy001' and preliminary else False
 if stage!='policy001':passed=preliminary and wrong['TS']['vIoU']['mean_delta_pp']>0 and any(a in wrong and wrong[a][k]['mean_delta_pp']>0 for a,k in [('T','tIoU'),('S','sIoU')] if a in positives)
 report=dict(status='completed_dual_geometry_pending_root_crosscheck',arms={a:dict(rows=r,summary=summarize_arm(r,parents[a])) for a,r in results.items()},comparisons=comps,query_tails=tails,B1_good_retention=ret,preliminary_pass=preliminary,positive_corresponding_branches=positives,TS_systematic_collapse=collapse,correct_minus_wrong=wrong,final_pass=passed,
  decision='run_only_registered_matched_wrong' if passed is None else ('qualified_source_policy_next_registered_OPD' if passed else 'STOP_DESTA3D_privileged_correction_OPD_line'),scalar_tensor_max_abs=err,CPU_seconds=time.monotonic()-start,
  scope='16 exposed source parents, one query each; possible LLaVA-ST training overlap; no target/fresh claims, descriptive unadjusted CIs; one fixed provider for both evidence roles; no optimized parameters.')
 write(dest/'REPORT.json',report);write(dest/'COMPLETE.json',dict(report_sha=sha(dest/'REPORT.json')));print('SCORED',report['decision'],comps['TS'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['policy001','wrong001']);a=p.parse_args()
 try:main(a.stage)
 except BaseException as e:
  write(D/(a.stage+'_SCORER_FAILURE.json'),dict(error=repr(e),traceback=traceback.format_exc()));raise
