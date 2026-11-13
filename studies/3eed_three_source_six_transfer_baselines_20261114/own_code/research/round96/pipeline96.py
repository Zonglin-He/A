"""Explicit one-shot serial training/evaluation task; no timer or recurring automation."""
from pathlib import Path
import json,subprocess,sys,hashlib,datetime,time
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';PY=R/'crossplatform/env/bin/python';rd=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def phase(name,**kw):
 s=rd(H/'STATE.json');s.update(state='R96_RUNNING_'+name.upper(),current_stage=name,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());s.update(kw);dump(H/'STATE.json',s);print('PIPELINE96',name,flush=True)
def child(script,args,log):
 with log.open('a') as f:r=subprocess.run([str(PY),str(script),*map(str,args)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{script.name} exited {r.returncode}; see {log}')
def main():
 reg=rd(H/'experiment/ACTIVE_TRAINING_REGISTRATION.json');assert rd(H/'experiment/ENGINEERING_REVIEW.json')['state'].startswith('PASS_');assert rd(H/'experiment/RESUME_AUDIT.json')['state'].startswith('PASS_')
 old=rd(H/'experiment/INITIAL_RESEARCH_STATE_PRIVATE.json');started=time.time()
 try:
  for i,pl in enumerate(['drone','quad']):
   p=Path(reg[pl]['train']['path']);assert sha(p)==reg[pl]['train']['sha256'];out=B/(pl+'_source100');phase('training_'+pl,active_source=pl,queued_sources=['quad'] if i==0 else [],expected_epochs=100)
   if (out/'TRAINING_COMPLETE.json').exists():
    co=rd(out/'TRAINING_COMPLETE.json');assert co['primary_epoch']==100 and co['trained_platforms']==[pl] and co['plan_sha256']==sha(p);continue
   args=['--plan',p,'--plan-sha256',sha(p),'--output',out]
   if out.exists():
    resume=out/'resume_latest.pth';assert resume.exists(),'Incomplete first epoch has no complete-epoch resume; manual diagnosis required, no rerun or deletion.';args+=['--resume',resume]
   child(B/'train_source.py',args,H/(pl+'_formal100.log'));co=rd(out/'TRAINING_COMPLETE.json');assert co['primary_epoch']==100 and co['trained_platforms']==[pl]
  phase('auditing_completed_source_models',active_source=None,queued_sources=[]);child(H/'audit_training96.py',[],H/'training_audit.log')
  phase('six_frozen_cross_platform_predictions');child(H/'run_matrix96.py',[],H/'matrix.log')
  phase('postseal_frozen_matrix_scoring');child(H/'score_matrix96.py',[],H/'matrix_score.log')
  summary=rd(H/'frozen_scores_attempt001/ALL_METHOD_ROWS.json');assert sum(x['arm']=='last_text_logodds' for x in summary)==6
  state='R96_TRAINING_AND_SIX_FROZEN_BASELINES_COMPLETE_PENDING_PUBLICATION';phase('complete_pending_publication',state=state,execution_open=False,GPU_stage_authorized=False,CPU_scoring_authorized=False,source_training_authorized=False,training_completed_sources=['drone','quad'],frozen_directions_completed=6,publication_pending=True,TTA_performed=False,total_pipeline_seconds=time.time()-started)
  g=rd(R/'research/STATE.json')
  for k in ['avis','affordance']:assert g['directions'][k]==old['directions'][k]
  g.update(run_state=state,current_stage='R96_complete_pending_publication',next_action='independent final result review and public export');g['directions']['crossplatform'].update(state=state,report='round96/REPORT.md',latest_completed_round='round96_official_three_source_six_transfer',active_experiment='none_queued_after_R96_source_baselines',R96_TTA_performed=False);dump(R/'research/STATE.json',g)
  with (R/'research/ARCHIVE.md').open('a') as f:f.write('\n\n## R96 — source100训练与六方向冻结基线完成，待公开同步\n\nSee round96/REPORT.md, TRAINING_AUDIT.json, frozen_scores_attempt001/SCORING_RECEIPT.json. No TTA claims; protected frames excluded.\n')
 except BaseException as e:
  phase('failed_preserved',failure_type=type(e).__name__,failure_message=str(e),all_partial_artifacts_retained=True);raise
if __name__=='__main__':main()
