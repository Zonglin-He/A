"""Write anonymous A0.4 report from already audited, sealed terminal evidence."""
import sys,time,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a04_screen import D,A3
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,total_prior

def main():
    reports={str(n):read(D/f'S{n}/PUBLIC_REPORT.json') for n in [200,2000]}
    audits={str(n):read(D/f'S{n}/ROOT_READBACK.json') for n in [200,2000]}
    assert all(a['status']=='passed' for a in audits.values())
    assert reports['200']['decision']=='continue_same_trajectory_to_2000'
    assert reports['2000']['decision']=='stop_offline_predictor_tuning_next_R16_optimization_feasibility'
    receipts={str(n):dict(worker=read(OUT/f'a04_s{n}/RECEIPT.json')['seconds'],wrapper=read(OUT/f'a04_s{n}_wrapper/RECEIPT.json')['seconds']) for n in [200,2000]}
    total=sum(sum(x.values()) for x in receipts.values());free=shutil.disk_usage(ROOT).free
    summary=dict(status='completed_independently_audited',screen='A0.4 Fixed Shared-R16 Factorized Predictor',
      train=dict(queries=128,parents=95),dev=dict(queries=64,parents=16,exposed=True),
      settings=dict(local_hidden=128,feature_dim=128,state_dim=33,evidence_dim=8,rank=16,trainable_parameters=76688,seed=20260928,batch=4,lr=.001,wd=0,clip=1,radius=.13545580427763146),
      stages=reports,audits=audits,actual_unique_Adam_steps=2000,new_steps_by_stage=[200,1800],
      coefficient_fields_audited=384,receipts=receipts,GPU_allocation_and_nonoverlap_wrapper_seconds=total,
      cumulative_GPU_seconds=total_prior(),cap=None,CPU_target_preparation_seconds=read(D/'REGISTRATION.json')['CPU_prepare_seconds'],
      CPU_audit_seconds=sum(a['CPU_seconds'] for a in audits.values()),free_bytes=free,
      scientific_worker_failures=0,wrapper_status_failure_count=1,wrapper_status_repair='isolated v2 launcher; original scientific worker and pins retained; no GPU replay',
      receipt_limit='S200 original wrapper receipt measures completed child and waiting overhead; subsequent FileExistsError traceback tail was not timed. Original receipt retained, not falsely labeled process-success evidence.',
      native_run=False,new_PTD_loads=0,new_labels=False,fresh_read=False,new_full_training=False,
      decision=reports['2000']['decision'],next_stage='proposed/untested R16 per-query coefficient optimization feasibility; no new objective or run registered',
      interpretation='Current local offline predictor still fails Train128 direction gate after output256->16 and bounded same-trajectory rescue. This does not invalidate the measured R16 source oracle gain or prove all offline prediction impossible.')
    write(D/'PUBLIC_COMPLETION_REPORT.json',summary)
    lines=['# A0.4 Fixed Shared-R16 Factorized Predictor — completed / independently audited','',
      'The original local H128 predictor did not learn the R16 coefficient field on Train128. The predeclared direction gate failed at200 and after continuation to2000 actual Adam steps. No PTD/native evaluation was run.','',
      'Only output128→256 changed to128→16. Frozen source features128, state33, evidence8, union256, Train-only B16, seed, AdamW.001/wd0/clip1, batch4 and query-global cosine remained fixed. 76688 trainable parameters. Source-GT cached targets; Train128/95parents and repeatedly exposed Dev64/16parents. Fresh31/388 untouched.','',
      '| Endpoint | Split | R16 cosine mean | Median | >.1 | >.3 | Full-oracle cosine mean | Full median |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for endpoint,r in reports.items():
        for split,v in r['results'].items():lines.append(f"| S{endpoint} | {split} | {v['mean']:.9f} | {v['median']:.9f} | {v['count_gt_point1']}/{v['defined']} | {v['count_gt_point3']}/{v['defined']} | {v['full_oracle_mean']:.9f} | {v['full_oracle_median']:.9f} |")
    lines+=['','Gate: Train median>=.3 AND Dev median>=.1. S200 Train<.3 authorized restoring the same model/Adam/RNG/sample-order trajectory for1800 additional steps. No new seed, best-step or fresh restart. At2000 the gate still fails; stop offline predictor architecture tuning. No rank32/width/step/attention search.','',
      'S200 terminal training batch loss .963999152; train/dev mean direction loss .972460554/.979875919. S2000 terminal batch loss1.000358611; train/dev mean direction loss .963047714/.982668277. A batch loss is not the whole training set loss. S200 gradient norm [.0170942,.868863], clip0/200; cumulative S2000 norm [.00621910,24.1491184], final2.000422, clipping503/2000. This optimization behavior is retained; the result is not proof that every offline function class is incapable.','',
      'All384 complete terminal coefficient fields were independently recomputed with NumPy. Maximum reported cosine/summary error2.77556e-15; full-cosine = R16-cosine × sqrt(projected-energy) error1.94289e-16 per query. FP64-projected target to FP32 rounding relative L2<=3.54255e-8. B16 equals A0.3 columns1–16 exactly; no basis refit, sign change, or Dev fitting. Hidden initialization equals original A0; all8 trainable tensors changed and all3 buffers remained exact. Integer Adam keys/live-Parameter state and actual counters1–2000, entire sample-order RNG and continuation passed. Three synthetic CPU controls are distinct from this real GPU execution.','',
      'S200 science worker finished and sealed normally. Its original wrapper then attempted to overwrite mutable ACTIVE using a write-once helper and raised FileExistsError. The original code/pins/status/failure remain. An isolated v2 launcher changed only mutable status handling and continued the saved trajectory; no GPU replay. The original receipt is child completion accounting, not proof of wrapper process exit0; the later traceback tail has no captured duration.','',
      f"Measured worker and nonoverlap waiting/wrapper times: S200 {receipts['200']['worker']:.9f}+{receipts['200']['wrapper']:.9f}s, continuation {receipts['2000']['worker']:.9f}+{receipts['2000']['wrapper']:.9f}s; total {total:.9f}s. Cumulative ledger {total_prior():.11f}s, cap=null. Target preparation and independent audits are CPU-only and separately recorded. No research artifacts deleted.",'',
      'A0.3 Oracle-R16 still retains its measured +10.938378pp Dev vIoU and83.3888% full-oracle gain; those are privileged source oracle results, not A0.4 learned results. A useful shared action subspace does not ensure that this local offline predictor can infer the required query-dependent THW coefficients.','',
      'Next candidate, not implemented/registered/run: R16 coefficient-field per-query optimization feasibility. A source supervised controllability check and a genuinely label-free TTA objective are different questions. Existing oracle targets cannot be described as available at test time. No trust/no-op gate is entered because direction learnability did not pass. No native/full447/full618/fresh/expert/OPD/target expansion.']
    (D/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (D/'DECISION_AND_NEXT.md').write_text('# A0.4 decision\n\n'+summary['interpretation']+'\n\nPredeclared Case D: stop offline predictor tuning. Next candidate only: per-query R16 coefficient-field optimization feasibility. Before execution lock the actual objective and distinguish source-GT supervision from any label-free TTA signal; no existing protocol here authorizes silently reusing sourceGT at deployment or inventing an unsupervised objective. No native run or larger cache generation follows this failed direction gate.\n')
    write(D/'ROOT_COMPLETION_SUMMARY.json',dict(status='completed_independently_audited',report_sha=sha(D/'PUBLIC_COMPLETION_REPORT.json'),native_run=False,cumulative_GPU_seconds=total_prior(),cap=None,free_bytes=free))
    write(D/'COMPLETE.json',dict(status='completed',decision=summary['decision'],report_sha=sha(D/'PUBLIC_COMPLETION_REPORT.json')))
    state=dict(status='completed_independently_audited',time=time.time(),decision=summary['decision'],active_GPU=False,next='R16 optimization feasibility proposed only',native_run=False)
    write(D/'CURRENT_HANDOFF.json',state);(D/'ACTIVE.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['stages','audits']},indent=2))
if __name__=='__main__':main()
