"""Anonymous completed direction-screen report; a passing signal cannot be marked done here."""
import sys,json,shutil,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a05_signal import D,SIGNALS
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,total_prior

def main():
    r=read(D/'ROOT_SIGNAL_READBACK.json');cross=read(D/'ROOT_SUMMARY_CROSSCHECK.json')
    assert r['status']==cross['status']=='passed' and not r['passing_signals'],'Passing signals require conditional native first'
    receipts={s:dict(worker=read(OUT/('a05_'+s)/'RECEIPT.json')['seconds'],wrapper=read(OUT/('a05_'+s+'_wrapper')/'RECEIPT.json')['seconds']) for s in ['probe001','rest001']}
    for s in receipts:
        assert read(OUT/('a05_'+s)/'RECEIPT.json')['status']=='completed' and read(OUT/('a05_'+s)/'EXIT.json')['code']==0
    total=sum(sum(v.values()) for v in receipts.values());missing={s:{b:sum(x['native_missing'][b] for x in r['results'][s]) for b in ['event','spatial']} for s in SIGNALS}
    pub=dict(status='completed_independently_audited',stage='A0.5 R16 Unlabeled Direction Signal Audit',queries=16,parents=16,
      selection='SHA256 one per previously exposed Dev64 parent; no outcome selection',signals=r['aggregates'],cases=r['results'],counts=cross['counts'],
      native_missing=missing,B1_format_support=cross['B1_format_support'],
      gates=dict(median_cosine=.10,each_GT_available_branch_positive_descent_fraction=.65),
      decision=r['decision'],passing_signals=[],actual_backwards=r['actual_backwards'],optimizer_steps=0,new_native_predictions=0,new_GT_oracle_backwards=0,
      GT_use='Sealed source GT branch gradients/Oracle-R16 only in CPU qualification and readout; no labels in GPU losses. Fixed subspace learned earlier from source GT. Not an unseen-target claim.',
      audit={k:r[k] for k in ['max_scalar_loss_error','max_first_chain_relative_error','max_raw_norm_direction_error','max_reference_projection_relative_error']},
      summary_crosscheck={k:cross[k] for k in ['max_torch_geometry_aggregate_error','checks','selection_independently_reconstructed','basis_reconstruction_max_abs','basis_gram_max_abs']},
      receipts=receipts,GPU_allocation_and_wrapper_seconds=total,cumulative_GPU_seconds=total_prior(),cap=None,
      CPU_prepare_seconds=read(D/'REGISTRATION.json')['CPU_prepare_seconds'],CPU_first_audit_seconds=read(D/'ROOT_FIRST_READBACK.json')['CPU_seconds'],CPU_all_audit_seconds=r['CPU_seconds'],CPU_summary_check_seconds=cross['CPU_seconds'],
      failed_GPU_runs=0,GPU_replays=0,engineering_failures=['Pre-GPU FP32 identity-KL gradient assertion overly strict; preserved and replaced with dtype-aware bound','Pre-GPU auditor syntax error and dependent missing preflight receipt; original code/partial preparation preserved, fresh registration after fix'],
      fresh_read=False,target_read=False,PTD_parameters_updated=False,free_bytes=shutil.disk_usage(ROOT).free,
      next='proposed/untested expert pseudo-gradient qualification; no expert/OPD/GPU stage auto-started',
      limitations=['One repeatedly exposed source query per16parents; exactly two predeclared objectives at C0','Branch balancing in R16 differs from old oracle full-F balancing before projection','Consistency gradient from mild F is transported to observed F for GT local-descent comparison','Direction/local first-order signs are not finite native utility; no failed signal received native inference','No loss/augmentation/weight/radius grid; finite precision/autograd limitations remain'])
    write(D/'PUBLIC_COMPLETION_REPORT.json',pub)
    lines=['# A0.5 R16 Unlabeled Direction Signal Audit — completed / independently audited','',
      'Neither predeclared signal passed all three direction gates. No finite native intervention was run; stop internal self-supervised objective tuning and make expert pseudo-gradient qualification the next proposed stage.','',
      f"Frozen official PTD4B/B1, union256 and A0.3 Train-only B16; C0=0 in the shared R16 coefficient field. Original v2 mild RGB brightness1.05/contrast.95 for consistency, observed input for entropy. Detached teacher and both student branches use the same original B1 semantic reference/interval/anchors/native block schedule. Spatial loss includes all 152775 classes, temporal includes actual observed endpoint classes. No GT-validity filtering of unlabeled coordinates. {r['actual_backwards']} actual backwards, 0 optimizer; no offline predictor training.",'',
      '| Signal | Mean cosine | Median cosine | Positive temporal GT descent | Positive spatial GT descent | Direction gate |',
      '|---|---:|---:|---:|---:|---|']
    for s,a in r['aggregates'].items():
        pe=a['positive_GT_descent'];av=a['available_GT'];lines.append(f"| {s} | {a['cosine_mean']:.9f} | {a['cosine_median']:.9f} | {pe['event']}/{av['event']} | {pe['spatial']}/{av['spatial']} | {'PASS' if a['passed'] else 'FAIL'} |")
    lines+=['','Gate was locked at median>=.10 and both supported-branch positive descent fractions>=.65. Strict positive dots; missing GT support only changes that branch denominator. All16 queries and both signals retained. Zero direction has undefined mathematical cosine and neutral0 only for the declared gate; no fabricated supervision.','',
      '| Anonymized query | Consistency cosine | C temporal dot | C spatial dot | Entropy cosine | E temporal dot | E spatial dot |','|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return 'missing' if v is None else f'{v:.9g}'
    for i in range(16):
        c=r['results']['U-Consistency'][i];e=r['results']['U-Entropy'][i]
        lines.append('| Q%02d | %s |'%(i+1,' | '.join(fmt(v) for v in [c['cosine'],c['local_descent']['event'],c['local_descent']['spatial'],e['cosine'],e['local_descent']['event'],e['local_descent']['spatial']])))
    lines+=['','The normalization order is explicit: requested unlabeled branch gradients are normalized *after projecting into C*, whereas the historical analytic oracle normalized full-F gradients before projection. Both use equal-branch balancing but are not the same numeric construction. The saved Oracle-R16 direction is reused unchanged; no new GT oracle backward. GT branch gradients are projected only in a separate CPU reference/readout path and are absent from signal computation.','',
      'For consistency, the mild-view gradient is transported to the observed common THW/R16 coordinate system. A positive GT local dot at observed F is a first-order diagnostic, not measured finite tube improvement. Entropy confidence can reinforce an incorrect answer. This screen does not establish universal failure of all self-supervision or impossibility of label-free adaptation.','',
      f"Independent CPU validation: scalar losses max error {r['max_scalar_loss_error']:.8g}; first real-query coefficient chain relative error {r['max_first_chain_relative_error']:.8g}; raw norm/direction error {r['max_raw_norm_direction_error']:.8g}; original GT projection relative error {r['max_reference_projection_relative_error']:.8g}. Second Torch/standard-library aggregation checked {cross['checks']} values with max error {cross['max_torch_geometry_aggregate_error']:.8g}. Observed replay logits exact at every available branch, physical input/context identity and frozen parameter scope verified. Complete mild student logits and all coefficient gradients are sealed locally.",'',
      'Engineering failures retained: initial synthetic identity-KL gradient residual1.163e-10 exceeded an arbitrary1e-10 test bound; fixed before GPU to a dtype-aware bound. An auditor syntax typo prevented preflight receipt creation, and a subsequent prepare stopped with partial config/input/basis files; original code/partial preserved and fresh registration used. Neither was a GPU run or a scientific gate change. No GPU failure/replay.','',
      f"Measured GPU worker/nonoverlap wrapper total {total:.9f}s; cumulative {total_prior():.11f}s cap=null. CPU preparation/audits separately recorded. Free disk {pub['free_bytes']}bytes exceeds8GiB. All research workers exited, no artifact deletion.",'',
      'Next proposal only: TVG/SVG expert pseudo-target gradient qualification on locked native support and the same R16 reference. It needs its own provider/provenance/token-support protocol before running. No additional entropy/augmentation/weight variants, iterative updates, full447/full618/fresh388/target or OPD were started.']
    (D/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (D/'DECISION_AND_NEXT.md').write_text('# A0.5 decision\n\n'+r['decision']+'\n\nNo signal passed all direction gates; failed signals receive no finite native inference. Stop internal objective/augmentation/loss-weight variants. Next is expert pseudo-gradient qualification, proposed/untested. Define actual provider, unavailable/invalid pseudo-target handling, fixed student-native support and matching gradients before a separate run; do not substitute incompatible expert token logits or jump to OPD. Preserve all positive/negative case evidence and source exposure.\n')
    write(D/'ROOT_COMPLETION_SUMMARY.json',dict(status='completed_independently_audited',report_sha=sha(D/'PUBLIC_COMPLETION_REPORT.json'),cumulative_GPU_seconds=total_prior(),native_predictions=0))
    write(D/'COMPLETE.json',dict(status='completed',decision=r['decision'],report_sha=sha(D/'PUBLIC_COMPLETION_REPORT.json')))
    state=dict(status='completed_independently_audited',decision=r['decision'],active_GPU=False,time=time.time(),next='expert pseudo-gradient qualification proposed only')
    write(D/'CURRENT_HANDOFF.json',state);(D/'ACTIVE.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({k:v for k,v in pub.items() if k not in ['cases','B1_format_support']},indent=2))
if __name__=='__main__':main()
