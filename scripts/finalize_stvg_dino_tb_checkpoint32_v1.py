"""Publication-ready interpretation and execution receipt; no model/GT actions."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
from scripts.run_stvg_dino_tb_checkpoint32_v1 import D,P,verify

def main():
 verify(True);a=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert a['status']=='pass'
 unit=subprocess.check_output(['systemctl','--user','show','stvg-dino-tb-checkpoint32-001.service','-p','ActiveState','-p','SubState','-p','MainPID','-p','Result','-p','ExecMainStatus','-p','MemoryPeak','-p','CPUUsageNSec'],text=True);unit=dict(line.split('=',1)for line in unit.splitlines());assert unit['ActiveState']=='inactive'and unit['MainPID']=='0'and unit['Result']=='success'and unit['ExecMainStatus']=='0'
 put(P/'EXECUTION_READBACK.json',dict(status='pass',owned_unit=unit,barrier= {k:v for k,v in read(D/'GLOBAL_PREDICTION_BARRIER.json').items()if k!='files'},CPU_scoring_completed=True,CPU_independent_audit=a['CPU_seconds'],other_services_not_signalled=True))
 put(P/'INTERFACE_READBACK.json',dict(T_B_image_processor_exact=True,all108_actual_model_inputs_text_span_RGB_exact=True,context_branch_calls=100,old_phrase_fallback_calls=8,unresolved_slots=20,actual_top_rule='context: highest target score in mixed-NMS top3 retained pool; fallback: original mixed-score top1 singleton',posterior_probabilities_not_claimed=True,all32_cached_T_E_exact=True,first_T4_replay_box_score_exact=True,raw_B_CPU_score_max_error=a['max_errors']['raw_scores'],interpretation_requires_same_score_threshold_not_same_probability_calibration=True))
 report=P/'REPORT.md';s=report.read_text().replace('Actual top means the highest target-score candidate after the unchanged mixed-score NMS/top3 path.', 'Actual top follows the unchanged branch:100 context calls use the highest target-score candidate after mixed-score NMS/top3;8 phrase-fallback calls use their original mixed-score top1 singleton.')
 s+='''
## Explicit decision and representative mechanisms

Do not replace the current T configuration or start OPD from this result. B has actual localized advantages, especially on the eight Vid parents with known called diagnostic support, but the complete-tube gain is not established at2–3pp and both main paired intervals cross zero. This is not an OPD experiment and cannot establish OPD failure.

- NH012: T admits0/4 despite high-IoU top boxes at several positions; B admits4/4. Full v improves+37.50577pp (3.38430%→40.89007%). This combines improved disambiguation/admission with candidate changes, not just tighter coordinates.
- NH003: B admits4/4 but all four actual tops have IoU≤.109; complete v falls−25.75847pp. A larger checkpoint can reinforce the wrong target under the same text/span and thresholds. High score is not correctness.
- NV012: T actual tops are already near.90 but E chooses zero-IoU alternatives. B selects its actual tops and restores the complete trajectory, +26.32300pp full v. This is not evidence that T lacked the correct object; changed scores/candidate geometry affect the estimator's choice.
- NV011 position21: B actual top IoU.956369 is present and admitted; E selects alternative index2 with IoU0, and output remains0. NH001 position42 likewise top.847178→selected0. These are exact retained-evidence selection losses, not missing teacher observations.
- NV001: fixed temporal overlap is0, so complete v stays0 while all-known unobserved actual s drops−72.72386pp. Preserve this spatial damage; do not hide it by restricting the main actual-node metric to the predicted interval.

Signed full-v counts: HC7 improve/9 worsen; Vid4 improve/3 worsen/9 unchanged. Sensitivity only: omitting NH012 gives the remaining15 HC mean−1.91496pp; omitting NV012 gives the remaining15 Vid mean−.23044pp. Main results keep every parent. The mean positives are concentrated and do not justify an automatic independent confirmation or checkpoint promotion.

The concrete surviving finding is that actual B evidence can be lost at reference selection, while B also creates severe errors. Any later OPD proposal would still need reliable object/reference supervision and matched Direct/Fixed/Live evidence; this completed comparison does not authorize that next experiment. See POST_SEAL_ANALYSIS.json and DECISION.json.
'''
 report.write_text(s)
 (P/'ROUND_REVIEW.md').write_text('''# Root review: official T/B checkpoint-only32

Completed and measurement valid. Official B revision and LFS bytes/hash verified. T scientific baseline exact reconstruct32; fresh T4 matches all inputs, scores and candidate identities exactly. B108 original full-RGB calls preserve all text/span/processor tensors. Five unresolved what queries unchanged; normal fallback8 and context100 both retained. Frozen weights before/after equal, no new Source/training/OPD updates. All32 double outputs seal before new GT.

Independent CPU replay checks108 raw logits/NMS/gates,64 E trajectories,108 input identities,192 official-versus-dense metrics,128 actual-node metrics,132 frame qualities and10 full parent bootstrap contrasts. Maximum raw score discrepancy1.19e−7 and metric5.55e−16, E reconstruction exactly0. CPU audit only, not an independent second GPU run. No inference/scoring failure or scientific protocol recovery occurred. The official HF download took305.14s, separate from78.95s GPU allocation. Four T timing calls are a small cold-start-sensitive timing check, not enough for a B/T throughput claim.

B−T full v HC+.54883pp CI[−4.92027,6.87322], Vid+1.42915[−.73890,5.07156]; actual unobserved HC+2.39270[−5.07466,11.77933], Vid−.01151[−13.87719,15.15023]. All main intervals cross0. Actual top Vid+15.49812pp conditional on8 parents is a positive diagnostic; not an all16 deployed gain. HC actual top+4.15056 interval crosses0. Correct and wrong admissions, changed E selection, successful NH012/NV012 and severe NH003/NV001 kept. No dual-domain admission gate is imposed.

Decision: no current checkpoint replacement, OPD run, threshold sweep or automatic follow-up. Preserve the actual teacher advantages and exact reference-selection loss counterexamples; neither claim reliable2–3pp nor universal B/OPD failure. Model capacity and pretraining differ together, and equal score thresholds do not equal calibration. All data were previously exposed developer parents. Formal cohorts and CURRENT/MAIN/PAPER stay unchanged.

Implementation, locked configuration/protocol, complete signed/unknown32 parent and128 slot scalar results, provenance, cost and negative findings exported explicitly; raw original IDs/text/media/annotations/coordinates/weights/tensors excluded. Publication receipt is verified separately, never inferred from staging.
''')
 f=D/'PUBLIC_FILES_FINAL.json';m=read(f);m['files']+=['scripts/analyze_stvg_dino_tb_checkpoint32_v1.py','scripts/finalize_stvg_dino_tb_checkpoint32_v1.py',str((P/'EXECUTION_READBACK.json').relative_to(ROOT)),str((P/'INTERFACE_READBACK.json').relative_to(ROOT))];assert len(m['files'])==len(set(m['files']));put(f,m,True);print('finalized full public report and',len(m['files']),'files plus publisher/review')

if __name__=='__main__':main()
