"""Render complete signed checkpoint comparison and public manifest."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_dino_tb_checkpoint32_v1 import D,P,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put

def stat(s):return f"{s['mean']:+.4f} [{s['CI95'][0]:+.4f}, {s['CI95'][1]:+.4f}]"if s['mean']is not None else 'unknown'

def main():
 verify(True);assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';z=read(P/'RESULTS_PUBLIC.json');cost=read(P/'COST_BREAKDOWN.json');g=z['groups'];lines=['# GroundingDINO-T / GroundingDINO-B checkpoint comparison32','','Completed, measured, independently checked; original NH16/NV16 exposed development parents, one query each. Frozen original TA-STVG Source, full-image four physical observation positions, target context/span, processor, NMS/admission/E source calibration/finalMBR unchanged. Official HF B checkpoint replaces T; no training, OPD, new Source or teacher ensemble. No promotion or automatic next run.','', 'All32 predictions sealed before new offline GT. All parents retained, unknown annotations not fabricated. The five unresolved Vid what queries still have0 calls; B performs108 forwards, not128. Four fresh T NH001 forwards are a numerical/input and small timing control; T scientific results otherwise come from exact cached baseline.','', '## Full-method results','','Mean changes are percentage points, parent-macro; intervals are paired10000 bootstrap, seed20261018.','', '| B E − T E | HC | Vid |','|---|---:|---:|']
 for ep in ['v','t','s','observed_actual_s','unobserved_actual_s']:lines.append('|'+ep+'|'+stat(g['hc2']['B_minus_T'][ep])+'|'+stat(g['vidstg']['B_minus_T'][ep])+'|')
 lines+=['','| Absolute complete vIoU (%) | HC | Vid |','|---|---:|---:|']
 for arm in ['E_MBR','B_E_MBR']:lines.append('|'+arm+'|'+f"{100*g['hc2']['means'][arm]['v']:.5f}"+'|'+f"{100*g['vidstg']['means'][arm]['v']:.5f}"+'|')
 lines+=['','## Actual teacher and E transfer','','Actual top means the highest target-score candidate after the unchanged mixed-score NMS/top3 path. It is not a GT oracle. Paired top changes average known shared called positions within query, then average parents with that diagnostic support; this support is narrower than the main16.','', '| Actual top B − T (IoU pp) | HC | Vid |','|---|---:|---:|','| change |'+stat(g['hc2']['actual_top_B_minus_T'])+'|'+stat(g['vidstg']['actual_top_B_minus_T'])+'|','']
 for ds in ['hc2','vidstg']:
  lines+=['### '+ds,'','| Teacher | Called / no call | Accepted | Parents with references | Known called / selected | Wrong admitted IoU<.3 / <.5 |','|---|---:|---:|---:|---:|---:|']
  for teacher in ['T','B']:
   s=g[ds]['stages'][teacher];lines.append(f"|{teacher}|{s['called']} / {s['no_call']}|{s['accepted']}|{s['parents_with_reference']}|{s['known_called']} / {s['known_selected']}|{s['wrong_accepted_lt03']} / {s['wrong_accepted_lt05']}|")
   lines+=[]
  lines+=['','| E transfer on own known selected support | T | B |','|---|---:|---:|']
  for k in ['selected_minus_top','propagated_minus_selected']:lines.append('|'+k+'|'+stat(g[ds]['stages']['T'][k])+'|'+stat(g[ds]['stages']['B'][k])+'|')
  lines+=['','Own-selected supports can differ: these are conditional stage diagnostics, not a paired whole-method comparison. Rejection reason counts and all128 planned slots, signed/unknown frame results and all32 parent outputs are saved in JSON. Empty B pools use Source via zero E correction, never an implicit T fallback.','']
 lines+=['## Signed cases and cost','','| Parent | B E − T E vIoU pp | Unobserved actual sIoU pp | Accepted T / B |','|---|---:|---:|---:|']
 for r in z['rows']:
  dv=100*(r['values']['B_E_MBR']['v']-r['values']['E_MBR']['v']);x=r['values']['B_E_MBR']['unobserved_actual_s'];y=r['values']['E_MBR']['unobserved_actual_s'];ss='unknown'if x is None or y is None else f'{100*(x-y):+.5f}';lines.append(f"|{r['alias']}|{dv:+.5f}|{ss}|{r['accepted_T']} / {r['accepted_B']}|")
 lines+=['',f"GPU allocation conservative upper bound {cost['GPU_allocation_seconds']:.4f}s includes load, numerical replay and final weight/hash checks plus3s guard. B108 synchronized forwards total {cost['B_forward_seconds']:.4f}s; observation pipeline {cost['B_observation_seconds']:.4f}s; peak allocated {cost['B_peak_allocated_bytes']/2**30:.4f}GiB. T4 NH001 forward {cost['T_timing_replay']['forward_seconds']:.4f}s vs B NH001 {cost['B_NH001_forward_seconds']:.4f}s; T peak {cost['T_timing_replay']['peak_allocated_bytes']/2**30:.4f}GiB. First forwards include cold-start effects; this four-frame timing is not a full matched benchmark. All-T historical forward time {cost['old_T_forward_seconds']:.4f}s is reported separately, not used to claim a controlled speed ratio. Official HF download/verification {cost['download_cost']:.2f}s and old cache acquisition are separate. No student updates or model training.",'','## Interpretation and limits','','The checkpoint replacement changes backbone capacity and pretraining together. Equal thresholds are a fixed protocol, not proof of calibration equivalence; no target-GT calibration or prompt sweep was done. This experiment tests actual B teacher capability and unchanged E transfer, not OPD. Direct readout failure cannot be called OPD failure. Dataset-specific decisions and any remaining potential must be stated from measured contrasts; no B/Direct/Fixed/Live continuation is automatic. All developer exposure remains disclosed.','', 'See RESULTS_PUBLIC.json, FRAME_DIAGNOSTICS_PUBLIC.json, ROOT_INDEPENDENT_AUDIT.json, MODEL_PROVENANCE.json and COST_BREAKDOWN.json.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 files=['scripts/run_stvg_dino_tb_checkpoint32_v1.py','scripts/audit_stvg_dino_tb_checkpoint32_v1.py','scripts/report_stvg_dino_tb_checkpoint32_v1.py','protocols/stvg_dino_tb_checkpoint32_v1.md']+[str(f.relative_to(ROOT))for f in sorted(P.glob('*.json'))]+[str((P/'REPORT.md').relative_to(ROOT))]
 put(D/'PUBLIC_FILES_FINAL.json',dict(files=files));print('rendered',len(files),'allowlist files')

if __name__=='__main__':main()
