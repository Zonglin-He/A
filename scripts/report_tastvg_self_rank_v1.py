"""Post-seal A2 report; no final-method promotion."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_self_rank_v1 import OUT,verify
from scripts.audit_tastvg_self_rank_public_v1 import run as audit

def run():
 p=verify();a=read(OUT/'ACROSS_ORDERS.json');write(OUT/'PUBLIC_AUDIT.json',audit(OUT))
 lines=['# Self-Rank: student spatial preference control','', '16 exposed VidSTG sources, five locked orders, clean and five 5% conditions. Frozen Final unchanged. Each row is mean ± sample SD over the five shared-source orders.','', '| Arm | Whole Δv pp | Future Δs pp | Future Δv pp | Future arm−Final pp |','|---|---:|---:|---:|---:|']
 for arm in ['Frozen','Fast-only','Slow-only','Final','random_rank']+p['arms']:
  cells=[]
  for sub,key in [('all','delta_v'),('nonexpert','delta_s'),('nonexpert','delta_v'),('nonexpert','minus_final_v')]:
   x=a[arm]['corruption'][sub][key];cells.append(f"{100*x['mean']:+.6f} ± {100*x['sample_std']:.6f}")
  lines.append('| '+arm+' | '+' | '.join(cells)+' |')
 lines+=['', 'Self-Rank uses descending ranks of the detached student probability; the nine candidate boxes and rank teacher remain detached. Temporal reranking, nine probes, 1792 parameters, SGD .005 and the 25% schedule are unchanged. No Sa2VA output was read by this arm.', '', 'Admission caveat (declared before execution): Self-Rank attempts all 120 scheduled updates, whereas Final and Random-Rank made 102 updates because Sa2VA had no valid mask at 18 positions. Therefore this is a spatial-teacher-free control, not a perfectly rank-only causal intervention. No extra teacher-derived validity gate was introduced.', '', 'Metrics reproduce the sampled-grid development endpoint. The five orders share the same 16 exposed sources and do not establish performance on fresh sources. All order-level values, clean results, harms and update diagnostics are retained. A result here cannot prove external knowledge is universally necessary or identify the fraction of gain due to knowledge transfer.', '']
 paired={sub:{metric:[100*(a['Final']['corruption'][sub][metric]['values'][j]-a['self_rank']['corruption'][sub][metric]['values'][j]) for j in range(5)] for metric in ['delta_s','delta_v']} for sub in ['all','nonexpert']}
 write(OUT/'PAIRED_EXPERT_MINUS_SELF.json',paired)
 (OUT/'REPORT.md').write_text('\n'.join(lines));alloc=[read(f) for f in OUT.glob('**/allocations/*.json')]
 write(OUT/'RESOURCES.json',dict(GPU_seconds=sum(x['seconds'] for x in alloc),failed_attempts=sum(x['status']!='completed' for x in alloc),accepted_arrivals=480,all_attempt_completed_arrivals=sum(x['done'] for x in alloc),new_expert_inferences=0))
 write(OUT/'COMPLETION.json',dict(status='completed',publication='pending',frozen_method_unchanged=True,claim_scope='Fixed exposed-panel Self-Rank with admission caveat; no winner promotion',audit_sha256=sha(OUT/'AUDIT.json'),time=time.time()))
if __name__=='__main__':run()
