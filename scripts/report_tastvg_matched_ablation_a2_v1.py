"""Post-seal A2 report; no final-method promotion."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_matched_ablation_a2_v1 import OUT,verify
from scripts.audit_tastvg_matched_ablation_a2_public_v1 import run as audit

def run():
 p=verify();a=read(OUT/'ACROSS_ORDERS.json');write(OUT/'PUBLIC_AUDIT.json',audit(OUT))
 lines=['# A2: matched raw/rank and pairwise/distributional preference','', '16 exposed VidSTG sources, five locked orders, clean and five 5% conditions. Frozen Final unchanged. Each row is mean ± sample SD over the five shared-source orders.','', '| Arm | Whole Δv pp | Future Δs pp | Future Δv pp | Future arm−Final pp |','|---|---:|---:|---:|---:|']
 for arm in ['Frozen','Fast-only','Slow-only','Final']+p['arms']:
  cells=[]
  for sub,key in [('all','delta_v'),('nonexpert','delta_s'),('nonexpert','delta_v'),('nonexpert','minus_final_v')]:
   x=a[arm]['corruption'][sub][key];cells.append(f"{100*x['mean']:+.6f} ± {100*x['sample_std']:.6f}")
  lines.append('| '+arm+' | '+' | '.join(cells)+' |')
 lines+=['','Raw-RKL reproduces the original S1 raw-IoU softmax teacher (temperature 1). Pairwise averages logistic loss over strictly expert-ordered candidate pairs and excludes ties. Both keep SGD .005 and all other frozen settings. Loss scale was not tuned or norm matched; conclusions apply to this configuration. Clean, every order, harm counts and source effects are retained in the companion JSON. On-policy superiority remains unsupported by A1.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines));alloc=[read(f) for f in (OUT/'allocations').glob('*.json')]
 write(OUT/'RESOURCES.json',dict(GPU_seconds=sum(x['seconds'] for x in alloc),failed_attempts=sum(x['status']!='completed' for x in alloc),new_arrivals=960,new_expert_inferences=0))
 write(OUT/'COMPLETION.json',dict(status='completed',publication='pending',frozen_method_unchanged=True,claim_scope='Fixed matched losses; no winner promotion',audit_sha256=sha(OUT/'AUDIT.json'),time=time.time()))
if __name__=='__main__':run()
