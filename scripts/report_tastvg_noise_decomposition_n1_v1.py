"""N1 readout with explicit oracle and pairwise-objective boundaries."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_noise_decomposition_n1_v1 import OUT,verify,PRIMARY,MATCHED
from scripts.audit_tastvg_noise_decomposition_n1_public_v1 import run as audit

def run():
 p=verify();a=read(OUT/'ACROSS_ORDERS.json');s=read(OUT/'SIGNAL_SUMMARY.json');write(OUT/'PUBLIC_AUDIT.json',audit(OUT));tier=read(OUT/'TIER0_READBACK.json')
 lines=['# N1: GT-audited teacher preference noise decomposition','', '**Mechanism oracle; not deployable TTA.** GT filters the original teacher signs on 16 already-exposed sources. The five main arms use pairwise softplus, not the frozen method\'s reverse KL. No method promotion or Paper48 restart.','', 'Five shared-source orders; each order has clean and five 5% transient conditions. Values below are corruption source-macro percentage-point changes versus Frozen, mean ± sample SD across orders. Orders share the same sources.','', '| Arm | Future ΔsIoU pp | Future ΔvIoU pp | Whole ΔvIoU pp |','|---|---:|---:|---:|']
 names={'all':'All (pairwise reference)','useful':'Useful-only','noisy':'Noisy-only','useful_positive':'Useful-Positive','useful_negative':'Useful-Negative','useful_matched':'Useful-Matched','noisy_matched':'Noisy-Matched','RKL-Final':'Frozen recipe RKL-Final (context)','Fast-only':'Temporal Fast-only (context)'}
 for arm in PRIMARY+MATCHED+['RKL-Final','Fast-only']:
  values=[]
  for sub,k in [('nonexpert','delta_s'),('nonexpert','delta_v'),('all','delta_v')]:
   x=a[arm]['corruption'][sub][k];values.append(f"{100*x['mean']:+.6f} ± {100*x['sample_std']:.6f}")
  lines.append('| '+names[arm]+' | '+' | '.join(values)+' |')
 lines+=['','## Signal exposure during corruption streams','', 'Counts pool the five order repeats; they are not independent signal samples. Supports and labels are recomputed at each arm\'s own evolving state.','', '| Arm | Selected signals | Updated arrivals | Mean gradient norm on updates |','|---|---:|---:|---:|']
 for arm in PRIMARY+MATCHED:
  v=s[arm]['corruption'];lines.append(f"| {names[arm]} | {v['selected_signals']} | {v['updates']} | {v['mean_gradient_norm']} |")
 lines+=['','## Historical Tier-0 check','',f"Frozen S0.6 support: corruption {tier['groups']['corruption']['comparisons']} comparisons, {tier['groups']['corruption']['decisive']} decisive, correct positive/negative 208/208 and noisy positive/negative 78/74. Noisy fraction 26.7606%; clean 25%. These are source-state statistics, not assumed to apply unchanged to adapted trajectories.",'','## Interpretation limits','', 'Useful/Noisy retain Sa2VA\'s original signs; GT chooses membership only. All retains every teacher-decisive signal, including GT ties. No valid spatial expert or empty subset means no update. The supplementary pair chooses the same count at each scheduled arrival, using the smaller of its two current eligible pools and a fixed hash. It subsamples Noisy when required and must not be confused with the primary unfiltered Noisy arm.', '', 'Each selected loss is averaged over its subset. SGD is fixed at .005, without gradient-norm matching or loss-scale tuning. Differences can include changed gradient scale, state-dependent eligibility and empty-update frequency. Exact-count matching addresses signal counts/update opportunities but does not make the state trajectories equal.', '', 'The within-N1 All arm is the positive control for these pairwise-loss statements. Comparing its absolute effect to historical RKL does not establish RKL\'s mechanism. In particular, low Noisy effect cannot establish that RKL absorbs noise, and positive Noisy effect alone cannot identify self-regularization. GT-filtered oracle gains do not demonstrate a GT-free reliability rule.', '', 'Paper48 remains paused at temporal 1039/2658 and spatial 2658/2658. Its original budget files and all completed artifacts are preserved. No baselines, new models or hyperparameter sweeps were launched.', '']
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 alloc=[read(f) for f in OUT.glob('**/allocations/*.json')]
 write(OUT/'RESOURCES.json',dict(GPU_seconds=sum(x['seconds'] for x in alloc),failed_attempts=sum(x['status']!='completed' for x in alloc),accepted_arrivals=3360,new_expert_inferences=0,GT_filter_queries=16))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
 axes[0].bar(['Correct +','Correct −','Noisy +','Noisy −'],[208,208,78,74],color=['#198754','#198754','#ce5a45','#ce5a45']);axes[0].set_ylabel('Center-to-probe comparisons');axes[0].set_title('Frozen-source teacher directions\n568 decisive; 26.76% noisy')
 labels=['All','Useful','Noisy','Useful +','Useful −','Useful M','Noisy M'];values=[100*a[k]['corruption']['nonexpert']['delta_v']['mean'] for k in PRIMARY+MATCHED];err=[100*a[k]['corruption']['nonexpert']['delta_v']['sample_std'] for k in PRIMARY+MATCHED]
 axes[1].bar(labels,values,yerr=err,capsize=3,color=['#657786','#198754','#ce5a45','#3286b7','#734fa3','#74b68c','#d99a8e']);axes[1].axhline(0,color='black',linewidth=.8);axes[1].tick_params(axis='x',rotation=25);axes[1].set_ylabel('Future ΔvIoU (percentage points)');axes[1].set_title('GT-filtered pairwise mechanism oracle\nMean ± SD, five shared-source orders')
 fig.savefig(OUT/'N1_ANALYSIS.png',dpi=160);fig.savefig(OUT/'N1_ANALYSIS.pdf');plt.close(fig)
 write(OUT/'COMPLETION.json',dict(status='completed',publication='pending',GT_assisted_oracle=True,deployable=False,frozen_method_unchanged=True,paper48_remains_paused=True,audit_sha256=sha(OUT/'AUDIT.json'),time=time.time()))
if __name__=='__main__':run()
