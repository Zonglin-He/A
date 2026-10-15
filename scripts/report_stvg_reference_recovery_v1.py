"""Anonymous full positive/negative/null report of the fixed recovery study."""
import sys,math,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_reference_recovery_v1';OUT=ROOT/'results/stvg_reference_recovery/2026-10-15'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 for stage in ['Optimizer_replay','LN_carry']:
  b=BASE/stage;assert read(b/'ROOT_READBACK.json')['status']=='pass'
  for name in ['POSTSEAL_ROWS.json','POSTSEAL_SUMMARY.json','PARENT_PAIRED_DELTAS.json']:
   d=read(b/name)
   if name=='POSTSEAL_ROWS.json':
    for row in d['rows']:row['support']={k:v for k,v in row['support'].items()if not isinstance(v,list)}
   write(OUT/(stage+'_'+name),d)
  root=read(b/'ROOT_READBACK.json');root.pop('records',None);write(OUT/(stage+'_ROOT_READBACK.json'),root)
  write(OUT/(stage+'_CONFIGURATION.json'),read(b/'DESIGN_LOCK.json'))
 inventory=read(BASE/'E1_SOURCE_INVENTORY.json')
 write(OUT/'E1_CACHE_FEASIBILITY.json',dict(status='complete_cache_inventory',summary=inventory['summary'],limitations=inventory['limitations'],full_private_inventory_sha256=sha(BASE/'E1_SOURCE_INVENTORY.json'),new_teacher_inference=0,new_matched_reference_swap_fits=0))
 write(OUT/'ORDERS.json',dict(status='locked',records=[{k:e[k]for k in ['dataset','order','case_ids']}for e in read(BASE/'LN_carry/ORDERS.json')['records']],independent_source_count=32,arrival_count=96,GT_order_selection=False))
 bindings=[]
 for stage in ['Optimizer_replay','LN_carry']:
  r=read(BASE/stage/'RUNTIME.json');bindings.append(dict(stage=stage,runtime_sha256=sha(BASE/stage/'RUNTIME.json'),local_static_python_closure_count=r['local_static_python_closure_count'],source_pins={p:h for p,h in r['pins'].items()if p.endswith('.py')and p.startswith(('scripts/','vg_tta/','methods/','models/','util/','datasets/'))},protected=r['protected']))
 write(OUT/'SOURCE_BINDINGS.json',dict(status='pass',records=bindings))
 write(OUT/'EXECUTION_COST.json',dict(status='complete',optimizer=read(BASE/'Optimizer_replay/STUDENT_GLOBAL_BARRIER.json'),carry={k:v for k,v in read(BASE/'LN_carry/STUDENT_GLOBAL_BARRIER.json').items()if k not in ['files','streams']},teacher_calls=0,old_queue_restarted=False,raw_end_to_end_teacher_latency_remeasured=False,limitation='Saved teacher/proposal/temporal/episodic costs are reused; this is actual incremental recovery-stage cost. Optimizer files are hashed private artifacts, not public weights.'))
 # Stage execution and clean exits are actual records, not launch snapshots.
 for stage,execution in [('Optimizer_replay','execution'),('LN_carry','execution002')]:
  exit_record=read(BASE/stage/execution/'WORKER_EXIT.json');assert exit_record['exit_code']==0
  assert not list((BASE/stage/execution).glob('*RESOURCE_STOP_RECEIPT.json'))
  write(OUT/(stage+'_WORKER_EXIT.json'),dict(exit_record,own_resource_guard_stop=False))
 # Only counts and custody hashes from the optimizer barrier are public.
 cost=read(OUT/'EXECUTION_COST.json');cost['optimizer'].pop('files',None)
 (OUT/'EXECUTION_COST.json').write_text(__import__('json').dumps(cost,indent=2)+'\n')
 for src,dst in [('Optimizer_replay/ROOT_INDEPENDENT_OPTIMIZER_READBACK.json','ROOT_INDEPENDENT_OPTIMIZER_READBACK.json'),('ROOT_OPTIMIZER_SCALAR_READBACK.json','ROOT_OPTIMIZER_SCALAR_READBACK.json'),('ROOT_CARRY_SCALAR_READBACK.json','ROOT_CARRY_SCALAR_READBACK.json'),('LN_carry/ROOT_STREAM_CPU_READBACK.json','ROOT_INDEPENDENT_CARRY_STATE_READBACK.json'),('LN_carry/ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json','ROOT_INDEPENDENT_CARRY_MATH_READBACK.json')]:
   audit=read(BASE/src);write(OUT/dst,{k:v for k,v in audit.items()if k not in ['files','records','bindings','pins']})
 e2=read(BASE/'Optimizer_replay/POSTSEAL_SUMMARY.json')['summary'];e3=read(BASE/'LN_carry/POSTSEAL_SUMMARY.json')['summary']
 lu={(r['dataset'],r['arm'],r['metric']):r['mean']for r in e2};lc={(r['dataset'],r['order'],r['arm'],r['metric']):r['mean']for r in e3}
 lines=['# Fixed-reference and LN-state recovery','',
 'This experiment separates an update-configuration package from persistent LN state using 32 previously exposed development sources, 16 per domain. It is not an untouched confirmation or full Table 1. All 49 frozen Weight-free Top-1 anchors across 23 sources are retained. No new teacher, candidate generator, GT-assisted reference, bootstrap or CURRENT promotion.','',
 '## E1: strictly matched cached reference swap','',inventory['summary']['conclusion'],'',
 'No reference-swap fit was executed: missing same-frame cached expert evidence is an execution support limitation, not evidence that stronger expert references would fail.','',
 '## E2: original configuration package, fixed reference and Source reset','',
 '| Arm | HC vIoU (%) | VidSTG vIoU (%) |','|---|---:|---:|']
 for arm in ['F','T','S_current','TS_current','S_original','TS_original']:lines.append(f"| {arm} | {100*lu['hc2',arm,'m_vIoU']:.6f} | {100*lu['vidstg',arm,'m_vIoU']:.6f} |")
 lines+=['',
 'The original HC package is lr .01, sigma .025, tau .05, 40 steps; Vid is .03, .1, .25, 10 steps. Both retain 32 antithetic samples, the current validated-IoU gate and the same reference weights/loss. The controls reuse exact parent current-five-step predictions. New Source-reset fits and actual complete Student readouts were executed only for the 23 sources with references. Thus this changes the full configuration package, not a single parameter, and does not reproduce the original DINO/online system.','',
 '## E3: original LN writeback, current five-step update','',
 '| Dataset | Order | Frozen | Temporal | Episodic S | Online Before S | Online After S | Episodic T+S | Online Before T+S | Online After T+S |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 arms=['F','T','Episodic_Native','Before_Native','After_Native','Episodic_T','Before_T','After_T']
 for ds in ['hc2','vidstg']:
  for order in ['order1','order2','order3']:lines.append('| '+ds+' | '+order+' | '+' | '.join(f"{100*lc[ds,order,a,'m_vIoU']:.6f}"for a in arms)+' |')
 lines+=['','Across the three prelocked orders, Online T+S is 34.700258% HC and 10.906455% Vid, versus Episodic 34.643557%/10.784159%: +0.056701/+0.122296 pp. Every order is positive in this matched contrast. Dense spatial IoU rises by +0.057029/+0.423407 pp. This is a small development-stream effect and does not reproduce the old full-population inherited-state advantage. The original configuration package is harmful on HC and helpful on Vid relative to current5; no universal rollback is justified.','']
 lines+=['',
 'Each stream uses the exact cohort subsequence of an original P1 ordinal order; these are six 16-query streams, not full historical streams. Every arrival resets Query256 and Adam; only LN1536 carries. The commit is incoming LN + alpha*(accepted fit LN - incoming LN), with alpha .0625 HC and .125 Vid. Each new stream resets to Source. The query-key random sampling is identical to the saved Episodic fit. Empty-reference arrivals preserve and actually decode their incoming memory.','',
 'Before and After are independently executed complete Student readouts for every arrival, verified against the saved decoder replay and Source model restoration. Native and fixed Offset-Trust intervals share identical spatial boxes. F/T and Episodic controls are exact saved reuse, not three new independent Episodic runs. There are 32 independent sources and 96 arrivals, not 96 independent sources.','',
 'The three primary descriptive contrasts are Before-Frozen, After-Before, and OnlineAfter-Episodic. All per-source gains/losses, order sensitivity and dense/supervised/unsupervised spatial metrics are in the anonymous JSON. Full vIoU cannot be equated with anchor IoU or dense sIoU. These descriptive contrasts do not prove that inherited state has semantically correct knowledge.','',
 '## Validation and limits','',
 'Every stage sealed all predictions and passed independent source, saved policy/reward/softmax, state-chain/Adam/rollback and complete-readout checks before GT scoring. GT was never used in reference construction, acceptance, carry, stream order or update configuration. Independent saved arithmetic does not constitute an independent backward recomputation. No best-order/step/source selection and no bootstrap.','',
 'Root corrected the unpinned draft carry worker ABI (parent arm names, Source capture, readout calling convention and stable seed) before any GPU execution. The locked code is the actual executed revision. The actual resources and wall times are recorded separately; no historical teacher latency is counted as zero cost for the whole method.','',
 'Current conclusions apply to this fixed Weight-free reference pool and these short development streams. Do not generalize a mean or a negative order to all online LN memory, all OPD, or a new target cohort. CURRENT and the human-paused full evaluation remain unchanged.','',
 '![Carry effect on full tube](../results/stvg_reference_recovery/2026-10-15/LN_CARRY_RESULTS.png)','',
 '[Locked protocol](../protocols/stvg_reference_recovery_v1.md) · [Complete anonymous outputs](../results/stvg_reference_recovery/2026-10-15).','']
 (ROOT/'docs/STVG_REFERENCE_RECOVERY_V1.md').write_text('\n'.join(lines))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10.3,4.1))
 for ax,ds,title in zip(axs,['hc2','vidstg'],['HC-STVG-v2','VidSTG']):
  x=list(range(3));w=.36
  for shift,a,b,label,color in [(-w/2,'After_Native','Episodic_Native','Native time','#708b9d'),(w/2,'After_T','Episodic_T','Fixed Temporal','#b47c56')]:
   y=[100*(lc[ds,o,a,'m_vIoU']-lc[ds,o,b,'m_vIoU'])for o in ['order1','order2','order3']];bars=ax.bar([z+shift for z in x],y,width=w,label=label,color=color)
   for bar,v in zip(bars,y):ax.annotate(f'{v:+.3f}',(bar.get_x()+bar.get_width()/2,v),ha='center',va='bottom'if v>=0 else'top',xytext=(0,3 if v>=0 else -3),textcoords='offset points',fontsize=8)
  ax.set_xticks(x,['Order 1','Order 2','Order 3']);ax.axhline(0,color='#777',lw=.8);ax.set_title(title+' (16 exposed sources)');ax.set_ylabel('Online After - Episodic vIoU (pp)');ax.spines[['top','right']].set_visible(False);ax.margins(y=.3);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(OUT/'LN_CARRY_RESULTS.png',dpi=180);plt.close(fig)
 print('recovery report saved')
if __name__=='__main__':main()
