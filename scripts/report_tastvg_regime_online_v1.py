"""Report the fixed O2 experiment without promoting a method or tuning a scale."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_regime_online_capture_v1 import OUT,verify
from scripts.audit_tastvg_regime_online_public_v1 import audit


def run():
    p=verify();rows=read(OUT/'analysis/ROWS.json');s=read(OUT/'analysis/SUMMARY.json');d=read(OUT/'analysis/READOUT_DIAGNOSTIC.json');rr=[r for r in rows if not r['expert']]
    primary=s['macro']['nonexpert']['comparisons']['Online Slow-Fast - Budgeted Rerank'];dt=primary['tIoU']['mean'];dv=primary['vIoU_corrected']['mean'];changed=[r for r in rr if r['selected']['Online Slow-Fast']!=0]
    if dt>0 and dv>0:conclusion='Positive mean transfer appears in this fixed coherent-regime screen. Its size, changed cases and conditional intervals delimit the support; locality is not uniquely identified as the cause.'
    elif dt==0 and dv==0:conclusion='The unchanged linear slow state yields exactly zero mean non-expert tIoU/vIoU gain in these five coherent streams. Examine actual selection changes before interpreting this as a failure of transferable representations.'
    elif dt<=0 and dv<=0:conclusion='The unchanged linear slow state does not yield positive mean non-expert transfer in this coherent-regime screen. The result does not by itself identify representation quality or rule out Slow-Fast adaptation.'
    else:conclusion='The coherent-regime screen is mixed across temporal and tube transfer. It does not establish a consistent task improvement.'
    resources=[read(f) for f in (OUT/'gpu_allocations').glob('*.json')]
    resource=dict(total_gpu_process_seconds=sum(z['seconds'] for z in resources),allocations=resources,new_frozen_capture_cells=80,new_student_offset_forwards=160,new_model_backwards=0,new_expert_calls=0,new_online_SGD_steps=20,online_CPU_seconds=read(OUT/'ONLINE_BARRIER.json')['seconds'],logical_calls={'Frozen':0,'Budgeted Rerank':20,'Online Slow-Fast':20,'Full Rerank':80},cached_online_expert=20,cached_full_extra=60,includes_model_loading_and_CPU_decode=True,excludes_development_scoring_and_historical_cache_cost=True,launcher_incident='Initial direct invocation of the non-executable CUDA wrapper exited126 before Python/model launch; rerun through bash succeeded. Original log retained; GPU failure count excludes this shell-only attempt.')
    write(OUT/'RESOURCES.json',resource);write(OUT/'PUBLIC_AUDIT.json',audit(OUT/'analysis'))
    count=lambda sign,m:sum(sign*(r['arms']['Online Slow-Fast'][m]-r['arms']['Budgeted Rerank'][m])>1e-12 for r in rr)
    decision=dict(status='completed',measurement='audited',conclusion=conclusion,primary_delta=primary,nonexpert_changed=len(changed),nonexpert_total=60,positive={m:count(1,m) for m in ['tIoU','vIoU_corrected']},negative={m:count(-1,m) for m in ['tIoU','vIoU_corrected']},production_changed=False,followon='none; no prototype, normalization, gate, LR or alpha sweep',scope='five fixed exposed16-source trajectories, four expert writes each; O1 used32 sources/eight writes, so comparison is not an isolated causal estimate of coherence')
    write(OUT/'DECISION.json',decision)
    old=read(ROOT/'artifacts/tastvg_sparse_online_o1_v1/PROVENANCE.json');old.update(parents=16,arrivals=80,conditions=p['conditions'],source_repetitions=4,source_occurrences=5,stream_orders=1,coherent_streams=5,state_resets=5,expert_writes_per_stream=4,alpha=1.,historical_development=True,GT_for_adaptation=False,production_changed=False,student_state_sha256=read(OUT/'CAPTURE_BARRIER.json')['model_state_sha256']);write(OUT/'PROVENANCE.json',old)
    lines=['# O2: Regime-Coherent Online Transfer','',conclusion,'','Same original16 C3 sources in five persistent regimes: frame drop, frame freeze, motion blur, occlusion and exposure, each using the already-fixed5% random-burst coverage. Same source-only O1 hash order restricted to these16, repeated across conditions. Four expert positions1/5/9/13 per stream; state resets to zero for each condition.','',
      '## Primary: non-expert arrivals','', '| Condition | N | Frozen/Budgeted tIoU (%) | Online tIoU (%) | Delta t (pp) | Frozen/Budgeted vIoU (%) | Online vIoU (%) | Delta v (pp) | Changed |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in [*p['conditions'],'macro']:
        z=s[name]['nonexpert'];a=z['arms'];delta=z['comparisons']['Online Slow-Fast - Budgeted Rerank'];nchanged=len(changed) if name=='macro' else d[name]['nonexpert_changed']
        values=[a['Budgeted Rerank']['tIoU']['mean'],a['Online Slow-Fast']['tIoU']['mean'],delta['tIoU']['mean'],a['Budgeted Rerank']['vIoU_corrected']['mean'],a['Online Slow-Fast']['vIoU_corrected']['mean'],delta['vIoU_corrected']['mean']]
        lines.append('| '+name+f" | {60 if name=='macro' else 12} | "+' | '.join(f'{v*100:.4f}' for v in values)+f' | {nchanged} |')
    lines+=['','Macro equally weights the five conditions. Since their source order and expert schedule coincide, uncertainty resamples12 non-expert parents AFTER averaging each parent across five conditions. This is60 cells, not60 independent sources. All-arrival secondary macro uses16 parents. The10000-sample seed20260929 paired intervals condition on these fixed shared trajectories; they do not rerun state evolution.','',
      '| Condition | Delta t 95% interval (pp) | Delta v 95% interval (pp) | Full agreement Online / Native | Final weight norm | Expert losses down |','|---|---:|---:|---:|---:|---:|']
    for name in [*p['conditions'],'macro']:
        delta=s[name]['nonexpert']['comparisons']['Online Slow-Fast - Budgeted Rerank'];ci=lambda m:'['+', '.join(f'{v*100:+.4f}' for v in delta[m]['ci95'])+']'
        tail=f"{d[name]['full_agreement']}/12 / {d[name]['native_full_agreement']}/12 | {d[name]['final_norm']:.8f} | {d[name]['loss_decreased']}/4" if name!='macro' else f"{sum(x['full_agreement'] for x in d.values())}/60 / {sum(x['native_full_agreement'] for x in d.values())}/60 | per-stream | {sum(x['loss_decreased'] for x in d.values())}/20"
        lines.append(f"| {name} | {ci('tIoU')} | {ci('vIoU_corrected')} | {tail} |")
    lines+=['','Positive means are concentrated in two nonexpert cases; three regimes have exactly unchanged predictions and both macro intervals touch zero. This is a weak local positive signal, not evidence that coherent shifts solve slow transfer. Neither useful change exactly matches Full Rerank, so top1 agreement remains1/60.','',
      '| Condition | Residual favors Full non-native choice | Median native gap | Median residual advantage |','|---|---:|---:|---:|']
    for condition in p['conditions']:
        z=d[condition];items=[v for v in z['details'] if v['full_selected']!=0]
        lines.append(f"| {condition} | {z['positive_residual_toward_full']}/{len(items)} | {np.median([v['full_native_gap'] for v in items]):.8f} | {np.median([v['full_residual_advantage'] for v in items]):.8f} |")
    lines+=['','## Four-arm reference and negative tails','', '| Arm | Logical expert calls | Nonexpert macro tIoU (%) | Nonexpert macro vIoU (%) | All-arrival macro tIoU (%) | All-arrival macro vIoU (%) |','|---|---:|---:|---:|---:|---:|']
    for arm in resource['logical_calls']:
        vals=[s['macro'][g]['arms'][arm][m]['mean']*100 for g in ['nonexpert','all'] for m in ['tIoU','vIoU_corrected']]
        lines.append('| '+arm+f" | {resource['logical_calls'][arm]} | "+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['',f"Online versus Budgeted nonexpert vIoU gains/losses/unchanged: {count(1,'vIoU_corrected')}/{count(-1,'vIoU_corrected')}/{60-count(1,'vIoU_corrected')-count(-1,'vIoU_corrected')}. >5pp cell harms: {sum(r['arms']['Online Slow-Fast']['vIoU_corrected']-r['arms']['Budgeted Rerank']['vIoU_corrected']<-.05 for r in rr)}. All per-condition and parent-macro harms remain in SUMMARY.json.",'','Full Rerank is a higher-budget reference, not a correctness oracle. Its agreement is diagnostic only. The public ROWS.json retains all candidate metrics, teacher/native/arrival scores, selections, and both useful and harmful cases.','',
      '| Changed nonexpert | Parent | Condition | Frozen t/v (%) | Online t/v (%) | Full t/v (%) |','|---|---|---|---:|---:|---:|']
    for r in changed:
        value=lambda a:'/'.join(f"{r['arms'][a][m]*100:.4f}" for m in ['tIoU','vIoU_corrected'])
        lines.append(f"| {r['stream_position']} | Q{r['parent']+1:02} | {r['condition']} | {value('Frozen')} | {value('Online Slow-Fast')} | {value('Full Rerank')} |")
    if not changed:lines.append('| None | — | all five | unchanged | unchanged | see all rows |')
    lines+=['','## Mechanism and limits','',
      'The method is unchanged from O1: raw768D final temporal hidden start/end/span mean, native-envelope base score, zero w+b, FP64 CPU strict-pair logistic loss, one plain SGD .001 per expert, alpha1. Emit expert output before writing future state; no current teacher read at nonexpert arrivals. No normalization, gate, prototype, LR/alpha search, or H/model updates. All20 sparse teacher outputs were loaded separately; extra60 reference outputs were accessed only after all online states/output were sealed.','',
      'READOUT_DIAGNOSTIC.json reports the ACTUAL O2 residual advantage and native gap toward Full on each nonexpert arrival. It does not amplify residuals or rerun a trajectory. O1.1 showed that old O1 directions could move choices when amplified; that does not guarantee sufficient readout scale after four unchanged O2 writes. Zero gain with unchanged selections is therefore not a decisive representation-failure result.','',
      'O2 changes stream construction while keeping method parameters fixed, as requested, but its16-source/four-write design differs from O1\'s32-source/eight-write design. Outcomes cannot uniquely attribute any difference to regime coherence. These historically exposed development sources and single fixed orders do not establish robust deployment transfer.','',
      '## Verification, cost and reproduction','',
      '80 captures exactly match old pixels/preprocessed inputs, boxes and both offset logits; frozen model state is unchanged. Native zero-state selection holds on80/80. Independent NumPy audit reconstructs all80 arrival states and20 SGD writes, including all five resets, feature definitions, hash chronology and absence of nonexpert teacher reads. Predictions for all arms were sealed before reading original16 GT keys. Each candidate metric was checked using two independent metric routines, and all80 critic scores were recomputed from proposals. Public audit reconstructs486 mean/interval checks and matched controls.','',
      f"New GPU process time: {resource['total_gpu_process_seconds']:.3f}s (80 frozen inputs /160 offset forwards, including loading/CPU decoding). No new expert computation;20 online +60 reference evidence items reused exactly. Online CPU loop: {resource['online_CPU_seconds']:.3f}s. Cache reuse preserves logical expert budgets and does not represent fresh-input end-to-end latency. Initial shell wrapper permission failure occurred before Python/model startup and was resolved by invoking the existing wrapper with bash; retained launch_failure.log. No changes to system permissions.",'',
      'Existing O1 three CPU contract tests pass. Resource metadata includes exact stage allocation and any model-worker failure. Production CURRENT and old evidence are unchanged. No follow-on experiment launched.','',
      'Reproduction: run_tastvg_regime_online_capture_v1.py prepare/run -> run_tastvg_regime_online_teacher_v1.py online -> run_tastvg_regime_online_v1.py online -> audit_tastvg_regime_online_v1.py -> teacher full -> online full -> analyze_tastvg_regime_online_v1.py -> report_tastvg_regime_online_v1.py. Public readback: audit_tastvg_regime_online_public_v1.py <result-directory>. Raw media, labels, features and state tensors stay local.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'COMPLETION.json',dict(status='completed',files={n:sha(OUT/n) for n in ['REPORT.md','STATE_AUDIT.json','UTILITY_AUDIT.json','PUBLIC_AUDIT.json','RESOURCES.json','DECISION.json','PROVENANCE.json','analysis/ROWS.json','analysis/SUMMARY.json']},time=time.time()))
    print(conclusion);print('PRIMARY',primary)

if __name__=='__main__':run()
