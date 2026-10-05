"""Recompute anonymous public tables without labels, weights, media or private artifacts."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_public_result_io_v1 import read
from scripts.score_decota_identity_commitment_v1 import public_check
from scripts.decota_duration_bias_v1 import panel
from scripts.decota_matrix_common_v1 import write
from scripts.score_decota_optimizer_posterior_v1 import stats
import numpy as np

def run(folder):
    p=Path(folder);assert public_check(p/'matched','matched') and public_check(p/'online','online')
    rows=read(p/'duration/ROWS.json');summary=read(p/'duration/SUMMARY.json');checks=0
    for r in rows:
        assert abs(r['r_expert']-(r['expert_log_duration']-r['native_log_duration']))<1e-12
        assert abs(r['r_GT']-(r['GT_log_duration']-r['native_log_duration']))<1e-12;checks+=2
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            for dom in ['same','cross']:
                for name in ['corruption','clean']:
                    rr=[r for r in rows if r['dataset']==ds and r['split']==sp and r['domain']==dom and (r['condition']=='clean')==(name=='clean')]
                    assert panel(rr)==summary[ds][sp][dom][name];checks+=1
    online=read(p/'online/ROWS.json');sums=read(p/'online/SUMMARY.json');rob=read(p/'online/SCHEDULE_ROBUSTNESS.json');cfg=read(p/'online/CONFIGURATION.json')
    from scripts.derive_decota_identity_persistence_v1 import derive
    pr,ps=derive(online);assert pr==read(p/'online/MATCHED_PERSISTENCE_ROWS.json') and ps==read(p/'online/MATCHED_PERSISTENCE_SUMMARY.json');checks+=len(pr)
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            for rate in [25,50]:
                streams=[f'seed{s}_budget{rate}' for s in range(5)];rr=[r for r in online if r['dataset']==ds and r['split']==sp and r['stream'] in streams and r['condition']!='clean']
                z=rob[ds][sp][str(rate)];means=[sums[ds][sp][s]['corruption']['metrics']['vs_frozen_v']['mean'] for s in streams]
                assert z['mean']==float(np.mean(means)) and z['range']==[min(means),max(means)]
                assert z['source_bootstrap_conditional_on_five_schedules']==stats(rr,['vs_frozen_v','before_vs_frozen_v','vs_before_v','vs_episodic_v']);checks+=3
            for seed in range(5):
                a=cfg['schedules'][ds][sp][f'seed{seed}_budget25'];b=cfg['schedules'][ds][sp][f'seed{seed}_budget50'];assert set(a)<=set(b)
    decision=read(p/'matched/DECISION.json');gate={}
    ms=read(p/'matched/SUMMARY.json')
    for ds in ['vidstg','hc2']:
        z=ms[ds]['search']['map_contrastive']['corruption'];gate[ds]=z['tails']['harm_gt20pp']==0 and z['metrics']['vs_top1_v']['mean']>=0 and z['metrics']['vs_top1_v']['ci95'][0]>=-.005
    assert decision['search_gate']==gate and decision['arm']==('map_contrastive' if all(gate.values()) else 'top1') and not decision['uses_confirmation']
    audit=dict(status='pass',rows=3456+13824+540,public_scalar_and_aggregate_checks=checks,search_selection_independently_recomputed=True,bootstrap_and_schedule_tables_recomputed=True,private_assets_required=False)
    if (p/'PUBLIC_GLOBAL_AUDIT.json').exists():assert read(p/'PUBLIC_GLOBAL_AUDIT.json')==audit
    else:write(p/'PUBLIC_GLOBAL_AUDIT.json',audit)
    print('PUBLIC_IDENTITY_AUDIT_PASS',audit,flush=True)
if __name__=='__main__':run(sys.argv[1])
