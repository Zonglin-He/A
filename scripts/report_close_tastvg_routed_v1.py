"""Publish measured closing readouts without altering the frozen experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_routed_common_v1 import *

def run():
    from scripts.score_tastvg_best_quick_v1 import source_summary
    from scripts.audit_tastvg_routed_public_v1 import audit
    root=read(BASE/'ROOT_CLOSING_READBACK.json');assert root['status']=='pass'
    public_audit=audit(PUBLIC);write(BASE/'PUBLIC_SCALAR_AUDIT.json',public_audit);write(PUBLIC/'PUBLIC_AUDIT.json',public_audit)
    for ds in DATASETS:
        ar={a:read(PUBLIC/ds/a/'ROWS.json') for a in ['A','R']};pairs=[]
        for a,r in zip(ar['A'],ar['R']):
            if not a['expert_scheduled']:continue
            assert (a['parent'],a['condition'],a['order'],a['arrival'])==(r['parent'],r['condition'],r['order'],r['arrival'])
            pairs.append({k:a[k] for k in ['source_id','condition','order','arrival']})
            pairs[-1]['delta_R_A_local_update']=r['delta_post_fixed_time']-a['delta_post_fixed_time']
        local={g:source_summary([r for r in pairs if (r['condition']!='clean')==(g=='corruption')],['delta_R_A_local_update']) for g in ['corruption','clean']}
        write(PUBLIC/ds/'LOCAL_UPDATE_PAIRED_ROWS.json',pairs);write(PUBLIC/ds/'LOCAL_UPDATE_PAIRED_SUMMARY.json',local)
        tokens=read(PUBLIC/ds/'TOKEN_ROWS.json');ordered=sorted([r for r in tokens if r['condition']!='clean' and r['delta_T_S_v'] is not None],key=lambda r:r['delta_T_S_v'])
        write(PUBLIC/ds/'TOKEN_CASES.json',dict(negative=ordered[:5],positive=[r for r in ordered if r['delta_T_S_v']>0][-5:],posthoc_examples_not_used_for_online_decisions=True))
    recovery=dict(all_failures_retained_privately=True,scientific_changes=0,GT_used_for_recovery_selection=False,records=[
        dict(id='code_receipt_schema_001',failure='Sa2VA official-code receipt values are metadata objects, not direct digest strings',phase='after A parity; before first R prediction or specialist inference',fix='verify nested sha256',revision='online001'),
        dict(id='restart_launch_002',failure='immutable LAUNCH file already exists on preserved restart',phase='controller initialization, no predictions',fix='mutable launch status update, without overwriting saved failed attempts',revision='online002'),
        dict(id='token_compact_schema_003',failure='compact old slow prediction omits logits',phase='first token replay, boxes/interval already parity, no token scores',fix='use same sealed arrival full first-step native prediction logits',revision='token001'),
        dict(id='token_manifest_metadata_004',failure='seal contains additive revision metadata as well as prediction receipts',phase='before GT scoring',fix='verify metadata hash separately, never edit the original seal',revision='token002'),
        dict(id='token_auditor_box_precision_005',failure='independent scalar ROI auditor computed FP32 corners while runtime converts boxes to FP64 first',phase='after token seal and first HC diagnostic GT opening',fix='convert auditor box to FP64 before corner arithmetic; retain 1e-12 criterion; no model or score change',revision='token003',GT_read_before_fix=True),
        dict(id='online_report_import',failure='missing repository root on Python import path',phase='offline report assembly',fix='add root to sys.path; preserve first failed log')],resource_scope='successfully completed worker wall time includes loading/decode/IO; saved unsuccessful startup durations not included; not pure GPU kernel time')
    write(PUBLIC/'ENGINEERING_RECOVERIES.json',recovery)
    decision=dict(status='completed_pending_verified_publication',online_R_improvement_established=False,token_P0_eligible={d:read(BASE/'token/ST_ELIGIBILITY.json')[d]['eligible'] for d in DATASETS},conditional_ST='not_triggered',combined_online_token='not_triggered',memory='not_run',method_promotion=False,scope='local negative qualification does not prove all event-aware routing or token binding impossible')
    write(PUBLIC/'DECISION.json',decision)
    status(BASE/'STATUS.json',dict(**decision,new_R_arrivals=768,reused_A_arrivals=768,total_online_scored=1536,token_cells=60,token_candidates=540,GT_read=True,time=time.time()))
    print('Closing readouts written; publication remains pending')
if __name__=='__main__':run()
