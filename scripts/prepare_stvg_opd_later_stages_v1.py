"""Pre-score matched rosters for P2–P6; preparation is not experiment execution."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *

def prepare():
    if (BASE/'LATER_DESIGN_LOCK.json').exists():return read(BASE/'LATER_DESIGN_LOCK.json')
    d=read(BASE/'DESIGN_LOCK.json');stages={}
    for ds in DATASETS:
        data=d['datasets'][ds];cross=data['source'];same='vidstg' if ds=='vidstg' else 'hcstvg2'
        panel=data['confirmation_query_ordinals'];orders=data['confirmation_orders'];cfg=data['config']
        common=dict(dataset=ds,parents=panel,observation_budget=4,configuration=cfg,
            input_cohort='exact original locked confirmation roster',GT_inference=False)
        for phase,arms in [('P2',['direct_L1_GIoU','shuffled_feedback','frozen_rollout','on_policy']),
                           ('P3',['query_only','LN_only','joint_alpha0','on_policy'])]:
            for setting in ['cross_clean','same_5percent']:
                conds=['clean'] if setting=='cross_clean' else [f'{f}_5' for f in FAMILIES]
                chosen_orders=orders if setting=='cross_clean' else {'order1':orders['order1']}
                stages[f'{phase}_{ds}_{setting}']=dict(**common,source=cross if setting=='cross_clean' else same,
                    split=setting,orders=chosen_orders,conditions=conds,arms=arms,
                    adapted_logical_arrivals=len(conds)*sum(map(len,chosen_orders.values()))*len(arms),
                    Frozen_logical_arrivals=len(conds)*sum(map(len,chosen_orders.values())),
                    variant_configs={a:{**cfg,'writeback':0.} if a=='joint_alpha0' else cfg for a in arms},
                    reuse_identical_presealed_streams=True)
        conds=d['later_stages']['P4']['conditions']
        stages[f'P4_{ds}']=dict(dataset=ds,source=same,split='same_domain_robustness',
            parents=data['balanced_one_query_per_parent'],orders={'order1':data['balanced_order']},
            conditions=conds,arms=['on_policy'],variant_configs={'on_policy':cfg},
            observation_budget=4,adapted_logical_arrivals=len(conds)*len(data['balanced_order']),
            cohort_unit='parent source',GT_inference=False,reuse_identical_presealed_streams=True)
        for k in [1,2,4,8]:
            stages[f'P5_{ds}_K{k}']=dict(**{**common,'observation_budget':k},source=cross,
                split='budget_cross_clean',orders=orders,conditions=['clean'],arms=['on_policy'],
                variant_configs={'on_policy':cfg},adapted_logical_arrivals=256,
                observation_budget_selection_for_main=False,reuse_identical_presealed_streams=True)
        diag=sorted(panel,key=lambda q:digest(['opd-paper-temporal-diagnostic-v1',ds,q]))[:32]
        stages[f'P6_{ds}']=dict(dataset=ds,source=cross,split='matched_offline_diagnosis',
            parents=diag,orders={'order1':[q for q in orders['order1'] if q in diag]},
            conditions=['clean'],observation_budget=4,config=cfg,
            representative_temporal_signal='existing final-method actionness projection and temporal-head update',
            GT_head_oracle='offline after all deployable diagnostic arm predictions sealed',
            Native_WHEN_and_spatial_OPD='reuse exact P0 order1 input and fit when identical',
            new_temporal_branch=False,source_selection_without_current_P0_GT=True)
    result=dict(status='locked_plans_not_executed',stages=stages,
        P0_gate_precedes_expensive_main_execution=True,original_design_sha256=sha(BASE/'DESIGN_LOCK.json'),
        parent_panel_shared=True,all_conditions_reset_from_source=True,time=time.time())
    write(BASE/'LATER_DESIGN_LOCK.json',result)
    return result

if __name__=='__main__':
    result=prepare();print('LATER_ROSTERS_LOCKED_NOT_EXECUTED',len(result['stages']))
