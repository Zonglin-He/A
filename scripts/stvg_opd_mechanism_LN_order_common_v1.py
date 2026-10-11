"""Only the missing LN-only streams; immutable existing three-arm streams reused."""
import copy,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,save,load
from scripts.stvg_opd_design_ablation_common_v1 import scientific,activate_old,clone_state
BASE=ROOT/'artifacts/stvg_opd_mechanism_review_v1/cross_parent_LN'
PUB=ROOT/'results/stvg_opd_mechanism_review/2026-10-11/cross_parent_LN'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
PAPER=ROOT/'artifacts/decota_paper_experiments_v1'
PREVIOUS=ROOT/'artifacts/stvg_opd_supplement_v1/orders'
PYTHON=ROOT/'.conda/tubedetr/bin/python';ARMS=['LN_only'];QUERY='spatial.query_residual';ALLOCATOR='expandable_segments:True'

def audit(fit,expert):
    from vg_tta.stvg_opd_p3_scope_precision001 import audit as qualified
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit as original
    return qualified(fit,expert,original)

def initial_state(source,previous,arm):
    import torch
    if arm!='LN_only':raise ValueError('Unknown scope')
    if previous is not None and (set(source)!=set(previous) or any(source[n].shape!=previous[n].shape for n in source)):raise ValueError('Different full state schema')
    result=clone_state(source if previous is None else previous);result[QUERY]=torch.zeros_like(result[QUERY]);return result

def committed(initial,final,alpha,arm):
    import torch,math
    if arm!='LN_only' or isinstance(alpha,bool) or not isinstance(alpha,(float,int)) or not math.isfinite(alpha) or not 0<alpha<1:raise ValueError('Invalid fixed consolidation')
    if set(initial)!=set(final) or any(initial[n].shape!=final[n].shape for n in initial):raise ValueError('Different full state schema')
    if not torch.equal(initial[QUERY],final[QUERY]):raise ValueError('Frozen query changed')
    return {n:torch.zeros_like(v) if n==QUERY else v+(final[n]-v)*alpha for n,v in initial.items()}

def observation_extent(ids,native,arm):
    if arm!='LN_only' or not ids or len(native)!=2 or any(isinstance(x,bool) or int(x)!=x for x in native) or not 0<=native[0]<=native[1]<len(ids):raise ValueError('Invalid Native WHEN')
    return list(native)

def stages():return read(BASE/'DESIGN_LOCK.json')['stages']

def old_record(stage,condition,order,at):
    q=stage['orders'][order][at];original_at=stage['input_order'].index(q)
    b=read(OLD/'stages'/stage['matched_full_stage']/'PREDICTION_BARRIER.json')
    z=b['logical_records'][f'{condition}/order1/on_policy/{original_at:05}'];p=OLD/z['path'];rc=read(p.with_suffix('.json'))
    assert sha(p)==z['sha256']==rc['sha256'] and p.stat().st_size==rc['bytes'] and not rc['GT_read'];return p,z

def verify():
    r=read(BASE/'RUNTIME_LOCK.json')
    assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256'] and sha(BASE/'CPU_CONTRACTS.json')==r['CPU_contracts_sha256']
    for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
    return r

def contracts():
    import torch
    src={QUERY:torch.zeros(256),'LN':torch.linspace(-1,1,1536)};fin={QUERY:src[QUERY].clone(),'LN':src['LN']+.25};good=bad=0
    for alpha in [1/16,1/8]:
        out=committed(src,fin,alpha,'LN_only');init=initial_state(src,out,'LN_only')
        assert torch.count_nonzero(init[QUERY])==0 and sum(v.numel() for v in out.values())==1792
        assert torch.equal(out['LN'],src['LN']+(fin['LN']-src['LN'])*alpha);good+=3
    for alpha in [True,0,1,float('nan'),-.1,2]:
        try:committed(src,fin,alpha,'LN_only')
        except ValueError:bad+=1
        else:raise AssertionError('Wrong consolidation accepted')
    for final,arm in [({**fin,QUERY:torch.ones(256)},'LN_only'),({**fin,'LN':torch.zeros(1)},'LN_only'),(fin,'full')]:
        try:committed(src,final,1/16,arm)
        except ValueError:bad+=1
        else:raise AssertionError('Wrong scope accepted')
    return dict(status='pass',scope='frozen-query and full stored-state lifecycle; not GPU qualification',valid_contracts=good,invalid_contracts_rejected=bad,GT_read=False)

def prepare():
    if (BASE/'RUNTIME_LOCK.json').exists():return verify()
    original=read(PREVIOUS/'DESIGN_LOCK.json');assert read(PREVIOUS/'ROOT_ALL_FITS_AND_STATISTICS.json')['status']=='pass'
    later=read(OLD/'LATER_DESIGN_LOCK.json')['stages'];defs={}
    for name,s in original['stages'].items():
        z=copy.deepcopy(s);cfg=later[z['matched_full_stage']]['variant_configs']['LN_only']
        z.update(arms=ARMS,variant_configs={'LN_only':cfg},original_full_config=s['variant_configs']['full'],expected_new_fits=384)
        assert len(z['orders'])==3 and all(len(seq)==128 for seq in z['orders'].values());defs[name.replace('O_','M_')]=z
    write(BASE/'DESIGN_LOCK.json',dict(status='preGPU_locked',stages=defs,new_formal_fits=768,qualification_arrivals=12,actual_qualification_GPU_fits=24,
       reused_formal_fits=2304,existing_orders_runtime_sha256=sha(PREVIOUS/'RUNTIME_LOCK.json'),order_seeds=original['order_seeds'],active_LN_parameters=1536,
       frozen_query_parameters=256,stored_state_parameters=1792,original_scientific_LN_only=True,all_new_deployment_global_seal_before_GT=True,
       scientific_parameters_changed=False,historically_exposed_parents=True,production_promotion=False,time=time.time()))
    write(BASE/'CPU_CONTRACTS.json',contracts())
    files=['protocols/stvg_opd_mechanism_review_v1.md','scripts/stvg_opd_mechanism_LN_order_common_v1.py','scripts/run_stvg_opd_mechanism_LN_order_v1.py',
      'scripts/continue_stvg_opd_mechanism_LN_order_v1.py','scripts/score_stvg_opd_mechanism_LN_order_v1.py','scripts/finalize_stvg_opd_mechanism_LN_order_v1.py',
      'vg_tta/stvg_opd_p3_scope_precision001.py','vg_tta/stvg_opd_paper_ablations_v1.py','vg_tta/stvg_opd_paper_component_audit_v1.py',
      'scripts/stvg_opd_legacy_input_bridge003.py','scripts/stvg_opd_design_ablation_common_v1.py','methods/CURRENT_METHOD.json',
      str((PREVIOUS/'GLOBAL_PREDICTION_BARRIER.json').relative_to(ROOT)),str((PREVIOUS/'ROOT_ALL_FITS_AND_STATISTICS.json').relative_to(ROOT)),
      str((PREVIOUS/'DESIGN_LOCK.json').relative_to(ROOT)),str((OLD/'FINAL_COMPLETION.json').relative_to(ROOT))]
    pins={**read(OLD/'P3_scope_engineering_revision001/REVISION_RUNTIME.json')['pins'],**read(PREVIOUS/'RUNTIME_LOCK.json')['pins'],**{p:sha(ROOT/p) for p in files}}
    write(BASE/'RUNTIME_LOCK.json',dict(status='locked_before_actual_qualification_and_GT',pins=pins,design_sha256=sha(BASE/'DESIGN_LOCK.json'),CPU_contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_waiting_for_existing_serial_work',CPU_only=True,new_formal_fits=0,new_GPU_fits=0,GT_read=False,time=time.time()))
    return verify()

if __name__=='__main__':prepare();print('MECHANISM_LN_ORDER_LOCKED',sha(BASE/'RUNTIME_LOCK.json'),flush=True)
