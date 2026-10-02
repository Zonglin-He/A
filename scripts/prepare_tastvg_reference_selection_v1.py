"""Hash-only cohort selection, all input/implementation contracts before GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from tastvg_reference_common_v1 import *
import hashlib,numpy as np
from vg_tta.tastvg_reference_selection_v1 import student_frames,decision

def run():
    assert not BASE.exists(),'Never overwrite a qualification namespace'
    assert read(PRIOR/'FINAL_COMPLETION.json')['status']=='completed'
    inputs={};plans={};labels={}
    def bind(f):inputs[str(f.relative_to(ROOT))]=sha(f)
    bind(PRIOR/'FINAL_COMPLETION.json');bind(PRIOR/'GLOBAL_PREDICTION_BARRIER.json');bind(ROOT/'methods/CURRENT_METHOD.json')
    for ds in DATASETS:
        pf=PRIOR/ds/'PLAN.json';p=read(pf);bind(pf);bind(PRIOR/ds/'A/PREDICTION_BARRIER.json');bind(POOL/ds/'CAPTURE_BARRIER.json')
        choices={}
        for order,seq in p['splits']['search']['orders'].items():
            for at,parent in enumerate(seq):
                if at%4==0:choices.setdefault(parent,(order,at))
        ranked=sorted(choices,key=lambda i:hashlib.sha256((ds+':'+p['rows'][i]['source']).encode()).hexdigest())[:10]
        assert len(ranked)==10 and len({p['rows'][i]['source'] for i in ranked})==10
        corrupt=[c for c in p['conditions'] if c!='clean'];assert len(corrupt)==5
        cells=[]
        for source_id,parent in enumerate(ranked):
            order,at=choices[parent];row=p['rows'][parent]
            conditions=['clean',corrupt[source_id%5],corrupt[(source_id+2)%5]]
            for cond in conditions:
                f=PRIOR/ds/'A/online'/cond/order/f'{at:05}.pt';rr=read(f.with_suffix('.json'));assert sha(f)==rr['sha256'];bind(f);bind(f.with_suffix('.json'))
                x=load(f);assert x['expert_scheduled'] and not x['GT_read']
                candidates=[c['prediction']['boxes'] for c in x['update_steps'][0]['candidates']]
                assert len(candidates)==9 and np.array_equal(candidates[0],x['slow']['boxes'])
                erp=POOL/ds/'experts/spatial'/cond/f'{parent:05}.json';er=read(erp);ec=POOL/ds/'experts'/er['cache'];bind(erp);bind(ec);assert sha(ec)==er['cache_sha256'] and er['pixel_sha256']==x['pixel_sha256']
                ev=load(ec);uniform=np.rint(np.linspace(0,len(row['frame_ids'])-1,5)).astype(int).tolist();assert ev['positions']==uniform
                router=student_frames(x['temporal']['candidates'],row['frame_ids'])
                dc=decision(candidates,ev['boxes'],ev['valid'])
                original=x['update_steps'][0]['rewards']
                if original is None:assert dc['rewards'] is None
                else:np.testing.assert_allclose(dc['rewards'],original,atol=1e-12,rtol=0)
                cells.append(dict(cell=len(cells),source_id=source_id,parent=parent,condition=cond,order=order,arrival=at,
                    payload=str(f.relative_to(ROOT)),uniform_receipt=str(erp.relative_to(ROOT)),uniform_cache=str(ec.relative_to(ROOT)),
                    pixel_sha256=x['pixel_sha256'],pre_state_sha256=x['pre_sha'],router=router,uniform_positions=uniform,
                    uniform_decision=dc,specialist_input_changes=True))
        plan=dict(dataset=ds,cells=cells,sources=10,queries=10,corrupt_cells=20,clean_cells=10,
            cohort_rule='first10 sha256(dataset+colon+source), eligible if scheduled in either A order; prefer order1; clean+rotating corrupt indices j and j+2 modulo5',
            row_plan=str(pf.relative_to(ROOT)),params=p['params'],checkpoint_state_sha256=read(POOL/ds/'CAPTURE_BARRIER.json')['checkpoint_state_sha256'],
            old_expert_fraction=.25,not_an_online_trajectory=True,historical_exposure=True,
            uniform_parity_cell=0,new_routed_calls=30,new_uniform_parity_calls=1)
        write(BASE/ds/'PLAN.json',plan);bind(BASE/ds/'PLAN.json');plans[ds]=sha(BASE/ds/'PLAN.json')
        labels[ds]=dict(path=str((POOL/ds/'GT_LABELS_search.json').relative_to(ROOT)),sha256=sha(POOL/ds/'GT_LABELS_search.json'))
    model=ROOT/'checkpoints/Sa2VA-4B'
    for f in ['DOWNLOAD_RECEIPT.json','OFFICIAL_CODE_RECEIPT.json','config.json','tokenizer_config.json']:bind(model/f)
    for f,h in read(model/'OFFICIAL_CODE_RECEIPT.json')['files'].items():bind(model/f)
    names=['vg_tta/tastvg_reference_selection_v1.py','scripts/tastvg_reference_common_v1.py','scripts/prepare_tastvg_reference_selection_v1.py',
        'scripts/run_tastvg_reference_selection_v1.py','scripts/score_tastvg_reference_selection_v1.py','scripts/audit_tastvg_reference_public_v1.py',
        'scripts/test_tastvg_reference_selection_v1.py','protocols/tastvg_reference_selection_v1.md','docs/tastvg_reference_selection_v1/EXECUTION.md',
        'scripts/run_tastvg_full_b1_experts_v1.py','scripts/run_final_simplification_v1.py','scripts/with_local_cuda.sh',
        'vg_tta/exact_frame_decode_audit_v2.py','vg_tta/tastvg_paper48_hc2_decode_v1.py','vg_tta/unanchored_dense_shift_data_v1.py',
        'vg_tta/tastvg_deployment_corruption_v2.py','scripts/c1_controlled_corruption_v1.py','vg_tta/tastvg_spatial_expansion_s0_v1.py',
        'vg_tta/tastvg_spatial_critic_s06_v1.py','vg_tta/box_stability_diagnostics_v1.py','scripts/diagnose_tastvg_pipeline_cpu_v1.py',
        'scripts/score_tastvg_best_quick_v1.py','vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py','vg_tta/tastvg_paper_readouts_v1.py']
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in names},inputs=inputs,diagnostic_labels=labels,
        max_new_specialist_calls=62,cap_includes_parity=True,GT_interpreted=False,plans=plans,time=time.time()))
    status(BASE/'STATUS.json',dict(status='locked_ready_for_specialist',done=0,total=62,GT_read=False))
    print('LOCKED20sources/60matchedcells/62newcalls including uniform parity',flush=True)
if __name__=='__main__':run()
