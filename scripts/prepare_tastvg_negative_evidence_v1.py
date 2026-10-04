"""Bind previous A evidence and cached H without opening GT contents."""
from scripts.tastvg_negative_evidence_common_v1 import *
def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists();budget()
    cells=read(ROOT/'artifacts/tastvg_dta_expert_r2_v1/COHORT.json')['cells'];assert len(cells)==1152 and sum(c['scheduled'] for c in cells)==288
    write(BASE/'COHORT.json',dict(cells=cells,historically_exposed=True,local_experts=288,reset_u_arrivals=1152,GT_read=False))
    inputs={};gt={}
    def pin(f):inputs[str(f.relative_to(ROOT))]=sha(f)
    pin(BASE/'COHORT.json')
    for ds in DATASETS:
        pin(VIEW/ds/'PLAN.json');pin(POOL/ds/'CAPTURE_BARRIER.json')
        p=plan(ds);assert len(p['rows'])==48 and all(p['params'][k]==v for k,v in BUNDLES[ds].items())
        for split in ['search','confirm']:
            f=POOL/ds/f'GT_LABELS_{split}.json';gt[str(f.relative_to(ROOT))]=sha(f)
    for c in cells:
        f=ROOT/c['old_payload'];pin(f);pin(f.with_suffix('.json'))
        f=POOL/c['dataset']/'capture'/c['condition']/f"{c['parent']:05}.json";pin(f)
        if c['scheduled']:
            for typ in ['spatial','temporal']:
                f=POOL/c['dataset']/'experts'/typ/c['condition']/f"{c['parent']:05}.json";pin(f)
                r=read(f);assert r['pixel_sha256']==c['pixel_sha256'];cf=f.parents[2]/r['cache'];assert sha(cf)==r['cache_sha256'];pin(cf)
    inherited=read(VIEW/'RUNTIME_LOCK.json')['pins']
    own=['scripts/tastvg_negative_evidence_common_v1.py','scripts/prepare_tastvg_negative_evidence_v1.py','scripts/run_tastvg_negative_evidence_v1.py',
         'scripts/continue_tastvg_negative_evidence_v1.py','vg_tta/tastvg_negative_evidence_v1.py','tests/test_tastvg_negative_evidence_v1.py','protocols/tastvg_negative_evidence_v1.md']
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in sorted(set(list(inherited)+own))},inputs=inputs,GT_inputs=gt,
        params=BUNDLES,negative_lambda=1.,producer='new spatial negative evidence and separate reset-u batch',GT_read=False,time=time.time(),
        CURRENT_METHOD_sha256=sha(ROOT/'methods/CURRENT_METHOD.json')))
    status(BASE/'STATUS.json',dict(status='prepared_pending_live_parity',GT_read=False,time=time.time()))
    archive('科学规则、1152名单与原A/专家收据已冻结，待真实parity和本轮预测')
    print('PREPARED',len(cells),'arrivals; expert local',sum(c['scheduled'] for c in cells),flush=True)
if __name__=='__main__':prepare()
