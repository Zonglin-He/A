"""No-GT metadata lock and actual same-media query availability audit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_correction_scope_common_v1 import *

def prepare():
    import hashlib
    assert not BASE.exists(); budget()
    old = ROOT / 'artifacts/tastvg_negative_evidence_v1'
    assert read(old / 'FINAL_COMPLETION.json')['status'] == 'completed_verified_publication'
    cells = read(old / 'COHORT.json')['cells']
    assert len(cells) == 1152 and sum(c['scheduled'] for c in cells) == 288
    inputs = {}; gt = {}; alternatives = {}; availability = {}
    def pin(f): inputs[str(f.relative_to(ROOT))] = sha(f)
    for ds in DATASETS:
        p = plan(ds); pin(VIEW / ds / 'PLAN.json'); pin(POOL / ds / 'CAPTURE_BARRIER.json')
        assert len(p['rows']) == 48
        for k, v in BUNDLES[ds].items(): assert p['params'][k] == v
        full = ROOT / 'artifacts/tastvg_best_full_v1' / ds / 'PLAN.json'; pin(full)
        roster = read(full)['rows']; donors = sorted({c['parent'] for c in cells if c['dataset'] == ds and c['scheduled']})
        available = 0
        for parent in donors:
            row = p['rows'][parent]
            options = [r for r in roster if r['input']['video_sha256'] == row['input']['video_sha256']
                       and r['input']['caption'] != row['input']['caption']]
            if not options: continue
            available += 1
            alt = min(options, key=lambda r: hashlib.sha256(r['key'].encode()).hexdigest())
            assert alt['source'] == row['source']
            sf = full.parent / 'subjects' / f"{alt['ordinal']:05}.json"
            assert sf.exists(); pin(sf)
            subj = read(sf)
            assert subj['caption_sha256'] == hashlib.sha256(alt['input']['caption'].encode()).hexdigest()
            alternatives[str(parent)] = dict(row=alt, subject=subj['parses']['subject'],
                    subject_receipt=str(sf.relative_to(ROOT)), subject_sha256=sha(sf))
        availability[ds] = dict(expert_sources=len(donors), same_video_other_query=available)
        if ds == 'hc2': assert available == 0
        for split in ('search', 'confirm'):
            f = POOL / ds / f'GT_LABELS_{split}.json'; gt[str(f.relative_to(ROOT))] = sha(f)
    assert alternatives and availability['vidstg']['same_video_other_query'] == availability['vidstg']['expert_sources']
    for c in cells:
        pin(ROOT / c['old_payload']); pin((ROOT / c['old_payload']).with_suffix('.json'))
        pin(POOL / c['dataset'] / 'capture' / c['condition'] / f"{c['parent']:05}.json")
        if c['scheduled']:
            f = Path(str(local_payload_path(c)) + '.pt'); pin(f); pin(f.with_suffix('.json'))
            for stage in ('spatial', 'temporal'):
                f = POOL / c['dataset'] / 'experts' / stage / c['condition'] / f"{c['parent']:05}.json"
                pin(f); r = read(f); pin(f.parents[2] / r['cache'])
    write(BASE / 'COHORT.json', dict(cells=cells, alternatives=alternatives, availability=availability,
          arrivals=1152, expert_writes=288, new_alternative_captures=len(alternatives)*6,
          historical_exposure=True, GT_read=False, time=time.time()))
    pin(BASE / 'COHORT.json')
    names = ['scripts/tastvg_correction_scope_common_v1.py', 'scripts/prepare_tastvg_correction_scope_v1.py',
             'scripts/run_tastvg_correction_scope_v1.py', 'scripts/continue_tastvg_correction_scope_v1.py',
             'protocols/tastvg_correction_scope_v1.md']
    inherited = read(old / 'RUNTIME_LOCK.json')['pins']
    write(BASE / 'RUNTIME_LOCK.json', dict(pins={f:sha(ROOT/f) for f in sorted(set(list(inherited)+names))},
        inputs=inputs, GT_inputs=gt, bundles=BUNDLES, parameters=1792, GT_read=False,
        CURRENT_METHOD_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'), time=time.time()))
    status(BASE / 'STATUS.json', dict(status='prepared_pending_smoke', GT_read=False, time=time.time()))
    archive('科学规则和原A写/名单/专家收据已锁，尚未执行本轮预测')
    print('LOCKED', availability, 'new alternative captures', len(alternatives)*6, flush=True)

if __name__ == '__main__': prepare()
