"""Pre-GT evaluation schema correction: proposal XYXY versus adopted CXCYWH."""
from pathlib import Path
import inspect, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import score_stvg_spatial_controls256_v1 as old
BASE, read, put, sha = old.BASE, old.read, old.put, old.sha

def xyxy_to_cxcywh(box):
    x0, y0, x1, y1 = map(float, box)
    assert x1 >= x0 and y1 >= y0
    return [(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0]

def corrected_source():
    src = inspect.getsource(old.main)
    before = "diagnostic=geom(t['center_box'], g)"
    after = "diagnostic=geom(xyxy_to_cxcywh(t['center_box']), g)"
    assert src.count(before) == 1
    return src.replace(before, after)

def main(arm):
    for p, h in read(BASE/'EVALUATION_SCHEMA_RECOVERY002.json')['pins'].items():
        assert sha(ROOT/p) == h, p
    ns = dict(old.__dict__)
    ns['xyxy_to_cxcywh'] = xyxy_to_cxcywh
    exec(compile(corrected_source(), str(Path(__file__)), 'exec'), ns)
    ns['main'](arm)
    p = BASE/'trials'/arm/'POSTSEAL_RESULTS.json'
    d = read(p)
    d['evaluation_schema_recovery_sha256'] = sha(BASE/'EVALUATION_SCHEMA_RECOVERY002.json')
    # Write a separate bound result; preserve the original scorer output and receipt.
    put(p.with_name('POSTSEAL_RESULTS002.json'), d)
    put(p.with_name('CPU_COMPLETION002.json'), dict(status='complete', count=256, GT_read=True,
        row_count=len(d['rows']), result_sha256=sha(p.with_name('POSTSEAL_RESULTS002.json')),
        original_cpu_completion_sha256=sha(p.with_name('CPU_COMPLETION.json')),
        evaluation_schema_recovery_sha256=sha(BASE/'EVALUATION_SCHEMA_RECOVERY002.json')))

def contracts():
    from scripts.score_stvg_B_transfer_v1 import geom
    old.contracts()
    for xy in ([.2,.3,.6,.8], [0.,0.,1.,1.], [.1,.2,.2,.4], [.3,.4,.31,.41]):
        d = geom(xyxy_to_cxcywh(xy), xy)
        assert abs(d['IoU']-1) < 1e-10 and abs(d['target_coverage']-1) < 1e-10
    assert geom([.2,.3,.6,.8], [.2,.3,.6,.8])['IoU'] < .9
    assert corrected_source().replace('geom(xyxy_to_cxcywh(t[\'center_box\']), g)',
        "geom(t['center_box'], g)") == inspect.getsource(old.main)
    print('SCHEMA002_CONTRACTS_PASS_PROPOSAL_XYXY_ADOPTED_CXCYWH_NO_GT')

def lock():
    files = [Path(__file__), ROOT/'scripts/score_stvg_spatial_controls256_v1.py',
             BASE/'EVALUATION_LOCK.json', ROOT/'scripts/stvg_instance_spatial32_v1.py',
             ROOT/'scripts/prepare_stvg_lite_reclocked_inputs002.py']
    put(BASE/'EVALUATION_SCHEMA_RECOVERY002.json', dict(status='locked', count=256, GT_read=False,
        pins={str(p.relative_to(ROOT)): sha(p) for p in files},
        correction='candidate center_box is normalized XYXY; convert only for CXCYWH geometry diagnostic helper; '
                   'adopted anchors already CXCYWH remain unchanged',
        actual_old_scorer_runs=0, old_GT_reads=0, inference_or_parameter_changes=False,
        original_source_and_lock_preserved=True, latest_entrypoint=str(Path(__file__).relative_to(ROOT))))
    print('EVALUATION_SCHEMA002_LOCKED_BEFORE_GT')

if __name__ == '__main__':
    m = sys.argv[1]
    contracts() if m == 'contracts' else lock() if m == 'lock' else main(m)
