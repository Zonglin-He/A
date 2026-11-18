"""Register the official validation population without opening protected payloads."""
from collections import Counter
from datetime import datetime
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT/'crossplatform/round165'
PROTOCOL = HERE/'experiment/PROTOCOL.json'
PAPER = HERE/'experiment/PAPER_TABLE2.json'
SOURCE_PINS = ROOT/'crossplatform/single_frame_sources.json'
R97 = ROOT/'research/round97'
W97 = ROOT/'crossplatform/round97'
R96 = ROOT/'research/round96'
W96 = ROOT/'crossplatform/round96'
DATA = ROOT/'crossplatform/dataset_full/3eed_dataset'
OLD_FINAL = ROOT/'research/NIGHT_20261101/input/FINAL384_PRIVATE.json'
sys.path.insert(0,str(ROOT/'research/round162'))
from manifest162 import atomic_json, canonical, read, sha, PLATFORMS

ORDER = tuple((s,t) for s in PLATFORMS for t in PLATFORMS if s!=t)
PINNED = ('crossplatform/ground_single_frame.py',
          'crossplatform/single_frame_sources.json',
          'crossplatform/round3/p0/upstream/src/joint_det_dataset.py',
          'crossplatform/round3/p0/upstream/src/grounding_evaluator.py',
          'crossplatform/round3/p0/upstream/utils/eval_det.py',
          'research/round66/interface66.py','research/round76/model76.py','research/round76/prepare76.py',
          'research/round88/routing88.py','research/round89/core89.py','research/round90/core90.py',
          'research/round158/grounding158.py','research/round91/core91.py','crossplatform/replication36_surface/common.py')


def check_lock():
    entry = read(HERE/'INPUT_LOCK.json')
    assert sha(PROTOCOL)==entry['protocol_sha256'] and sha(PAPER)==entry['paper_transcription_sha256']
    assert all(sha(ROOT/path)==val for path,val in entry['dependency_hashes'].items())
    return entry


def protected_locators():
    import ijson
    # Only compare frame identities. Never decode raw descriptions, images, or target boxes here.
    with OLD_FINAL.open('rb') as stream:
        out = {Path(x).resolve() for x in ijson.items(stream,'item.metapath')}
    assert len(out)==384
    return out


def ordinary():
    contract = read(R97/'experiment/DESIGN_LOCK.json')['inputs_sha256']
    protected = protected_locators()
    data = {}
    for target,n in read(PROTOCOL)['official_population']['initial_eligible'].items():
        loc = R97/f'input/TARGET_{target.upper()}_PRIVATE.json'
        assert sha(loc)==contract[str(loc.relative_to(ROOT))]
        items = read(loc)
        assert len(items)==n
        for row in items:
            assert Path(row['metapath']).resolve() not in protected
            assert row['caption']==canonical(row['caption']) and row['platform']==target
        data[target] = items
    return data


def paper_package():
    import zipfile
    release = ROOT/'crossplatform/downloads/baseline_ckpt_pred.zip'
    expected = 'c9ab8ab90517a44379976a3a9a136a1d84bade4abf6dc2c7123b8b33d2dd79ef'
    assert sha(release)==expected
    with zipfile.ZipFile(release) as zipped:
        import json
        config = json.loads(zipped.read('final_6384/config.json'))
        n = len([f for f in zipped.namelist() if f.endswith(('.pth','.pt','.safetensors'))])
        assert n==1 and set(config['dataset'])==set(PLATFORMS)
    return dict(sha256=expected, recognized_checkpoints=n, training_platforms=sorted(config['dataset']),
                author_single_source_checkpoint_in_package=False)


def fusion_identity():
    import numpy as np
    pairs = {}
    pins = read(SOURCE_PINS)
    for p in PLATFORMS:
        if p=='waymo':
            path = ROOT/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz'
        else:
            r = read(W97/p/'SOURCE_COMPLETE.json')['fusion_selected']
            path = Path(r['state_path'])
            assert sha(path)==r['state_sha256']
        combined = ROOT/pins[p]['matcher']
        with np.load(path,allow_pickle=False) as a,np.load(combined,allow_pickle=False) as b:
            keys = [k for k in b.files if not k.startswith('interface.')]
            assert set(keys).issubset(a.files)
            changed = [k for k in keys if not np.array_equal(a[k],b[k])]
            if changed:
                raise ValueError('Source preparation differs outside the R91 interface: '+p+' '+str(changed))
        pairs[p] = dict(fusion_sha256=sha(path),combined_sha256=sha(combined),state_arrays_identical=len(keys))
    return pairs


def begin():
    assert not (HERE/'STATE.json').exists()
    rows = ordinary()
    p_aggregate = Counter(p.parent.parent.parent.name for p in protected_locators())
    # Platform directory is three levels above meta_info.json, through recording and frame.
    assert dict(p_aggregate)=={'drone':192,'quad':192}
    r96 = read(R96/'experiment/DATA_AUDIT.json')
    for t in PLATFORMS:
        expected = r96['platforms'][t]['val']['split_sha256']
        assert sha(DATA/'splits'/f'{t}_val.txt')==expected
    files = {p:sha(ROOT/p) for p in PINNED}
    atomic_json(HERE/'INPUT_LOCK.json',dict(protocol_sha256=sha(PROTOCOL),paper_transcription_sha256=sha(PAPER),
                dependency_hashes=files,ordinary_counts={p:len(x) for p,x in rows.items()},
                ordinary_directions=sum(len(r) for r in rows.values())*2,protected_frame_counts=dict(p_aggregate),
                paper_release=paper_package(),fusion_identity=fusion_identity(),
                official_matrix_directions=len(ORDER),protected_payload_access=False))
    stamp = datetime.now().astimezone().isoformat()
    overall_path = ROOT/'research/STATE.json'
    overall = read(overall_path)
    assert overall['R161']['state']=='paused_by_user'
    status = dict(round='R165',state='running',protocol='round165/experiment/PROTOCOL.json',
                  ordinary_direction_records=18620,full_direction_records_expected=19388,
                  FINAL_access=False,requires_confirmation_to_include_previously_protected=True,updated=stamp)
    atomic_json(HERE/'STATE.json',status)
    overall['R165']=status
    overall.update(current_round='R165',manual_active_state='round165/STATE.json',current_stage='R165_OFFICIAL_CROSS_PLATFORM_ABLATION',
                   run_state='R165_RUNNING',updated_local=stamp)
    overall['directions']['crossplatform'].update(active_round='R165',execution_open=True)
    atomic_json(overall_path,overall)
    print('Official Validation registered, 18620 unprotected direction records',flush=True)


if __name__=='__main__':
    begin()
