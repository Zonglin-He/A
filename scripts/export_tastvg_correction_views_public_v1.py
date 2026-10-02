"""Build the explicit anonymous-public allowlist and immutable integrity receipt."""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.tastvg_correction_views_common_v1 import *


def run():
    for stage in ['round1','round2']:
        for split in ['search','confirm']:
            assert read(BASE/f'{stage}_{split}_ROOT_AUDIT.json')['status']=='pass'
            assert (PUBLIC/stage/split/'CONTRASTS.json').exists()
    verify()
    assert sha(ROOT/'methods/CURRENT_METHOD.json')=='bd75706cf8377823ac6af488b1bd993af5fd7480e7e45a047c6fdde5e81df8d2'
    p=read(BASE/'RUNTIME_LOCK.json');pins=dict(p['pins'])
    revisions=[]
    for f in sorted((BASE/'revisions').glob('*.json'),key=lambda f:int(re.search(r'_(\d+)\.json$',f.name).group(1))):
        z=read(f);pins.update(z['pin_overrides']);revisions.append(dict(name=f.name,**z))
    scoring=read(BASE/'SCORING_RUNTIME_LOCK.json');score_pins=dict(scoring['pins']);score_revisions=[]
    for f in sorted((BASE/'scoring_revisions').glob('*.json')):
        z=read(f);score_pins.update(z['pin_overrides']);score_revisions.append(dict(name=f.name,**z))
    for f,h in score_pins.items():assert sha(ROOT/f)==h
    write(PUBLIC/'RUNTIME_PROVENANCE.json',dict(
        model_worker_pins=pins, initial_model_worker_pins=p['pins'],runtime_revisions=revisions,
        scoring_pins=score_pins,scoring_revisions=score_revisions,
        original_private_runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        locked_private_input_files=len(p['inputs']),
        private_inputs_manifest_sha256=hashlib.sha256(json.dumps(p['inputs'],sort_keys=True).encode()).hexdigest(),
        private_inputs_withheld='captions, media, annotations, H, states, gradients, masks and checkpoints',
        CURRENT_METHOD_unchanged_sha256=sha(ROOT/'methods/CURRENT_METHOD.json')))
    cohorts={}
    for ds in DATASETS:
        q=plan(ds);cohorts[ds]=dict(
            dataset=ds,params=q['params'],conditions=q['conditions'],
            source_checkpoint_state_sha256=read(POOL/ds/'CAPTURE_BARRIER.json')['checkpoint_state_sha256'],
            splits={k:dict(orders=v['orders'],anonymous_source_count=v['sources'],arrivals=v['total']) for k,v in q['splits'].items()},
            historical_exposure=True,confirmation_disjoint_within_batch=True,
            query_count_per_source=1,frame_sampling='original Paper48 observed frames; no Fig1 uniform64 change')
    write(PUBLIC/'COHORT_CONFIG.json',cohorts)
    evidence={}
    for f in ['round1_search_GLOBAL_PREDICTION_BARRIER.json','round2_search_GLOBAL_PREDICTION_BARRIER.json',
              'round1_confirm_GLOBAL_PREDICTION_BARRIER.json','round2_confirm_GLOBAL_PREDICTION_BARRIER.json']:
        evidence[f]=read(BASE/f)
    controls={ds:dict(A_live=read(BASE/ds/'A_LIVE_PARITY.json'),
                     temporal_live={sp:read(BASE/ds/f'{sp}_TEMPORAL_LIVE_PARITY.json')
                                    for sp in ['search','confirm']}) for ds in DATASETS}
    for split in ['search','confirm']:
        f=BASE/f'{split}_SPATIAL_LIVE_PARITY.json'
        if f.exists():controls[split+'_spatial_live']=read(f)
    # Payload hashes are integrity commitments; raw private payloads are withheld.
    write(PUBLIC/'BARRIERS_AND_CONTROLS.json',dict(barriers=evidence,controls=controls,
        prediction_seal_precedes_scoring=True,confirmation_selection_sha256=sha(BASE/'FINAL_SELECTION.json')))
    incidents=[]
    for f in sorted((BASE/'recovery').glob('*/INCIDENT.json')):
        z=read(f)
        incidents.append(dict(name=f.parent.name,scope=z.get('scope'),reason=z.get('reason',z.get('error')),
            scientific_change=z.get('scientific_change',False),original_record_sha256=sha(f)))
    write(PUBLIC/'ENGINEERING_HISTORY.json',dict(incidents=incidents,
        interpretation='Preserved engineering attempts; no data, target, method or selection revision from outcomes',
        temporal_live_bound='original phase0 exact cached; live head <.1 physical frame, <1e-4 score, same selected student candidate'))
    code=[
        'vg_tta/tastvg_current_correction_views_v1.py',
        'scripts/tastvg_correction_views_common_v1.py','scripts/prepare_tastvg_correction_views_v1.py',
        'scripts/run_tastvg_correction_experts_v1.py','scripts/run_tastvg_current_correction_v1.py',
        'scripts/continue_tastvg_correction_views_v1.py','scripts/score_tastvg_correction_views_v1.py',
        'scripts/test_tastvg_correction_views_v1.py','scripts/report_tastvg_correction_views_v1.py',
        'scripts/audit_tastvg_correction_views_public_v1.py','scripts/diagnose_tastvg_correction_views_v1.py',
        'scripts/export_tastvg_correction_views_public_v1.py',
        'protocols/tastvg_current_correction_views_v1.md','docs/tastvg_current_correction_views_v1/EXECUTION.md',
        'docs/TA_CURRENT_CORRECTION_VIEWS_REVIEW.md']
    history=BASE/'implementation_history'
    dest=PUBLIC/'implementation_history';dest.mkdir(parents=True,exist_ok=True)
    for f in history.iterdir():
        if f.suffix in ['.py','.json']:shutil.copy2(f,dest/f.name)
    files=[ROOT/f for f in code]+sorted(f for f in PUBLIC.rglob('*') if f.is_file())
    assert not any(f.suffix in ['.pt','.sqlite','.safetensors'] for f in files)
    receipt=[]
    for f in files:
        b=f.read_bytes();receipt.append(dict(path=str(f.relative_to(ROOT)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),
            git_blob_sha1=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest(),binary=f.suffix in ['.png','.pdf']))
    assert len({z['path'] for z in receipt})==len(receipt)
    write(BASE/'PUBLIC_MANIFEST.json',dict(files=receipt,file_count=len(receipt),total_bytes=sum(z['bytes'] for z in receipt),
        exclusions='credentials, personal conversation records, captions, videos, annotations, weights, raw tensors and caches'))
    print('Anonymous export ready',len(receipt),sum(z['bytes'] for z in receipt))


if __name__=='__main__':run()
