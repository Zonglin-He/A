"""Freeze the development protocol and import compatible HC64 cached inputs."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.boxfixed_tuning.capture import OUT,prepare as capture_prepare
from scripts.tube_temporal.run import read,sha,atomic,savept

def run():
    lock,rows=capture_prepare()
    from scripts.interval_pool import development
    development.validate()
    files={}
    for r in rows['hc2']:
        key=r['id'];old=read(ROOT/f'outputs/interval_pool/development/receipts/{key}.json')
        post=ROOT/old['posterior_path'] if str(old['posterior_path']).startswith('outputs/')else development.TRIAL/old['posterior_path']
        # Old receipts bind the original native capture and its saved six-layer readout.
        pp=torch.load(post,map_location='cpu',weights_only=False)
        cp=ROOT/old['capture_path'] if str(old['capture_path']).startswith('outputs/')else ROOT/'outputs/tube_temporal_hc64'/old['capture_path']
        cap=torch.load(cp,map_location='cpu',weights_only=False)
        with np.load(development.TRIAL/'predictions'/f'{key}.npz',allow_pickle=False)as z:
            native=z['Source_interval'].copy();assert np.array_equal(cap['Source'],z['Source'])
        path=OUT/'inputs/hc2'/f'{key}.pt'
        value=dict(Source=cap['Source'],E=cap['E'],frame_ids=cap['frame_ids'],positions8=cap['planned_positions'],
            records=cap['records'],geometry=cap['geometry'],logits=pp['logits'],native_interval=native,GT_read=False)
        if path.exists():assert sha(path)==read(OUT/'hc2_inputs.json')['files'][key]['sha256']
        else:savept(path,value)
        files[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),old_capture_sha256=sha(cp),old_posterior_sha256=sha(post))
    atomic(OUT/'hc2_inputs.json',dict(files=files,GT_read=False,queries=64,new_network_calls=0))
    plan=dict(schema=1,user_authorization='2026-10-22 finite two-direction tuning followed by NEW full runs and one Luna max monitor',
        method='BoxFixed complete native propagation + six-layer old hull + original physical MBR',
        development_queries_per_direction=64,maximum_unique_trials_per_direction=18,
        axes=['temperature','observed_weight','guard_weight','link_weight'],
        coarse_values=dict(temperature=[.5,1.,2.],observed_weight=[.25,.5,.75],guard_weight=[.05,.1,.2],link_weight=[0.,.25,.5]),
        stages='Sequential one-axis coarse search; root reviews actual errors after every stage. Up to4 mechanism-led neighboring refinements and4 leave-one-parameter-out attribution controls; at most1 interaction trial within18 total. No automatic exhaustive grid.',
        selection='Primary full vIoU; reject if mean tIoU or unobserved actual sIoU more than .1pp below baseline or severe >5pp losses vs Source exceed baseline by2. Improvement<=.02pp retains current. Use all64, no seed selection.',
        confirmation='Freeze both profiles then score prior parent-isolated32 per direction ONCE. Require full vIoU not below original baseline, t/s not more than .1pp below, severe vs Source at most baseline+2; otherwise fallback to original profile for that domain. No retuning on confirmation.',
        exposure='All cohorts historically exposed development evidence, not pristine independent confirmation. Full official results are evaluation after profiles lock; no selection from full scores.',
        fixed='Source/ReferDINO weights, eight observations, initial gamma .5/1, spatial geometry/candidates/confidence, frame IDs/precision/readout-observation rule and MBR. No448/conditional decoder/OPD/training.',
        capture_lock_sha256=sha(OUT/'capture_lock.json'),labels_sha256=sha(ROOT/'data/evaluation/development_labels.jsonl'),
        sources={str(p.relative_to(ROOT)):sha(p)for p in Path(__file__).parent.glob('*.py')},GT_read=False)
    path=OUT/'plan.json'
    if path.exists():assert read(path)==plan
    else:atomic(path,plan)
    atomic(ROOT/'results/boxfixed_tuning_v1/protocol.json',plan)
    print(json.dumps(dict(status='prepared',hc2_cache_queries=64,plan_sha256=sha(path))))

if __name__=='__main__':run()
