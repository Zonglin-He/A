"""Explicit cached CPU migration regression, never a GT/accuracy experiment.

The production package does not use these historical paths. This optional
checker verifies the standalone arithmetic against sealed engineering assets.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT',Path(__file__).resolve().parents[1])).expanduser().resolve()
sys.path.insert(0,str(ROOT))
from stvg.evidence import SourceEvidence,extract_visual_grids
from stvg.spatial import build_geometry,evidence_dp


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def run(root,output):
    torch.set_num_threads(1);start=time.perf_counter()
    snaproot=root/'outputs/maintenance/boxfixed_switch_20261021';snap=read(snaproot/'snapshot.json')
    legacy=snaproot/'legacy'
    for path,digest in snap['sources'].items():assert sha(legacy/path)==digest,path
    bankroot=root/'outputs/box_propagation_hc64';oldstage=root/'outputs/propagation_v0_hc64/stage1'
    layout=root/'outputs/layout_reward_dev';lock=read(bankroot/'lock.json');seal=read(bankroot/'sealed.json')
    assert seal['queries']==64 and seal['GT_read']is False and seal['lock_sha256']==sha(bankroot/'lock.json')
    assert sha(oldstage/'sealed.json')==lock['old_seal_sha256']
    oldseal=read(oldstage/'sealed.json');checks=0;cases=[]
    fields=('E','boxes','features','E_features','confidence','observed','frame_ids',
        'source_anchors','anchor_distance','nearest_anchor','side_maps','side_reliability',
        'side_available','side_reference_boxes','side_prototypes','seed_features','seed_valid',
        'merged_targets','projection','guard','link','extent_scores','association_scores')
    for index,item in enumerate(lock['inputs']):
        key=item['id'];receipt=read(bankroot/'receipts'/f'{key}.json');stage_receipt_path=oldstage/'predictions'/f'{key}.json'
        assert sha(stage_receipt_path)==oldseal['files'][key]['receipt_sha256']
        sr=read(stage_receipt_path);ep=oldstage/sr['evidence_path'];cp=layout/'student_cache'/f'{key}.pt';mp=layout/'research/native_masks'/item['mask_path']
        assert sha(ep)==sr['evidence_sha256']and sha(cp)==item['cache_sha256']and sha(mp)==item['native_mask_sha256']
        assert sha(bankroot/receipt['bank_path'])==receipt['bank_sha256']and sha(bankroot/receipt['trace_path'])==receipt['trace_sha256']
        oldev=torch.load(ep,map_location='cpu',weights_only=False);cache=torch.load(cp,map_location='cpu',weights_only=False)
        native=torch.load(mp,map_location='cpu',weights_only=False);oldbank=torch.load(bankroot/receipt['bank_path'],map_location='cpu',weights_only=False)
        assert all(x['GT_read']is False for x in (oldev,cache,native,oldbank))
        f,v,frames,shape,_=extract_visual_grids(cache['spatial_inputs'],cache['frame_ids'])
        assert torch.equal(f,oldev['features'])and torch.equal(v,oldev['valid'])
        selected=int(cache['teacher']['selected_query']);assert selected==int(native['selected_query'])==int(oldev['metadata']['selected_native_query'])
        masks=native['raw_out']['pred_masks'][0,:,selected]
        teacher_shape=cache['teacher']['input_shape'][-2:]
        ev=SourceEvidence.from_native_masks(f,v,frames,cache['planned_positions'],masks,teacher_shape,shape,selected_query=selected)
        for name in ('seeds','frame_ids','observed','adjacent_forward_indices','adjacent_forward_weights'):
            assert torch.equal(getattr(ev,name),oldev[name]),(index,name)
            checks+=1
        geometry=build_geometry(ev,cache['E'],cache['teacher']['boxes'])
        assert not hasattr(geometry,'event')and not hasattr(geometry,'active')
        bank=geometry.for_interval(cache['interval'])
        for name in fields:
            assert torch.equal(getattr(bank,name),oldbank[name]),(index,name)
            checks+=1
        for name in ('event','active','valid','candidate_origin','candidate_quality','score_unary','score_pair'):
            assert torch.equal(getattr(bank,name),oldbank[name]),(index,name)
            checks+=1
        assert bank.score_counts==oldbank['score_counts']
        out=evidence_dp(bank)
        oldtrace=torch.load(bankroot/receipt['trace_path'],map_location='cpu',weights_only=False)['BoxFixed']
        assert torch.equal(out['final'],oldtrace['final'])
        assert torch.equal(out['actions'],oldtrace['trace']['actions'])
        assert out['score']==oldtrace['score']
        with np.load(bankroot/seal['files'][key]['path'],allow_pickle=False)as predictions:
            assert np.array_equal(out['final'].numpy(),predictions['BoxFixed'])
            assert np.array_equal(bank.interval.numpy(),predictions['Strong_interval'])
        # Extension verification only: this does not score a new prediction.
        full=geometry.for_full_timeline()
        assert full.event.all()and full.score_counts==dict(observed=8,unobserved=geometry.frames-8,event_edges=geometry.frames-1,all_nodes=geometry.frames)
        assert torch.equal(full.boxes,bank.boxes)and torch.equal(full.link,bank.link)
        r=geometry.temporal_reliability();assert ((r>=0)&(r<=1)).all()
        cases.append(dict(query=f'H{index+1:03d}',tensor_checks=len(fields)+12,interval_boxes_actions_score_bit_exact=True,
            original_time_interval_bit_exact=True,full_timeline_independent=True))
        del oldev,cache,native,oldbank,ev,geometry,bank,out,oldtrace,full,f,v
        gc.collect()
    assert len(cases)==64 and not torch.cuda.is_initialized()
    for path,digest in snap['sources'].items():assert sha(legacy/path)==digest,path
    result=dict(status='all64_BoxFixed_engineering_migration_bit_exact',queries=64,GT_read=False,model_calls=0,
        decoder_calls=0,expert_calls=0,RGB_reads=0,accuracy_experiment=False,legacy_snapshot_source_checks=len(snap['sources']),
        legacy_snapshot_sha256=sha(snaproot/'snapshot.json'),old_bank_lock_sha256=sha(bankroot/'lock.json'),
        old_bank_seal_sha256=sha(bankroot/'sealed.json'),tensor_checks=checks,
        source_files={p:sha(root/p)for p in ('stvg/evidence.py','stvg/spatial.py','stvg/_spatial_math.py')},
        native_masks_to_Source_seeds_bit_exact=True,physical_visual_grids_bit_exact=True,
        adjacent_transport_bit_exact=True,geometry_and_score_bit_exact=True,DP_actions_boxes_score_bit_exact=True,
        Strong_interval_bit_exact=True,full_timeline='New explicit extension; all-node denominators, no accuracy claim',
        seconds=time.perf_counter()-start,cases=cases)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='cases'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--root',default=str(ROOT));parser.add_argument('--output',default='results/boxfixed_migration.json')
    args=parser.parse_args()
    if args.run:
        root=Path(args.root).expanduser().resolve();p=Path(args.output)
        run(root,p if p.is_absolute()else root/p)
    else:parser.print_help()
