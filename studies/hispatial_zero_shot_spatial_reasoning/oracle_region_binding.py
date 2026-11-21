"""Reconstruct the full candidate bank and score box-matched geometry offline."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import time
import warnings

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from download import DATA, STUDY, read_json, sha256, write_json
from geometry_audit import paired_stats
from object_region_audit import ious, option_box
from object_region_plan import BASELINE, CONFIG as BANK_CONFIG, DIAG, RUN as BANK, index_rows, proposal_regions, read_rgb, valid_xyz
import score


RUN = BANK.parent / 'hispatial_oracle_binding_20261121'
CONFIG = STUDY / 'ORACLE_REGION_BINDING.json'
ARM_NAMES = ('original_hispatial', 'marked_prototype', 'oracle_region', 'oracle_window', 'gt_box')
POLICIES = {'primary': ('joint', 0.25), 'independent25': ('independent', 0.25),
            'positive_overlap': ('joint', 0.0), 'strict50': ('joint', 0.50)}


def verify_files(root, name):
    seal = read_json(root / name)
    for filename, expected in seal['files'].items():
        assert sha256(root / filename) == expected, filename
    return seal


def pool_assets():
    source_input = verify_files(BANK, 'input_seal.json')
    source_pred = verify_files(BANK, 'prediction_seal.json')
    prior = read_json(DIAG / 'predictions_seal.json')
    assert source_input['protocol'] == sha256(BANK_CONFIG)
    assert source_pred['inputs'] == sha256(BANK / 'input_seal.json')
    assert sha256(DIAG / 'predictions_seal.json') == source_input['diagnosis_predictions']
    assert sha256(DIAG / 'frames.json') == read_json(DIAG / 'inputs_seal.json')['files']['frames.json']
    assert sha256(DIAG / 'pointmap_sha256.json') == prior['files']['pointmap_sha256.json']
    assert source_input['frames'] == 897 and source_input['rows'] == 1206
    return read_json(BANK_CONFIG), read_json(BANK / 'regions.json'), read_json(DIAG / 'frames.json')


def region_stats(xyz, bank, cfg, saved, packed):
    stride = cfg['grid_stride']
    valid = valid_xyz(xyz)
    tiled = xyz[:3, stride//2::stride, stride//2::stride].repeat(stride, axis=1).repeat(stride, axis=2)
    support = valid & (np.linalg.norm(xyz[:3]-tiled, axis=0) <=
                       cfg['max_relative_xyz_step']*np.maximum(xyz[2], tiled[2]))
    z = xyz[2]
    idx_by_region = {r['id']: i for i, r in enumerate(saved)}
    measurements = []
    for r in bank:
        cx0, cy0, cx1, cy1 = r['box']
        rough = r['grid_mask'][cy0//stride:cy1//stride, cx0//stride:cx1//stride]
        keep = rough.repeat(stride, 0).repeat(stride, 1) & support[cy0:cy1, cx0:cx1]
        sy, sx = np.nonzero(keep)
        assert len(sx) > 0
        x0, y0 = int(sx.min()) + cx0, int(sy.min()) + cy0
        x1, y1 = int(sx.max()) + cx0 + 1, int(sy.max()) + cy0 + 1
        mean_z = float(z[sy+cy0, sx+cx0].mean(dtype=np.float64))
        window_z = float(z[y0:y1, x0:x1][valid[y0:y1, x0:x1]].mean())
        item = dict(id=r['id'], box=[x0, y0, x1, y1], area=len(sx), region_z=mean_z, window_z=window_z)
        if r['id'] in idx_by_region:
            idx = idx_by_region[r['id']]
            known = saved[idx]
            full = np.zeros_like(valid)
            full[cy0:cy1, cx0:cx1] = keep
            np.testing.assert_array_equal(np.packbits(full), packed[idx])
            assert item['box'] == known['box'] and item['area'] == known['area']
            assert np.isclose(mean_z, known['region_z'], rtol=1e-13, atol=1e-13)
            assert window_z == known['window_z']
        measurements.append(item)
    return measurements


def recover_pool():
    cfg, saved, frame_metadata = pool_assets()
    packed_seal = read_json(BANK / 'views_sha256.json')
    started = time.monotonic()
    depths = {}
    mask_checks = 0
    cv2.setNumThreads(1)
    for ordinal, (key, frame) in enumerate(saved.items(), 1):
        path = DIAG / 'pointmaps' / (key + '.npy')
        assert sha256(path) == frame['xyz_sha256']
        xyz = np.load(path, allow_pickle=False)
        image, size = read_rgb(frame_metadata[key])
        assert size == frame['size']
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Got image with third dimension of 6', category=RuntimeWarning)
            proposals, shown = proposal_regions(image, xyz, cfg)
        assert frame['bank'] == [{k: p[k] for k in ('id','box','scale','grid_area','rank')} for p in proposals], key
        assert [r['id'] for r in shown] == [r['id'] for r in frame['marked']]
        masks = BANK / 'views' / (key + '.support.npy')
        assert sha256(masks) == packed_seal[key]['supports']
        packed = np.load(masks, allow_pickle=False)
        depths[key] = region_stats(xyz, proposals, cfg, frame['marked'], packed)
        mask_checks += len(shown)
        if ordinal % 100 == 0 or ordinal == len(saved):
            print('recovered', ordinal, '/', len(saved), 'frames', round(time.monotonic()-started,1), 's', flush=True)
    write_json(RUN / 'region_depth_bank.json', depths)
    receipt = dict(physical_frames=len(saved), proposals=sum(map(len, depths.values())),
                   stored_marked_masks_verified=mask_checks, seconds=time.monotonic()-started,
                   configuration_sha256=sha256(CONFIG), prior_input_sha256=sha256(BANK/'input_seal.json'),
                   candidate_depths_sha256=sha256(RUN/'region_depth_bank.json'))
    write_json(RUN / 'rebuild_seal.json', receipt)
    return depths, saved, receipt


def assignment(matrix, mode, threshold):
    if matrix.shape[1] < matrix.shape[0]:
        return None
    permitted = matrix >= threshold if threshold > 0 else matrix > 0.0
    if mode == 'independent':
        indices = matrix.argmax(axis=1)
        if len(set(indices.tolist())) != len(indices):
            return None
        rows = np.arange(matrix.shape[0])
    else:
        # All feasible complete matchings score >=0. Missing an eligible edge is worse than any feasible complete match.
        rows, indices = linear_sum_assignment(np.where(permitted, matrix, -matrix.shape[0]-1), maximize=True)
    if not permitted[rows, indices].all():
        return None
    return indices.tolist()


def extreme(indices, values, relation):
    if len(indices) < 2 or any(x is None or not np.isfinite(x) for x in values):
        return None
    v = sorted(zip(values, indices, strict=True), reverse=(relation=='far'))
    return 'ABCD'[v[0][1]] if abs(v[0][0] - v[1][0])>1e-6 else None


def bind(depths, saved):
    prior = read_json(DIAG/'audit_summary.json')
    box_path = DIAG/'emb_box_depth.jsonl'
    assert sha256(box_path) == prior['result_files'][box_path.name]
    geo = index_rows(box_path)
    roster = index_rows(BANK/'roster.jsonl')
    old = index_rows(BANK/'predictions.jsonl')
    rows = []
    for q, item in roster.items():
        boxes = geo[q]['boxes']
        assert item['options'] == geo[q]['options'] and geo[q]['size'] == saved[item['key']]['size']
        named = [i for i, name in enumerate(item['options']) if name.strip().casefold()!='unknown']
        rectangle = [option_box(boxes[i], geo[q]['size']) for i in named]
        initial = old[q]['answers']['original_hispatial']
        gt_means = [geo[q]['depth'][i][0]['mean_z'] if geo[q]['depth'][i] and geo[q]['depth'][i][0] else None for i in named]
        gt_pick = extreme(named, gt_means, item['topic'])
        bank = depths[item['key']]
        candidate_rectangles = [r['box'] for r in bank]
        matrix = np.stack([ious(b,candidate_rectangles) for b in rectangle]) if all(b is not None for b in rectangle) else None
        outcomes = {}
        for name, (mode, threshold) in POLICIES.items():
            found = assignment(matrix,mode,threshold) if matrix is not None else None
            if found is None:
                outcomes[name] = dict(status='no_assignment',bindings=[],region=initial,window=initial)
                continue
            assigned = [bank[k] for k in found]
            region = extreme(named,[r['region_z'] for r in assigned],item['topic'])
            window = extreme(named,[r['window_z'] for r in assigned],item['topic'])
            outcomes[name] = dict(status='used' if region and window else 'readout_incomplete',
                                  bindings=[{'option':o,'iou':float(matrix[i,k]),**bank[k]} for i,(o,k) in enumerate(zip(named,found,strict=True))],
                                  region=region if region and window else initial,
                                  window=window if region and window else initial)
        primary = outcomes['primary']
        answers = dict(original_hispatial=initial, marked_prototype=old[q]['answers']['region_depth'],
                       oracle_region=primary['region'], oracle_window=primary['window'],gt_box=gt_pick or initial)
        rows.append(dict(id=q, topic=item['topic'], scene=geo[q]['scene'], source=geo[q]['source'], key=item['key'],
                         answers=answers, gt_box_used=gt_pick is not None,gt_box_depths=gt_means,policies=outcomes))
    score.jsonl(RUN/'predictions.jsonl',rows)
    seal = dict(rows=len(rows), no_gold_answer_used_in_matching=True, model_calls=0,
                bank_sha256=sha256(RUN/'rebuild_seal.json'), protocol_sha256=sha256(CONFIG),
                predictions_sha256=sha256(RUN/'predictions.jsonl'))
    write_json(RUN/'prediction_seal.json',seal)
    print('locked offline bindings',len(rows),flush=True)


def replay(official_source, rows):
    sealed = read_json(RUN/'prediction_seal.json')
    assert sealed['predictions_sha256'] == sha256(RUN/'predictions.jsonl')
    official_dir = BASELINE/'embspatial'
    base = read_json(official_dir/'inputs_seal.json')
    baseline_seal=read_json(official_dir/'predictions_seal.json')
    assert sha256(official_dir/'predictions.jsonl')==baseline_seal['predictions_sha256']
    assert sha256(official_dir/'inputs.jsonl')==base['inputs_sha256']==baseline_seal['inputs_sha256']
    source_file=DATA/'embspatial/source.json'
    assert sha256(source_file)==read_json(DIAG/'inputs_seal.json')['baseline']['embspatial']['dataset_source']
    for file in read_json(source_file)['files']:
        assert sha256(source_file.parent/file['file'])==file['sha256']
    native = index_rows(official_dir/'predictions.jsonl')
    prompts = {i:v['prompt'] for i,v in index_rows(official_dir/'inputs.jsonl').items()}
    official, official_hash = score.import_official('embspatial',official_source)
    scores = {}
    for name in ARM_NAMES:
        combined={**native,**{i.removeprefix('embspatial/'):dict(response=r['answers'][name]) for i,r in rows.items()}}
        out=RUN/'official_replay'/name
        (out/'official_replay').mkdir(parents=True,exist_ok=True,mode=0o700)
        records,_,metrics=score.score_emb(official,out,combined,prompts)
        write_json(out/'official_metrics.json',metrics)
        scores[name]={'embspatial/'+r['id']:r for r in records if 'embspatial/'+r['id'] in rows}
    assert all(x.keys() == rows.keys() for x in scores.values())
    assert sum(r['correct'] for r in scores['original_hispatial'].values())==786
    assert sum(r['correct'] for r in scores['marked_prototype'].values())==773
    return scores, official_hash


def analyze(official_source):
    rows=index_rows(RUN/'predictions.jsonl')
    index_panel={k:r['order_control'] for k,r in index_rows(BANK/'roster.jsonl').items()}
    result, script_hash = replay(official_source, rows)
    scored=[]
    for i, row in rows.items():
        gold = result['original_hispatial'][i]['official_answer']
        row['gold'] = gold
        row['conditions'] = {m:dict(correct=result[m][i]['correct']) for m in ARM_NAMES}
        assert all(row['conditions'][m]['correct'] == (gold in row['answers'][m][:2]) for m in ARM_NAMES)
        for k,p in row['policies'].items():
            if k!='primary':
                for d in ('region','window'):
                    name=k+'_'+d
                    row['conditions'][name]=dict(correct=gold in p[d][:2])
        scored.append(row)
    score.jsonl(RUN/'scored_questions.jsonl',scored)
    active=[r for r in scored if r['policies']['primary']['status']=='used']
    scopes={'all':scored,'close':[r for r in scored if r['topic']=='close'],
            'far':[r for r in scored if r['topic']=='far'], 'matched_common':active,
            'matched_common_gt_box':[r for r in active if r['gt_box_used']],
            'marked_panel96':[r for r in scored if index_panel[r['id']]]}
    for source in sorted({r['source'] for r in scored}):
        scopes['source_'+source]=[r for r in scored if r['source']==source]
    accuracy = {k:dict(count=len(rs),correct={name:sum(r['conditions'][name]['correct'] for r in rs) for name in ARM_NAMES})
                for k,rs in scopes.items()}
    arms=[('original_hispatial',name) for name in ARM_NAMES[1:]]+[('oracle_window','oracle_region'),('gt_box','oracle_region')]
    paired=[dict(scope=k,**paired_stats(rs,'correct',a,b)) for k,rs in scopes.items() if rs for a,b in arms]
    first=[r for r in paired if r['scope']=='all' and r['to_condition']=='oracle_region' and r['from_condition']=='original_hispatial'][0]
    decision=first['delta_pp']>0 and first['ci95_pp'][0]>0
    sensitivity={}
    for m in POLICIES:
        valid=[r for r in scored if r['policies'][m]['status']=='used']
        if m=='primary':
            ri,wi='oracle_region','oracle_window'
        else:
            ri,wi=m+'_region',m+'_window'
        sensitivity[m]=dict(used=len(valid),region_correct=sum(r['conditions'][ri]['correct'] for r in scored),
                            window_correct=sum(r['conditions'][wi]['correct'] for r in scored),
                            native_correct_on_used=sum(r['conditions']['original_hispatial']['correct'] for r in valid),
                            region_correct_on_used=sum(r['conditions'][ri]['correct'] for r in valid))
    gg=index_rows(DIAG/'emb_box_depth.jsonl')
    known=[r for r in scored if not r['conditions']['original_hispatial']['correct'] and
           gg[r['id']]['bbox_moge_prediction']==gg[r['id']]['bbox_center_prediction']==r['gold']]
    joint=read_json(RUN/'rebuild_seal.json')
    export=dict(cohorts=accuracy,paired=paired,sensitivity=sensitivity,gt_box_used=sum(r['gt_box_used'] for r in scored),
                primary_status=dict(Counter(r['policies']['primary']['status'] for r in scored)),
                known_geometry_correct_prior_errors=dict(n=len(known),primary_used=sum(r['policies']['primary']['status']=='used' for r in known),
                  recovered_region=sum(r['conditions']['oracle_region']['correct'] for r in known),
                  recovered_window=sum(r['conditions']['oracle_window']['correct'] for r in known)),
                official_source_sha256=script_hash, bank_rebuild=joint, model_calls=0,
                semantic_verification_gate=decision, sealed_predictions_sha256=sha256(RUN/'prediction_seal.json'),
                scored_questions_sha256=sha256(RUN/'scored_questions.jsonl'))
    with (RUN/'paired.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,list(paired[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(paired)
    write_json(RUN/'summary.json',export)
    print('all',accuracy['all'],'common',accuracy['matched_common'],'gate',decision,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','score'))
    parser.add_argument('--official-source',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':
        RUN.mkdir(parents=True,exist_ok=True,mode=0o700)
        if (RUN/'prediction_seal.json').exists():
            raise FileExistsError(RUN/'prediction_seal.json')
        depths,saved,_=recover_pool()
        bind(depths,saved)
    else:
        if not args.official_source:
            parser.error('--official-source is required for score')
        analyze(args.official_source)
