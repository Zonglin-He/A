"""Replay the authored last-layer BBF after sealing the deployable predictions."""
import argparse
import os
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from transformers import RobertaTokenizerFast

from predict165 import RAW_ARMS, axis_box, points_for
from scope165 import HERE, ORDER, ROOT, WORK, atomic_json, check_lock, ordinary, read, sha
from common160 import array_sha, npz_atomic
from crossplatform.ground_single_frame import orient_box
from decode160 import annotated_mask, grounding_evaluator, GroundingEvaluator, skip_metric, top_ten

ARMS = ('official_bbf_axis', 'official_bbf_pca', *RAW_ARMS)


def official_selection(raw, query, tokens, mask, native, stub, device):
    outputs = dict(
        last_proj_queries=torch.as_tensor(query, device=device)[None],
        proj_tokens=torch.as_tensor(tokens, device=device)[None],
        last_center=torch.as_tensor(raw[:, :3], device=device)[None],
        last_pred_size=torch.as_tensor(raw[:, 3:], device=device)[None].clamp(min=1e-6),
        box_label_mask=torch.ones((1, 1), device=device),
        positive_map=torch.as_tensor(mask, device=device)[None, None],
    )
    top, ties = top_ten(outputs, outputs['positive_map'][0])
    native.evaluate_bbox_by_contrast(stub, outputs, 'last_')
    first = int(top[0])
    official_box = np.array(native.prediction_records.pop()['pred_box'], np.float32)
    axis = axis_box(raw[first])
    np.testing.assert_array_equal(official_box, axis[:6])
    return first, axis, ties


def decode_block(block, target, rows, tok, native, stub, device):
    endpoints, ids, flags = [], [], []
    for j, ordinal in enumerate(block['ordinals']):
        row = rows[int(ordinal)]
        pm = annotated_mask({'cohort':'official_val', 'target':target, 'r':row}, row['object_index'], tok)
        raw = block['raw'][j]
        start, stop = block['offsets'][j:j+2]
        who, axis, ties = official_selection(raw, block['q'][j], block['tokens'][start:stop], pm, native, stub, device)
        chosen = block['winners'][j]
        previous = np.flatnonzero(chosen == who)
        if previous.size:
            pca = block['boxes'][j, previous[0] + 1]
        else:
            pts, ix = points_for(row)
            assert array_sha(pts) == block['points_sha'][j].tobytes().hex()
            assert array_sha(ix) == block['indices_sha'][j].tobytes().hex()
            pca = orient_box(pts, raw[who])
        np.testing.assert_array_equal(pca[:6], axis[:6])
        endpoints.append(np.concatenate((np.stack((axis, pca)), block['boxes'][j]), axis=0))
        ids.append(np.r_[np.int16(who), np.int16(who), chosen])
        flags.append((ties, not previous.size, who != chosen[0], np.count_nonzero((raw[:, 3:] < 1e-6).any(axis=1))))
    return np.stack(endpoints), np.stack(ids), np.asarray(flags, np.int16)


@torch.inference_mode()
def decode(cohort, device):
    lock=check_lock()
    forward = HERE / {'pilot':'PILOT_SEAL.json','ordinary':'ORDINARY_FORWARD_SEAL.json'}[cohort]
    proof = read(forward)
    assert proof['paired_frozen_forward'] and not proof['FINAL_access']
    rows = ordinary()
    tok = RobertaTokenizerFast.from_pretrained(str(ROOT/'crossplatform/upstream/data/roberta_base'), local_files_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    native = GroundingEvaluator(prefixes=['last_'])
    stub = dict(gt_bboxes=torch.zeros((1,1,7), device=device), meta_path=['stub/p/f/m'], utterances=[''])
    total, checks = 0, {}
    for source, target in ORDER:
        name = f'{source}_to_{target}'
        folder = WORK/cohort/name
        direction = read(folder/'SEAL.json')
        assert direction['protocol_sha256'] == lock['protocol_sha256']
        assert not direction['oracle_target_class_read'] and not direction['protected_payload_access']
        output = folder/f'MATCHED_BOXES_{device.upper()}_PRIVATE.npz'
        receipt = folder/f'BBF_{device.upper()}_RECEIPT.json'
        upstream_digest = sha(folder/'SEAL.json')
        if receipt.exists():
            registered = read(receipt)
            assert registered['output_sha256'] == sha(output)
            assert registered['direction_forward_seal'] == upstream_digest
            assert registered['n']==direction['n']==proof['by_direction'][name]
            checks[name]=registered
            total+=registered['n']
            continue
        boxes, winners, flags, ordinals = [], [], [], []
        with patch.object(grounding_evaluator, 'iou3d_rotated_vs_aligned', skip_metric):
            for file, digest in sorted(direction['files'].items()):
                path = folder/file
                assert sha(path) == digest
                with np.load(path, allow_pickle=False) as data:
                    assert tuple(data['arms']) == RAW_ARMS and str(data['protocol']) == lock['protocol_sha256']
                    e, w, trace = decode_block(dict(data), target, rows[target], tok, native, stub, device)
                    ordinals.extend(data['ordinals'].tolist())
                boxes.append(e)
                winners.append(w)
                flags.append(trace)
        n = len(ordinals)
        assert n == direction['n'] == proof['by_direction'][name]
        if cohort == 'ordinary':
            assert ordinals == list(range(len(rows[target])))
        trace = np.concatenate(flags)
        npz_atomic(output, arms=np.asarray(ARMS), ordinals=ordinals, boxes=np.concatenate(boxes),
                   winners=np.concatenate(winners), flags=trace)
        checks[name] = dict(n=n, output_sha256=sha(output), direction_forward_seal=upstream_digest,
                            native_ties=int((trace[:,0]>1).sum()), pca_replayed=int(trace[:,1].sum()),
                            annotated_span_changes_native_winner=int(trace[:,2].sum()),
                            rows_with_clamped_raw_dimensions=int((trace[:,3]>0).sum()))
        atomic_json(receipt,checks[name])
        total += n
        print('R165 official last-layer BBF sealed', name, n, flush=True)
    assert proof['n'] == total
    atomic_json(HERE/f'{cohort.upper()}_{device.upper()}_ENDPOINT_SEAL.json', dict(
        n=total, cohorts=[cohort], forward_seal_sha256=sha(forward), per_direction=checks, official_device=device,
        official_positive_maps_match_loader=total, official_top1_matches_evaluator=total,
        target_box_label_read=False, target_class_used_by_official_only=True, FINAL_access=False, arms=ARMS))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cohort', choices=('pilot','ordinary'))
    p.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    args = p.parse_args()
    decode(args.cohort,args.device)
