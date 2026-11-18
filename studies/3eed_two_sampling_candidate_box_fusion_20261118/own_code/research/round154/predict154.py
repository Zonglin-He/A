"""Two-sampling, 256-output box-fusion transplant. No task labels or evaluation imports."""
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'research/round154'
PARENT = ROOT / 'crossplatform/round145/method'
CAPTURE = ROOT / 'crossplatform/round154/captures'
OUTPUT = ROOT / 'crossplatform/round154/predictions'
PARENT_ARMS = ['Frozen_A', 'norm_joint_zero', 'norm_direct', 'norm_TTA', 'norm_wrong_TTA']


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def match_boxes(anchor, alternate):
    """Descending physical IoU; omit invalid extents; ties resolved by slot order."""
    a = anchor[:, None, :]
    b = alternate[None, :, :]
    lo = np.maximum(a[..., :3] - a[..., 3:]/2, b[..., :3] - b[..., 3:]/2)
    hi = np.minimum(a[..., :3] + a[..., 3:]/2, b[..., :3] + b[..., 3:]/2)
    iv = np.prod(np.maximum(hi - lo, 0), axis=-1)
    uv = np.prod(a[..., 3:], axis=-1) + np.prod(b[..., 3:], axis=-1) - iv
    valid = np.all(a[..., 3:] > 0, axis=-1) & np.all(b[..., 3:] > 0, axis=-1)
    overlap = np.divide(iv, uv, out=np.zeros_like(iv), where=valid & (uv > 0))
    left, right = np.where(overlap > .55)
    a_to_b = np.full(len(anchor), -1, dtype=int)
    used = np.zeros(len(alternate), dtype=bool)
    for edge in np.lexsort((right, left, -overlap[left, right])):
        i, j = left[edge], right[edge]
        if a_to_b[i] < 0 and not used[j]:
            a_to_b[i] = j
            used[j] = True
    return a_to_b, np.flatnonzero(~used)


def fuse(anchor, alternate, parent_scores, alt_score, permutation):
    anchor, alternate = np.asarray(anchor, np.float64), np.asarray(alternate, np.float64)
    pzero, ptta = parent_scores['norm_joint_zero'], parent_scores['norm_TTA']
    a_to_b, extra = match_boxes(anchor, alternate)
    keep = np.flatnonzero(a_to_b >= 0)
    partners = a_to_b[keep]
    summed = pzero[keep] + alt_score[partners]
    fraction = np.divide(alt_score[partners], summed, out=np.full_like(summed, .5), where=summed > 0)
    moved = anchor.copy()
    moved[keep] = anchor[keep] + fraction[:, None] * (alternate[partners] - anchor[keep])
    coordinates = np.vstack((moved, alternate[extra]))
    representatives = np.vstack((anchor, alternate[extra]))

    def rank(anchor_scores, second_scores):
        votes = anchor_scores.copy()
        votes[keep] += second_scores[partners]
        votes = np.r_[votes, second_scores[extra]] / 2.
        return np.argsort(-votes, kind='stable')[:256], int(np.argmax(votes[:256]))

    order_zero, _ = rank(pzero, alt_score)
    order_tta, best_anchor = rank(ptta, alt_score)
    order_wrong, _ = rank(ptta, alt_score[permutation])
    winners = [int(parent_scores[name].argmax()) for name in PARENT_ARMS]
    selected = np.vstack([anchor[winners], alternate[alt_score.argmax()],
                          moved[pzero.argmax()], moved[ptta.argmax()], anchor[best_anchor],
                          representatives[order_tta[0]], coordinates[order_zero[0]],
                          coordinates[order_tta[0]], coordinates[order_wrong[0]]])
    pools = np.stack([anchor, alternate, moved, coordinates[order_zero], coordinates[order_tta]])
    return dict(chosen_boxes=selected, pools=pools, A_to_alt=a_to_b,
                fusion_TTA_winner_group=order_tta[0], fusion_zero_winner_group=order_zero[0],
                n_matched=len(keep), n_unmatched_alt_in_TTA_pool=int((order_tta >= 256).sum()))


def main():
    start = monotonic()
    lockpath = HERE / 'experiment/DESIGN_LOCK.json'
    lock = json.loads(lockpath.read_bytes())
    for name, sha in lock['inputs'].items():
        assert digest(ROOT/name) == sha, name
    registry = json.loads((ROOT/'research/round145/INPUT_REGISTRY_PRIVATE.json').read_bytes())
    parent = json.loads((PARENT/'SEALED.json').read_bytes())['files']
    capture_seal = CAPTURE/'SEALED_PRIVATE.json'
    alternate = json.loads(capture_seal.read_bytes())['files']
    assert len(registry) == len(alternate) == 1798
    OUTPUT.mkdir(exist_ok=False)
    files = {}
    for e in registry:
        assert not e['protected_alias_fallback']
        src, tgt, sid = e['source'], e['target'], e['row']['id']
        directory = src + '_to_' + tgt
        old_key = f'{directory}/samples_PRIVATE/{sid}.npz'
        key = f'{directory}/{sid}.npz'
        assert digest(PARENT/old_key) == parent[old_key] and digest(CAPTURE/key) == alternate[key]
        with np.load(PARENT/old_key, allow_pickle=False) as old, np.load(CAPTURE/key, allow_pickle=False) as new:
            ps = dict(zip(old['arm_names'].tolist(), old['scores']))
            seed = int.from_bytes(hashlib.sha256(f'R154-shuffle|{src}|{tgt}|{sid}'.encode()).digest()[:8], 'little')
            output = fuse(old['source_boxes'], new['new_boxes'], ps, new['new_p'],
                          np.random.default_rng(seed).permutation(256))
        assert output['chosen_boxes'].shape == (len(lock['arm_names']), 6)
        assert output['pools'].shape == (len(lock['pool_names']), 256, 6)
        path = OUTPUT/key
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **output, arm_names=lock['arm_names'], pool_names=lock['pool_names'],
                            has_B=e['has_B'])
        files[key] = digest(path)
    seal = dict(n=len(files), files=files, arm_names=lock['arm_names'], pool_names=lock['pool_names'],
                design_sha256=digest(lockpath), capture_seal_sha256=digest(capture_seal),
                code_sha256=digest(Path(__file__)), seconds=monotonic()-start, GT_read=False)
    (OUTPUT/'SEALED.json').write_text(json.dumps(seal, indent=2)+'\n')
    print('Sealed', len(files), 'inputs,', len(lock['arm_names']), 'arms', flush=True)


if __name__ == '__main__':
    main()
