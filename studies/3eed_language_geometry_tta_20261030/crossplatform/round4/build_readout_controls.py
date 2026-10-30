"""No-update controls from frozen BBF/BBS heads; never opens numeric labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
S = R / 'source'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def probability(v):
    v = np.asarray(v, dtype=float)
    assert np.isfinite(v).all() and (v >= 0).all()
    total = v.sum()
    return v / total if total > 0 else np.full(len(v), 1 / len(v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', choices=['dev', 'confirm'], required=True)
    ap.add_argument('--bbs', type=Path, required=True); a = ap.parse_args()
    source = S / (a.cohort + '_source'); ss = json.loads((source / 'sealed.json').read_text())
    bs = json.loads((a.bbs / 'sealed.json').read_text())
    assert sha(source / 'sealed.json') == bs['source_seal_sha256']
    for folder, seal in [(source, ss), (a.bbs, bs)]:
        for filename, digest in seal['files'].items():
            assert sha(folder / filename) == digest
    manifest = S / 'inputs' / (a.cohort + '_manifest.json')
    assert sha(manifest) == ss['manifest_sha256']
    out = R / (a.cohort + '_readout_controls'); out.mkdir(exist_ok=False)
    (out / 'CODE_INPUT_LOCK.json').write_text(json.dumps({'source_sha256': sha(Path(__file__)),
        'source_seal_sha256': sha(source / 'sealed.json'), 'BBS_seal_sha256': sha(a.bbs / 'sealed.json'),
        'model_calls': 0, 'updates': 0, 'numeric_GT_access': False,
        'fusion': 'Simple arithmetic mean of per-view query-normalized semantic probabilities after frozen box correspondence. '
                  'Distinct from geometry-IoU-weighted adaptation augmentation if present; both explicitly retained.'}, indent=2) + '\n')
    rows = []
    for sample in json.loads(manifest.read_text()):
        sid = sample['id']; full = []; noun = []; bbs = []
        for j in range(3):
            with np.load(source / f'{sid}_v{j}_full.npz', allow_pickle=False) as z:
                alignment = z['aligned_to_full_view0']; full.append(probability(z['semantic_scores'][alignment]))
                if j == 0:
                    boxes = z['boxes'].copy(); pca = z['boxes_pca'].copy()
                    original = z['semantic_scores'].copy()
            with np.load(source / f'{sid}_v{j}_noun.npz', allow_pickle=False) as z:
                noun.append(probability(z['semantic_scores'][z['aligned_to_full_view0']]))
            with np.load(a.bbs / f'{sid}_v{j}_full.npz', allow_pickle=False) as z:
                bbs.append(probability(z['bbs_semantic_scores'][alignment]))
                if j == 0:
                    bbs0 = z['bbs_semantic_scores'].copy()
        chosen = {'frozen_raw': int(original.argmax()), 'frozen_PCA': int(original.argmax()),
                  'simple_TTAug_PCA': int(np.mean(full, axis=0).argmax()),
                  'nounonly_mean_PCA': int(np.mean(noun, axis=0).argmax()),
                  'frozen_BBS_PCA': int(bbs0.argmax()), 'simple_BBS_mean_PCA': int(np.mean(bbs, axis=0).argmax())}
        arrays = {name: (np.r_[boxes[index], np.float32(0)] if name == 'frozen_raw' else pca[index]).copy()
                  for name, index in chosen.items()}
        np.savez_compressed(out / (sid + '.npz'), **arrays)
        rows.append({'id': sid, 'platform': sample['platform'], 'split': sample['split'], 'scene': sample['scene'],
                     'chosen_queries': chosen, 'bbs_zero_total': bool(bbs0.sum() == 0)})
    (out / 'trace.json').write_text(json.dumps(rows, indent=2) + '\n')
    sealed = {'n': len(rows), 'arms': list(arrays), 'files': {p.name: sha(p) for p in out.glob('*.npz')},
              'trace_sha256': sha(out / 'trace.json'), 'code_lock_sha256': sha(out / 'CODE_INPUT_LOCK.json'),
              'numeric_GT_access': False}
    (out / 'sealed.json').write_text(json.dumps(sealed, indent=2) + '\n')
    print(json.dumps({'cohort': a.cohort, 'n': len(rows), 'arms': list(arrays), 'no_update_controls_sealed': True}))


if __name__ == '__main__':
    main()
