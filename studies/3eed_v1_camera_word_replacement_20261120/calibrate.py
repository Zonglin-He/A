"""Fit the camera word likelihood on source-own annotation centers."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from factor import ROOT, horizontal, image_word
from crossplatform.round7.camera_projection import raw_to_model_points

HERE = Path(__file__).resolve().parent
WORK = HERE/'run_PRIVATE'
OUT = HERE/'results'
SUP = HERE.parent/'3eed_v1_supervised_instance_grounding_20261119'
SOURCE = SUP/'inputs_PRIVATE'
ORD = HERE.parent/'3eed_v2_ordinary_val_subject_ranking_20261119'
PLATFORMS = ('waymo','drone','quad')


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def csv_file(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def source_rows(src, split):
    registry = read(SOURCE/'SEAL.json')
    n = f'{src}_{split}.json'
    assert digest(SOURCE/n) == registry['files'][n]
    return read(SOURCE/n)


def coordinates(rows):
    x, y = [], []
    counts = Counter()
    for row in rows:
        counts['n'] += 1
        text = row['caption']
        word = image_word(text)
        if word.sign == 0:
            continue
        inverse = image_word(word.flipped)
        assert inverse.sign == -word.sign and inverse.flipped == text
        counts['parsed'] += 1
        item = read(row['metapath'])['ground_info'][row['object_index']]
        assert ' '.join(item['caption'].replace(',',' ,').split()) == ' '.join(text.replace(',',' ,').split())
        index = int(row['suffix'] or 0)
        center = raw_to_model_points(np.asarray(item['bbox_3d'][:3])[None,:], row['platform'], index)
        u, valid = horizontal(center, row['platform'], index, row['metapath'], row['imagepath'])
        if valid[0]:
            counts['visible'] += 1
            x.append(u[0])
            y.append(word.sign)
    return np.array(x), np.array(y), counts


def slope_fit(a):
    # Derivative of sum softplus(-k*a) + k^2/(2*10^2), strictly increasing.
    def gradient(k):
        return k / 100 - np.sum(a * expit(-k*a))
    if gradient(0) >= 0:
        return 0.
    hi = 1.
    while gradient(hi) < 0:
        hi *= 2
    return float(brentq(gradient, 0., hi))


def main():
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(mode=0o700, exist_ok=True)
    assert digest(ROOT/'crossplatform/single_frame_sources.json') == read(HERE/'PROTOCOL.json')['registered_sources']
    metrics, slopes = [], {}
    total_loss, total_n = 0., 0
    for src in PLATFORMS:
        seen = set()
        for split in ('fit','calibration'):
            rows = source_rows(src, split)
            recs = {r['recording'] for r in rows}
            assert not (seen & recs)
            seen.update(recs)
            x, y, count = coordinates(rows)
            a = x * y
            if split == 'fit':
                slopes[src] = slope_fit(a)
            k = slopes[src]
            loss = np.logaddexp(0., -k*a)
            metrics.append(dict(source=src,split=split,n=len(rows),recordings=len(recs),
                                parsed=count['parsed'],visible=len(y),right=int((y>0).sum()),
                                slope=k,sign_correct=int((a>0).sum()),log_loss=float(loss.mean()) if len(a) else None,
                                null_loss=float(np.log(2))))
            if split == 'calibration':
                total_loss += loss.sum()
                total_n += len(y)
    csv_file(OUT/'camera_source.csv', metrics)
    save(OUT/'calibration.json', dict(slopes=slopes,heldout=total_n,loss=total_loss/total_n if total_n else None,
                                      improves_null=bool(total_n and total_loss < total_n*np.log(2)),
                                      protocol=digest(HERE/'PROTOCOL.json'),matcher=digest(HERE/'factor.py'),
                                      calibration_code=digest(Path(__file__)),source_seal=digest(SOURCE/'SEAL.json')))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
