"""Audit sealed edge energies and endpoints through exhaustive CPU enumeration."""
import hashlib

import numpy as np

from edge_predict import HERE, WORK, sha, atomic_json
from edge_score import verify


def separate(x):
    """Compute each coefficient from a cross-difference over all alternative endpoints."""
    if not x.size:
        return x
    rows, columns = x.shape
    return (x*rows*columns - columns*x.sum(0)[None,:] -
            rows*x.sum(1)[:,None] + x.sum())/(rows*columns)


def enumeration(score):
    return max(np.ndindex(score.shape), key=score.__getitem__)


def run():
    from predict171 import read_npz

    local, seal, prior = verify()
    max_error = 0.
    selected_deltas = {"vs_original_subject":0,"vs_original_reference":0,"vs_additive_subject":0}
    for label,digest in seal['files'].items():
        assert sha(local/label)==digest
        a = read_npz(local/label)
        d = read_npz(WORK/'predictions'/label)
        subject, ref = a['subject_valid'],a['reference_valid']
        cross = np.clip((a['source_forward']+a['source_reverse'])*.5,-1.,1.)
        row = np.zeros(len(d['subject_boxes']))
        col = np.zeros(len(d['D_native_unary']))
        middle = np.zeros_like(d['native_left'])
        negative, one_way, other_way = middle.copy(),middle.copy(),middle.copy()
        if cross.size:
            ix = np.ix_(subject,ref)
            mean = cross.sum()/cross.size
            row[subject] = cross.sum(axis=1)/len(ref)-mean
            col[ref] = cross.sum(axis=0)/len(subject)-mean
            middle[ix] = separate(cross)
            step = 1+int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'little')%(len(ref)-1)
            negative[ix] = middle[np.ix_(subject,np.roll(ref,step))]
            one_way[ix] = separate(np.clip(a['source_forward'],-1,1))
            other_way[ix] = separate(np.clip(a['source_reverse'],-1,1))
        np.testing.assert_allclose(row,a['row_offset'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(col,a['column_offset'],rtol=0,atol=1e-12)
        for found, want in [(middle,a['edge']),(negative,a['mismatched'])]:
            np.testing.assert_allclose(found,want,rtol=0,atol=1e-12)
            max_error = max(max_error,float(np.max(np.abs(found-want))))
        base = d['subject_unary'][:,None]+d['D_native_unary']
        geometry = d['sign']*d['native_left']
        independent = base+row[:,None]+col
        joint = independent+geometry
        expected = [d['choices'][0],enumeration(base),enumeration(base+geometry),enumeration(independent),
                    enumeration(joint),enumeration(joint+middle),enumeration(joint+negative),
                    enumeration(joint+one_way),enumeration(joint+other_way)]
        np.testing.assert_array_equal(expected,a['choices'])
        selected_deltas['vs_original_subject'] += int(a['choices'][5,0]!=a['choices'][2,0])
        selected_deltas['vs_original_reference'] += int(a['choices'][5,1]!=a['choices'][2,1])
        selected_deltas['vs_additive_subject'] += int(a['choices'][5,0]!=a['choices'][4,0])
    path = HERE/'results/edge/INDEPENDENT_AUDIT.json'
    atomic_json(path,dict(rows=len(prior['files']),enumerated_arms=9,maximum_edge_roundoff=max_error,
                          choices=selected_deltas,source_weights_unchanged=seal['unchanged_weights'],
                          protected_data_access=False))
    print(path.read_text(),flush=True)


if __name__ == '__main__':
    run()
