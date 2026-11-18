"""Compute frozen candidate assignments without reading target labels."""
import hashlib

import numpy as np

from binding_collect import ARMS, HERE, PRIVATE, WORK, R171, atomic_json, read, sha
from binding_fit import open_checked
from edge_predict import OLD_ARMS


def shifted_terms(a, name):
    rc = a['reference_common']
    edge = a['edge']
    shifted = edge.copy()
    if len(rc):
        turn = 1 + int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little') % (len(rc)-1)
        shifted[:, :, rc] = np.roll(edge[:, :, rc], turn, axis=2)
    return shifted


def assignments(old, a, weights, name):
    from assignment171 import assign

    su, ru = old['subject_unary'], old['D_native_unary']
    left, sign = old['native_left'], old['sign']
    pure = su[:,None] + ru[None,:]
    base = pure + sign * left
    np.testing.assert_array_equal(base,a['baseline'])
    marginal = a['offset'].sum(0)
    additive = base+marginal
    edge = a['edge']
    shuffled = shifted_terms(a,name)
    choose = lambda energy: np.unravel_index(energy.argmax(),energy.shape)
    predictions = [old['choices'][0], assign(su,ru,left,sign,'independent'), assign(su,ru,left,sign,'joint'),
                   choose(pure+marginal),choose(additive),choose(additive+edge.sum(0)),
                   choose(additive+shuffled.sum(0)),choose(additive+np.einsum('k,ksr->sr',weights,edge)),
                   choose(additive+np.einsum('k,ksr->sr',weights,shuffled))]
    sb,rb = old['subject_boxes'],old['D_natural_boxes']
    boxes = np.stack([np.stack((sb[i],rb[j] if 0<=j<len(rb) else np.full(7,np.nan))) for i,j in predictions])
    choices = np.asarray(predictions,dtype=np.int16)
    indices = [OLD_ARMS.index(arm) for arm in ('r158_pca','D_native_independent','D_native_joint')]
    np.testing.assert_array_equal(old['choices'][indices],choices[:3])
    np.testing.assert_array_equal(old['method_boxes'][indices],boxes[:3])
    return dict(boxes=boxes, choices=choices, shuffled_edge=shuffled)


def main():
    from common160 import npz_atomic
    target = PRIVATE / 'target'
    proof, npz = open_checked(target)
    source_fit = PRIVATE / 'SOURCE_FIT.json'
    params = read(source_fit)
    assert params['source'] == sha(PRIVATE / 'source/SEAL.json')
    assert params['design'] == sha(PRIVATE / 'SOURCE_FIT_LOCK.json')
    assert not params['target_label_used'] and not params['protected_data_access']
    output = PRIVATE / 'predictions'
    output.mkdir(mode=0o700,parents=True,exist_ok=True)
    assert not (output / 'SEAL.json').exists()
    lock = dict(files={p.name:sha(p) for p in (HERE / 'binding.py',HERE / 'binding_predict.py',HERE / 'binding_protocol.json')},
                target=sha(target / 'SEAL.json'),source=sha(source_fit))
    atomic_json(output / 'LOCK.json',lock)
    prior = read(R171 / 'PREDICTION_SEAL.json')
    assert proof['files'].keys() == prior['files'].keys()
    files = {}
    for name,h in proof['files'].items():
        assert sha(target / name) == h and sha(WORK / 'predictions' / name) == prior['files'][name]
        source = name.split('_')[-1].removesuffix('.npz')
        previous = npz(WORK / 'predictions' / name)
        features = npz(target / name)
        variant = params['coefficients'][source]
        prediction = assignments(previous,features,np.asarray(variant['weights']),name)
        end = output/name
        end.parent.mkdir(exist_ok=True,mode=0o700)
        assert not end.exists()
        npz_atomic(end, **prediction, arms=np.asarray(ARMS), f_hash=h, source_fit=variant['source_fit_gate'])
        files[name] = sha(end)
    assert len(files) == 168
    atomic_json(output/'SEAL.json',dict(files=files,n=len(files),lock=sha(output/'LOCK.json'),target_labels_read=False,
                                        protected_data_access=False))
    print('Sealed role-assignment variants',len(files),flush=True)


if __name__ == '__main__':
    main()
