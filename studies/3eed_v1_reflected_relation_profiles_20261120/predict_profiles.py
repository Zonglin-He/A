"""Seal all official and control decisions before opening the annotation inventory."""
import numpy as np

from observe import CONFIG, HERE, WORK, archive_npz, digest, read, save
from profiles import ARMS, predict

DECISIONS = WORK/'decisions'
SIGNATURE = DECISIONS/'SEALED.json'
FIELDS = ('subject','reference','joint','original_choice','active','original_ref_valid')


def seal():
    data = read(WORK/'SEALED.json')
    assert data['n']==len(data['files'])==501 and not data['labels_used'] and not data['FINAL_access']
    assert data['inputs']['protocol']==digest(CONFIG)
    DECISIONS.mkdir(mode=0o700,exist_ok=True)
    all_hashes={}
    active=0
    max_error=0.
    for name,pinned in sorted(data['files'].items()):
        assert digest(WORK/name)==pinned
        with np.load(WORK/name,allow_pickle=False) as arrays:
            pred=predict(**{k:arrays[k] for k in FIELDS})
            active+=bool(arrays['active'])
            max_error=max(max_error,float(arrays['point_inverse_error']))
            np.testing.assert_array_equal(pred['raw'][0],arrays['subject'][0,int(arrays['original_choice'][0])])
            for a in ('joint_raw','weighted_boxes'):
                np.testing.assert_array_equal(pred['choice'][ARMS.index(a)],pred['choice'][-1])
            np.testing.assert_array_equal(pred['raw'][ARMS.index('fixed_pair_geometry')],pred['raw'][0])
            assert np.isfinite(pred['boxes']).all()
            if not arrays['active']:
                np.testing.assert_array_equal(pred['boxes'],np.repeat(pred['raw'][:1],len(ARMS),axis=0))
        archive_npz(DECISIONS/name,**pred)
        all_hashes[name]=digest(DECISIONS/name)
    record=dict(n=len(all_hashes),active=active,max_inverse_point_error=max_error,
                observation_seal=digest(WORK/'SEALED.json'),protocol=digest(CONFIG),
                code={name:digest(HERE/name) for name in ('profiles.py','predict_profiles.py')},
                files=all_hashes,arms=ARMS,labels_used=False,FINAL_access=False)
    save(SIGNATURE,record)
    print('Sealed',record['n'],'active',active,'max inverse point error',max_error,flush=True)


if __name__=='__main__':
    seal()

