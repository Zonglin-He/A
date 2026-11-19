"""Reproduce sealed raw outputs through a live single-frame entrypoint."""

import numpy as np
import torch

from ground181 import ARMS, CalibratedGrounder
from capture181 import HERE, load_r180
from score181 import CHOICES, RESULTS
from scope import PLATFORMS, evaluation_rows, file_sha, read, save
from features import observation
from train import prepare_device


def verify():
    prepare_device()
    rows=evaluation_rows()
    old=load_r180(rows)
    receipt=read(CHOICES/'SEALED.json')
    assert receipt['n']==3072 and not receipt['target_labels_used'] and not receipt['FINAL_access']
    arms=('frozen_joint','calibrated_subject','calibrated_joint','calibrated_mismatch')
    check=exact=0
    largest=0.
    for source in PLATFORMS:
        grounder=CalibratedGrounder(source)
        for target in PLATFORMS:
            if target==source:
                continue
            name=f'{source}_to_{target}.npz'
            pin=CHOICES/name
            assert file_sha(pin)==receipt['files'][name]
            with np.load(pin,allow_pickle=False) as content:
                chosen=content['chosen']
            block=old[source,target]
            altered=np.flatnonzero((chosen[:,4:]!=chosen[:,:1]).any(axis=1))
            second=int(altered[0]) if len(altered) and altered[0]!=0 else 1
            for i in (0,second):
                frame=observation(rows[int(block['ordinals'][i])])
                for arm in arms:
                    value=grounder.predict(arm=arm,**frame)
                    c=int(chosen[i,ARMS.index(arm)])
                    expected=block['pool'][i,block['slate'][i,6,c]]
                    actual=np.asarray(value['box'])
                    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
                    assert value['retains_v1']==(c==chosen[i,0])
                    largest=max(largest,float(np.abs(actual-expected).max()))
                    exact+=int(np.array_equal(actual,expected))
                    check+=1
        del grounder
        torch.cuda.empty_cache()
    assert check==48
    save(RESULTS/'runtime.json',dict(checked=check,bit_exact=exact,max_abs_error=largest,
         calibration_choices_sha256=file_sha(CHOICES/'SEALED.json'),
         entrypoint_sha256=file_sha(HERE/'ground181.py'),target_labels_read=False,FINAL_access=False))


if __name__=='__main__':
    verify()
