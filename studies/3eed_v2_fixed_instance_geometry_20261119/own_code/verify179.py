"""Compare the live geometry selector to sealed choices without annotations."""

import numpy as np
import torch

from ground179 import FixedInstanceGeometryGrounder
from run179 import (HERE, PREDICTIONS, SELECTORS, PLATFORMS, evaluation_rows,
                    file_sha, protocol, read, save)
from features import observation


def verify():
    rows = evaluation_rows()
    seal = read(PREDICTIONS/'SEALED.json')
    assert seal['n']==protocol()['evaluation']['direction_records']
    maximum = 0.
    checks = exact = samples = 0
    for source in PLATFORMS:
        model = FixedInstanceGeometryGrounder(source)
        for target in PLATFORMS:
            if target==source:
                continue
            name = f'{source}_to_{target}.npz'
            path = PREDICTIONS/name
            assert file_sha(path)==seal['files'][name]['sha256']
            with np.load(path,allow_pickle=False) as cache:
                # Check a fixed input and an exercised choice where this direction permits one.
                switched = np.flatnonzero(cache['switches'][:,-1])
                second = int(switched[0]) if len(switched) and switched[0]!=0 else 1
                for k in (0,second):
                    frame = observation(rows[int(cache['ordinals'][k])])
                    for selector in SELECTORS:
                        output = model.predict(selector=selector,**frame)
                        arm = protocol()['arms'].index(selector)
                        actual = np.asarray(output['box'])
                        expected = cache['boxes'][k,arm]
                        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
                        assert output['carrier']==cache['carrier'][k]
                        assert (output['geometry']=='refined')==cache['switches'][k,arm]
                        assert output['support_eligible']==cache['supported'][k]
                        maximum=max(maximum,float(np.max(np.abs(actual-expected))))
                        exact+=int(np.array_equal(actual,expected))
                        checks+=1
                    samples+=1
        del model
        torch.cuda.empty_cache()
    assert checks==36 and samples==12
    save(HERE/'results/runtime.json',dict(samples=samples,predictions=checks,exact=exact,
         max_abs_error=maximum,all_carriers_match=True,all_support_and_switches_match=True,
         grounder_sha256=file_sha(HERE/'ground179.py'),
         prediction_seal_sha256=file_sha(PREDICTIONS/'SEALED.json'),
         target_labels_read=False,FINAL_access=False))


if __name__=='__main__':
    verify()
