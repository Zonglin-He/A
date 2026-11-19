"""Check actual raw-box entrypoint results against sealed K predictions."""

import numpy as np
import torch

from retrieval180 import ARMS, HERE, RetrievalGrounder
from source180 import PLATFORMS, calibrated, prepare_device
from evaluate180 import OUT, archive_lock, previous
from features import observation
from scope import evaluation_rows, file_sha, read, save


def run():
    prepare_device()
    selection=calibrated()
    old=archive_lock()
    pins=read(OUT/'SEALED.json')
    assert pins['n']==3072 and not pins['target_labels_read'] and not pins['FINAL_access']
    rows=evaluation_rows()
    count=passes=0
    max_error=0.
    for source in PLATFORMS:
        model=RetrievalGrounder(source)
        for target in PLATFORMS:
            if source==target:
                continue
            path=OUT/f'{source}_to_{target}.npz'
            assert file_sha(path)==pins['files'][path.name]['sha256']
            ordinals,_,pool,_=previous(rows,source,target,old)
            with np.load(path,allow_pickle=False) as d:
                chosen=d['chosen']
            switches=np.flatnonzero(chosen[:,6]!=chosen[:,0])
            second=int(switches[0]) if len(switches) and switches[0]!=0 else 1
            expected_k=selection['sources'][source]['selected_k']
            for sample in (0,second):
                index=int(ordinals[sample])
                for label,args,k in (('trained_source_K',{},expected_k),('trained16',{'k':16},16),
                                     ('native8',{'k':8,'retrieval':'native'},8)):
                    val=model.predict(**args,**observation(rows[index]))
                    expect=pool[sample,chosen[sample,ARMS.index(label)]]
                    actual=np.array(val['box'])
                    np.testing.assert_allclose(actual,expect,atol=1e-12,rtol=0)
                    assert val['k']==k and val['incumbent_in_shortlist']
                    max_error=max(max_error,float(np.abs(actual-expect).max()))
                    count+=1
                    passes+=int(np.array_equal(actual,expect))
        del model
        torch.cuda.empty_cache()
    assert count==36
    save(HERE/'results/runtime.json',dict(checked=count,bit_exact=passes,max_abs_error=max_error,
         incumbent_retained=count,source_k_verified=True,target_labels_read=False,FINAL_access=False,
         runtime_entrypoint_sha256=file_sha(HERE/'ground180.py'),
         prediction_seal_sha256=file_sha(OUT/'SEALED.json')))


if __name__=='__main__':
    run()
