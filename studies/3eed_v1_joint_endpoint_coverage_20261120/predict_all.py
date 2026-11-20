"""Make and seal all full-pool and control predictions without opening evaluation annotations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import numpy as np

from collect import CONFIG,HERE,PRIVATE,archive_npz,digest,read,save
from joint_coverage import ARMS,predict

DECISIONS=PRIVATE/'decisions'
SEALED=DECISIONS/'SEALED.json'


def run_one(job):
    name,observed,code=job
    f=PRIVATE/name
    assert digest(f)==observed
    dst=DECISIONS/name
    proof=dst.with_suffix('.json')
    if proof.exists():
        old=read(proof)
        assert old['observed']==observed and old['code']==code and digest(dst)==old['sha256']
        return name,old
    with np.load(f,allow_pickle=False) as z:
        predictions=predict(z)
        np.testing.assert_array_equal(predictions['boxes'][0],z['native_s'][int(z['before'][0])])
        assert np.isfinite(predictions['boxes']).all() and (predictions['boxes'][:,3:6]>0).all()
        if not predictions['applied']:
            np.testing.assert_array_equal(predictions['boxes'],np.repeat(predictions['boxes'][:1],len(ARMS),axis=0))
        active=bool(z['active'])
    archive_npz(dst,**predictions)
    receipt=dict(observed=observed,code=code,sha256=digest(dst),active=active,applied=bool(predictions['applied']))
    save(proof,receipt)
    return name,receipt


def run(nworkers):
    obs=read(PRIVATE/'OBSERVED.json')
    assert obs['n']==501 and not obs['labels'] and not obs['FINAL_access']
    assert list(ARMS)==read(CONFIG)['arms'] and obs['inputs']['protocol']==digest(CONFIG)
    DECISIONS.mkdir(mode=0o700,exist_ok=True)
    code={f:digest(HERE/f) for f in ('joint_coverage.py','predict_all.py','collect.py')}
    start=perf_counter();records={}
    jobs=[(name,pin,code) for name,pin in sorted(obs['files'].items())]
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        for i,(name,item) in enumerate(pool.map(run_one,jobs,chunksize=1),1):
            records[name]=item
            if i%100==0:
                print(i,len(jobs),round(perf_counter()-start,1),flush=True)
    save(SEALED,dict(n=len(records),files={k:v['sha256'] for k,v in records.items()},
        applied=sum(v['applied'] for v in records.values()),active=sum(v['active'] for v in records.values()),
        code=code,protocol=digest(CONFIG),observations=digest(PRIVATE/'OBSERVED.json'),arms=ARMS,
        cpu_seconds_elapsed=perf_counter()-start,source_training=0,target_updates=0,FINAL_access=False,labels=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',default=4,type=int)
    run(p.parse_args().workers)

