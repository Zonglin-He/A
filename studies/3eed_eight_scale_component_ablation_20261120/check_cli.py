"""Call the installed CLI with no research paths, from an unrelated working directory."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import numpy as np

from common import ROOT, OUT, prior, read, digest, save, catalog, row_id, ARMS, WORK
from scope183 import ordinary, frame


def run(package_root,receipt):
    roster=ordinary()
    sums=[]
    root=prior.FULL/'predictions'
    active_rows={(r['source'],r['target'],r['ordinal']):r for r in catalog()['rows'] if r['scope']=='ordinary'}
    for src,tgt,active,scales in [('waymo','drone',True,8),('quad','waymo',True,8),('drone','quad',False,8),('drone','waymo',True,4)]:
        folder=root/(src+'_to_'+tgt)
        assert digest(folder/'all.npz')==read(folder/'SEALED.json')['all']
        with np.load(folder/'all.npz') as a:
            ix=int(np.flatnonzero(a['registered']==active)[0])
            expected=a['completed'][ix,prior.old.ARMS.index('shared_cut')]
        if scales==4:
            from score_roles import sensor_frame_box
            saved=active_rows[src,tgt,ix]
            with np.load(WORK/'boxes'/(row_id(saved)+'.npz')) as short:
                expected=sensor_frame_box(short['subject'][ARMS.index('four_scales')],tgt,roster[tgt][ix]['suffix'])
        inp=frame(roster[tgt][ix])
        cmd=[sys.executable,'-I',str(package_root/'crossplatform/ground_regions.py'),'--asset-root',str(ROOT),'--source',src]
        if scales!=8:cmd+=['--scales',str(scales)]
        for k,v in inp.items():
            cmd.extend(('--'+k.replace('_','-'),str(v)))
        with TemporaryDirectory() as tmp:
            env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
            res=subprocess.run(cmd,cwd=tmp,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if res.returncode:raise RuntimeError(res.stderr[-3500:])
        output=json.loads(res.stdout)
        assert output['frame']=='target_lidar'
        err=float(abs(np.asarray(output['box'])-expected).max());assert err<1e-5
        sums.append(dict(source=src,target=tgt,active=active,scales=scales,max_error=err))
    save(OUT/receipt,dict(cases=sums,isolated_python=True,package_root='deployment package',FINAL_access=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--receipt',default='cli_replay.json')
    a=p.parse_args();run(a.root,a.receipt)
