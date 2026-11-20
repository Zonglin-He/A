"""Time cache-warm full predictions using identical examples in separate releases."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np


def worker(package,asset_root,manifest,out):
    import torch
    sys.path.insert(0,str(package.resolve()))
    import crossplatform.ground as frozen
    from crossplatform.ground_regions import Grounder
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    current=engine=None
    results=[]
    for example in json.loads(manifest.read_text()):
        src,form,inputs=example['source'],example['form'],example['inputs']
        if src!=current:
            engine=None
            engine=Grounder(src,asset_root=asset_root)
            current=src
        expected=np.array(example['box'])
        calls=[]
        hook=engine._decoder.register_forward_hook(lambda *_:calls.append(1))
        try:
            with patch.object(frozen,'decoder_evidence',wraps=frozen.decoder_evidence) as evidence:
                np.testing.assert_array_equal(engine.predict(**inputs),expected)
        finally:
            hook.remove()
        ms=[]
        for _ in range(3):
            torch.cuda.synchronize()
            start=perf_counter_ns()
            outbox=engine.predict(**inputs)
            torch.cuda.synchronize()
            ms.append((perf_counter_ns()-start)/1e6)
            np.testing.assert_array_equal(outbox,expected)
        results.append(dict(source=src,target=inputs['target'],form=form,ms=ms,decoder_calls=len(calls),
                            decoder_evidence_calls=evidence.call_count))
    out.write_text(json.dumps(results))


def compare(before,after):
    from replay import OUT, ROOT, save
    from check_four_default import example_cases
    OUT.mkdir(exist_ok=True)
    env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
    with TemporaryDirectory() as temp:
        folder=Path(temp)
        manifest=folder/'cases.json'
        cases=[dict(source=s,form=f,inputs=x,box=y.tolist()) for s,f,x,y in example_cases()]
        assert len(cases)==14
        manifest.write_text(json.dumps(cases,default=str))
        runs=[[],[]]
        for n,version in enumerate((0,1,1,0)):
            result=folder/'run.json'
            subprocess.run([sys.executable,'-I',__file__,'worker',str((before,after)[version]),
                            str(ROOT),str(manifest),str(result)],check=True,cwd=folder,env=env)
            runs[version].append(json.loads(result.read_text()))
            print('Finished paired raw timing block',n+1,flush=True)
        rows=[]
        for j,case in enumerate(cases):
            previous=runs[0][0][j]
            now=runs[1][0][j]
            assert all(r[j]['decoder_evidence_calls']==r[j]['decoder_calls']==int(i==0 or case['form']=='fallback')
                       for i,twice in enumerate(runs) for r in twice)
            old_ms,new_ms=(float(np.median([t for r in twice for t in r[j]['ms']])) for twice in runs)
            rows.append(dict(source=case['source'],target=case['inputs']['target'],form=case['form'],
                             previous_ms=old_ms,tree_ms=new_ms,saved_ms=old_ms-new_ms,
                             decoder_calls_before=previous['decoder_calls'],decoder_calls_after=now['decoder_calls']))
        with (OUT/'end_to_end.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        stats={}
        for form in ('relation','ordered','fallback','all'):
            block=[r for r in rows if form=='all' or r['form']==form]
            stats[form]=dict(rows=len(block),**{k:float(np.median([r[k] for r in block]))
                             for k in ('previous_ms','tree_ms','saved_ms')},
                             faster=sum(r['saved_ms']>0 for r in block))
        save(OUT/'entry.json',dict(cases=rows,by_form=stats,blocks=['previous','tree','tree','previous'],
             device='RTX 5090',cpu_threads_per_process=1,cuda_synchronized=True,repeats_per_case_per_version=6,
             warmup_per_block_per_case=1,model_initialization_included=False,same_seven_coordinates=True,
             FINAL_access=False))
        checks=[]
        for case in cases:
            form=case['form']
            if form in {v['form'] for v in checks}:continue
            cmd=[sys.executable,'-I',str(after/'crossplatform/ground_regions.py'),'--asset-root',str(ROOT),
                 '--source',case['source']]
            for k,v in case['inputs'].items():
                cmd.extend(['--'+k.replace('_','-'),str(v)])
            run=subprocess.run(cmd,check=True,capture_output=True,text=True,cwd=folder,env=env)
            result=json.loads(run.stdout)
            assert result['frame']=='target_lidar'
            np.testing.assert_array_equal(result['box'],case['box'])
            checks.append(dict(source=case['source'],target=case['inputs']['target'],form=form,coordinate_error=0))
        assert len(checks)==3
        save(OUT/'cli.json',dict(cases=checks,isolated=True,FINAL_access=False))


if __name__=='__main__':
    if sys.argv[1]=='worker':
        worker(*map(Path,sys.argv[2:]))
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('previous_package',type=Path)
        parser.add_argument('current_package',type=Path)
        a=parser.parse_args()
        compare(a.previous_package,a.current_package)
