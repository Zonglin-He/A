"""Exercise the two late fallback paths after a successfully parsed relation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


def worker(package,root,manifest,output):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(package.resolve()))
    from crossplatform.ground_regions import Grounder, regions, selection
    import crossplatform.ground as frozen
    case=json.loads(manifest.read_text())
    engine=Grounder(case['source'],asset_root=root)
    out={}
    for name,owner,method,value in (
            ('empty_inventory',regions,'build_inventory',{'parent':np.array([],int)}),
            ('no_feasible_pair',selection,'select',(-1,-1))):
        calls=[]
        hook=engine._decoder.register_forward_hook(lambda *_:calls.append(1))
        try:
            with (patch.object(owner,method,return_value=value) as inject,
                  patch.object(frozen,'decoder_evidence',wraps=frozen.decoder_evidence) as evidence):
                b=engine.predict(**case['inputs'])
            assert inject.call_count==evidence.call_count==len(calls)==1
        finally:
            hook.remove()
        out[name]=b.tolist()
    output.write_text(json.dumps(out))


def check(before,after):
    from replay import OUT,ROOT,save
    from check_four_default import example_cases
    src,_,inputs,_=next(item for item in example_cases() if item[1]=='relation')
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)
        manifest=folder/'case.json'
        manifest.write_text(json.dumps(dict(source=src,inputs=inputs),default=str))
        versions=[]
        for package in (before,after):
            out=folder/'boxes.json'
            subprocess.run([sys.executable,'-I',__file__,'worker',str(package),str(ROOT),str(manifest),str(out)],
                           cwd=folder,check=True,env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'})
            versions.append(json.loads(out.read_text()))
        for key in versions[0]:
            np.testing.assert_array_equal(versions[0][key],versions[1][key])
    save(OUT/'late_fallback.json',dict(model_check='same existing relation frame in both releases',
         simulated_branches=['empty_inventory','no_feasible_pair'],each_decoder_and_input_build_called_once=True,
         max_coordinate_difference=0.,ordinary_metric_rows_added=0,FINAL_access=False))


if __name__=='__main__':
    if sys.argv[1]=='worker':
        worker(*map(Path,sys.argv[2:]))
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('previous_package',type=Path)
        parser.add_argument('current_package',type=Path)
        a=parser.parse_args()
        check(a.previous_package,a.current_package)
