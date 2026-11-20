"""Exercise the deployed Python and isolated CLI entry points on archived raw frames."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from replay import OUT, prior, save
from check_four_default import example_cases
from crossplatform.ground_regions import Grounder, regions, selection


def check(package=None):
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    results = []
    engine = source = None
    for src,form,inputs,expected in example_cases():
        audit = {}
        if package is None:
            if source != src:
                engine = None
                engine = Grounder(src)
                source = src
            with (patch.object(regions,'complete',wraps=regions.complete) as complete,
                  patch.object(selection,'select',wraps=selection.select) as select,
                  patch.object(engine._frontend,'score_frame',wraps=engine._frontend.score_frame) as front):
                out = engine.predict(**inputs)
            assert all(call.kwargs['read_reference'] is False for call in select.call_args_list)
            assert 0 <= complete.call_count <= select.call_count <= 1
            assert front.call_count == (1 if form=='fallback' else 2)
            audit = dict(box_completions=complete.call_count,subject_decisions=select.call_count,
                         extra_proposal_forwards=front.call_count-1)
        else:
            if any(r['form']==form for r in results):
                continue
            cmd = [sys.executable,'-I',str(package/'crossplatform/ground_regions.py'),
                   '--asset-root',str(prior.ROOT),'--source',src]
            for k,v in inputs.items():
                cmd.extend(['--'+k.replace('_','-'),str(v)])
            with TemporaryDirectory() as folder:
                p = subprocess.run(cmd,cwd=folder,env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'},
                                   capture_output=True,text=True,check=True)
            output = json.loads(p.stdout)
            assert output['frame']=='target_lidar'
            out = np.asarray(output['box'])
        np.testing.assert_array_equal(out,expected)
        results.append(dict(source=src,target=inputs['target'],form=form,max_error=0.,**audit))
    assert len(results)==(3 if package else 14)
    OUT.mkdir(exist_ok=True)
    save(OUT/('cli.json' if package else 'entry.json'),dict(cases=results,
         all_seven_coordinates_equal=True,target_label_selection=False,FINAL_access=False))
    print('Verified raw input cases:',len(results),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path)
    check(p.parse_args().package)
