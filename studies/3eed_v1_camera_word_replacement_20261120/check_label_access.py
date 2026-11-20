"""Replay sealed source examples with a calibration file containing no annotations."""
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from calibrate import OUT, WORK, digest, read, save
from crossplatform.round7.camera_projection import CAMERA_KEYS, read_camera_allowlist
from predict import Grounder, frame, inputs, observe


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    source_dir=WORK/'source'
    seal=read(source_dir/'SEALED.json')
    slopes=read(OUT/'calibration.json')['slopes']
    checks=[]
    with tempfile.TemporaryDirectory(dir=WORK) as temporary:
        for (source,target), rows in inputs('source').items():
            name=f'{source}_to_{target}/00000.npz'
            assert digest(source_dir/name)==seal['files'][name]['sha256']
            with np.load(source_dir/name,allow_pickle=False) as saved:
                position=int(np.flatnonzero(saved['valid'].sum(1)>=2)[0])
                reference={k:saved[k][position] for k in saved.files}
            model=Grounder(source)
            model._frontend.parser=lru_cache(maxsize=2048)(model._frontend.parser)
            inp=frame(rows[position])
            allowed=read_camera_allowlist(inp['calibration'])
            assert 'ground_info' not in allowed and set(allowed)<=set(CAMERA_KEYS)
            path=Path(temporary)/(source+'.json')
            path.write_text(json.dumps(allowed,default=float))
            result=observe(model,dict(inp,calibration=str(path)),slopes[source])
            for k in reference:
                np.testing.assert_array_equal(reference[k],result[k])
            checks.append(dict(source=source,exposures=2,comparable_slots=int(reference['valid'].sum()),
                               reproduced=sorted(reference),calibration_keys=sorted(allowed)))
            del model
            torch.cuda.empty_cache()
    save(OUT/'annotation_access_check.json',dict(seal=digest(source_dir/'SEALED.json'),checks=checks,
         n_source_observations=len(checks),equal_all_boxes_scores_and_choices=True,
         original_annotation_files_written=False,FINAL_access=False))


if __name__=='__main__':
    main()
