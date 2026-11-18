"""Run the shipped CLI from another working directory with relative sensor paths."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'crossplatform/round7'))
from camera_projection import read_camera_allowlist


def verify(entry):
    extra = ROOT/'crossplatform/round158'
    rows = json.loads((extra/'EXTRA_REGISTRY_PRIVATE.json').read_text())
    i = next(i for i,r in enumerate(rows) if r['source']=='waymo' and r['target']=='drone')
    r = rows[i]['row']
    with np.load(extra/'EXTRA_PRIVATE.npz', allow_pickle=False) as data:
        expected = data['boxes'][i,1].astype(float)
    expected[2] -= 1.8
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        (work/'points.bin').symlink_to(Path(r['pointpath']).resolve())
        (work/'frame.jpg').symlink_to(Path(r['imagepath']).resolve())
        cal = read_camera_allowlist(r['metapath'])
        (work/'camera.json').write_text(json.dumps(cal, default=float))
        command = [sys.executable, str(entry), '--source','waymo','--target','drone','--points','points.bin',
                   '--rgb','frame.jpg','--calibration','camera.json','--text',r['caption'],'--seed',str(r['seed'])]
        run = subprocess.run(command,cwd=work,capture_output=True,text=True,timeout=90)
        if run.returncode:
            raise RuntimeError('CLI failed: '+run.stderr[-700:])
        output = json.loads(run.stdout)
        if (set(output) != {'box','frame','length_unit','yaw_unit'} or output['frame'] != 'target_lidar'
                or output['length_unit'] != 'm' or output['yaw_unit'] != 'rad'
                or not np.allclose(output['box'],expected,rtol=0,atol=1e-5)):
            raise ValueError('Unexpected CLI output')
    record = dict(status='PASS', real_sensor_CLI='waymo_to_drone', external_cwd_relative_paths=True,
                  only_single_json_on_stdout=True, selected_box_max_error=float(np.max(np.abs(output['box']-expected))),
                  target_ground_truth_access=False, FINAL_access=False)
    return record


if __name__=='__main__':
    record = verify(ROOT/'crossplatform/ground_single_frame.py')
    (HERE/'CLI_REPLAY.json').write_text(json.dumps(record,indent=2)+'\n')
    print(record['status'])
