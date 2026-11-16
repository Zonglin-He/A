"""Pin fixed execution programs after engineering, never target metric selection."""
from pathlib import Path
import json,hashlib,argparse
R=Path(__file__).resolve().parents[2];H=R/'research/round97'
files=list(H.glob('*.py'))+[R/'tools/manual_r97_guard.py']
for folder in ['round76','round79','round88','round89','round90','round91']:
 files+=list((R/'research'/folder).glob('core*.py'))
files += [R/f for f in ['research/round76/model76.py','research/round76/prepare76.py','research/round88/routing88.py','research/round91/targets91.py','research/round66/interface66.py','crossplatform/round51/model51.py','crossplatform/round7/camera_projection.py','crossplatform/round7/experiment/runner_helpers.py','crossplatform/round96/factory.py','crossplatform/round48/parser48.py','research/round85/evidence85.py','research/round87/reference_roles87.py']]
files += [R/f for f in ['research/round76/common76.py','research/round66/common66.py','research/round86/evidence86.py','crossplatform/round7/regression_adapter.py']]
a=argparse.ArgumentParser();a.add_argument('--engineering',action='store_true');args=a.parse_args();p=H/'experiment'/('ENGINEERING_CODE_LOCK.json' if args.engineering else 'TARGET_CODE_LOCK.json');assert not p.exists();p.write_text(json.dumps({'method':'fixed R91 physical_bt','original_checkpoint_and_boxes_frozen':True,'no_target_metric_selection':True,'code_sha256':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(files))}},indent=2)+'\n');print({'pinned_code':len(set(files))})
