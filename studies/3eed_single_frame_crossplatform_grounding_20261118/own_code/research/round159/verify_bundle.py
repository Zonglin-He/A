"""Start from an empty tree; provide the first-party runtime and the operator's model assets."""
import json
import shutil
import tempfile
from pathlib import Path

from verify_cli import HERE, ROOT, verify


def main():
    files = json.loads((HERE/'FIRST_PARTY_RUNTIME.json').read_text()) + ['crossplatform/single_frame_sources.json']
    pins = json.loads((ROOT/'crossplatform/single_frame_sources.json').read_text())['waymo']
    assets = [pins['epoch100'], pins['matcher'], 'crossplatform/round3/p0/upstream',
              'crossplatform/upstream', 'crossplatform/round20/runtime',
              'crossplatform/round20/assets/STANZA_ASSET_RECEIPT.json','crossplatform/round20/assets/stanza']
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for rel in files:
            dest = root/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/rel,dest)
        for rel in assets:
            dest = root/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to((ROOT/rel).resolve(),target_is_directory=(ROOT/rel).is_dir())
        report = verify(root/'crossplatform/ground_single_frame.py')
    report.update(first_party_files=len(files),only_required_source_weight_pair_present=True,
                  archived_predictions_and_role_files_in_bundle=0)
    (HERE/'BUNDLE_REPLAY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])


if __name__=='__main__':
    main()
