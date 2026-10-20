"""Compare selected RGB to an independent streaming decode on physical IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(path):
    from stvg.inputs import validate_metadata
    from stvg.tube_decode import decode, asset_identity
    from stvg.runtime import sha
    request = json.loads(Path(path).read_text())
    meta, ds = request['metadata'], request['dataset']
    validate_metadata(meta)
    frames, ids = decode(meta, ds)
    selected_hash = hashlib.sha256(frames.tobytes()).hexdigest()
    del frames
    asset = asset_identity()
    cmd = [asset['ffmpeg_path'], '-v', 'error', '-i', meta['video_path']]
    if ds == 'vidstg':
        cmd += ['-vsync', '0']
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    size = meta['height']*meta['width']*3
    expected = set(ids)
    reference = hashlib.sha256()
    count = selected = 0
    with tempfile.TemporaryFile() as err:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                frame = process.stdout.read(size)
                if not frame:
                    break
                if len(frame) != size:
                    raise RuntimeError('Incomplete reference RGB frame')
                if count in expected:
                    reference.update(frame)
                    selected += 1
                count += 1
            if process.wait() != 0:
                raise RuntimeError('Independent streaming decoder failed')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
    if selected != len(ids) or reference.hexdigest() != selected_hash:
        raise RuntimeError('Selected RGB does not match physical full-decode frame IDs')
    return dict(status='pass', dataset=ds, request_sha256=sha(path), selected_frames=selected,
                decoded_frames=count, physical_grid_sha256=hashlib.sha256(np.array(ids, dtype=np.int64).tobytes()).hexdigest(),
                rgb_sha256=selected_hash, decoder_asset=asset,
                independent_streaming_selection_bitexact=True, GT_read=False,
                scope='physical frame selection under this pinned build; historical-build parity not established')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from stvg.tube_runner import write
    result = check(args.request)
    write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
