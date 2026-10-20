"""Decode every required original video before full evaluation; no GT access."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--ffprobe', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    rows = [json.loads(s) for s in Path(args.manifest).read_text().splitlines()]
    videos = {}
    for row in rows:
        m = row['metadata']
        key = m['video_sha256']
        if key not in videos:
            videos[key] = dict(metadata=m, maximum_frame=max(m['frame_ids']))
        else:
            videos[key]['maximum_frame'] = max(videos[key]['maximum_frame'], max(m['frame_ids']))

    def check(item):
        key, value = item
        m = value['metadata']
        if sha(m['video_path']) != key:
            raise ValueError('Media byte identity changed')
        r = subprocess.run([args.ffprobe, '-v', 'error', '-threads', '2', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=codec_name,width,height,nb_read_frames', '-of', 'json',
            m['video_path']], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        streams = json.loads(r.stdout).get('streams', [])
        if r.returncode or len(streams) != 1:
            raise ValueError('Full video decode failed: '+key+' '+r.stderr.decode(errors='replace')[-400:])
        s = streams[0]
        if (s['width'], s['height']) != (m['width'], m['height']) or int(s.get('nb_read_frames', 0)) <= value['maximum_frame']:
            raise ValueError('Video geometry/physical support mismatch: '+key+' '+str(s))
        return dict(video_sha256=key, decoded_frames=int(s['nb_read_frames']),
                    maximum_requested_frame=value['maximum_frame'], codec=s['codec_name'],
                    width=s['width'], height=s['height'])

    values = []
    with ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(check, item) for item in videos.items()]):
            values.append(future.result())
            if len(values) % 50 == 0:
                print('Media full-decode preflight', len(values), '/', len(videos), flush=True)
    result = dict(status='pass', queries=len(rows), videos=len(videos), no_queries_filtered=True,
        ffprobe_sha256=sha(args.ffprobe), manifest_sha256=sha(args.manifest),
        codecs=dict(Counter(v['codec'] for v in values)),
        all_actual_decoded_frame_counts_cover_fixed_input_grid=True,
        media_inventory_sha256=hashlib.sha256(json.dumps(sorted(values, key=lambda v:v['video_sha256']),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(), labels_read=False)
    write(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
