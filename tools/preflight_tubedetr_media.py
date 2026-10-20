"""Decode every required original video before full evaluation; no GT access."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write


def hc_output_support(ffmpeg, metadata):
    """Count the RGB output of the unchanged HC decoder, including muxer sync.

    ffprobe counts source frames. HC indexes the default rawvideo output,
    which may contain timestamp-driven duplicates. The null sink avoids
    retaining hundreds of MB per video; progress also verifies byte size.
    """
    r = subprocess.run([str(ffmpeg), '-v', 'error', '-i', metadata['video_path'],
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-progress', 'pipe:1', '-nostats',
        '-y', os.devnull], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    fields = dict(line.split('=', 1) for line in r.stdout.decode().splitlines() if '=' in line)
    frames = int(fields.get('frame', 0))
    size = int(fields.get('total_size', -1))
    if (r.returncode or fields.get('progress') != 'end' or frames <= 0
            or size != frames * metadata['width'] * metadata['height'] * 3):
        raise ValueError('HC default RGB output failed: '+r.stderr.decode(errors='replace')[-400:])
    return dict(decoded_frames=frames, output_bytes=size,
                duplicated_frames=int(fields.get('dup_frames', 0)),
                dropped_frames=int(fields.get('drop_frames', 0)))


def validate_support(metadata, maximum_frame, stream, output_frames):
    if ((stream['width'], stream['height']) != (metadata['width'], metadata['height'])
            or output_frames <= maximum_frame):
        raise ValueError('Video geometry/physical support mismatch: '+metadata['video_sha256']+' '+
                         str(dict(stream=stream, output_frames=output_frames,
                                  maximum_requested_frame=maximum_frame)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--ffprobe', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    rows = [json.loads(s) for s in Path(args.manifest).read_text().splitlines()]
    if not rows or len({r['dataset'] for r in rows}) != 1:
        raise ValueError('One nonempty single-dataset manifest is required')
    dataset = rows[0]['dataset']
    if dataset not in {'hc2', 'vidstg'}:
        raise ValueError('Unknown physical frame convention')
    ffmpeg = None
    if dataset == 'hc2':
        from stvg.tube_decode import asset_identity
        ffmpeg = Path(asset_identity()['ffmpeg_path'])
        if Path(args.ffprobe).resolve() != ffmpeg.with_name('ffprobe').resolve():
            raise ValueError('HC preflight must use the registered decoder tool pair')
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
        native_frames = int(s.get('nb_read_frames', 0))
        output = hc_output_support(ffmpeg, m) if dataset == 'hc2' else dict(decoded_frames=native_frames)
        validate_support(m, value['maximum_frame'], s, output['decoded_frames'])
        return dict(video_sha256=key, native_decoded_frames=native_frames, **output,
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
        frame_index_domain='HC default FFmpeg RGB output' if dataset == 'hc2' else 'native source frame indices',
        ffmpeg_sha256=sha(ffmpeg) if ffmpeg else None,
        videos_with_output_count_different_from_native=sum(v['decoded_frames'] != v['native_decoded_frames'] for v in values),
        codecs=dict(Counter(v['codec'] for v in values)),
        all_actual_decoded_frame_counts_cover_fixed_input_grid=True,
        media_inventory_sha256=hashlib.sha256(json.dumps(sorted(values, key=lambda v:v['video_sha256']),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(), labels_read=False)
    write(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
