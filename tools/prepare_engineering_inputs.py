"""Deterministic input-only engineering fixtures; full population is built separately."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write
from tools.build_manifests import hc_grid, vid_grid, VID_SHA
from tools.remote_archive import hc_zip, extract_member, SplitHTTPFile
from tools.fetch_assets import get_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/evaluation')
    args = parser.parse_args()
    output = ROOT/args.output
    hc = ROOT/'data/annotations/tastvg_release/data/hc-stvg2/annos/test.json'
    if not hc.exists():
        # Exact member of the author's uncompressed public data.tar, including
        # an independent saved hash. This exact 1902-record release was subsequently confirmed by the user.
        payload, _ = get_range('https://huggingface.co/Gstar666/TASTVG/resolve/main/data.tar',
                              181358080, 190237292, 6592634880)
        hc.parent.mkdir(parents=True, exist_ok=True)
        hc.write_bytes(payload)
    if sha(hc) != '61f7a2a74d39c13dd380540d4ac819edb7abcde1826d44229e539d128bf3260c':
        raise ValueError('Author engineering annotation release changed')
    raw = json.loads(hc.read_text())
    name = min(raw, key=lambda n: hashlib.sha256(('tubedetr-interface-v1|'+n).encode()).hexdigest())
    a = raw[name]
    path = ROOT/'data/engineering/hc2'/name
    if not path.exists():
        with hc_zip() as z:
            member = next(i.filename for i in z.infolist() if Path(i.filename).name == name)
            extract_member(z, member, path)
    write(output/'engineering_hc2.json', dict(dataset='hc2', key='tubedetr-interface-hc2-v1', metadata=dict(
        caption=a['English'].lower(), index=0, width=a['width'], height=a['height'], fps=a['img_num']/20,
        original_video_id=name, frame_count=a['img_num'], frame_ids=hc_grid(a['img_num']),
        video_path=str(path.resolve()), video_sha256=sha(path))))
    ann = ROOT/'data/annotations/vidstg/test_annotations.json'
    if not ann.exists():
        ann.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve('https://raw.githubusercontent.com/Guaranteer/VidSTG-Dataset/master/annotations/test_annotations.json', ann)
    if sha(ann) != VID_SHA:
        raise ValueError('Canonical VidSTG annotation release changed')
    raw = json.loads(ann.read_text())
    index = min(range(len(raw)), key=lambda i: hashlib.sha256(('tubedetr-interface-v1|'+str(i)).encode()).hexdigest())
    a = raw[index]
    with zipfile.ZipFile(ROOT/'data/vidor/validation-annotation.zip') as z:
        v = json.loads(z.read(next(n for n in z.namelist() if Path(n).stem == a['vid'])))
    path = ROOT/'data/engineering/vidstg'/Path(v['video_path']).name
    if not path.exists():
        with zipfile.ZipFile(SplitHTTPFile([('https://huggingface.co/datasets/shangxd/vidor/resolve/main/validation-video.zip', 3056763915)])) as z:
            member = next(n for n in z.namelist() if Path(n).name == path.name)
            extract_member(z, member, path)
    write(output/'engineering_vidstg.json', dict(dataset='vidstg', key='tubedetr-interface-vidstg-v1', metadata=dict(
        caption=a['captions'][0]['description'], index=0, width=a['width'], height=a['height'], fps=a['fps'],
        original_video_id=a['vid'], frame_count=a['frame_count'],
        frame_ids=vid_grid(a['used_segment']['begin_fid'], a['used_segment']['end_fid']),
        video_path=str(path.resolve()), video_sha256=sha(path))))
    write(output/'engineering_exposure.json', dict(status='registered_engineering_exposure',
        selection='fixed hash of annotation key/index before native outputs; no GT selection',
        requests={ds: sha(output/f'engineering_{ds}.json') for ds in ['hc2', 'vidstg']},
        HC1902_formal_release_confirmed_by_user=True))
    print('Two deterministic engineering inputs prepared; both registered as development exposure')


if __name__ == '__main__':
    main()
