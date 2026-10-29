"""Verify every explicit export file against immutable public commit bytes."""
import argparse, concurrent.futures, hashlib, json, time
from pathlib import Path
import requests

def main():
    p=argparse.ArgumentParser();p.add_argument('commit');p.add_argument('output',type=Path);a=p.parse_args()
    base=Path(__file__).resolve().parents[1]
    root=base if (base/'MANIFEST.json').is_file() else base/'public_export'
    manifest=json.loads((root/'MANIFEST.json').read_text())
    entries=manifest['files'];assert entries, 'An empty allowlist cannot establish verification'
    allowed={x['path'] for x in entries};assert len(allowed)==len(entries), 'Duplicate paths'
    assert 'MANIFEST.json' not in allowed
    for entry in entries:
        rel=Path(entry['path']);assert not rel.is_absolute() and '..' not in rel.parts
        f=root/rel;assert f.is_file() and not f.is_symlink()
        b=f.read_bytes();assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256']
    allowed.add('MANIFEST.json')
    actual={x.relative_to(root).as_posix() for x in root.rglob('*') if x.is_file()}
    assert actual==allowed, 'Unexpected or missing export files'
    files=[root/rel for rel in sorted(allowed)]
    def check(f):
        rel=f.relative_to(root).as_posix();b=f.read_bytes();last=None
        for attempt in range(4):
            try:
                r=requests.get(f'https://raw.githubusercontent.com/Zonglin-He/A/{a.commit}/studies/three_direction_tta_20261029/{rel}',timeout=45)
                r.raise_for_status();assert r.content==b,rel
                return {'path':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'remote_bytes_exact':True}
            except Exception as e:
                last=e
                if attempt<3:time.sleep(1+attempt)
        raise RuntimeError(rel) from last
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(check,files))
    obj={'commit':a.commit,'prefix':'studies/three_direction_tta_20261029/','all_remote_bytes_exact':True,'files':rows,'count':len(rows)}
    a.output.write_text(json.dumps(obj,indent=2)+'\n');print(json.dumps({'commit':a.commit,'count':len(rows),'all_remote_bytes_exact':True}))

if __name__=='__main__':main()
