"""Read a small suffix of the public official archive, never print guest URLs/cookies."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests
import time
import os

ROOT = Path(__file__).resolve().parents[1] / 'recovery_workspace'
URL = 'https://1drv.ms/u/c/3c9af704fb61931d/EVOs609SGMxLsbvVzVJHAa4Bmnu4GVZGjqYHQxDz0NKTew?download=1'


def main():
    ROOT.mkdir(exist_ok=True)
    session = requests.Session()
    response = session.get(URL, headers={'Range':'bytes=0-0'}, stream=True, timeout=(30,60))
    response.raise_for_status()
    assert response.status_code == 206
    total = int(response.headers['Content-Range'].split('/')[-1])
    assert total == 10473330966
    direct = response.url
    etag = response.headers.get('ETag')
    response.close()
    mib=int(os.environ.get('AVIS_TAIL_MIB','8'))
    assert mib in [8,16]
    chunk, length = 1<<20, mib<<20
    offset = total-length
    old=json.loads((ROOT/'suffix_source.json').read_text()) if mib==16 else None
    if old and old.get('etag') and etag:assert old['etag']==etag

    def fetch(index):
        path = ROOT/(f'prefix16_part_{index:02d}.bin' if mib==16 else f'part_{index:02d}.bin')
        if path.exists() and path.stat().st_size==chunk:
            return
        a,b = offset+index*chunk, offset+(index+1)*chunk-1
        client = requests.Session()
        client.cookies.update(session.cookies)
        for attempt in range(5):
            try:
                r = client.get(direct, headers={'Range':f'bytes={a}-{b}'}, timeout=(30,75))
                if r.status_code==401:
                    r=client.get(URL,headers={'Range':f'bytes={a}-{b}'},timeout=(30,75))
                r.raise_for_status()
                assert r.status_code==206 and r.headers['Content-Range']==f'bytes {a}-{b}/{total}'
                assert len(r.content)==chunk
                path.write_bytes(r.content)
                print('tail chunk', index, 'verified', flush=True)
                return
            except Exception as e:
                print('retry',index,attempt,type(e).__name__,flush=True)
                time.sleep(min(2**attempt,16))
        raise RuntimeError(f'Failed tail chunk {index}')

    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        list(executor.map(fetch,range(8)))
    path=ROOT/f'tail_{mib}MiB.bin'
    if mib==16:
        prior=(ROOT/'tail_8MiB.bin').read_bytes()
        assert hashlib.sha256(prior).hexdigest()==old['sha256']
        path.write_bytes(b''.join((ROOT/f'prefix16_part_{i:02d}.bin').read_bytes() for i in range(8))+prior)
    else:path.write_bytes(b''.join((ROOT/f'part_{i:02d}.bin').read_bytes() for i in range(8)))
    info={'official_public_url':URL,'archive_bytes':total,'offset':offset,'bytes':length,
          'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'etag':etag,
          'claim':'Authenticated HTTPS range bytes only; no decompression or annotation claim yet.'}
    (ROOT/('suffix_source_16.json' if mib==16 else 'suffix_source.json')).write_text(json.dumps(info,indent=2))
    print('suffix complete',length,info['sha256'],flush=True)


if __name__=='__main__':main()
