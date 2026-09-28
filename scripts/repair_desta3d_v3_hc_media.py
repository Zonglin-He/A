"""Recover only failed HC1 clips from official HC2 when annotations are exact.

No renaming/replacing a query, no evaluation outcomes, no model. Old damaged
archive extracts remain untouched. Different media bytes are explicitly logged.
"""
import hashlib,json,shutil,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_desta3d_v3_source import read,put,sha,HC,BASE
from scripts.audit_desta3d_v3_media import probe,OUT as QA
OUT=ROOT/'artifacts/desta3d_v3/hc_media_repair_v1';DATA=ROOT/'data/desta3d_v3_source/hc1_repaired'
def main():
    finished=read(QA/'COMPLETE.json');assert not (OUT/'COMPLETE.json').exists()
    h1=read(HC/'train.json');h2root=BASE/'hcstvg/datasets/HC-STVG2/HC-STVG';h2=read(h2root/'anno_v2/train_v2.json')
    failed=[r for r in finished['failures'] if r['key'].startswith('hc1-')]
    put(OUT/'REGISTRATION.json',dict(time=time.time(),failures=len(failed),GPU_seconds=0,rule='same exact official train clip key, caption, every bbox, st_frame, img_num, dimensions; actual compatible metadata and successful probe required; never replace query',
       caveat='HC2 media may be re-encoded. Byte equivalence to damaged HC1 is not claimed. Both originals retained.',
       pins={str(p):sha(p) for p in [Path(__file__),QA/'COMPLETE.json',HC/'train.json',h2root/'anno_v2/train_v2.json']}))
    index={}
    for p in sorted((h2root/'VIdeo').glob('*.zip')):
        with zipfile.ZipFile(p) as z:
            for i in z.infolist():
                if not i.is_dir():index[Path(i.filename).name]=dict(archive=str(p),member=i.filename,bytes=i.file_size,crc32=i.CRC)
    DATA.mkdir(parents=True,exist_ok=True);repairs={};unresolved=[]
    for f in failed:
        name=Path(f['source']['path']).name;a=h1[name];b=h2.get(name)
        checks=dict(official_train_name=b is not None,archive_available=name in index)
        if b is not None:checks.update(caption=a['caption']==b['English'],bbox=a['bbox']==b['bbox'],
             st_frame=a['st_frame']==b['st_frame'],img_num=a['img_num']==b['img_num'],size=[a['height'],a['width']]==b['img_size'][:2])
        if not all(checks.values()):unresolved.append(dict(key=f['key'],checks=checks));continue
        s=index[name];assert shutil.disk_usage(ROOT).free-s['bytes']>8*2**30;p=DATA/name;h=hashlib.sha256()
        with zipfile.ZipFile(s['archive']) as z,z.open(s['member']) as src,p.open('xb') as dst:
            for chunk in iter(lambda:src.read(8<<20),b''):h.update(chunk);dst.write(chunk)
        try:
            m=probe(p);assert (m['width'],m['height'])==(a['width'],a['height'])
            assert 0<=a['img_num']-m['frame_count']<=2 and a['st_frame']-1+len(a['bbox'])<=m['frame_count']
            r=dict(key=f['key'],status='probe_passed_full_decode_not_yet_checked',metadata=m,sha256=h.hexdigest(),source=dict(path=str(p),**s),
                original_failed_sha=f['sha256'],official_annotation_equality=checks,bytes_equal_to_original=h.hexdigest()==f['sha256'],
                zip_CRC_verified=True,scope='official same-clip media repair with exact full-frame annotation agreement; not an output-selected replacement')
            # Prefer persistent repaired file, not the bounded Vid archive cache.
            r['source'].pop('archive');r['official_repair_archive']=s
            repairs[f['key']]=r;put(OUT/'repairs'/(f['key']+'.json'),r)
        except Exception as e:unresolved.append(dict(key=f['key'],error=repr(e),new_path=str(p)))
    put(OUT/'COMPLETE.json',dict(time=time.time(),repairs=repairs,unresolved=unresolved,GPU_seconds=0,
        other_failed_media=[r for r in finished['failures'] if not r['key'].startswith('hc1-')],
        free_bytes=shutil.disk_usage(ROOT).free))
    print('REPAIRED',len(repairs),'UNRESOLVED',len(unresolved),flush=True)
if __name__=='__main__':main()
