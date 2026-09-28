"""CPU verification of all intended source media; no model and no scoring."""
import concurrent.futures,hashlib,json,os,shutil,subprocess,sys,time,traceback,zipfile
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_desta3d_v3_source import OUT as INTAKE,read,put,sha
OUT=ROOT/'artifacts/desta3d_v3/source_media_audit_v1'
TEMP=ROOT/'.cache/desta3d_v3_media_probe'
FFPROBE=ROOT/'.conda/tubedetr/bin/ffprobe'
def probe(p):
    r=subprocess.run([str(FFPROBE),'-v','error','-select_streams','v:0','-show_entries',
       'stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(p)],capture_output=True,text=True,timeout=90,check=True)
    d=json.loads(r.stdout)['streams'][0]
    if not str(d.get('nb_frames','')).isdigit():
        r=subprocess.run([str(FFPROBE),'-v','error','-select_streams','v:0','-count_frames','-show_entries',
            'stream=nb_read_frames','-of','json',str(p)],capture_output=True,text=True,timeout=120,check=True)
        d['nb_frames']=json.loads(r.stdout)['streams'][0]['nb_read_frames']
    fps=float(Fraction(d['avg_frame_rate']));n=int(d['nb_frames'])
    assert fps>0 and n>0
    return dict(width=int(d['width']),height=int(d['height']),fps=fps,frame_count=n)
def one(item):
    key,spec=item;p=None;temp=False;result=dict(key=key,source=spec)
    try:
        if 'archive' in spec:
            assert shutil.disk_usage(ROOT).free-spec['bytes']>8*2**30
            p=TEMP/(key+'.mp4');temp=True;h=hashlib.sha256()
            with zipfile.ZipFile(spec['archive']) as z,p.open('xb') as f,z.open(spec['member']) as stream:
                for b in iter(lambda:stream.read(8<<20),b''):h.update(b);f.write(b)
            assert p.stat().st_size==spec['bytes'];result['sha256']=h.hexdigest();result['zip_CRC_verified']=True
        else:
            p=Path(spec['path']);assert p.stat().st_size==spec['bytes'];result['sha256']=sha(p)
            assert result['sha256']==spec['sha256'];result['original_archive_SHA_verified']=True
        result.update(metadata=probe(p),status='probe_passed_full_decode_not_yet_checked')
    except Exception as e:result.update(status='failed',error=repr(e),stderr=getattr(e,'stderr',None))
    finally:
        if temp and p is not None and p.exists():p.unlink() # only this declared disposable extraction
    put(OUT/'media'/(key+'.json'),result)
    return result
def main():
    assert not (OUT/'STARTED.json').exists()
    rows=read(INTAKE/'MANIFEST.json');hc=read(INTAKE/'HC_MEDIA_COMPLETE.json')['members'];items={}
    for r in rows:
        key=('vid-'+r['parent']) if r['domain']=='Vid' else ('hc1-'+Path(r['annotation_key']).stem)
        items[key]=r['media'] if r['domain']=='Vid' else hc[r['annotation_key']]
    put(OUT/'REGISTRATION.json',dict(time=time.time(),media=len(items),GPU_seconds=0,
        reserve_bytes=8*2**30,concurrency=4,temporary_extractions='own Vid ZIP members deleted only after probe',
        failure_rule='keep every media failure; never replace a query or silently truncate frames; full selected pixel decode is checked by worker',
        pins={str(p):sha(p) for p in [Path(__file__),INTAKE/'MANIFEST.json',INTAKE/'HC_MEDIA_COMPLETE.json',FFPROBE]}))
    TEMP.mkdir(parents=True,exist_ok=True);put(OUT/'STARTED.json',dict(pid=os.getpid(),time=time.time()))
    t=time.monotonic();counts={};fail=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(one,sorted(items.items()))):
            counts[r['status']]=counts.get(r['status'],0)+1
            if r['status']=='failed':fail.append(r)
            if (i+1)%100==0:print('MEDIA_PROBED',i+1,len(items),counts,flush=True)
    put(OUT/'COMPLETE.json',dict(counts=counts,failures=fail,media=len(items),CPU_wall_seconds=time.monotonic()-t,GPU_seconds=0,
        free_bytes=shutil.disk_usage(ROOT).free,scope='all intended media hashes/ZIP CRC and header/count probes; not complete pixel decoding',
        pins={str(p):sha(p) for p in sorted((OUT/'media').glob('*.json'))}))
if __name__=='__main__':main()
