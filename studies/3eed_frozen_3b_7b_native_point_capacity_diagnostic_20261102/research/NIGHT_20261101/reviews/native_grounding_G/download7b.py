"""Bounded official metadata-pinned download only. No model imports or execution."""
import datetime,hashlib,json,os,re,shutil,signal,sys,time,urllib.error,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
REVIEW=Path(__file__).resolve().parent
DEST=ROOT/'crossplatform/assets/Qwen2.5-VL-7B-Instruct'
REV='cc594898137f460bfe9f0759e9844b3ce807cfb5'
CAP=8*1024*1024;CHUNK=1024*1024
START=time.monotonic();LIMIT=90*60
DEST.mkdir(parents=True,exist_ok=True)
EVENTS=DEST/'DOWNLOAD_EVENTS.jsonl'
MAN=DEST/'DOWNLOAD_MANIFEST.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def event(kind,**kw):
 row={'utc':now(),'kind':kind,**kw}
 with EVENTS.open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def save(d):
 p=MAN.with_suffix('.json.tmp');p.write_text(json.dumps(d,indent=2)+'\n');os.replace(p,MAN)
def stopped(*_):raise TimeoutError('owned_deadline_or_signal')
signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
def check():
 if time.monotonic()-START>=LIMIT:raise TimeoutError('90_minute_limit')
 state=json.loads((ROOT/'research/NIGHT_20261101/STATE.json').read_text())
 deadline=datetime.datetime.fromisoformat(state['deadline_utc'].replace('Z','+00:00'))
 if state['status']!='active' or datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('night_deadline_or_inactive')
def safe_error(e):return {'type':type(e).__name__,'http_code':getattr(e,'code',None),'message':re.sub(r'https?://\S+','[redacted_url]',str(e))[:240]}
def url(host,name):
 if host=='HF':return f'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/{REV}/{name}'
 return 'https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-VL-7B-Instruct/repo?Revision=master&FilePath='+name
def opening(host,name,offset=0,probe=False):
 headers={'User-Agent':'ResearchModelDownload/1.0','Accept-Encoding':'identity'}
 if offset or probe:headers['Range']=f'bytes={offset}-{offset+CAP-1}' if probe else f'bytes={offset}-'
 return urllib.request.urlopen(urllib.request.Request(url(host,name),headers=headers),timeout=20)
def probe(host,name):
 t=time.monotonic();n=0;first=None;result={'host':host,'limit_bytes':CAP,'body_bytes_read':0}
 try:
  check()
  with opening(host,name,probe=True) as r:
   headers=time.monotonic();result.update(status=r.status,handshake_header_seconds=headers-t,range_response=r.headers.get('Content-Range'),content_length=r.headers.get('Content-Length'))
   while n<CAP:
    check();b=r.read(min(CHUNK,CAP-n))
    if not b:break
    if first is None:first=time.monotonic()
    n+=len(b)
  elapsed=time.monotonic()-t;result.update(body_bytes_read=n,total_seconds=elapsed,end_to_end_bytes_per_second=n/elapsed,first_chunk_seconds=first-t if first else None,post_first_chunk_bytes_per_second=max(0,n-CHUNK)/max(time.monotonic()-(first or headers),1e-6))
 except Exception as e:result.update(body_bytes_read=n,total_seconds=time.monotonic()-t,error=safe_error(e))
 assert n<=CAP
 event('probe',**result);return result
def hashes(path,size):
 h=hashlib.sha256();g=hashlib.sha1();g.update(b'blob '+str(size).encode()+b'\0');n=0
 with path.open('rb') as f:
  while True:
   check();b=f.read(CHUNK)
   if not b:break
   h.update(b);g.update(b);n+=len(b)
 return n,h.hexdigest(),g.hexdigest()
def valid_hash(row,sha256,gitsha):return sha256==row['lfs']['sha256'] if 'lfs' in row else gitsha==row['blobId']
def get_file(row,hosts):
 name=row['rfilename'];size=row['size'];final=DEST/name;part=DEST/(name+'.part')
 if final.exists():
  n,h,g=hashes(final,size)
  if n==size and valid_hash(row,h,g):return {'filename':name,'size':n,'sha256':h,'git_blob_sha1':g,'reused_verified':True}
  os.replace(final,DEST/(name+'.bad_existing.'+str(int(time.time()))));event('preserve_bad_existing',filename=name)
 for attempt in range(1,5):
  host=hosts[(attempt-1)%len(hosts)];check();offset=part.stat().st_size if part.exists() else 0
  if offset>size:os.replace(part,DEST/(name+'.bad_oversize.'+str(int(time.time()))));offset=0
  if shutil.disk_usage(DEST).free<size-offset+2*1024**3:raise OSError('Insufficient_space_with_2GiB_margin')
  t=time.monotonic();networkbytes=0;last=t
  try:
   if offset<size:
    with opening(host,name,offset) as r:
     cr=r.headers.get('Content-Range')
     if offset and r.status==206:
      if not cr or not cr.startswith(f'bytes {offset}-'):raise ValueError('Incorrect_resume_Content_Range')
      mode='ab'
     elif r.status==200:offset=0;mode='wb'
     elif not offset and r.status==206:
      if not cr or not cr.startswith('bytes 0-'):raise ValueError('Incorrect_initial_Content_Range')
      mode='wb'
     else:raise ValueError('Unexpected_HTTP_status')
     event('file_started',filename=name,host=host,attempt=attempt,resume_offset=offset,status=r.status)
     with part.open(mode) as f:
      while True:
       check();b=r.read(min(CHUNK,size-offset-networkbytes+1))
       if not b:break
       if offset+networkbytes+len(b)>size:raise ValueError('Body_exceeds_pinned_size')
       f.write(b);networkbytes+=len(b)
       if time.monotonic()-last>=20:
        f.flush();last=time.monotonic();event('file_progress',filename=name,host=host,bytes_on_disk=offset+networkbytes,expected_size=size,network_bytes_per_second=networkbytes/(last-t))
      f.flush();os.fsync(f.fileno())
   n,h,g=hashes(part,size)
   if n!=size:raise ValueError('Incomplete_file_retained_for_resume')
   if not valid_hash(row,h,g):
    os.replace(part,DEST/(name+'.bad_hash.'+str(int(time.time()))));raise ValueError('Pinned_hash_mismatch_preserved')
   os.replace(part,final);out={'filename':name,'size':n,'sha256':h,'git_blob_sha1':g,'host':host,'attempt':attempt,'resume_offset':offset,'network_bytes':networkbytes,'seconds_including_verify':time.monotonic()-t,'reused_verified':False};event('file_verified',**out);return out
  except (TimeoutError,KeyboardInterrupt):raise
  except Exception as e:
   event('file_failed',filename=name,host=host,attempt=attempt,partial_bytes=part.stat().st_size if part.exists() else 0,error=safe_error(e))
   if attempt==4:raise
   time.sleep(min(2**attempt,8))
 raise AssertionError('unreachable')
def main():
 raw=REVIEW/'HF_METADATA_7B.json';hf=json.loads(raw.read_text())['metadata'];assert hf['sha']==REV and hf['private'] is False and hf['gated'] is False and hf['cardData']['license']=='apache-2.0'
 rows=hf['siblings'];assert not any(x['rfilename'].endswith('.py') for x in rows)
 assert len([r for r in rows if r['rfilename'].endswith('.safetensors')])==5
 ms=json.loads((REVIEW/'MODELSCOPE_METADATA_7B.json').read_text())['metadata']['Data']['Files'];mm={x['Path']:x for x in ms}
 for r in rows:
  assert '/' not in r['rfilename'] and r['rfilename'] not in ['.','..']
  if 'lfs' in r:assert mm[r['rfilename']]['Sha256']==r['lfs']['sha256'] and mm[r['rfilename']]['Size']==r['size']
 manifest={'state':'PROBING','started_utc':now(),'revision':REV,'official_repository':'Qwen/Qwen2.5-VL-7B-Instruct','license':'apache-2.0','public_ungated_verified':True,'metadata_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'pinned_files':rows,'max_probe_body_bytes_each_site':CAP,'max_stream_buffer_bytes':CHUNK,'max_seconds':LIMIT,'no_model_import_install_GPU_or_inference':True,'files':[]};save(manifest)
 try:
  probes=[probe(host,'model-00005-of-00005.safetensors') for host in ['HF','ModelScope']];manifest['probes']=probes
  ok=sorted([p for p in probes if p['body_bytes_read']==CAP and 'error' not in p],key=lambda p:p['end_to_end_bytes_per_second'],reverse=True)
  total=sum(r['size'] for r in rows);free=shutil.disk_usage(DEST).free
  if not ok:manifest.update(state='STOPPED_NO_COMPLETE_BOUNDED_PROBE',finished_utc=now());save(manifest);return 2
  rate=ok[0]['end_to_end_bytes_per_second']*.5;eta=total/rate;manifest.update(conservative_speed_fraction=.5,conservative_bytes_per_second=rate,conservative_ETA_seconds=eta,disk_free_before_bytes=free,total_expected_bytes=total)
  event('download_gate',conservative_ETA_seconds=eta,within_90min=eta<=LIMIT,disk_available=free,total_expected=total)
  if eta>LIMIT or free<2*total+2*1024**3:manifest.update(state='STOPPED_PROBE_COST_OR_SPACE_GATE',finished_utc=now());save(manifest);return 3
  manifest['state']='DOWNLOADING';save(manifest);hosts=[p['host'] for p in ok]
  # Both endpoints are official; failed probe endpoint is still a bounded fallback.
  hosts += [h for h in ['HF','ModelScope'] if h not in hosts]
  for r in sorted(rows,key=lambda r:r['rfilename'].endswith('.safetensors')):
   manifest['files'].append(get_file(r,hosts));save(manifest)
  assert len(manifest['files'])==len(rows)
  manifest.update(state='COMPLETE_VERIFIED_NO_INFERENCE',finished_utc=now(),elapsed_seconds=time.monotonic()-START,verified_total_bytes=sum(x['size'] for x in manifest['files']));save(manifest);event('complete',elapsed_seconds=manifest['elapsed_seconds'],verified_total_bytes=manifest['verified_total_bytes']);return 0
 except BaseException as e:
  manifest.update(state='STOPPED_PARTIAL_PRESERVED',finished_utc=now(),elapsed_seconds=time.monotonic()-START,error=safe_error(e));save(manifest);event('stopped',error=safe_error(e));return 4
if __name__=='__main__':sys.exit(main())
