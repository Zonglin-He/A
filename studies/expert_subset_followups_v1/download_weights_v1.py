"""Finite official checkpoints only; preserve partial/error state, never infer."""
from pathlib import Path
import urllib.request,json,hashlib,time,os,sys,traceback,shutil
R=Path(__file__).resolve().parents[2];D=R/'outputs/expert_subset_followups_v1'
URLS={
 'onlinerefer_r50':('onlinerefer_r50_refytvos.pth','https://github.com/wudongming97/OnlineRefer/releases/download/v1.0/ytvos-r50-checkpoint.pth'),
 'sgmg_videoswin_t':('sgmg_videoswin_t_refytvos.pth','https://drive.usercontent.google.com/download?id=1SiHl7oYqBabaN28nsrNOJeiZrJyhRixl&export=download&confirm=t')}
def download(expert):
 assert shutil.disk_usage(R).free>=80*2**30
 name,url=URLS[expert];dest=D/'vendor'/name;part=dest.with_suffix(dest.suffix+'.partial')
 assert not dest.exists()and not part.exists()
 t=time.perf_counter();h=hashlib.sha256();n=0
 receipt=dict(expert=expert,url=url,new_Source=0,new_expert=0,new_GPU=0,new_optimizer=0,new_GT=0,pid=os.getpid())
 try:
  request=urllib.request.Request(url,headers={'User-Agent':'STVG-official-checkpoint-download/1.0'})
  with urllib.request.urlopen(request,timeout=180)as response,part.open('wb')as f:
   length=response.headers.get('Content-Length');ctype=response.headers.get('Content-Type');assert 'text/html'not in(ctype or ''),ctype
   receipt.update(response_content_length=length,response_content_type=ctype,final_response_url=response.url)
   while True:
    b=response.read(4*1024**2)
    if not b:break
    f.write(b);h.update(b);n+=len(b)
   f.flush();os.fsync(f.fileno())
  assert n>10000000 and(length is None or n==int(length))
  os.replace(part,dest);receipt.update(status='official_checkpoint_downloaded_length_SHA',bytes=n,sha256=h.hexdigest(),elapsed_wall_seconds=time.perf_counter()-t)
  (D/(expert+'_download.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
 except BaseException:
  receipt.update(status='download_failed_no_model_or_inference',error=traceback.format_exc(),partial_bytes=n,partial_sha256=h.hexdigest(),elapsed_wall_seconds=time.perf_counter()-t)
  (D/(expert+'_download_failure.json')).write_text(json.dumps(receipt,indent=2)+'\n');raise
if __name__=='__main__':download(sys.argv[1])
