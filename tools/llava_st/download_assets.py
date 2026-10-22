"""Download the explicitly requested official checkpoint at a pinned revision."""
import hashlib,json,os,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stvg.tube_runner import write
import requests
from tools.llava_st.ranged_download import transfer
REPO='appletea2333/LLaVA-ST-Qwen2-7B';REV='2f7261b55b0368e9c2971120b62c77545b3a792d'
def main():
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
    target=ROOT/'checkpoints/llava-st/LLaVA-ST-Qwen2-7B';out=ROOT/'runtime/llava_st';tick=time.time()
    response=requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REV}',params={'blobs':'true'},timeout=60);response.raise_for_status();meta=response.json();assert meta['sha']==REV
    total=sum(x.get('size',0)for x in meta['siblings']);assert total<25*2**30
    if shutil.disk_usage(ROOT).free<total+10*2**30:raise RuntimeError('Insufficient free disk for checkpoint and reserve')
    write(out/'download_plan.json',dict(repo=REPO,revision=REV,expected_bytes=total,files=[dict(path=x['rfilename'],size=x['size'],lfs=x.get('lfs'),git_blob_id=x.get('blobId'))for x in meta['siblings']],target=str(target)))
    write(out/'download_status.json',dict(status='downloading',repo=REPO,revision=REV,pid=os.getpid(),expected_bytes=total))
    transfer(REPO,REV,meta["siblings"],target,out/"download_status.json")
    files={}
    for x in meta['siblings']:
        p=target/x['rfilename'];assert p.stat().st_size==x['size'];h=hashlib.sha256();g=hashlib.sha1();g.update(f'blob {p.stat().st_size}\0'.encode())
        with p.open('rb')as f:
            for b in iter(lambda:f.read(8*2**20),b''):h.update(b);g.update(b)
        if x.get('lfs'):assert h.hexdigest()==x['lfs']['sha256'],x['rfilename']
        elif x.get('blobId'):assert g.hexdigest()==x['blobId'],x['rfilename']
        files[x['rfilename']]=dict(bytes=p.stat().st_size,sha256=h.hexdigest(),publisher_hash_verified=True)
    write(out/'checkpoint_seal.json',dict(status='complete_verified',repo=REPO,revision=REV,bytes=total,files=files,seconds=time.time()-tick))
    write(out/'download_status.json',dict(status='complete_verified',repo=REPO,revision=REV,bytes=total,files=len(files)))
    print('CHECKPOINT VERIFIED',total,len(files),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/'runtime/llava_st/download_failure.json',dict(error=type(e).__name__,message=str(e)));raise
