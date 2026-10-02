"""Download only official frozen CLIP files and verify upstream LFS SHA256."""
import os,sys,time,requests
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_transfer_aligned_common_v1 import *
REPO='openai/clip-vit-base-patch16';REV='57c216476eefef5ab752ec549e440a49ae4ae5f3'
ASSET=ROOT/'checkpoints/clip_vit_b16_token_v1'
FILES=['config.json','merges.txt','preprocessor_config.json','pytorch_model.bin','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json']
def run():
 ASSET.mkdir(parents=True,exist_ok=True);budget();r=requests.get('https://huggingface.co/api/models/'+REPO+'/revision/'+REV,params={'blobs':'true'},timeout=30);r.raise_for_status();meta=r.json();assert meta['sha']==REV
 entries={x['rfilename']:x for x in meta['siblings']};rows=[];tick=time.monotonic()
 for name in FILES:
  f=ASSET/name;lfs=entries[name].get('lfs',{});expected=lfs.get('sha256')
  if not f.exists():
   temp=f.with_suffix(f.suffix+'.part');url='https://huggingface.co/'+REPO+'/resolve/'+REV+'/'+name
   with requests.get(url,stream=True,timeout=(30,60)) as resp:
    resp.raise_for_status()
    with temp.open('wb') as out:
     for block in resp.iter_content(4<<20):
      if block:out.write(block)
   if expected:assert sha(temp)==expected,name
   temp.replace(f)
  if expected:assert sha(f)==expected,name
  rows.append(dict(file=name,bytes=f.stat().st_size,sha256=sha(f),upstream_LFS_sha256=expected));print('CLIP ASSET',name,f.stat().st_size,flush=True)
 if not (ASSET/'DOWNLOAD_RECEIPT.json').exists():write(ASSET/'DOWNLOAD_RECEIPT.json',dict(repository=REPO,revision=REV,files=rows,seconds=time.monotonic()-tick,time=time.time(),source='official OpenAI converted Hugging Face checkpoint',training=False))
 status(BASE/'token/ASSET_STATUS.json',dict(status='completed',files=len(rows),bytes=sum(x['bytes'] for x in rows),time=time.time()))
if __name__=='__main__':
 try:run()
 except BaseException as e:status(BASE/'token/ASSET_STATUS.json',dict(status='failed',error=repr(e),time=time.time()));raise
