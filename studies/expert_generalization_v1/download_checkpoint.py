"""Official ReferFormer R50 Ref-YTVOS checkpoint only; no model construction."""
import urllib.request,time,json,hashlib,os
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'outputs/expert_generalization_v1';out=D/'vendor/r50_refytvos_finetune.pth';part=out.with_suffix('.pth.part');receipt=D/'checkpoint_download.json'
assert not out.exists()and not part.exists()and not receipt.exists()
url='https://drive.usercontent.google.com/download?id=1TBykzTRiOEbpM2YEYnDICyqjwkvlD2cR&export=download&confirm=t';start=time.perf_counter();h=hashlib.sha256();count=0
try:
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60)as response,part.open('xb')as f:
  expected=int(response.headers['Content-Length']);assert expected==1116919589 and response.headers.get_content_type()=='application/octet-stream'
  while True:
   chunk=response.read(2**20)
   if not chunk:break
   f.write(chunk);h.update(chunk);count+=len(chunk)
   if count%(16*2**20)<2**20:(D/'download_progress.json').write_text(json.dumps(dict(bytes=count,total=expected,seconds=time.perf_counter()-start)))
 assert count==expected;os.replace(part,out);v=dict(status='download_complete_verified_length_SHA',official_repository='https://github.com/wjn922/ReferFormer',official_README_model='Ref-Youtube-VOS ResNet50 model, not pretrained_weights or A2D finetune',url=url,bytes=count,sha256=h.hexdigest(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_GPU=0,new_expert=0,weight_file_private=True);receipt.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
except BaseException as e:
 (D/'download_failure_preserved.json').write_text(json.dumps(dict(error=repr(e),bytes=count,elapsed_wall_seconds=time.perf_counter()-start,partial_preserved=str(part))))
 raise
