"""Inference/TTA entry: input manifest has no GT fields; predictions sealed for later CPU score."""
import argparse,json,os,pathlib,sys,time,hashlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'adapted'))
from tta import adapt,configure,predict

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,nargs='+',default=[0,1,3]);p.add_argument('--objectives',nargs='+',default=['consistency','entropy']);a=p.parse_args()
 out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
 os.environ['GREAT_TEXT_MODEL']=str(ROOT/'assets/robbert-v2-dutch-base')
 from model.GREAT import GREAT
 ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 sd=ckpt.get('model',ckpt);sd={k.removeprefix('module.'):v for k,v in sd.items()}
 emb=sd['text_encoder.text_encoder.embeddings.word_embeddings.weight'];assert emb.shape[0]==40000,'Tokenizer identity must be re-audited'
 model=GREAT(pre_train=False);model.load_state_dict(sd,strict=True);del ckpt,sd
 model.to(a.device).eval();names=configure(model)
 meta={'checkpoint':str(pathlib.Path(a.checkpoint).resolve()),'checkpoint_sha256':hashlib.file_digest(open(a.checkpoint,'rb'),'sha256').hexdigest(),'tokenizer':'pdelobelle/robbert-v2-dutch-base; historical code and checkpoint vocab match; official paper identity unresolved','steps':a.steps,'objectives':a.objectives,'lr':1e-4,'seed':42,'cohort':a.manifest,'device':a.device,'updated_names':names,'state':'running'}
 (out/'protocol.json').write_text(json.dumps(meta,indent=2))
 manifest=json.loads(pathlib.Path(a.manifest).read_text());normal=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 for i,row in enumerate(manifest):
  img=normal(Image.open(row['image']).convert('RGB').resize((224,224))).unsqueeze(0).to(a.device)
  # Geometry-only file is exported by preparation; target column never enters this process.
  xyz=np.load(row['geometry'],allow_pickle=False).astype(np.float32);assert xyz.shape==(2048,3)
  xyz=xyz-xyz.mean(0);xyz=xyz/max(float(np.linalg.norm(xyz,axis=1).max()),1e-12)
  point=torch.from_numpy(xyz.T).unsqueeze(0).to(a.device);t0=time.perf_counter()
  for objective in a.objectives:
   for steps in a.steps:
    if objective=='entropy' and steps==0:continue
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();r=adapt(model,img,point,[row['human']],[row['object']],steps=steps,seed=42+i,objective=objective)
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    r['diagnostics'].update(seconds=time.perf_counter()-start,peak_memory_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None)
    np.savez_compressed(out/f'{i:04d}_{objective}_k{steps}.npz',source=r['source'],ensemble=r['ensemble'],adapted=r['adapted'])
    (out/f'{i:04d}_{objective}_k{steps}.json').write_text(json.dumps({'id':row['id'],**r['diagnostics']},indent=2))
  print(i,row['id'],time.perf_counter()-t0,flush=True)
 files=sorted(out.glob('*.npz'));seal={f.name:hashlib.file_digest(open(f,'rb'),'sha256').hexdigest() for f in files}
 (out/'SEALED.json').write_text(json.dumps(seal,indent=2));meta['state']='sealed';meta['count']=len(manifest);(out/'protocol.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
