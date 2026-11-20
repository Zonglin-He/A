"""Collect native per-object data for only the predeclared shared-frame nomination union."""
import argparse
from time import perf_counter
import numpy as np
import torch
from local_paths import *
from local_reader import LocalReader,observe_objects,check_crop
from collect import Grounder

def signatures():
    p=WORK/'PLANS.json';plan=read(p)
    assert plan['base']==base() and plan['code']==digest(HERE/'nominate.py') and not plan['labels_used']
    return dict(base=base(),plan=digest(p),capture=digest(Path(__file__)),reader=digest(HERE/'local_reader.py'))

@torch.inference_mode()
def observe(src,batch):
    check_crop();plan=read(WORK/'PLANS.json');pins=signatures()
    frames={k:v for k,v in catalog()['frames'].items() if v['source']==src}
    out=WORK/'observed'/src;out.mkdir(mode=0o700,parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    model=Grounder(src)._frontend.base;reader=LocalReader(model,batch)
    files={};nobjects=0;began=perf_counter()
    for j,f in enumerate(frames,1):
        path=WORK/'plans'/(f+'.npz');assert digest(path)==plan['files'][f]
        result=out/(f+'.npz');meta=result.with_suffix('.json')
        if meta.exists():
            stamp=read(meta);assert stamp['pins']==pins and stamp['hash']==digest(result)
        else:
            with np.load(path) as data:
                ids=data['crop_nodes'];xyz=data['xyz'];labels=data['assign'];box=data['boxes']
                sets=[(xyz[(labels==i).any(0)],box[i,6]) for i in ids]
                assert all(len(p) == data['size'][i] for (p,_),i in zip(sets,ids))
                data_=observe_objects(reader,sets)
            archive_npz(result,**data_,ids=ids)
            stamp=dict(pins=pins,hash=digest(result),count=len(ids));save(meta,stamp)
        files[f]=stamp['hash'];nobjects+=stamp['count']
        if j%50==0:print(src,j,len(frames),nobjects,round(perf_counter()-began),flush=True)
    save(out/'SEALED.json',dict(pins=pins,files=files,objects=nobjects,batches=reader.batches,seconds=perf_counter()-began,
        labels_used=False,source_updated=False,target_updated=False))
    print('sealed',src,len(frames),nobjects,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',choices=SOURCES);p.add_argument('--batch',type=int,default=16)
    a=p.parse_args();observe(a.source,a.batch)
