"""Recover the original absolute centers for each isolated word, with a direct native-query check."""
import argparse
from time import perf_counter
import numpy as np
import torch
from vote_paths import *
from observe_shape import prepare_nouns
from observe_lex import encode,decode
from common_lex import sample
from collect import Grounder

@torch.inference_mode()
def capture(source):
    doc=catalog()
    out=WORK/'votes'/source
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    engine=Grounder(source)
    model=engine._frontend.base
    assert not model.training and not any(x.requires_grad for x in model.parameters())
    device=next(model.parameters()).device
    prepared=prepare_nouns(model)
    forced=torch.arange(1024,device=device).reshape(256,4).T.contiguous()
    restore=forced.flatten().argsort()
    allframes={k:v for k,v in doc['frames'].items() if v['source']==source}
    rec={};reports=[];max_extent=0.;started=perf_counter()
    deps=dict(spec=digest(SPEC),catalog=digest(WORK/'CATALOG.json'),code=digest(Path(__file__)))
    for n,(uid,v) in enumerate(allframes.items(),1):
        dst=out/(uid+'.npz');cert=out/(uid+'.json')
        if cert.exists():
            meta=read(cert);assert meta['pins']==deps and digest(dst)==meta['hash']
        else:
            assert digest(v['word'])==v['wh'] and digest(v['lex'])==v['lh']
            with np.load(v['word']) as old_word,np.load(v['lex']) as lex:
                points=sample(v['f'])
                np.testing.assert_array_equal(points,lex['sample'])
                data=torch.from_numpy(points)[None].to(device)
                native=model.backbone_net(data)
                xyz,features=native['fp2_xyz'],native['fp2_features']
                np.testing.assert_array_equal(xyz[0].cpu().numpy(),old_word['xyz'])
                boxes=[];check=[]
                for text,words,pad in prepared:
                    vis,lang=encode(model,xyz,features,words,pad)
                    _,b=decode(model,xyz,vis,lang,pad,forced)
                    boxes.append(b.reshape(1024,6)[restore].cpu().numpy())
                    if n==1:
                        direct=model(dict(point_clouds=data,text=[text]))
                        _,b=decode(model,xyz,vis,lang,pad,direct['query_points_sample_inds'].long())
                        expected=torch.cat([direct['last_center'],direct['last_pred_size']],dim=-1)
                        error=float((b-expected).abs().max())
                        assert error<1.e-4
                        check.append(error)
                both=np.stack(boxes,axis=1);extent=float(np.max(abs(both[...,3:]-old_word['extent'])))
                assert extent<1.e-4 and np.isfinite(both).all(),extent
                archive_npz(dst,center=both[...,:3])
            meta=dict(pins=deps,hash=digest(dst),extent=extent,direct=check)
            save(cert,meta)
        rec[uid]=meta['hash'];reports.extend(meta['direct']);max_extent=max(max_extent,meta['extent'])
        if n%100==0:print(source,n,len(allframes),round(perf_counter()-started),flush=True)
    assert len(reports)==CATS
    save(out/'SEALED.json',dict(pins=deps,files=rec,original_query_max_error=max(reports),extent_max_error=max_extent,
        n=len(rec),seconds=perf_counter()-started,labels_used=False))
    print('sealed',source,len(rec),max(reports),max_extent,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',choices=SOURCES);capture(p.parse_args().source)

