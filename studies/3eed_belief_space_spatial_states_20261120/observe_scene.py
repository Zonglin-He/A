"""Capture one frozen region forest and one noun nomination per source/frame, without answers."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from io_scene import ROOT, PRIVATE, OUT, SPEC, PLATFORMS, read, write, digest

sys.path.insert(0, str(ROOT if (ROOT/'crossplatform').exists() else ROOT/'v1'))
from crossplatform.candidate_frontend import CandidateFrontend
from crossplatform.ground_regions import phrase_bank
from crossplatform.physical_regions import selection, regions, semantic
from crossplatform.physical_regions.language import CLASSES
from crossplatform.round7.camera_projection import model_to_camera_matrix

@torch.inference_mode()
def observe(source):
    panel=read(OUT/'panel_seal.json')
    assert digest(PRIVATE/'questions.json')==panel['questions'] and digest(SPEC)==panel['protocol']
    rows=[v for v in read(PRIVATE/'questions.json') if v['frame']['target']!=source]
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    front=CandidateFrontend(source)
    base=front.base
    assert not base.training and not any(p.requires_grad for p in base.parameters())
    prompts=semantic.prompts(base),semantic.prepare_nouns(base)
    folder=PRIVATE/'observations'/source
    folder.mkdir(exist_ok=True,parents=True)
    pins=dict(protocol=digest(SPEC),questions=panel['questions'],observer=digest(__file__),
              physical_regions=digest(regions.__file__),semantics=digest(semantic.__file__),
              nominations=digest(selection.__file__),source_assets=digest(ROOT/'crossplatform/single_frame_sources.json'))
    files={}
    for n,v in enumerate(rows,1):
        f=v['frame'];uid=v['uid']; path=folder/(uid+'.npz'); stamp=path.with_suffix('.json')
        if stamp.exists():
            old=read(stamp)
            assert old['pins']==pins and old['file']==digest(path)
        else:
            s,r=[CLASSES.index(v[role]) for role in ('subject','reference')]
            scene=front.score_frame(target=f['target'],points=f['points'],text='The '+v['subject']+'.',
                                    seed=f['seed'],waymo_view=f['waymo_view'])
            scores,words=semantic.observe(base,scene.sampled_points,*prompts)
            seg=regions.build(scene.sampled_points,scores,words['xyz'],f,native_logits=True)
            d=regions.build_inventory(seg,words)
            bank=phrase_bank(scene,scene,f,normalize=False)
            us=selection.overlap_prior(d,{'t_'+k:val for k,val in bank.items()},'t',regions.Returns(d['xyz']))
            if not len(us):us=np.ones(0)
            spec,_=regions.camera(f['target'],scene.view_index,f['calibration'],f['rgb'])
            def complete(c):
                return np.array([regions.complete(b,dim[c]) for b,dim in zip(d['boxes'],d['dimensions'])]).reshape(-1,7)
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,parent=d['parent'],stop=d['stop'],visible=np.isfinite(d['view'][:,0]),
                 size=d['size'],visible_count=np.nan_to_num(d['view'][:,4]),qs=d['p'][:,s],qr=d['p'][:,r],
                 nomination=us,subject_boxes=complete(s),reference_boxes=complete(r),
                 camera=model_to_camera_matrix(spec))
            tmp.replace(path)
            write(stamp,dict(file=digest(path),pins=pins))
        files[uid]=digest(path)
        if n%24==0:print(source,n,len(rows),flush=True)
    write(folder/'seal.json',dict(files=files,pins=pins,labels_accessed=False))
    print('sealed source',source,len(files),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',choices=PLATFORMS)
    observe(p.parse_args().source)
