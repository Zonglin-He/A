"""Acquire paired counterfactual descriptions with the registered frozen single-frame model."""
import argparse
from functools import lru_cache
from time import perf_counter

import numpy as np
import torch

from shared_paths import HERE, ROOT, NEW, OUT, SPEC, CFG, SOURCES, name, read, save, digest, archive_npz, anonymous
from collect import Grounder
from observe_witness import observe
from observe_lex import prompts, geometry, nouns


def provenance():
    assert digest(ROOT/'crossplatform/CURRENT_METHOD.json') == CFG['parent_model']
    assert digest(ROOT/'crossplatform/single_frame_sources.json') == CFG['source_registry']
    assert digest(NEW/'roster.json') == read(OUT/'panel.json')['observations']
    return dict(protocol=digest(SPEC), roster=digest(NEW/'roster.json'), code=digest(HERE/'observe_scene.py'))


@torch.inference_mode()
def acquire(source):
    pin=provenance()
    rows=[r for r in read(NEW/'roster.json') if r['source']==source]
    for d in (NEW/'role', NEW/'lexical'):
        d.mkdir(mode=0o700,exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    engine=Grounder(source)
    frontend=engine._frontend
    frontend.parser=lru_cache(maxsize=512)(frontend.parser)
    model=frontend.base
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    prepared=prompts(model)
    files={}; hashes={}; native=[]; history={}
    start=perf_counter()
    for i,row in enumerate(rows,1):
        f=row['frame']
        fid=anonymous((source,f['target'],f['points'],f['seed'],f['waymo_view']))
        dst=NEW/'lexical'/(fid+'.npz')
        if fid not in hashes:
            item=geometry(f,model,prepared,not native)
            assert np.isfinite(item['logits']).all() and np.isfinite(item['native_boxes']).all()
            if len(item['exact_native_check']):native.append(item['exact_native_check'].tolist())
            archive_npz(dst,**item)
            hashes[fid]=digest(dst)
        packet=observe(engine,f,history)
        with np.load(dst) as z:
            np.testing.assert_array_equal(z['xyz'],packet['seeds'])
        words=list(nouns(f,frontend))
        assert packet['active'] and min(words)>=0
        out=NEW/'role'/name(row)
        archive_npz(out,**packet)
        files[out.name]=dict(fid=fid,sha=digest(out),words=words)
        if i%20==0 or i==len(rows):print(source,i,len(rows),round(perf_counter()-start,1),flush=True)
    assert not rows or native
    save(NEW/(source+'.json'),dict(pin=pin,files=files,hashes=hashes,native=native))


def seal():
    pin=provenance()
    files={}; hashes={}; native={}
    for src in SOURCES:
        data=read(NEW/(src+'.json'))
        assert data['pin']==pin and not(files.keys() & data['files'].keys())
        files.update(data['files']);hashes.update(data['hashes']);native[src]=data['native']
    assert set(files)=={name(r) for r in read(NEW/'roster.json')}
    assert all(digest(NEW/'role'/n)==v['sha'] for n,v in files.items())
    assert all(digest(NEW/'lexical'/(fid+'.npz'))==h for fid,h in hashes.items())
    save(NEW/'SEALED.json',dict(**pin,files=files,hashes=hashes,native=native,labels_used=False,FINAL_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=(*SOURCES,'seal'))
    stage=parser.parse_args().stage
    seal() if stage=='seal' else acquire(stage)
