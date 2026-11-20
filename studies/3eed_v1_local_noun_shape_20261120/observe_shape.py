"""Isolated noun interventions through a frozen prediction head at identical physical seeds."""
import argparse
from functools import lru_cache
from time import perf_counter
import numpy as np
import torch

from paths_shape import HERE, ROOT, WORK, OUT, SPEC, OBS, SOURCES, CLASSES, tag, previous, name, read, save, digest, archive_npz
from collect import Grounder
from observe_witness import observe
from observe_lex import geometry, prompts, nouns, encode, decode
from common_lex import sample


def prepare_nouns(model):
    result=[]
    for noun in CLASSES[:-1]:
        text='The '+noun+'.'
        ids=model.tokenizer([text],return_tensors='pt').to(next(model.parameters()).device)
        words=model.text_projector(model.text_encoder(**ids).last_hidden_state)
        result.append((text,words,ids.attention_mask.ne(1)))
    return result


def measure(model,frame,prepared,check):
    data=torch.from_numpy(sample(frame))[None].to(next(model.parameters()).device)
    backbone=model.backbone_net(data)
    xyz,features=backbone['fp2_xyz'],backbone['fp2_features']
    ids=torch.arange(1024,device=xyz.device).reshape(256,4).T.contiguous()
    restore=ids.flatten().argsort()
    out=[]; error=[]
    for text,words,pad in prepared:
        vision,lang=encode(model,xyz,features,words,pad)
        _,boxes=decode(model,xyz,vision,lang,pad,ids)
        out.append(boxes.reshape(1024,6)[restore,3:].cpu().numpy())
        if check and not error:
            original=model(dict(point_clouds=data,text=[text]))
            origins=original['query_points_sample_inds'].long()
            _,direct=decode(model,xyz,vision,lang,pad,origins)
            error=[float((direct[...,3:]-original['last_pred_size']).abs().max())]
            assert error[0]<1.e-4
    extent=np.stack(out,axis=1)
    if extent.shape!=(1024,len(CLASSES)-1,3) or not np.isfinite(extent).all():
        raise ValueError('Invalid native extents')
    return dict(xyz=xyz[0].cpu().numpy(),extent=extent,native_error=np.asarray(error))


def pins():
    cfg=read(SPEC)
    assert digest(ROOT/'crossplatform/CURRENT_METHOD.json')==cfg['production_registry']
    assert digest(ROOT/'crossplatform/single_frame_sources.json')==cfg['source_registry']
    assert digest(WORK/'roster.json')==read(OUT/'panel.json')['roster']
    return dict(protocol=digest(SPEC),capture=digest(HERE/'observe_shape.py'),
                new_roster=digest(WORK/'roster.json'),old_roster=digest(previous.NEW/'roster.json'),
                previous_observations=digest(previous.NEW/'SEALED.json'))


@torch.inference_mode()
def run(source):
    pin=pins()
    engine=Grounder(source)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    engine._frontend.parser=lru_cache(maxsize=1024)(engine._frontend.parser)
    model=engine._frontend.base
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    prepared=prepare_nouns(model)
    multi=prompts(model)
    previous_rows=[item for item in previous.inputs() if item[0]['source']==source]
    new_rows=[r for r in read(WORK/'roster.json') if r['source']==source]
    for folder in (OBS,WORK/'lexical',WORK/'role'):
        folder.mkdir(exist_ok=True,mode=0o700)
    records={}; captures={}; history={}; new_geometry={}; checked=[]; start=perf_counter()
    for item in previous_rows+[(r,) for r in new_rows]:
        row=item[0]; f=row['frame']; n=name(row)
        if len(item)>1:
            _,path,sha,lex,lsha,fid,words,*_=item
            assert digest(path)==sha
            with np.load(path) as z:
                active=bool(z['active']); seed=z['seeds']
        else:
            fid=tag((source,f['target'],f['points'],f['seed'],f['waymo_view']))
            lex=WORK/'lexical'/(fid+'.npz')
            if fid not in new_geometry:
                archive_npz(lex,**geometry(f,model,multi,not new_geometry))
                new_geometry[fid]=digest(lex)
            lsha=new_geometry[fid]
            z=observe(engine,f,history)
            active=bool(z['active']);seed=z['seeds']
            assert active
            words=list(nouns(f,engine._frontend))
            path=WORK/'role'/n
            archive_npz(path,**z);sha=digest(path)
        if active and min(words)>=0 and fid not in captures:
            result=measure(model,f,prepared,not checked)
            np.testing.assert_array_equal(result['xyz'],seed)
            if len(result['native_error']):checked.append(float(result['native_error'][0]))
            dst=OBS/(fid+'.npz')
            archive_npz(dst,**result);captures[fid]=digest(dst)
        records[n]=dict(fid=fid,words=words,role=sha,lex=lsha,requires_noun=bool(active and min(words)>=0))
        if len(records)%100==0:
            print(source,len(records),len(captures),round(perf_counter()-start),flush=True)
    assert checked
    save(OBS/(source+'.json'),dict(pin=pin,records=records,hashes=captures,checks=checked,
         seconds=perf_counter()-start))


def seal():
    pin=pins()
    files={}; tensors={}; checks={}; seconds={}
    for src in SOURCES:
        d=read(OBS/(src+'.json'))
        assert pin==d['pin'] and not(files.keys()&d['records'].keys()) and not(tensors.keys()&d['hashes'].keys())
        files.update(d['records']);tensors.update(d['hashes']);checks[src]=d['checks'];seconds[src]=d['seconds']
    assert set(files)=={name(r) for r in previous.observed_roster()+read(WORK/'roster.json')}
    assert {v['fid'] for v in files.values() if v['requires_noun']}==set(tensors)
    assert all(digest(OBS/(fid+'.npz'))==h for fid,h in tensors.items())
    save(OBS/'SEALED.json',dict(**pin,files=files,hashes=tensors,native=checks,seconds=seconds,
        task_labels_used=False,source_updates=0,target_updates=0,FINAL_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=(*SOURCES,'seal'))
    stage=p.parse_args().stage
    seal() if stage=='seal' else run(stage)
