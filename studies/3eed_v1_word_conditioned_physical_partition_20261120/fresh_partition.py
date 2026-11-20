"""Run the same frozen operator on the content-disjoint nine-frame panel."""
from functools import lru_cache
import numpy as np
import torch
from partition_paths import HERE, WORK, archive, digest, read, save, signed, archive_npz
from predict_partition import controls, decision, outputs
from physical_partition import partition
from physical_evidence import Returns
from fresh_inference import FRESH as OLD
from observe_lex import prompts, geometry, nouns
from observe_shape import prepare_nouns, measure
from observe_witness import observe
from full_observe import key
from collect import Grounder


@torch.inference_mode()
def run():
    from method_local import ARMS as past_arms
    roster=read(OLD/'input.json')
    assert digest(OLD/'input.json')==read(archive.OUT/'fresh_panel.json')['roster']
    hashes={};files={};torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    dest=WORK/'fresh';dest.mkdir(exist_ok=True,mode=0o700)
    for src in sorted({r['source'] for r in roster}):
        frozen=read(OLD/src/'SEALED.json');assert not frozen['labels_read']
        assert digest(OLD/src/'SEALED.json')==read(OLD/'SEALED.json')['sources'][src]
        engine=Grounder(src);front=engine._frontend;model=front.base
        front.parser=lru_cache(maxsize=64)(front.parser)
        multi=prompts(model);lexical=prepare_nouns(model)
        cache={};history={}
        for r in (r for r in roster if r['source']==src):
            f=r['frame'];fid=key(f);uid=r['uid']
            if fid not in cache:
                obs=geometry(f,model,multi,not cache);word=measure(model,f,lexical,not cache)
                data=partition(obs,word,f)
                cache[fid]=data,controls(data),Returns(data['xyz'])
            data,ctl,points=cache[fid];native=observe(engine,f,history)
            words=list(nouns(f,front))[:2];assert native['active'] and min(words)>=0
            prior=OLD/src/(uid+'.npz');assert digest(prior)==frozen['files'][uid]
            with np.load(prior) as old:
                ix=[past_arms.index('v1'),past_arms.index('tree')]
                values=outputs({k:old[k][ix] for k in ('raw','anchored','ref','ref_anchored')})
                np.testing.assert_allclose(native['raw'],values['measured'][0],rtol=0,atol=1e-5)
                decision(data,ctl,native,words,values,points)
                values.update({k:old[k] for k in ('view','seeds','native_a')})
            p=dest/(src+'_'+uid+'.npz');archive_npz(p,**values);files[src+'_'+uid]=digest(p)
        hashes[src]=digest(OLD/src/'SEALED.json')
        del engine,model,front
        torch.cuda.empty_cache()
    assert len(files)==len(roster)==36
    save(dest/'SEALED.json',dict(files=files,code=signed(),runner=digest(HERE/'fresh_partition.py'),
        observed=digest(OLD/'input.json'),source=hashes,labels_read=False,FINAL=False))
    print('fresh sealed',len(files),flush=True)


if __name__=='__main__':run()
