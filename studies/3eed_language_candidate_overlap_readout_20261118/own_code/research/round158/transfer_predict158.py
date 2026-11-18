"""Sealed A-only comparisons on every additional eligible content group."""
import json
import sys
from pathlib import Path
import numpy as np
from grounding158 import select_grounding_box
from transfer_inputs158 import WORK,HERE,ROOT,read,digest
sys.path[:0]=[str(ROOT/'research/round156'),str(ROOT/'crossplatform/round3/p0/upstream')]
from pair156 import shared_points,fit,load_module,PCA_FILE, array_hash
from utils.transform_waymo import transform_to_front_view


def main():
    p=WORK/'EXTRA_REGISTRY_PRIVATE.json';en=read(WORK/'EXTRA_INPUT_SEAL.json')
    if digest(p)!=en['sha256']:raise ValueError('registry changed')
    rr=read(p); out=np.empty((len(rr),2,7),np.float32);ids=np.empty((len(rr),2),np.int16)
    files={};pca=load_module('extra_pca',PCA_FILE).refine
    for i,e in enumerate(rr):
        direction=e['source']+'_to_'+e['target'];row=e['row'];sid=row['id']
        dr=ROOT/'crossplatform/round97'/direction
        if direction not in files: files[direction]=read(dr/'SEALED.json')['files']
        sample='samples_PRIVATE/'+sid+'.npz';record='records_PRIVATE/'+sid+'.json'
        if digest(dr/sample)!=files[direction][sample]: raise ValueError('sample hash')
        if record in files[direction] and digest(dr/record)!=files[direction][record]: raise ValueError('sensor receipt hash')
        rec=read(dr/record)
        with np.load(dr/sample,allow_pickle=False) as f:
            dist=f['scores'][f['arm_names'].tolist().index('prepared_frozen')];boxes=f['source_boxes']
            ids[i]=[dist.argmax(),select_grounding_box(boxes,dist)[0]]
            if array_hash(f['point_indices'])!=rec['input_receipt']['point_indices_sha256']:raise ValueError('index hash')
        sampled,_=shared_points(e,rec['input_receipt'],transform_to_front_view)
        selected={j:fit(sampled,boxes[j],pca)[0] for j in set(ids[i])};out[i]=[selected[j] for j in ids[i]]
        if (i+1)%1000==0:print('extra A',i+1,flush=True)
    if not np.isfinite(out).all():raise ValueError('nonfinite boxes')
    dest=WORK/'EXTRA_PRIVATE.npz';tmp=dest.with_suffix('.tmp')
    if dest.exists():raise FileExistsError(dest)
    with tmp.open('xb') as f:np.savez_compressed(f,boxes=out,indices=ids,ordinal=np.arange(1,len(rr)+1))
    tmp.replace(dest)
    seal=dict(n=len(rr),registry=en['sha256'],sha256=digest(dest),target_GT_read=False,FINAL_access=False,
              condition='A-only complete frozen competition on prior physical candidates',
              code={n:digest(HERE/n) for n in ('transfer_inputs158.py','transfer_predict158.py','grounding158.py','reader158.py','experiment/TRANSFER_CHECK_LOCK.json')})
    (WORK/'EXTRA_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
    print('sealed additional comparisons',len(rr))

if __name__=='__main__':main()
