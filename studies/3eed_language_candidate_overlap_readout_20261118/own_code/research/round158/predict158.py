"""Select boxes without target annotations, verify parents, then seal all arms."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'research/round156'),str(ROOT/'crossplatform/round3/p0/upstream')]
from pair156 import fit,shared_points,load_module,PCA_FILE
from utils.transform_waymo import transform_to_front_view
from evidence158 import arrays,atomic_npz,digest,REG,LOCK,WORK
from reader158 import ARMS,select_from_description


def main():
    started=time.monotonic()
    plan=json.loads(LOCK.read_text()); code=json.loads((HERE/'experiment/CODE_LOCK.json').read_text())
    if any(digest(HERE/k)!=v for k,v in code.items()) or digest(REG)!=plan['registry_sha256']:
        raise ValueError('locked implementation or registry changed')
    entries=json.loads(REG.read_text()); n=len(entries)
    if n!=1798 or sum(e['has_B'] for e in entries)!=764:
        raise ValueError('cohort changed')
    pair=ROOT/'crossplatform/round156/PAIR_PRIVATE.npz'
    if digest(pair)!='1513f9addb268ebaa505d43337ae8a784905288cb094614a80e45991123795e9':
        raise ValueError('previous pair seal changed')
    oldpair=arrays(pair,['boxes','symmetric_A_pca','query'])
    base=ROOT/'crossplatform/round145/method'
    seal=base/'SEALED.json'
    if digest(seal)!='ac4a9cf70845431f1509c195afb2664a9593c75388fd48b6880483ff62a6ca31':
        raise ValueError('previous readout seal changed')
    pred_files=json.loads(seal.read_text())['files']
    m=len(ARMS);indices=np.empty((n,m),np.int16); endpoint=np.empty((n,m,7),np.float32)
    diag=[]; evidence_hashes=[]; pca=load_module('r158_pca',PCA_FILE).refine
    source_alias={'frozen_A':'Frozen_A','joint_zero':'norm_joint_zero','R145':'norm_TTA'}
    for i,e in enumerate(entries):
        row=e['row']; sid=row['id'];direction=e['source']+'_to_'+e['target']
        rp=WORK/'evidence_PRIVATE'/f'{i:04}.npz';evidence_hashes.append(digest(rp))
        with np.load(rp,allow_pickle=False) as f: ev=dict(f)
        k=direction+'/samples_PRIVATE/'+sid+'.npz'; p=base/k
        if digest(p)!=pred_files[k]:
            raise ValueError('original prediction hash')
        old=arrays(p,['scores','arm_names','source_boxes'])
        if not np.array_equal(old['source_boxes'],ev['boxes']):
            raise ValueError('different physical candidates')
        table=dict(zip(old['arm_names'],old['scores']))
        controls={k:table[v] for k,v in source_alias.items()}
        target=next((x['span'] for x in json.loads((ROOT/'crossplatform/round97/roles_PRIVATE'/(sid+'.json')).read_text())['graph']['entities'] if x['role']=='T'),None)
        winners,info=select_from_description(ev,row['caption'],target,controls)
        indices[i]=[winners[k] for k in ARMS]
        if indices[i,1]!=oldpair['query'][i,0]:
            raise ValueError('original top-one mismatch')
        receipt=json.loads((ROOT/'crossplatform/round131/formal'/direction/'records_PRIVATE'/(sid+'.json')).read_text())
        sampled,_=shared_points(e,receipt['source_prefix'],transform_to_front_view)
        memo={int(j):fit(sampled,ev['boxes'][j],pca)[0] for j in np.unique(indices[i])}
        endpoint[i]=[memo[int(j)] for j in indices[i]]
        if not np.array_equal(endpoint[i,1],oldpair['boxes'][i,0,1]) or not np.array_equal(endpoint[i,0],oldpair['symmetric_A_pca'][i]):
            raise ValueError('historical geometry mismatch')
        diag.append(info)
        if (i+1)%300==0: print('selected',i+1,flush=True)
    out=WORK/'PREDICTIONS_PRIVATE.npz'
    if out.exists(): raise FileExistsError(out)
    atomic_npz(out,boxes=endpoint,indices=indices,arms=np.asarray(ARMS),ordinal=np.arange(1,n+1),
               **{k:np.asarray([r[k] for r in diag]) for k in diag[0]})
    (WORK/'PREDICTION_SEAL.json').write_text(json.dumps(dict(n=n,arms=ARMS,sha256=digest(out),
        evidence_manifest_sha256=__import__('hashlib').sha256('\n'.join(evidence_hashes).encode()).hexdigest(),
        design_sha256=digest(LOCK),code=code,registered_B=764,seconds=time.monotonic()-started,
        target_GT_read=False,FINAL_access=False,verified_joint_top1_and_pca=n,verified_frozen_pca=n),indent=2)+'\n')
    print('sealed',n,flush=True)


if __name__=='__main__':main()
