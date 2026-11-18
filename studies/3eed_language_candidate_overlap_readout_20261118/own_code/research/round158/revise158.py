"""Seal the predeclared development revision before measuring its selections."""
import json
from pathlib import Path
import sys
import numpy as np
from context158 import select_contextual_winner,NAMES
from evidence158 import arrays,atomic_npz,digest,REG,HERE,ROOT,WORK
sys.path[:0]=[str(ROOT/'research/round156'),str(ROOT/'crossplatform/round3/p0/upstream')]
from pair156 import shared_points,fit,load_module,PCA_FILE
from utils.transform_waymo import transform_to_front_view


def main():
    parent=json.loads((WORK/'PREDICTION_SEAL.json').read_text())
    p=WORK/'PREDICTIONS_PRIVATE.npz'
    if digest(p)!=parent['sha256']:raise ValueError('base predictions differ')
    with np.load(p,allow_pickle=False) as f: all_data=dict(f)
    n=parent['n'];boxes=np.empty((n,len(NAMES),7),np.float32);inds=np.empty((n,len(NAMES)),np.int16)
    code_file=HERE/'experiment/REVISION_CODE_LOCK.json'
    for name,sha in json.loads(code_file.read_text()).items():
        if digest(HERE/name)!=sha:raise ValueError('revised code mismatch')
    files=json.loads((ROOT/'crossplatform/round145/method/SEALED.json').read_text())['files']
    entries=json.loads(REG.read_text()); pca=load_module('revised_pca',PCA_FILE).refine
    for i,e in enumerate(entries):
        row=e['row'];sid=row['id'];d=e['source']+'_to_'+e['target']
        k=d+'/samples_PRIVATE/'+sid+'.npz'; source=ROOT/'crossplatform/round145/method'/k
        if digest(source)!=files[k]:raise ValueError('historical control changed')
        old=arrays(source,['arm_names','scores','source_boxes']); ps=dict(zip(old['arm_names'],old['scores']))
        with np.load(WORK/'evidence_PRIVATE'/f'{i:04}.npz',allow_pickle=False) as f:ev=dict(f)
        if not np.array_equal(old['source_boxes'],ev['boxes']):raise ValueError('candidate order')
        roles=json.loads((ROOT/'crossplatform/round97/roles_PRIVATE'/(sid+'.json')).read_text())['graph']['entities']
        span=next((x['span'] for x in roles if x['role']=='T'),None)
        wrong=ps['norm_joint_wrong'] if e['has_B'] else ps['Frozen_A']
        pred=select_contextual_winner(ev['boxes'],row['caption'],span,ev['rect'],ev['seen'],
                ps['norm_joint_zero'],ev['A'][0],ev['B'][0] if e['has_B'] else None,ev['coverage'],
                diagnostics=True,competition_A=ps['Frozen_A'],competition_wrong_B=wrong)
        inds[i]=[pred[k] for k in NAMES]
        if not e['has_B'] and not pred['context_overlap']==pred['context_A_only']==pred['context_wrong_B']:
            raise ValueError('no-B bridge')
        memo=dict(zip(all_data['indices'][i],all_data['boxes'][i]))
        missing=set(inds[i])-set(memo)
        if missing:
            rc=json.loads((ROOT/'crossplatform/round131/formal'/d/'records_PRIVATE'/(sid+'.json')).read_text())
            sampled,_=shared_points(e,rc['source_prefix'],transform_to_front_view)
            memo.update({j:fit(sampled,ev['boxes'][j],pca)[0] for j in missing})
        boxes[i]=[memo[j] for j in inds[i]]
        old_w=all_data['indices'][i,1];new_w=pred['context_overlap']
        from reader158 import overlaps
        all_data['current_old_new_overlap'][i]=overlaps(ev['boxes'][[old_w,new_w]])[0,1]
    all_data['boxes']=np.concatenate([all_data['boxes'],boxes],axis=1)
    all_data['indices']=np.concatenate([all_data['indices'],inds],axis=1)
    all_data['arms']=np.concatenate([all_data['arms'],NAMES])
    output=WORK/'REVISION_PRIVATE.npz'
    if output.exists():raise FileExistsError(output)
    atomic_npz(output,**all_data)
    rec=dict(n=n,sha256=digest(output),parent=parent['sha256'],design_sha256=digest(HERE/'experiment/REVISION_LOCK.json'),
             code_sha256=digest(code_file),target_GT_read=False,FINAL_access=False,
             ordinary_cohort_previously_evaluated=True)
    (WORK/'REVISION_SEAL.json').write_text(json.dumps(rec,indent=2)+'\n')
    print('revision sealed',n,flush=True)

if __name__=='__main__':main()
