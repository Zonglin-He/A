"""Live replay of the chosen full-frame reader and same-rule A-only / wrong-B controls."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from evidence158 import (ROOT,HERE,WORK,REG,arrays,atomic_npz,base97,model97,
                         tensor_prefix,tensor,digest)
from grounding158 import ground_frame, select_grounding_box
from reader158 import overlaps
import hashlib
sys.path[:0]=[str(ROOT/'research/round156'),str(ROOT/'crossplatform/round3/p0/upstream')]
from pair156 import shared_points,fit,load_module,PCA_FILE
from utils.transform_waymo import transform_to_front_view

NAMES=('overlap_reader','overlap_A_only','overlap_wrong_B')


@torch.inference_mode()
def main():
    parent=json.loads((WORK/'REVISION_SEAL.json').read_text()); pf=WORK/'REVISION_PRIVATE.npz'
    if digest(pf)!=parent['sha256']:raise ValueError('missing sealed predictions')
    with np.load(pf) as f:p=dict(f)
    n=parent['n']; rr=json.loads(REG.read_text())
    choices=np.empty((n,3),np.int16); boxes=np.empty((n,3,7),np.float32)
    pca=load_module('r158_working_pca',PCA_FILE).refine
    strongest=list(p['arms']).index('old_scores_risk');oldtop=list(p['arms']).index('joint_zero')
    maxerr=0.;live=0
    for src in ('drone','quad','waymo'):
        base=base97(src);m,_=model97(base,src)
        for i,e in enumerate(rr):
            if e['source']!=src: continue
            row=e['row'];sid=row['id'];fn=sid+'.npz';direction=src+'_to_'+e['target']
            p131=ROOT/'crossplatform/round131/formal'/direction;p135=ROOT/'crossplatform/round135/formal'/direction
            bi=ROOT/'crossplatform/round145/image'/direction
            fields=['vis0','pos0','xyz','point_indices','query_points_sample_inds','lang0','textpad','token_mask','boxes']
            pos=['position'+str(k) for k in range(6)]
            pre=arrays(p131/'prefix_PRIVATE'/fn,fields+pos);z=tensor_prefix(pre)
            tag=hashlib.sha256(pre['xyz'].tobytes()+pre['point_indices'].tobytes()).hexdigest()
            with np.load(ROOT/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+tag[:20]+'.npz')) as f:g=dict(f)
            g['query_anchor_indices']=pre['query_points_sample_inds']
            with np.load(p135/'route_PRIVATE'/fn) as f:rt=dict(f)
            old=arrays(ROOT/'crossplatform/round145/method'/direction/'samples_PRIVATE'/fn,['arm_names','scores','source_boxes'])
            scores=dict(zip(old['arm_names'],old['scores']));coords=pre['boxes']
            if not np.array_equal(coords,old['source_boxes']):raise ValueError('candidate mismatch')
            slim={'id':sid}
            joint=None; bg=None
            if e['has_B']:
                source=arrays(p135/'features_PRIVATE'/fn,['B_vis','valid_B'])
                P=torch.where(tensor(source['valid_B'])[:,None],(z['vis0'][0]+tensor(source['B_vis']))/2,z['vis0'][0])
                bp=arrays(bi/'position_PRIVATE'/fn,['pos0','positions','physical_xyz'])
                joint=dict(z,pos0=tensor(bp['pos0']),vis0=P[None])
                joint.update({k:tensor(bp['positions'][k0]) for k0,k in enumerate(pos)})
                kk=['p2i','p2i_weight','i2p','i2p_weight','visible']
                bank=arrays(bi/'banks_PRIVATE'/fn,['norm_Bimage_'+k for k in kk])
                bg={k:bank['norm_Bimage_'+k] for k in kk}
                with Image.open(e['pair_PRIVATE']['B']['imagepath']) as im:
                    bg['image']=np.asarray(im.convert('RGB').resize((384,256),Image.Resampling.BILINEAR))
            idx,_,_,dist=ground_frame(base,m,coords,z,g,rt,slim,registered_prefix=joint,graph_B=bg)
            err=float(np.max(np.abs(dist-scores['norm_joint_zero'])));maxerr=max(err,maxerr)
            if idx!=p['indices'][i,strongest] or err>1e-7:raise ValueError(('live control replay',i,err))
            live+=1
            choices[i]=[idx,select_grounding_box(coords,scores['Frozen_A'])[0],
                        select_grounding_box(coords,scores['norm_joint_wrong'] if e['has_B'] else scores['Frozen_A'])[0]]
            if not e['has_B'] and len(set(choices[i]))!=1:raise ValueError('single frame handling')
            memo=dict(zip(p['indices'][i],p['boxes'][i]));missing=set(choices[i])-set(memo)
            if missing:
                rec=json.loads((p131/'records_PRIVATE'/(sid+'.json')).read_text())
                points,_=shared_points(e,rec['source_prefix'],transform_to_front_view)
                memo.update({j:fit(points,coords[j],pca)[0] for j in missing})
            boxes[i]=[memo[int(j)] for j in choices[i]]
            p['current_old_new_overlap'][i]=overlaps(coords[[p['indices'][i,oldtop],idx]])[0,1]
            if live%300==0:print('live API',live,flush=True)
        del base,m
    p['indices']=np.column_stack([p['indices'],choices]);p['boxes']=np.concatenate([p['boxes'],boxes],axis=1)
    p['arms']=np.concatenate([p['arms'],NAMES]);out=WORK/'WORKING_PRIVATE.npz'
    if out.exists():raise FileExistsError(out)
    atomic_npz(out,**p)
    rec=dict(n=n,sha256=digest(out),parent=parent['sha256'],live_ground_frame_winner_matches=live,
             max_live_full_competition_probability_difference=maxerr,target_GT_read=False,FINAL_access=False,
             design_sha256=digest(HERE/'experiment/WORKING_READOUT.json'),
             code={k:digest(HERE/k) for k in ('grounding158.py','reader158.py','check_working158.py')})
    (WORK/'WORKING_SEAL.json').write_text(json.dumps(rec,indent=2)+'\n')
    print('sealed all live frame selections',live,maxerr,flush=True)


if __name__=='__main__': main()
