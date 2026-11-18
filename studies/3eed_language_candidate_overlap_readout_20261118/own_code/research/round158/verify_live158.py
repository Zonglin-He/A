"""Replay public inference API for each source with and without registration."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from evidence158 import (REG,ROOT,HERE,WORK,arrays,base97,model97,tensor_prefix,
                         data_graph,tensor,rectangles,role_plan,head_span)
from runtime158 import observe_roles,ground_description


@torch.inference_mode()
def main():
    rows=json.loads(REG.read_text())
    if not (WORK/'PREDICTION_SEAL.json').exists():
        raise ValueError('prediction seal missing')
    with np.load(WORK/'PREDICTIONS_PRIVATE.npz') as p:
        expected=p['indices'][:,p['arms'].tolist().index('role_risk')]
    errors=[]; same=0
    for src in ('drone','quad','waymo'):
        base=base97(src);matcher,_=model97(base,src);matcher.route=None;matcher.use_interface=False
        for has in (False,True):
            idx=next(i for i,e in enumerate(rows) if e['source']==src and e['has_B']==has)
            e=rows[idx];row=e['row'];d=src+'_to_'+e['target'];sid=row['id'];file=sid+'.npz'
            positions=['position'+str(i) for i in range(6)]
            pre=arrays(ROOT/'crossplatform/round131/formal'/d/'prefix_PRIVATE'/file,
                       ['vis0','pos0','xyz','point_indices','query_points_sample_inds','boxes']+positions)
            z=tensor_prefix(pre);boxes=pre['boxes']
            h=hashlib.sha256(pre['xyz'].tobytes()+pre['point_indices'].tobytes()).hexdigest()
            fields=['p2i','i2p','p2i_weight','i2p_weight']
            g=arrays(ROOT/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+h[:20]+'.npz'),fields+['image'])
            a={'prefix':z,'visual':z['vis0'][0],'graph':data_graph(g,g['image'])};b=None
            if has:
                dr=ROOT/'crossplatform/round145/image'/d;f=arrays(ROOT/'crossplatform/round135/formal'/d/'features_PRIVATE'/file,['valid_B','B_vis'])
                valid=f['valid_B'];bank=arrays(dr/'banks_PRIVATE'/file,['norm_Bimage_'+k for k in fields+['visible']])
                bg={k:bank['norm_Bimage_'+k] for k in fields+['visible']}
                with Image.open(e['pair_PRIVATE']['B']['imagepath']) as im:
                    image=np.asarray(im.convert('RGB').resize((384,256),Image.Resampling.BILINEAR))
                norm=arrays(dr/'position_PRIVATE'/file,['pos0','positions'])
                bz=dict(z,pos0=tensor(norm['pos0']));bz.update({k:tensor(norm['positions'][i]) for i,k in enumerate(positions)})
                inside=np.all(np.abs(pre['xyz'][0][None]-boxes[:,None,:3])<=boxes[:,None,3:]/2,axis=-1)
                coverage=np.sum(inside&(valid&bg['visible'])[None],axis=1)/np.maximum(inside.sum(axis=1),1)
                b={'prefix':bz,'graph':data_graph(bg,image),'visual':torch.where(tensor(valid)[:,None],
                   (a['visual']+tensor(f['B_vis']))/2,a['visual']),'coverage':coverage}
            parsed=json.loads((ROOT/'crossplatform/round97/roles_PRIVATE'/(sid+'.json')).read_text())['graph']
            rect,vis=rectangles(e,boxes)
            scores_a,scores_b,_,_=observe_roles(base,matcher,row['caption'],parsed,a,b)
            with np.load(WORK/'evidence_PRIVATE'/f'{idx:04}.npz') as blind:
                errors.append(float(max(np.max(np.abs(scores_a-blind['A'])),np.max(np.abs(scores_b-blind['B'])))))
            picked=ground_description(base,matcher,row['caption'],parsed,boxes,rect,vis,a,b)
            same+=int(picked==expected[idx])
        del base,matcher
    out=dict(status='PASS' if same==6 and max(errors)<1e-3 else 'FAIL',checks=6,matching_top1=same,
             max_abs_role_logit_error=max(errors),logit_errors=errors,predictions_frozen=True,task_GT_read=False)
    (HERE/'LIVE_API_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
    if out['status']!='PASS':raise ValueError('cache/live mismatch')

if __name__=='__main__':main()
