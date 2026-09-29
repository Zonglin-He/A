"""Independent NumPy reconstruction of every saved oracle candidate loss."""
import sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.special import logsumexp
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_causal_round2_v1 import BASE,OUT,ARMS,verify


def loss_numpy(pred,gt,records,weights):
    x=np.asarray(pred['boxes'],dtype=np.float64);y=np.asarray(gt['boxes'],dtype=np.float64);valid=np.array(gt['valid'],bool)
    a=x[valid];b=y[valid];aa=np.c_[a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2];bb=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]
    inter=np.maximum(0,np.minimum(aa[:,2:],bb[:,2:])-np.maximum(aa[:,:2],bb[:,:2])).prod(1);union=a[:,2:].prod(1)+b[:,2:].prod(1)-inter;enclosing=(np.maximum(aa[:,2:],bb[:,2:])-np.minimum(aa[:,:2],bb[:,:2])).prod(1)
    giou=inter/union-(enclosing-union)/enclosing;s=float(weights[0]*np.abs(a-b).sum(1).mean()+weights[1]*(1-giou).mean()) if len(a) else 0.;tt=[]
    for z,record in zip(pred['logits'],records):
        z=np.asarray(z,dtype=np.float64);ids=record['frame_ids'];active=[i for i,f in enumerate(ids) if gt['interval'][0]<=f<gt['interval'][1]]
        bound=[active[0],active[-1]] if active else [min(range(len(ids)),key=lambda i:abs(ids[i]-gt['interval'][0])),min(range(len(ids)),key=lambda i:abs(ids[i]-(gt['interval'][1]-1)))]
        terms=[]
        for k,at in enumerate(bound):
            target=np.exp(-(np.arange(len(ids))-at)**2/8)+1e-6;target/=target.sum();lp=z[0,:,k]-logsumexp(z[0,:,k]);prob=np.exp(lp);terms.append(float(np.mean(prob*np.log((prob+1e-6)/target))))
        tt.append(sum(terms))
    t=float(np.mean(tt)*weights[2]);return dict(S=s,T=t,ST=s+t)


def run():
    torch.set_num_threads(4);p=verify();assert (OUT/'ORACLE_BARRIER.json').exists();gt=read(OUT/'GT_SUBSET.json');errors=[];total=0;near=[]
    for row in p['rows']:
        stem=row['key'].replace(':','_');base=load(BASE/'capture'/f'{stem}.pt');weights=p['config']['source_loss_weights'][row['cohort']];label=gt[row['key']]
        z=load(OUT/'oracle'/f'{stem}.pt');a0=loss_numpy(base['prediction'],label,base['records'],weights)
        for k in a0:errors.append(abs(a0[k]-z['initial']['losses'][k]))
        for arm in ARMS:
            a=load(OUT/'arms'/f'{stem}_{arm}.pt');pairs=[(a['prediction'],a['loss'])]
            for step in a['path']:
                for t in step['trials']:
                    pairs.append((t['prediction'],t['loss']))
                    if abs(t['loss'][step['task']]-step['current_loss'][step['task']])<5e-5:near.append(dict(key=row['key'],arm=arm,step=step['step'],alpha=t['alpha'],native_loss_delta=t['loss'][step['task']]-step['current_loss'][step['task']],accepted=t['accepted']))
            for pred,expected in pairs:
                value=loss_numpy(pred,label,base['records'],weights);total+=1
                for k,v in value.items():errors.append(abs(v-expected[k]))
        for task,zp in z['swapped_probes'].items():
            for t in zp['trials']:
                value=loss_numpy(t['prediction'],label,base['records'],weights);total+=1
                for k,v in value.items():errors.append(abs(v-t['loss'][k]))
        print('LOSS_AUDIT',row['key'],flush=True)
    assert max(errors)<5e-5,max(errors)
    write(OUT/'analysis/NATIVE_LOSS_NUMPY_AUDIT.json',dict(status='pass',queries=64,predictions=total,scalar_checks=len(errors),max_absolute_error=max(errors),tolerance=5e-5,precision='independent float64 formula vs native float32 official arithmetic',near_zero_native_loss_changes=len(near),acceptance_authority='native float32 loss with registered 1e-8 relative tolerance; double calculation is numeric audit, not reselection',script_sha256=sha(Path(__file__))))
    write(OUT/'analysis/NEAR_ZERO_LOSS_TRIALS.json',near)
    print('NUMPY LOSS PASS',len(errors),max(errors))
if __name__=='__main__':run()
