"""Unlabeled shared candidate bank and one global temperature per ROI reward."""
import json
import numpy as np
import torch
from stvg.runtime import ROOT,setup,sha
from stvg.tube_runner import write,digest
from stvg import policy
from tools.layout_trial.worker import checked_plan,records,reference,original,load_feature
from tools.layout_trial.reward import LayoutReward,temperature_from_unlabeled


def candidate_bank(old,teacher,alternatives,key,cfg):
    b=torch.as_tensor(teacher,dtype=torch.float64);positions=old['positions8'].astype(int)
    e=torch.as_tensor(old['E_Ref'][positions],dtype=torch.float64)
    feedback=policy.feedback(policy.chart(e),policy.diagonal(e),b,key,41,0,cfg['scales'],cfg['tau'])
    bank=[feedback['boxes'],b[:,None],torch.as_tensor(old['Source'][positions],dtype=torch.float64)[:,None]]
    kinds=[f'policy{i:02d}'for i in range(32)]+['teacher','source']
    for dim in [0,1]:
        for sign in [-1,1]:
            v=b.clone();v[:,dim]+=sign*.25*b[:,dim+2];bank.append(v[:,None]);kinds.append(f'shift{dim}_{sign}')
    for scale in [.7,1.3]:
        v=b.clone();v[:,2:]*=scale;bank.append(v[:,None]);kinds.append(f'scale{scale}')
    for ratio in [.7,1.3]:
        v=b.clone();v[:,2]*=ratio;v[:,3]/=ratio;bank.append(v[:,None]);kinds.append(f'aspect{ratio}')
    bank.append(torch.as_tensor(alternatives,dtype=torch.float64));kinds += [f'native_alternative_query{i}'for i in range(alternatives.shape[1])]
    return torch.cat(bank,1),kinds


def calibrate(plan_path):
    p=checked_plan(plan_path);setup();out=ROOT/p['output'];fseal=json.loads((out/'features_seal.json').read_text());root,seal=reference(p)
    all3=[];all1=[];geometry=[];files={}
    for index,row in enumerate(records(p)):
        old,prior=original(row,root,seal);f=load_feature(row,out,fseal);boxes,kinds=candidate_bank(old,f['boxes'],f['alternative_boxes'],row['key'],p['configuration'])
        feat=torch.from_numpy(f['features']);target=torch.from_numpy(f['boxes']).double();ratio=torch.from_numpy(f['valid_ratio'])
        reward3=LayoutReward(feat,target,3,ratio)(boxes).detach();reward1=LayoutReward(feat,target,1,ratio)(boxes).detach()
        geometric=policy.local_cost(boxes,target[:,None]);all3.append(reward3[:,:32]);all1.append(reward1[:,:32]);geometry.append(geometric[:,:32])
        dest=out/'diagnostic_candidates'/f"{row['id']}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise ValueError('Diagnostic candidates are immutable; do not overwrite')
        np.savez(dest,boxes=boxes.numpy(),layout3_reward=reward3.numpy(),layout1_reward=reward1.numpy(),geometry_reward=-geometric.numpy(),
                 observed_frame_ids=f['observed_frame_ids'],candidate_kinds=np.asarray(kinds))
        files[row['id']]=dict(path=str(dest.relative_to(out)),sha256=sha(dest),request_sha256=digest(row))
        write(out/'worker_progress.json',dict(stage='unlabeled_calibration',complete=index+1,total=128,labels_read=False))
    geom=torch.stack(geometry);cal=dict(status='complete',queries=128,target_GT_used=False,feature_seal_sha256=sha(out/'features_seal.json'),
        layout3=temperature_from_unlabeled(torch.stack(all3),geom,p['configuration']['tau']),
        layout1=temperature_from_unlabeled(torch.stack(all1),geom,p['configuration']['tau']))
    write(out/'temperature_calibration.json',cal)
    write(out/'diagnostic_seal.json',dict(status='complete',queries=128,files=files,labels_read=False,
        plan_sha256=sha(plan_path),feature_seal_sha256=sha(out/'features_seal.json')))
