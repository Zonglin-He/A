"""CPU reward pass, sealed before any cached GT-derived metrics are opened."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,load,sha
from vg_tta.tastvg_spatial_critic_s06_v1 import rewards,ALL_PAIRS,ANTITHETIC
S0=ROOT/'artifacts/tastvg_spatial_expansion_s0_v1';NATIVE=ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1';OUT=ROOT/'artifacts/tastvg_spatial_critic_s06_v1'

def prepare():
    p=read(NATIVE/'LOCK.json');paths=['protocols/tastvg_spatial_critic_s06_v1.md','vg_tta/tastvg_spatial_critic_s06_v1.py','scripts/run_tastvg_spatial_critic_s06_v1.py','methods/CURRENT_METHOD.json']
    write(OUT/'LOCK.json',dict(rows=p['rows'],conditions=p['conditions'],pins={f:sha(ROOT/f) for f in paths},native_barrier_sha256=sha(NATIVE/'PREDICTION_BARRIER.json'),expert_barrier_sha256=sha(S0/'EXPERT_BARRIER.json'),seed=20260929,GT_for_reward=False,new_GPU=0,new_experts=0,time=time.time()))

def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h
    assert sha(NATIVE/'PREDICTION_BARRIER.json')==p['native_barrier_sha256'];assert sha(S0/'EXPERT_BARRIER.json')==p['expert_barrier_sha256'];return p

def run():
    import torch
    torch.set_num_threads(4);start=time.monotonic();lock=verify();nb=read(NATIVE/'PREDICTION_BARRIER.json');eb=read(S0/'EXPERT_BARRIER.json')
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in ['GT_SUBSET','/ROWS.json','/SUMMARY.json','/annos/','/annotations/']):raise PermissionError('Reward worker forbids GT/metric files')
    sys.addaudithook(guard);rows=[];audits=0
    for r in lock['rows']:
        for cond in lock['conditions']:
            name=f"{r['ordinal']:03}.pt";nr=f'ta/{cond}/{name}';er=f'expert/{cond}/{name}';assert sha(NATIVE/nr)==nb['files'][nr] and sha(S0/er)==eb['files'][er]
            x=load(NATIVE/nr);e=load(S0/er);assert x['pixel_sha256']==e['pixel_sha256'];bs=[c['prediction']['boxes'].numpy() for c in x['trajectory']];score=rewards(bs,e['boxes'],e['valid'])
            if score is not None:
                # Independent rectangle implementation, including mean denominator.
                for k,b in enumerate(bs):
                    vals=[]
                    for j in np.flatnonzero(e['valid']):
                        a=np.asarray(b[j],float);q=np.asarray(e['boxes'][j],float);al=a[:2]-a[2:]/2;ah=a[:2]+a[2:]/2;bl=q[:2]-q[2:]/2;bh=q[:2]+q[2:]/2
                        ix=max(0,min(ah[0],bh[0])-max(al[0],bl[0]));iy=max(0,min(ah[1],bh[1])-max(al[1],bl[1]));inter=ix*iy;vals.append(inter/(a[2]*a[3]+q[2]*q[3]-inter))
                    assert abs(np.mean(vals)-score[k])<1e-12;audits+=1
            rows.append(dict(parent=r['ordinal'],condition=cond,rewards=score.tolist() if score is not None else None,selected=int(np.argmax(score)) if score is not None else 0,valid_expert_frames=int(np.sum(e['valid'])),native_prediction_sha256=nb['files'][nr],expert_sha256=eb['files'][er]))
    cuts={}
    for group in ['corruption','clean']:
        rr=[r for r in rows if r['rewards'] is not None and (r['condition']!='clean' if group=='corruption' else r['condition']=='clean')];cuts[group]={}
        for kind,pairs in [('all',ALL_PAIRS),('antithetic',ANTITHETIC)]:cuts[group][kind]=np.quantile([abs(r['rewards'][i]-r['rewards'][j]) for r in rr for i,j in pairs],[1/3,2/3]).tolist()
    write(OUT/'REWARDS.json',rows);write(OUT/'MARGIN_CUTS.json',cuts);write(OUT/'REWARD_BARRIER.json',dict(cells=96,files={n:sha(OUT/n) for n in ['REWARDS.json','MARGIN_CUTS.json']},GT_read=False,new_GPU=0,new_experts=0,independent_reward_checks=audits,seconds=time.monotonic()-start,time=time.time()));print('sealed',len(rows),'valid',sum(r['rewards'] is not None for r in rows),'checks',audits)

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
