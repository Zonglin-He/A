"""Matched conditional students on frozen O2 streams, no GT before seal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.decota_matrix_common_v1 import read,write,load,save,sha
from methods.decota_final_simplified_v1.tensors import state_hash
from vg_tta.tastvg_conditional_opd_v1 import create,scores,update
OLD=ROOT/'artifacts/tastvg_regime_online_o2_v1';OUT=ROOT/'artifacts/tastvg_conditional_opd_o3_v1'
MODES={'Conditional Pairwise':'pairwise','Conditional Reverse-KL':'reverse_kl'}


def prepare():
    assert not (OUT/'LOCK.json').exists();p=read(OLD/'LOCK.json');files={}
    for name in ['CAPTURE_BARRIER.json','TEACHER_ONLINE_BARRIER.json']:
        for f,h in read(OLD/name)['files'].items():assert sha(OLD/f)==h;files[f]=h
    for f in ['analysis/ROWS.json','analysis/SUMMARY.json']:
        h=read(OLD/'COMPLETION.json')['files'][f];assert sha(OLD/f)==h;files[f]=h
    pins={f:sha(ROOT/f) for f in ['protocols/tastvg_conditional_opd_o3_v1.md','vg_tta/tastvg_conditional_opd_v1.py','scripts/run_tastvg_conditional_opd_o3_v1.py','methods/CURRENT_METHOD.json']}
    rows=[dict(position=r['position'],stream_position=r['stream_position'],parent=r['ordinal'],condition=r['condition'],expert=r['expert']) for r in p['rows']]
    write(OUT/'LOCK.json',dict(rows=rows,conditions=p['conditions'],old_files=files,pins=pins,architecture=[768,128,1],activation='ReLU',lr=.001,steps=1,replay_capacity=4,replay_weight=1.,lambda_residual=1.,tau_E=1.,tau_S=1.,seed=20260929,GT_read=False,time=time.time()))


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    for f,h in p['old_files'].items():assert sha(OLD/f)==h,f
    return p


def online():
    t=time.monotonic();torch.set_num_threads(4);p=verify();out=[];access=[];initial=create();zero_hash=state_hash(initial.state_dict());save(OUT/'INITIAL.pt',initial.state_dict())
    clone=lambda net:{k:v.detach().clone() for k,v in net.state_dict().items()}
    for r in p['rows']:
        if r['stream_position']==1:models={a:create() for a in MODES};replay=[];prev_record=None
        pos=r['position'];x=load(OLD/'capture'/f'{pos:03}.pt');item=dict(position=pos,phi=x['phi'],base=x['base_score']);assert int(scores(initial,item).argmax())==0
        states={a:clone(net) for a,net in models.items()};arriving={a:scores(net,item).detach() for a,net in models.items()};pred={a:int(s.argmax()) for a,s in arriving.items()};budget=0;diag={};grads={};past=[z['position'] for z in replay]
        if r['expert']:
            e=load(OLD/'teacher/online'/f'{pos:03}.pt');access.append(pos);item['teacher']=torch.tensor(e['scores'],dtype=torch.float64);budget=e['selected'];pred={a:budget for a in MODES}
            for a,net in models.items():diag[a],grads[a]=update(net,item,replay,MODES[a])
            replay=(replay+[item])[-4:]
        after={a:clone(net) for a,net in models.items()}
        public=dict(**r,base_score=item['base'].tolist(),arrival_scores={a:z.tolist() for a,z in arriving.items()},selected={'Budgeted Rerank':budget,**pred},arrival_hash={a:state_hash(z) for a,z in states.items()},after_hash={a:state_hash(z) for a,z in after.items()},replay_before=past,replay_after=[z['position'] for z in replay],diagnostics=diag,output_before_update=True)
        f=OUT/'states'/f'{pos:03}.pt';save(f,dict(arrival=states,after=after,gradients=grads,previous_record_sha256=prev_record));prev_record=sha(f);public['state_file_sha256']=prev_record;out.append(public)
    assert access==[r['position'] for r in p['rows'] if r['expert']];assert not torch.cuda.is_initialized();verify();write(OUT/'ONLINE.json',out)
    write(OUT/'ONLINE_BARRIER.json',dict(sha256=sha(OUT/'ONLINE.json'),initial_sha256=sha(OUT/'INITIAL.pt'),initial_state_hash=zero_hash,positions=80,resets_per_arm=5,expert_reads=access,nonexpert_teacher_reads=0,updates_per_arm=20,total_new_SGD_steps=40,GT_or_cached_metric_read=False,seconds=time.monotonic()-t,time=time.time()))
    print('ONLINE SEALED:80 arrivals,40 SGD steps,20 expert reads,GPU0')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','online']);globals()[p.parse_args().stage]()
