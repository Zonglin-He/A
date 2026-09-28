"""F39 finite label-free worker. No evaluator, labels, or A4 training/imported plan."""
import argparse
import copy
import fcntl
import gc
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
from scripts.prepare_dense_support_v1 import OUT,F36,plan
from scripts.run_spatial10_components_v1 import capture_timed
from scripts.run_parametric_observation_v1 import cpu_tree
OWN=['vg_tta/dense_support_temporal_v1.py','scripts/run_dense_support_v1.py']

def dest(stage,key):return OUT/stage/(key.replace(':','_')+'.pt')
def record(path,value):
    save(path,cpu_tree(value));write(path.with_suffix('.json'),dict(sha256=sha(path),key=value['key'],
        code_pins={f:sha(ROOT/f) for f in OWN},completed=time.time()))
def existing(path):
    if not path.exists():return False
    receipt=read(path.with_suffix('.json'));assert sha(path)==receipt['sha256']
    for f,h in receipt['code_pins'].items():assert sha(ROOT/f)==h,f
    return True

def one(model,r,p,stage):
    import torch
    from vg_tta.parametric_observation_v1 import ObservationReplay
    from vg_tta.simplification_partial_v1 import FullInputHeadReplay
    from vg_tta.dense_support_temporal_v1 import teacher_from_logits,fit,OutputReplay
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    frames,_,records,views,qa,cost=capture_timed(model,r)
    it=FullInputHeadReplay(ObservationReplay(model,views,len(r['input']['frame_ids']),'head'))
    zero=it.zero;ids=r['input']['frame_ids'];teacher=teacher_from_logits(zero['actions'],records)
    oldpath=F36/'space'/(r['key'].replace(':','_')+'.pt');old=load(oldpath);a4=old['fits']['A4']
    assert qa==old['qa'] and torch.equal(zero['boxes'].cpu(),old['native_boxes'])
    assert all(torch.equal(a.cpu(),b) for a,b in zip(zero['logits'],old['native_logits']))
    native=dict(boxes=a4['final']['boxes'],logits=old['native_logits'],indices=old['native_indices'])
    # An original full-backbone FP32-suffix call verifies actual final dense
    # field identity, not just a similarly named internal intermediate.
    dense=[]
    hook=model.register_forward_hook(lambda m,a,o:dense.append(o['pred_actioness'].detach().cpu().clone()))
    try:native_audit=full_prediction(model,frames,ids,r['input'],r['subject'],a4['state'],native)['audit']
    finally:hook.remove()
    assert len(dense)==2 and all(torch.equal(a,b.cpu()) for a,b in zip(dense,zero['actions']))
    # T3 changes ONLY query text/subject, not recipient video, timestamps or size.
    dr=copy.deepcopy(r);donor=r['F39_wrong_query']
    dr['input']['caption']=donor['caption'];dr['subject']=donor['subject']
    _,_,drecords,dviews,dqa,dcost=capture_timed(model,dr)
    dit=FullInputHeadReplay(ObservationReplay(model,dviews,len(ids),'head'))
    wrong=teacher_from_logits(dit.zero['actions'],drecords)
    assert qa==dqa and [x['frame_ids'] for x in records]==[x['frame_ids'] for x in drecords]
    del dit,dviews
    rolled=[]
    for t in teacher:
        shift=len(t['a'])//2
        assert 0<shift<len(t['a'])
        rolled.append({**t,'a':torch.roll(t['a'],shift).detach().clone(),
            'raw_logits':torch.roll(t['raw_logits'],shift).detach().clone(),'roll':shift})
    fits={};full={}
    if stage=='dev':
        specs=[(f'T1_lr{lr:g}_b{b:g}','T1',lr,b) for lr in p['temporal']['lr'] for b in p['temporal']['beta']]
    else:
        selected=read(OUT/'TEMPORAL_SELECTION.json')['chosen'];lr,b=selected['lr'],selected['beta']
        names=['T2','T3','T4'] if stage=='dev_controls' else ['T1','T2','T3','T4']
        specs=[(n,n,lr,b) for n in names]
    if stage=='dev' and not list((OUT/'noops').glob(r['key'].split(':')[0]+'*.pt')):
        for name,steps,lr in [('steps0',0,.001),('lr0',1,0.)]:
            zz=fit(it,records,ids,teacher,lr=lr,beta=.1,steps=steps)
            assert zz['state_delta']==0 and zz['backwards']==steps
            expected={**zz['final'],'boxes':a4['final']['boxes']}
            audit=full_prediction(model,frames,ids,r['input'],r['subject'],{**a4['state'],**zz['state']},expected)['audit']
            record(dest('noops',r['key']+'_'+name),dict(key=r['key'],fit=zz,audit=audit))
    for name,arm,lr,b in specs:
        evidence=wrong if arm=='T3' else rolled if arm=='T4' else teacher
        interface=OutputReplay(zero) if arm=='T2' else it
        zz=fit(interface,records,ids,evidence,lr=lr,beta=b,steps=5,output_control=arm=='T2')
        for step in zz['path']:step['boxes']=a4['final']['boxes']
        zz['final']=zz['path'][zz['best_step']]
        if arm!='T2':
            assert zz['parameters']==66306
            full[name]=full_prediction(model,frames,ids,r['input'],r['subject'],{**a4['state'],**zz['state']},zz['final'])['audit']
        fits[name]=zz
    record(dest(stage,r['key']),dict(key=r['key'],cohort=r['key'].split(':')[0],group=r['group'],source=r['source'],
        role=r['f37_role'],query=r['input']['caption'],frame_ids=ids,records=records,T0=native,teacher=teacher,
        wrong_teacher=wrong,wrong_query=donor,rolled_teacher=rolled,qa=qa,capture_seconds=cost,wrong_capture_seconds=dcost,
        teacher_full_model_exact=True,native_full_model=native_audit,fits=fits,full_reinsertion=full,
        A4_reference=dict(path=str(oldpath),sha256=sha(oldpath)),GT_online=False,
        all_positions_used=[len(t['a']) for t in teacher],new_DINO=0))
    print('F39',stage,r['key'],{n:(z['best_step'],round(z['state_delta'],5),z['final']['physical_interval']) for n,z in fits.items()},flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['dev','dev_controls','eval']);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    import torch
    from scripts.run_decota_refine_v1 import configure
    from scripts.run_closure_v1 import model_for
    from vg_tta.foreground_runtime import state_digest
    configure();p=plan();n=0
    lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for c,rows in p['rows'].items():
            role='evaluation' if args.stage=='eval' else 'development'
            pending=[r for r in rows if r['f37_role']==role and not existing(dest(args.stage,r['key']))]
            if not pending:continue
            if args.limit and n>=args.limit:break
            model=model_for(c);digest=state_digest(model)
            for r in pending:
                if args.limit and n>=args.limit:break
                one(model,r,p,args.stage);assert state_digest(model)==digest;n+=1
                status(OUT/'STATUS.json',dict(stage=args.stage,last=r['key'],finished=False,completed_this_process=n))
            del model;gc.collect();torch.cuda.empty_cache()
        print('F39 stage complete',args.stage,n,flush=True)
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()

if __name__=='__main__':main()
