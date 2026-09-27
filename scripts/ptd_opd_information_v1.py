"""Finite information/position ablations. Inference and fitting never read labels."""
import argparse,gc,os,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts.ptd_dense_opd_v1 import rows,digest,parent_path,teacher_path,OUT as OLD
from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for,infer
from vg_tta.ptd_opd_information_v1 import ARMS,design,fit
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter
OUT=ROOT/'artifacts/ptd_opd_information_v1'

def file(row,folder,arm=None):return OUT/folder/(digest(row['key'])+('' if arm is None else '_'+arm)+'.pt')

def verify():
    reg=read(OUT/'REGISTRATION.json');pins=dict(reg['pins'])
    for f in sorted((OUT/'amendments').glob('*.json')):pins.update(read(f).get('pins',{}))
    for p,h in {**pins,**reg['protected'],**reg['parents']}.items():assert sha(p)==h,p
    return reg

class Budget:
    def __init__(self,out,stage,limit):
        from scripts.run_final_simplification_v1 import lease
        self.guard=lease();self.t=time.monotonic();self.out=out;self.stage=stage;self.limit=limit;self.calls=0;self.tokens=0
        self.prior=sum(read(f)['seconds'] for f in (out/'receipts').glob('*.json'))
        assert not (out/'ACTIVE_WORKER.json').exists(),'UNCLOSED_WORKER_ACCOUNTING'
        write(out/'ACTIVE_WORKER.json',dict(pid=os.getpid(),started=time.time(),stage=stage,prior_seconds=self.prior))
        def expired(*a):raise TimeoutError('CUMULATIVE_BUDGET')
        signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(limit-self.prior)-2))
        torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
        status(out/'STATUS.json',dict(stage=stage,state='running',pid=os.getpid(),time=time.time()))
    def check(self):
        assert self.prior+time.monotonic()-self.t<self.limit
        assert torch.cuda.max_memory_allocated()<31*2**30
    def close(self):
        signal.alarm(0)
        write(self.out/'receipts'/f'{time.time_ns()}.json',dict(stage=self.stage,seconds=time.monotonic()-self.t,prior_seconds=self.prior,calls=self.calls,tokens=self.tokens,peak_bytes=torch.cuda.max_memory_allocated()))
        (self.out/'ACTIVE_WORKER.json').unlink();self.guard.close()

def register():
    torch.set_num_threads(4)
    parents={str(p):sha(p) for r in rows() for p in [parent_path(r,'clean'),teacher_path(r,'plus'),teacher_path(r,'minus')]}
    for p,h in parents.items():assert h==read(Path(p).with_suffix('.json'))['sha']
    pins=['protocols/ptd_opd_information_v1.md','scripts/ptd_opd_information_v1.py','vg_tta/ptd_opd_information_v1.py']
    protected=['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json','artifacts/ptd_dense_opd_v1/PREDICTION_BARRIER.json','artifacts/c1_fresh_confirmation_v1/vid_train_audit/PROSPECTIVE_SOURCE_RESERVATION.json']
    write(OUT/'REGISTRATION.json',dict(time=time.time(),keys=[r['key'] for r in rows()],pins={str(ROOT/p):sha(ROOT/p) for p in pins},parents=parents,
        protected={str(ROOT/p):sha(ROOT/p) for p in protected},arms=ARMS,LR=.002,steps=3,primary_step=3,rank=16,seed=20260924,
        new_spatial_teacher_forwards=0,gpu_seconds=1800,GT_training=False,historical_development=True,old_result_sha=sha(OLD/'SUMMARY.json'),
        temporal_keys=[r['key'] for r in rows() if r['p0']],confirmation_rule='At most one of five: positive mean dv/ds versus D1, nonnegative dv in both domains, no additional severe source losses or native-good threshold loss. Rank by mean dv vs D1, then fixed ARMS order. If none, do not expand.'))
    for row in rows():
        z=load(parent_path(row,'clean'));qp,qm=[load(teacher_path(row,b))['logp'] for b in ['plus','minus']]
        d=design(z,qp,qm);save(file(row,'design'),d)
        write(file(row,'design').with_suffix('.json'),dict(sha=sha(file(row,'design')),**{k:v for k,v in d.items() if k!='qshift'}))
    write(OUT/'DESIGN_BARRIER.json',dict(time=time.time(),files={str(file(r,'design')):sha(file(r,'design')) for r in rows()},GT_read=False))

def train():
    torch.set_num_threads(4);start=time.monotonic()
    for row in rows():
        z=load(parent_path(row,'clean'));d=load(file(row,'design'))
        qp,qm=[load(teacher_path(row,b))['logp'] for b in ['plus','minus']]
        for arm in ARMS:
            dest=file(row,'fits',arm)
            if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
            target=qm if arm=='S1' else d['qshift'] if arm=='S2' else qp
            fitted=fit(z['h'],z['logits'],target,d['indices'][arm],seed=int(digest(row['key'])[:8],16))
            fitted.update(key=row['key'],arm=arm,positions=z['positions'],native_time=z['interval'])
            save(dest,fitted);write(dest.with_suffix('.json'),dict(sha=sha(dest)))
            print('FIT',row['key'],arm,fitted['history'][0]['loss'],fitted['history'][-1]['loss'],flush=True)
    write(OUT/'TRAIN_COMPLETE.json',dict(fits=80,optimizer_steps=240,CPU_seconds=time.monotonic()-start,teacher_forwards=0,GT_read=False))

def replay(budget):
    pr=processor_load();model=model_load()
    for row in rows():
        z=load(parent_path(row,'clean'));frames,_=frames_for(row,'clean');inputs,_=inputs_for(row,pr,frames)
        for arm in ARMS:
            dest=file(row,'predictions',arm)
            if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
            fitted=load(file(row,'fits',arm));adapter=Adapter(z['h'].shape[-1]).cuda();adapter.load_state_dict(fitted['history'][-1]['state'])
            result=infer(model,pr,inputs,fixed=z,adapter=adapter)
            assert torch.equal(result['h'],z['h']) and torch.equal(result['logits'],z['logits'])
            assert result['interval']==z['interval'] and result['semantic']==z['semantic']
            assert torch.equal(result['adapted_logits'].argmax(-1),fitted['history'][-1]['tokens'])
            result.update(key=row['key'],arm=arm,full_model_reinsertion_exact_tokens=True)
            save(dest,result);write(dest.with_suffix('.json'),dict(sha=sha(dest)))
            budget.calls+=1;budget.check();print('REPLAY',row['key'],arm,flush=True)
        del inputs,frames;gc.collect();torch.cuda.empty_cache()
    files={str(p):sha(p) for folder in ['design','fits','predictions'] for p in (OUT/folder).glob('*.pt')}
    write(OUT/'PREDICTION_BARRIER.json',dict(files=files,time=time.time(),predictions=80,GT_read=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['register','train','replay']);args=ap.parse_args()
    if args.action=='register':register();return
    verify()
    if args.action=='train':train();return
    b=Budget(OUT,'replay',1800)
    try:
        replay(b);status(OUT/'STATUS.json',dict(state='predictions_sealed',time=time.time()))
    except BaseException as e:
        failure=dict(stage='replay',error=repr(e),traceback=traceback.format_exc(),time=time.time())
        write(OUT/'failures'/f'{time.time_ns()}.json',failure);status(OUT/'STATUS.json',dict(state='failed',**failure));raise
    finally:b.close()
if __name__=='__main__':main()
