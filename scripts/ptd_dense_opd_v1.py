"""Finite dense OPD implementation/run. All inference/training stages exclude GT."""
import argparse,gc,hashlib,json,os,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts.ptd_spatial_adapter_ab_v1 import path as parent_path,processor_load,model_load,frames_for,inputs_for,infer
from vg_tta.ptd_dense_opd_v1 import fit,cpu_control
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter
OUT=ROOT/'artifacts/ptd_dense_opd_v1'
DENSE=ROOT/'artifacts/ptd_dense_teacher_audit_v1'
OLD=ROOT/'artifacts/ptd_visual_evidence_decomposition_v1'
def digest(key):return hashlib.sha256(key.encode()).hexdigest()
def rows():return read(DENSE/'INPUTS.json')
def student_path(row):return OUT/'student'/(digest(row['key'])+'.pt')
def fit_path(row,arm):return OUT/'fits'/(digest(row['key'])+'_'+arm+'.pt')
def teacher_path(row,branch):return OUT/'teacher'/(digest(row['key'])+'_'+branch+'.pt')

class Budget:
    def __init__(self,stage):
        from scripts.run_final_simplification_v1 import lease
        self.guard=lease();self.t=time.monotonic();self.calls=0;self.tokens=0;self.stage=stage
        self.prior=sum(read(f)['seconds'] for f in (OUT/'receipts').glob('*.json'))
        assert not (OUT/'ACTIVE_WORKER.json').exists(),'UNCLOSED_WORKER_ACCOUNTING'
        write(OUT/'ACTIVE_WORKER.json',dict(pid=os.getpid(),started=time.time(),stage=stage,prior_seconds=self.prior))
        def expired(*a):raise TimeoutError('CUMULATIVE_7200_SECONDS')
        signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(7200-self.prior)-2))
        torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
        status(OUT/'STATUS.json',dict(stage=stage,state='running',pid=os.getpid(),time=time.time()))
    def check(self):
        assert self.prior+time.monotonic()-self.t<7200,'CUMULATIVE_BUDGET'
        assert torch.cuda.max_memory_allocated()<31*2**30,'ALLOCATED_GPU_BUDGET'
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<8*2**30,'OUTPUT_BUDGET'
    def close(self):
        signal.alarm(0)
        write(OUT/'receipts'/f'{time.time_ns()}.json',dict(stage=self.stage,seconds=time.monotonic()-self.t,
            prior_seconds=self.prior,calls=self.calls,tokens=self.tokens,peak_bytes=torch.cuda.max_memory_allocated()))
        (OUT/'ACTIVE_WORKER.json').unlink();self.guard.close()

def register():
    torch.set_num_threads(4)
    pins=['protocols/ptd_dense_opd_v1.md','scripts/ptd_dense_opd_v1.py','scripts/ptd_dense_opd_teacher_v1.py',
          'vg_tta/ptd_dense_opd_v1.py','scripts/ptd_visual_evidence_student_v1.py','scripts/ptd_spatial_adapter_ab_v1.py',
          'vg_tta/ptd_spatial_adapter_ab_v1.py','scripts/score_ptd_spatial_adapter_ab_v1.py']
    protected=['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json',
               'artifacts/c1_fresh_confirmation_v1/vid_train_audit/PROSPECTIVE_SOURCE_RESERVATION.json']
    parents={str(parent_path(r,'clean')):sha(parent_path(r,'clean')) for r in rows()}
    for f,h in parents.items():assert h==read(Path(f).with_suffix('.json'))['sha']
    write(OUT/'REGISTRATION.json',dict(time=time.time(),protocol_sha=sha(ROOT/pins[0]),keys=[r['key'] for r in rows()],
        pins={p:sha(ROOT/p) for p in pins},protected={p:sha(ROOT/p) for p in protected},parents=parents,
        inputs_sha=sha(DENSE/'INPUTS.json'),teacher_manifest_sha=sha(ROOT/'checkpoints/Qwen3-VL-8B-Instruct/DOWNLOAD_MANIFEST.json'),
        native_positions=sum(len(load(parent_path(r,'clean'))['positions']) for r in rows()),arms=['D0','D1','D2','D3'],
        gpu_seconds=7200,output_bytes=8*2**30,primary_step=3,LR=.002,steps=3,rank=16,seed=20260924,
        prior_exposed_development=True,GT_training=False,cross_video_writeback=False))
    write(OUT/'CPU_CONTROL.json',cpu_control())

def verify():
    reg=read(OUT/'REGISTRATION.json');pins=dict(reg['pins'])
    for f in sorted((OUT/'amendments').glob('*.json')):pins.update(read(f).get('pins',{}))
    for p,h in {**pins,**reg['protected']}.items():assert sha(ROOT/p)==h,p
    assert sha(DENSE/'INPUTS.json')==reg['inputs_sha']
    return reg

def capture(budget):
    from scripts.ptd_visual_evidence_student_v1 import probe,strip_visual
    pr=processor_load();model=model_load()
    for row in rows():
        dest=student_path(row)
        if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
        z=load(parent_path(row,'clean'));old=OLD/'student'/(digest(row['key'])+'.pt')
        if old.exists():
            assert sha(old)==read(old.with_suffix('.json'))['sha'];prior=load(old)
            assert torch.equal(prior['plus']['h'],z['h']) and torch.equal(prior['plus']['logits'],z['logits'])
            result=dict(key=row['key'],minus=prior['minus'],reused=str(old),reused_sha=sha(old),GT_used=False)
        else:
            frames,_=frames_for(row,'clean');inputs,_=inputs_for(row,pr,frames)
            plus=probe(model,pr,inputs,z);assert torch.equal(plus['h'],z['h']) and torch.equal(plus['logits'],z['logits'])
            text,removed=strip_visual(inputs,pr.tokenizer);minus=probe(model,pr,text,z,text_only=True)
            repeat=probe(model,pr,text,z,text_only=True)
            assert torch.equal(minus['logits'],repeat['logits']) and torch.equal(minus['h'],repeat['h'])
            result=dict(key=row['key'],minus=minus,removed=removed,true_replay_exact=True,minus_repeat_exact=True,GT_used=False)
            budget.calls+=3;del inputs,text,plus,repeat,frames
        save(dest,result);write(dest.with_suffix('.json'),dict(sha=sha(dest)))
        print('CAPTURE',row['key'],flush=True);budget.check()
    write(OUT/'STUDENT_COMPLETE.json',dict(count=len(rows()),GT_used=False))

def train_row(row):
    torch.set_num_threads(4)
    files=[teacher_path(row,b) for b in ['plus','minus']]
    if not all(f.exists() for f in files) or not student_path(row).exists():return
    for f in files+[student_path(row)]:assert sha(f)==read(f.with_suffix('.json'))['sha']
    z=load(parent_path(row,'clean'));m=load(student_path(row))['minus']
    qp,qm=[load(f)['logp'] for f in files]
    for arm in ['D1','D2','D3']:
        dest=fit_path(row,arm)
        if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
        res=fit(z,m,qp,qm,arm,seed=int(digest(row['key'])[:8],16))
        res.update(key=row['key'],positions=z['positions'],native_time=z['interval'],semantic=z['semantic'])
        save(dest,res);write(dest.with_suffix('.json'),dict(sha=sha(dest),steps=3,parameter_count=res['parameter_count']))
        print('FIT',row['key'],arm,'loss',res['history'][0]['losses'],res['history'][-1]['losses'],flush=True)

def teach(budget,smoke=False):
    from scripts.ptd_dense_opd_teacher_v1 import Teacher
    teacher=Teacher(budget);audits=[]
    for row in rows():
        z=load(parent_path(row,'clean'))
        for branch in ['plus','minus']:
            dest=teacher_path(row,branch)
            if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
            inp,text=teacher.inputs(row,z,branch);p=teacher.prepare(inp);cache,_=teacher.prefill(p);items=[]
            for j,pos in enumerate(z['positions']):
                for c,name in enumerate(['x1','y1','x2','y2']):
                    point=OUT/'teacher_points'/digest(row['key'])/f'{branch}_{j}_{c}.pt'
                    if not smoke and point.exists():
                        assert sha(point)==read(point.with_suffix('.json'))['sha']
                        items.append(load(point));continue
                    selector=f'Frame {pos+1}, {name}: '
                    base=teacher.pr.tokenizer.encode(text+selector,add_special_tokens=False)
                    assert base==teacher.pr.tokenizer.encode(text,add_special_tokens=False)+teacher.pr.tokenizer.encode(selector,add_special_tokens=False)
                    for v in [0,9,10,417,999,1000]:assert teacher.pr.tokenizer.encode(text+selector+str(v)+'\n',add_special_tokens=False)==base+teacher.events[v]
                    start=time.monotonic();result=teacher.distribution(p,cache,selector)
                    if j==0 and c==0:
                        # Release the shared prefix to bound peak during independent full replay.
                        del cache;gc.collect();torch.cuda.empty_cache()
                        errors=[]
                        for v in [0,9,10,417,999,1000]:
                            full=teacher.independent(p,selector,v);errors.append(dict(value=v,trie=float(result['raw'][v]),full=full,error=abs(full-float(result['raw'][v]))))
                        aud=dict(key=row['key'],branch=branch,errors=errors,maximum=max(v['error'] for v in errors),prefix_tokens=result['prefix_tokens'])
                        write(OUT/'numerical'/f'{digest(row["key"])}_{branch}_{time.time_ns()}.json',aud)
                        assert aud['maximum']<=.1,('NUMERICAL_EQUIVALENCE',aud)
                        audits.append(aud);cache,_=teacher.prefill(p)
                    result.update(position=pos,coordinate=name,seconds=time.monotonic()-start)
                    items.append(result)
                    if not smoke:
                        save(point,result);write(point.with_suffix('.json'),dict(sha=sha(point)))
                    print('SCORE',row['key'],branch,pos,name,round(result['seconds'],3),flush=True)
                    if smoke:
                        save(OUT/'smoke'/f'{branch}_{time.time_ns()}.pt',result)
                        break
                if smoke:break
            if not smoke:
                result=dict(key=row['key'],branch=branch,positions=z['positions'],logp=torch.stack([x['logp'] for x in items]).reshape(-1,4,1001),
                    raw=torch.stack([x['raw'] for x in items]).reshape(-1,4,1001),items=[{k:v for k,v in x.items() if k not in ['logp','raw']} for x in items],
                    prompt=text,prompt_sha=digest(text),input_ids=inp['input_ids'],precision='FP32 language decoder/projection, BF16 original vision, FP64 normalization',GT_used=False)
                save(dest,result);write(dest.with_suffix('.json'),dict(sha=sha(dest)))
            del p,cache,inp,items;gc.collect();torch.cuda.empty_cache();budget.check()
        if smoke:break
        train_row(row)
    if smoke:write(OUT/f'SMOKE_{time.time_ns()}.json',dict(audits=audits,passed=True,seconds=time.monotonic()-budget.t))
    else:write(OUT/'TEACHER_COMPLETE.json',dict(count=len(rows()),GT_used=False))

def replay(budget):
    pr=processor_load();model=model_load()
    for row in rows():
        z=load(parent_path(row,'clean'));frames,_=frames_for(row,'clean');inputs,_=inputs_for(row,pr,frames)
        for arm in ['D0','D1','D2','D3']:
            dest=OUT/'predictions'/(digest(row['key'])+'_'+arm+'.pt')
            if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
            adapter=Adapter(z['h'].shape[-1]).cuda()
            fitted=None
            if arm!='D0':
                fitted=load(fit_path(row,arm));adapter.load_state_dict(fitted['history'][-1]['state'])
            result=infer(model,pr,inputs,fixed=z,adapter=adapter)
            assert torch.equal(result['h'],z['h']) and torch.equal(result['logits'],z['logits'])
            assert result['interval']==z['interval'] and result['semantic']==z['semantic']
            expected=z['base_tokens'] if fitted is None else fitted['history'][-1]['tokens']
            # FP32 CPU vs GPU tiny rounding differences must not alter coordinate readout.
            assert torch.equal(result['adapted_logits'].argmax(-1),expected),'REPLAY_TOKENS'
            if arm=='D0':assert result['completion']==z['completion']
            result.update(key=row['key'],arm=arm,full_model_reinsertion_exact_tokens=True)
            save(dest,result);write(dest.with_suffix('.json'),dict(sha=sha(dest)))
            budget.calls+=1;budget.check();print('REPLAY',row['key'],arm,flush=True)
        del inputs,frames;gc.collect();torch.cuda.empty_cache()
    files={str(p):sha(p) for folder in ['student','teacher','fits','predictions'] for p in (OUT/folder).glob('*.pt')}
    write(OUT/'PREDICTION_BARRIER.json',dict(files=files,time=time.time(),count=64,GT_read=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['register','capture','smoke','teacher','train','replay']);args=ap.parse_args()
    if args.action=='register':register();return
    verify()
    if args.action=='train':
        for row in rows():train_row(row)
        return
    budget=Budget(args.action)
    try:
        if args.action=='capture':capture(budget)
        elif args.action in ['smoke','teacher']:teach(budget,smoke=args.action=='smoke')
        else:replay(budget)
        status(OUT/'STATUS.json',dict(stage=args.action,state='completed',time=time.time()))
    except BaseException as e:
        failure=dict(stage=args.action,error=repr(e),traceback=traceback.format_exc(),time=time.time())
        write(OUT/'failures'/f'{time.time_ns()}.json',failure);status(OUT/'STATUS.json',dict(stage=args.action,state='failed',error=repr(e),time=time.time()));raise
    finally:budget.close()

if __name__=='__main__':main()
