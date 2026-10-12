"""Same Q0 science; correct lease metadata and exact reuse of already qualified HC."""
import inspect,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_common_v1 import BASE,read,write,status,sha
import scripts.run_stvg_opd_query_q0_v1 as original
REC=BASE/'serial_GPU_context002'


def setup():
    previous=original.verify
    def verify():
        r=previous()
        for p,h in read(REC/'REVISION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
        return r
    original.verify=verify
    s=inspect.getsource(original.gpu_lease)
    s=s.replace("BASE/'PRIORITY_PAUSE_RECEIPT.json'","REC/'PRIORITY_PAUSE_RECEIPT.json'")
    ns=dict(original.__dict__,REC=REC);exec(compile(s,str(Path(__file__)), 'exec'),ns);original.gpu_lease=ns['gpu_lease']
    s=inspect.getsource(original.run)
    needle='        for r in records:\n';assert s.count(needle)==1
    insertion='''        for r in records:
            existing=dest/f"{r['index']:05}.pt"
            if qualification and existing.with_suffix('.json').exists():
                oldrc=read(existing.with_suffix('.json'))
                assert sha(existing)==oldrc['sha256'] and not oldrc['GT_read']
                prior=load(existing);assert prior['index']==r['index'] and prior['dataset']=='hc2'
                repeat=dest/f"REPEAT_{r['index']:05}.pt"
                assert sha(repeat)==prior['repeat']['sha256']
                counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                equal(prior['episode'],load(repeat),counts)
                files[str(existing.relative_to(ROOT))]=sha(existing)
                continue
'''
    s=s.replace(needle,insertion)
    ns=dict(original.__dict__,REC=REC);exec(compile(s,str(Path(__file__)), 'exec'),ns);original.run=ns['run']
    return verify()


if __name__=='__main__':
    job=sys.argv[1]
    try:
        setup()
        if job in ['qualify','formal']:original.run(job=='qualify')
        elif job in ['qualification_root','deployment_root']:original.root(job=='qualification_root')
        else:raise ValueError(job)
    except BaseException:
        p=REC/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',stage=job,revision='serial002',evidence=str(p.relative_to(ROOT)),time=time.time()));raise
