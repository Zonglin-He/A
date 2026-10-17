"""Receipt namespace recovery only; no changed support, update, output or original pin."""
from pathlib import Path
import inspect,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_semantic_preference_4anchor32_v1 as original
BASE=original.BASE
read,write,sha=original.read,original.write,original.sha

def verify():
    r=read(BASE/'EXECUTION_RUNTIME002.json')
    for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
    assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch'],'same original four-anchor deadline'
    return r

def run(stage,arm=None):
    verify();original.verify=verify
    if stage=='worker':
        s=original.study();src=inspect.getsource(s.worker);assert src.count("'semantic_pref32_' + arm")==1
        src=src.replace("'semantic_pref32_' + arm","'semantic_pref4anchor32_' + arm")
        ns=dict(s.__dict__);exec(compile(src,__file__,'exec'),ns);return ns['worker'](arm)
    if stage=='lock':
        for arm in ['GEOM','PREF']:
            p=BASE/'arms'/arm/'trials/G32/RUNTIME.json';r=read(p)
            # Preserve the original immutable trial lock; independent root check
            # additionally verifies this addendum without mutating that lock.
            for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
        return
    if stage=='score':
        s=original.study();oldwrite=s.write
        def labelled_write(path,data,**kw):
            if Path(path).name=='POSTSEAL_RESULTS.json':data=dict(data,scope='exposed32 four-anchor fixed-support matched comparison; not independent confirmation')
            return oldwrite(path,data,**kw)
        s.write=labelled_write
        return s.score()
    return original.run(stage,arm)
if __name__=='__main__':run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
