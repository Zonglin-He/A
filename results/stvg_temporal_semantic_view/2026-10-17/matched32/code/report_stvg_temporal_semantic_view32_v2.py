"""Public numeric cost field alias; private-text denylist remains unchanged."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_temporal_semantic_view32_v1 as old
BASE,OUT=old.BASE,old.OUT

def public_write(path,data,**kw):
    if Path(path).name=='COST.json':
        def numeric_alias(x):
            if isinstance(x,dict):
                if 'caption' in x:assert isinstance(x['caption'],(int,float))
                return {('visual_description' if k=='caption' else k):numeric_alias(v) for k,v in x.items()}
            if isinstance(x,list):return [numeric_alias(v) for v in x]
            return x
        data=numeric_alias(data)
    return old.write(path,data,**kw)
def report():
    if OUT.exists():
        dest=BASE/'recovery/public_numeric_alias001/partial_export';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();OUT.rename(dest)
    src=inspect.getsource(old.report);ns=dict(old.__dict__);ns['write']=public_write;exec(compile(src,__file__,'exec'),ns);ns['report']()
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    p=OUT/'code'/Path(__file__).name;p.write_text(public_source_text(Path(__file__).read_bytes()));m=BASE/'publication/matched32/EXPORT_MANIFEST.json';d=old.read(m);paths=set(d['paths'])|{str(p.relative_to(ROOT))};verify_privacy(paths);old.write(m,dict(d,paths=sorted(paths),files=[dict(path=p,sha256=old.sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]),mutable=True)
if __name__=='__main__':report()
