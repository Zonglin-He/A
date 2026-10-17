"""Use the existing exact-manifest/CAS/content verification public workflow."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_coverage224_v1 import BASE
def main(group,mode,pixel=None):
    if mode=='close':
        from scripts import close_stvg_referent_abc_publication_v1 as close
        source=inspect.getsource(close.main);assert source.count('count=16')==1;source=source.replace('count=16','count='+str(384 if group=='decomposition' else 256));ns=dict(close.__dict__);ns['BASE']=BASE;exec(compile(source,__file__,'exec'),ns);return ns['main'](group,pixel)
    from scripts import publish_stvg_controls256_v1 as old
    old.SCOPES['global32']=BASE.name;return old.main('global32',group,mode,pixel)
if __name__=='__main__':main(sys.argv[1],sys.argv[2],sys.argv[4] if len(sys.argv)>4 else None)
