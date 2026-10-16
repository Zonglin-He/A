"""Extend established exact-manifest publication closure to native-eight scope."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import publish_stvg_overnight_group_v1 as old
def main(group,mode,pixel=None):
    if group=='closing':
        if mode=='close':return close_custom(group,128,pixel)
        old.old.SCOPES['global32']='stvg_overnight_consolidation_20261016';return old.old.main('global32',group,mode,pixel)
    if group!='tubedetr_qualification8':return old.main(group,mode,pixel)
    if mode=='close':
        from scripts import close_stvg_referent_abc_publication_v1 as close
        src=inspect.getsource(close.main);assert src.count('count=16')==1;src=src.replace('count=16','count=8');ns=dict(close.__dict__);ns['BASE']=ROOT/'artifacts/stvg_overnight_consolidation_20261016';exec(compile(src,__file__,'exec'),ns);return ns['main'](group,pixel)
    old.old.SCOPES['global32']='stvg_overnight_consolidation_20261016';return old.old.main('global32',group,mode,pixel)
def close_custom(group,count,pixel):
    from scripts import close_stvg_referent_abc_publication_v1 as close
    src=inspect.getsource(close.main);assert src.count('count=16')==1;src=src.replace('count=16','count='+str(count));ns=dict(close.__dict__);ns['BASE']=ROOT/'artifacts/stvg_overnight_consolidation_20261016';exec(compile(src,__file__,'exec'),ns);return ns['main'](group,pixel)
if __name__=='__main__':main(sys.argv[1],sys.argv[2],sys.argv[4] if len(sys.argv)>4 else None)
