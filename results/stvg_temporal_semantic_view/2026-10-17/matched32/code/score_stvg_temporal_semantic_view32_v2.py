"""Metric-support correction only; sealed outputs unchanged, no repeated models."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import score_stvg_temporal_semantic_view32_v1 as old
BASE=old.BASE

def same_write(p,d,**kw):
    if Path(p).exists():assert old.read(p)==d;return
    return old.write(p,d,**kw)
def main():
    src=inspect.getsource(old.main)
    needle="            if name=='Native':native_s=v['m_sIoU_dense_GT']\n            elif name not in ['SEM_FULL_DIRECT_OPD','T2_R1_DIRECT_OPD']:assert v['m_sIoU_dense_GT']==native_s"
    assert src.count(needle)==1
    src=src.replace(needle,"            cv,_=metric._score_student_arm(dict(boxes=pred['boxes'],interval=preds['Native']['interval']),c,targets[cid],support,official,independent)\n            v['m_sIoU_common_Native_clock_GT']=cv['m_sIoU_dense_GT']\n            if name=='Native':native_s=cv['m_sIoU_dense_GT']\n            elif name not in ['SEM_FULL_DIRECT_OPD','T2_R1_DIRECT_OPD']:assert cv['m_sIoU_dense_GT']==native_s")
    ns=dict(old.__dict__);ns['write']=same_write;exec(compile(src,__file__,'exec'),ns);return ns['main']()
if __name__=='__main__':main()
