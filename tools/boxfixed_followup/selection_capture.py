"""Inspect the exact frozen execution bank without modifying prior source files."""
from tools.boxfixed_tube.profiles import predict as original_predict
from tools.boxfixed_tube.spatial import evidence_dp
from tools.boxfixed_followup import refinement


def capture_bank(base,evidence,source,raw,interval,p):
    if p['mode']=='original':
        q={k:p[k]for k in ['gamma','admission','guard','link']}
        # Capture the scoring view from the unmodified original predictor.
        import tools.boxfixed_tube.profiles as module
        function=original_predict
    else:
        module=refinement;function=refinement.predict
    saved=module.BoxFixedBank;seen=[]
    def hook(*args,**kwargs):
        b=saved(*args,**kwargs);seen.append(b);return b
    try:
        module.BoxFixedBank=hook
        r=function(base,evidence,source,raw,interval,p if p['mode']!='original' else q)
    finally:
        module.BoxFixedBank=saved
    assert len(seen)==1
    b=seen[0];exact=evidence_dp(b)
    assert exact['score']==r['score'] and exact['actions'].equal(r['actions']) and exact['final'].equal(r['final'])
    return b,r
