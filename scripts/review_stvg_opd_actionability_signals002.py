"""Separate independent all1024 test-visible signal readback; no GT or inference."""
import os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_evidence_actionability_audit_v1 import BASE,read,write,load,sha,require_seal
from scripts.audit_stvg_opd_actionability_public_v1 import near
def independent(z,j):
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    training=z['anchors'][j];held=[a for a in z['anchors'][1-j] if a['position'] not in set(z['visited'][j])]
    clip=z['dataset']=='hc2'
    def boxes(x):
        b=xyxy(x,1,1);return np.maximum(b,0) if clip else b
    def agreement(pred,anchors):
        if not anchors:return None
        a=boxes(np.asarray(pred)[[v['position'] for v in anchors]])
        b=boxes(np.array([v['box'] for v in anchors]));return float(box_iou(a,b).sum()/len(anchors))
    before=z['before'];after=z['after'][j];tb=agreement(before,training);ta=agreement(after,training);hb=agreement(before,held);ha=agreement(after,held)
    if len(training)>1:
        b=boxes(np.array([a['box'] for a in training]));mat=box_iou(b[:,None,:],b[None,:,:]);ii,jj=np.triu_indices(len(b),1);cons=float(mat[ii,jj].sum()/len(ii))
    else:cons=None
    displacement=after-before;norm=np.sqrt(np.sum(displacement*displacement,axis=1))
    # Match the sealed original float32 mean, not a promoted float64 quotient.
    score=-float(np.divide(norm.sum(dtype=np.float32),np.float32(len(norm)),dtype=np.float32))
    return dict(holdout_anchors=len(held),training_anchors=len(training),holdout_before_agreement=hb,holdout_after_agreement=ha,
        holdout_agreement_gain=None if hb is None else ha-hb,detector_score=float(sum(a['score'] for a in training)/len(training)) if training else None,
        before_training_agreement=tb,training_agreement_gain=None if tb is None else ta-tb,teacher_geometric_consistency=cons,negative_output_displacement=score)
def run():
    import torch
    torch.set_num_threads(2);b=require_seal();assert not torch.cuda.is_initialized();checks=0
    for path,digest in b['files'].items():
        assert sha(ROOT/path)==digest;z=load(ROOT/path)
        for j in [0,1]:checks+=near(independent(z,j),z['signals'][j])
        assert sha(ROOT/path)==digest
    assert not torch.cuda.is_initialized()
    write(BASE/'ROOT_EXTERNAL_SIGNAL_READBACK.json',dict(status='pass',scope='all1024 test-visible signals independently recomputed with original box geometry helper, vectorized held-out pairing/pairwise consistency/output displacement; all512 readout double hashes; no GT/model/GPU',saved_updates=1024,cells=512,independent_scalar_checks=checks,second_hash_checks=512,review_source_sha256=sha(Path(__file__)),new_GPU_fits=0,new_model_forward_calls=0,new_DINO_calls=0,CPU_only=True,GT_read=False,time=time.time()))
    print('P9_ALL1024_INDEPENDENT_SIGNALS_PASS',checks)
if __name__=='__main__':run()
