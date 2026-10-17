"""Resume only missing readouts; known two retired prefixes get exact frozen replay."""
from pathlib import Path
import gc,hashlib,inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_pairwise_caption256_v1 as original
BASE=original.BASE;read=original.read;write=original.write;sha=original.sha
REC=BASE/'recovery/readout002';CACHE={}
def verify():
    original.verify()
    for p,h in read(REC/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
def init():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    assert read(BASE/'FAILURE.json')['status']=='technical_failure';REC.mkdir(parents=True,exist_ok=False)
    rr=read(BASE/'ROSTER_LOCK.json')['records'];partial=[]
    missing=[r['probe_alias']for r in rr if not (ROOT/'artifacts/stvg_pipeline_speed_v1/source_features'/(r['probe_alias']+'.receipt.json')).exists()]
    assert sorted(missing)==sorted(read(original.B2/'METHOD_LOCK.json')['missing_prefix_cases'])
    for r in rr:
        p=BASE/'predictions'/(r['probe_alias']+'.pt')
        if not p.exists():continue
        c,_=load_case(r);v=torch.load(p,map_location='cpu',weights_only=False);assert v['case_id']==r['probe_alias']and v['dataset']==r['dataset']and np.array_equal(v['arms']['Frozen']['boxes'],c['native_boxes_xyxy_normalized'])
        for kind in ['R1','QWEN']:
            f=torch.load(BASE/kind/'fits'/(r['probe_alias']+'.pt'),map_location='cpu',weights_only=False);assert f['source_case_sha256']==c['source_case_sha256'];assert all(torch.equal(a,b)for a,b in zip(v['actual_integrated_head_logits'][kind],f['fit']['final_logits']))
        partial.append(dict(case_id=r['probe_alias'],path=str(p.relative_to(ROOT)),sha256=sha(p)));del c,v;gc.collect()
    assert len(partial)==49
    write(REC/'PARTIAL_REUSE_LOCK.json',dict(status='pass',count=49,files=partial,actual_prior_outputs_fits_Source_exact=True,GT_read=False));write(REC/'METHOD_LOCK.json',dict(status='locked_before_resume',count=256,remaining=207,missing_prefix_cases=missing,missing_matches_original_B2_prelock=True,only_change='readout loader uses established exact frozen-prefix fallback for two retired Source feature files',no_new_Teacher_or_fits=True,all_partial_outputs_preserved=True,deadline_inherited=True,GT_read=False))
    ps=[Path(__file__),ROOT/'scripts/continue_stvg_pairwise_caption256_recovery002.py',REC/'PARTIAL_REUSE_LOCK.json',REC/'METHOD_LOCK.json',BASE/'ACTIVE_EXECUTION.json']+[BASE/k/'FIT_BARRIER.json'for k in ['R1','QWEN']]
    write(REC/'EXECUTION_RUNTIME.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in ps},GT_read=False))
def cached_source(model,c):
    import numpy as np,torch
    from scripts import stvg_temporal_semantic_grounder32_v1 as old,stvg_factorial_native256_v1 as rgb
    from scripts.stvg_pipeline_speed_v1 import to_device
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    cid=c['case_id']
    if cid not in read(REC/'METHOD_LOCK.json')['missing_prefix_cases']:return old.cached_source(model,c)
    if cid in CACHE:
        views=to_device(CACHE.pop(cid),next(model.parameters()).device);base=SpatialReplay(model,views,len(c['frame_ids']))
    else:
        rgb.BASE=REC/'fresh_prefix';rgb.BASE.mkdir(exist_ok=True);req=read(ROOT/'artifacts/stvg_spatial_controls256_v1/temporal/requests'/(cid+'.json'));rgb.CURRENT_CASE=c;pixels=rgb.native_rgb(req,set(map(int,c['frame_ids'])));frames=np.stack([pixels[int(i)]for i in c['frame_ids']]);assert hashlib.sha256(frames.tobytes()).hexdigest()==c['pixel_sha256'];rec=next(r for r in read(BASE/'ROSTER_LOCK.json')['records']if r['probe_alias']==cid);row=old.row_for(rec);assert model._fixed_full_state_hash==c['source_model_state_sha256']
        with torch.inference_mode():_,records,base=frozen_forward(model,frames,row)
        assert records==c['records'];CACHE[cid]=to_device(base.views,'cpu');write(REC/'fresh_prefix'/(cid+'.json'),dict(status='pass',fresh_frozen_Source_prefix=True,source_pixel_sha256=c['pixel_sha256'],GT_read=False))
    assert torch.equal(base.zero['boxes'].cpu(),c['zero']['boxes'])and all(torch.equal(z.cpu(),v)for z,v in zip(base.zero['logits'],c['zero']['logits']))and all(torch.equal(z[-1,0].cpu(),v)for z,v in zip(base.temporal_inputs,c['temporal_hidden']))
    return base
def readout():
    verify();partial={p['case_id']:p for p in read(REC/'PARTIAL_REUSE_LOCK.json')['files']}
    def reuse(cid):
        p=partial.get(cid)
        if p is not None:assert sha(ROOT/p['path'])==p['sha256']
        return p
    src=inspect.getsource(original.readout).replace('from scripts.stvg_temporal_semantic_grounder32_v1 import cached_source','from scripts.stvg_pairwise_caption256_recovery002 import cached_source')
    needle="verify();cid=r['probe_alias'];c,_=load_case(r);src=";assert src.count(needle)==1
    src=src.replace(needle,"verify();cid=r['probe_alias'];oldfile=reuse(cid)\n        if oldfile is not None:\n            files.append(oldfile);continue\n        c,_=load_case(r);src=")
    ns=dict(original.__dict__,verify=verify,reuse=reuse);exec(compile(src,__file__,'exec'),ns);ns['readout']();assert not CACHE
score=original.score
if __name__=='__main__':globals()[sys.argv[1]]()
