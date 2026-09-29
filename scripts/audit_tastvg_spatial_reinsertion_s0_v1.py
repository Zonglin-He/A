"""Replay two GT-blind nonempty-evidence cells and verify genuine edited-H full reinsertion."""
import sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_spatial_expansion_s0_v1 import OUT,verify,observation

def run():
    import torch,numpy as np
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from scripts.run_final_simplification_v1 import lease
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.tastvg_spatial_expansion_s0_v1 import expand
    from vg_tta.tastvg_temporal_fourarm_v1 import reinsert
    start=time.monotonic();guard=lease();p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');hb=read(OUT/'H_BARRIER.json');eb=read(OUT/'EXPERT_BARRIER.json')
    chosen=[]
    for r in p['rows']:
        for cond in ['clean','frame_drop_5']:
            rel=f"expert/{cond}/{r['ordinal']:03}.pt";assert sha(OUT/rel)==eb['files'][rel]
            if load(OUT/rel)['valid'].any():chosen.append((r,cond))
            if len(chosen)==2:break
        if len(chosen)==2:break
    assert len(chosen)==2
    torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;install_clean_loader()
    from scripts.run_spatial_regression_alignment_v1 import model_load
    model=model_load('hcstvg1_test');model.eval().requires_grad_(False);mh=state_hash(model.state_dict());rows=[]
    for r,cond in chosen:
        name=f"{r['ordinal']:03}.pt";hrel=f'capture/{cond}/{name}';rel=f'ta/{cond}/{name}';assert sha(OUT/hrel)==hb['files'][hrel] and sha(OUT/rel)==bar['files'][rel]
        data=device_tree(load(OUT/hrel),'cuda');saved=load(OUT/rel);assert mh==saved['model_state_sha256']
        actual,fields,ev=expand(model,data,load(OUT/'expert'/cond/name))
        assert actual['diagnostics'][-1]['relative_to_native']>0
        for a,b in zip(actual['trajectory'],saved['trajectory']):
            assert torch.equal(a['prediction']['boxes'],b['prediction']['boxes'])
            assert all(torch.equal(x,y) for x,y in zip(a['prediction']['logits'],b['prediction']['logits']))
        frames,_=decode(r['input']);shifted,_=observation(r,cond,frames);receipt=reinsert(model,shifted,r,fields,ev)
        rows.append(dict(parent=r['ordinal'],condition=cond,**receipt,relative_to_native=actual['diagnostics'][-1]['relative_to_native'],saved_trajectory_exact=True));del data,actual,fields,ev,saved,frames,shifted;gc.collect();torch.cuda.empty_cache()
    assert state_hash(model.state_dict())==mh;seconds=time.monotonic()-start
    write(OUT/'REINSERTION_AUDIT.json',dict(status='pass',GT_read=False,selection='First two clean/drop cells with nonempty evidence in locked roster order',cells=rows,additional_replay_backward_calls=6,seconds=seconds))
    prior=sum(read(f)['seconds'] for f in (OUT/'allocations').glob('*.json'));assert prior+seconds<p['cap_seconds']
    write(OUT/'allocations'/f'{time.time_ns()}.json',dict(stage='reinsertion_audit',status='completed',done=2,seconds=seconds,prior_seconds=prior,failure=None,runner_sha256=sha(Path(__file__)),time=time.time()));guard.close();print('EDITED REINSERTIONS VERIFIED',rows)

if __name__=='__main__':run()
