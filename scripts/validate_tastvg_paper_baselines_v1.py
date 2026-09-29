"""Exclusive-GPU two-old-fixture integration for persistent TENT/SAR/ViTTA ports."""
import sys,time,gc,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,save,sha
OUT=ROOT/'artifacts/tastvg_paper_matrix_v1/baseline_smoke'

def run():
    tick=time.monotonic()
    assert read(ROOT/'artifacts/tastvg_full_b1_v1/COMPLETION.json')['status']=='completed'
    import torch,numpy as np
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from methods.decota_final_simplified_v1.tensors import state_hash
    from methods.decota_final_simplified_v1.backbone import make_batch
    from vg_tta.native_baselines_paper_v1 import live_output
    from vg_tta.vitta_paper_v1 import live_views,decoder_scope,validate_source,FEATURE_LAYERS
    from vg_tta.tastvg_baseline_expansion import decoder_layernorm_scope
    from vg_tta.tastvg_paper_online_baselines_v1 import OnlineOptimizer
    from vg_tta.exact_frame_decode_audit_v2 import decode
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)) and any(s in str(args[0]) for s in ['GT_SUBSET','labels_diagnostic','/ROWS.json','/SUMMARY.json']):raise PermissionError('Engineering smoke forbids GT/scores')
    sys.addaudithook(guard);install_clean_loader();torch.set_num_threads(4);torch.manual_seed(20260930);np.random.seed(20260930);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    with lease():
        model=model_load('hcstvg1_test').eval().requires_grad_(False);initial=state_hash(model.state_dict())
        p=read(ROOT/'artifacts/tastvg_schedule_j01_v1/LOCK.json');seq=p['orders']['order1'][:2];rows=[next(r for r in p['rows'] if r['ordinal']==i) for i in seq]
        source_dir=ROOT/'artifacts/decota_paper_execution_20260917/vitta/vid_to_hc1/source'
        receipt=read(source_dir/'COMPLETION.json');assert sha(source_dir/'statistics.pt')==receipt['statistics_sha256']
        source=load(source_dir/'statistics.pt');validate_source(source,checkpoint_sha256=read(ROOT/'methods/tastvg_dual_evidence_j0_v1/config.json')['source_checkpoint_sha256'],source_dataset='vidstg',layer_names=FEATURE_LAYERS)
        audits=[]
        for method in ['TENT','SAR','ViTTA']:
            if method=='ViTTA':names,params=decoder_scope(model)
            else:names,params,_=decoder_layernorm_scope(model)
            policy=OnlineOptimizer(params,method,source=source if method=='ViTTA' else None)
            try:
                for arrival,row in enumerate(rows):
                    assert time.monotonic()-tick<900
                    frames,ids=decode(row['input']);batch=make_batch(frames,ids,row['input'],model);subject=row['parses']['subject']
                    def inference():return live_output(model,batch,ids,subject)[1]
                    def closure():return live_views(model,batch,ids,subject)[0] if method=='ViTTA' else live_output(model,batch,ids,subject)[0]
                    # Source no-update equality uses original flags, not tolerance on logits.
                    if arrival==0:
                        cache=load(ROOT/'artifacts/tastvg_spatial_expansion_s0_v1/capture/clean'/f"{row['ordinal']:03}.pt")
                        with torch.no_grad():baseline=inference()
                        assert torch.equal(baseline['boxes'],cache['prediction']['boxes'])
                    pre,post,a=policy.arrive(closure,inference,ids);assert a['arrival']==arrival+1
                    audits.append(dict(method=method,arrival=arrival,parameters=sum(x.numel() for x in params),**{k:v for k,v in a.items() if k not in ['method','arrival']}))
                    save(OUT/method/f'{arrival:02}.pt',dict(pre=pre,post=post,audit=a,state=policy.state_dict(),GT_read=False))
                    del frames,batch,pre,post;gc.collect();torch.cuda.empty_cache()
            finally:policy.reset()
            assert state_hash(model.state_dict())==initial
        write(OUT/'AUDIT.json',dict(status='pass',methods=['TENT','SAR','ViTTA'],source_state_exact=True,GT_read=False,new_query_exposure=False,records=audits,seconds=time.monotonic()-tick,scope='Two exposed fixtures and persistent state; not full baseline results'))
if __name__=='__main__':run()
