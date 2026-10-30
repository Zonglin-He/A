"""R10 source-only forward and input observations; no GT or pseudo-target API."""
from pathlib import Path
import argparse, hashlib, json, os, random, sys, time
import numpy as np
D=Path(__file__).resolve().parent
ROOT=D.parents[1]
sys.path[:0]=[str(D.parent/'round9'),str(D.parent/'round7'),str(D.parent/'round7/experiment')]
from run_diagnostic import load_model, mask_for, fingerprint, save_source_cache, STRICT_FLAGS
from runner_helpers import sha, read, validate_manifest, sample_points, array_sha

def main(a):
    a.plan=a.plan.resolve();a.manifest=a.manifest.resolve();a.output=a.output.resolve()
    assert sha(a.plan)==a.plan_sha256
    plan=read(a.plan)
    assert plan['stage']=='source_capture' and plan['GPU_execution_authorized']
    assert plan['strict_FP32_flags']==STRICT_FLAGS
    assert sha(a.manifest)==plan['manifest_sha256']
    rows=validate_manifest(read(a.manifest))
    assert len(rows)==plan['expected_samples']
    for name,digest in plan['capture_program_sha256'].items():
        assert sha(ROOT/name)==digest, name
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
    (a.output/'manifest_input.json').write_bytes(a.manifest.read_bytes())
    (a.output/'observations').mkdir()
    import torch
    from regression_adapter import capture_source,parameter_sha
    assert plan['seed']==0
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    start=time.perf_counter();model,entries=load_model(plan)
    baseline=fingerprint(model);head=parameter_sha(model)
    files={};traces=[];pointnames=[]
    for i,row in enumerate(rows):
        tick=time.perf_counter();pc,ix,n=sample_points(row)
        text=row['caption']+plan['sentinel'];mask,info=mask_for(model,text,plan['noun_fallback'])
        torch.cuda.reset_peak_memory_stats()
        cache=capture_source(model,{'point_clouds':torch.from_numpy(pc)[None].cuda(),'text':[text]},mask)
        torch.cuda.synchronize()
        assert cache.query_index==int(cache.text_scores[0].argmax())
        rel='source_cache/'+row['id']+'.npz'
        save_source_cache(a.output/rel,cache,sample_id=row['id'],point_sha=array_sha(pc),index_sha=array_sha(ix),model_text=text,manifest_row=row,checkpoint_sha=plan['source_checkpoint_sha256'])
        files[rel]=sha(a.output/rel)
        obs='observations/'+row['id']+'.npz'
        np.savez_compressed(a.output/obs,sampled_points_model=pc,raw_indices=ix,
                            point_sample_sha256=np.frombuffer(bytes.fromhex(array_sha(pc)),np.uint8),
                            raw_indices_sha256=np.frombuffer(bytes.fromhex(array_sha(ix)),np.uint8))
        files[obs]=sha(a.output/obs);pointnames.append(obs)
        if i==0:
            repeat=capture_source(model,{'point_clouds':torch.from_numpy(pc)[None].cuda(),'text':[text]},mask)
            assert torch.equal(cache.boxes_raw6,repeat.boxes_raw6)
            assert torch.equal(cache.text_scores,repeat.text_scores)
        traces.append({'id':row['id'],'platform':row['platform'],'scene':row['scene'],
                       'q0':cache.query_index,'raw_points':n,'unique_sample_indices':len(np.unique(ix)),
                       'text_readout':info,'seconds':time.perf_counter()-tick,
                       'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                       'cache_sha256':files[rel],'observation_sha256':files[obs],
                       'point_file_sha256':sha(row['pointpath']),
                       'image_sha256':sha(row['imagepath']),
                       'camera_container_sha256':sha(row['metapath'])})
        assert parameter_sha(model)==head
        if i%20==0 or i+1==len(rows):
            print(json.dumps({'completed':i+1,'n':len(rows),'elapsed_seconds':time.perf_counter()-start}),flush=True)
    assert fingerprint(model)==baseline
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not m.training for m in model.modules())
    (a.output/'trace.json').write_text(json.dumps(traces,indent=2)+'\n')
    for name in ['trace.json','registered_plan.json','manifest_input.json']:files[name]=sha(a.output/name)
    seal={'state':'sealed_R10_source_forward_and_input_observations_no_G_no_GT',
          'n_samples':len(rows),'n_source_cache_files':len(rows),'n_observation_files':len(rows),
          'manifest_sha256':sha(a.manifest),'plan_sha256':sha(a.plan),
          'source_checkpoint_sha256':plan['source_checkpoint_sha256'],
          'source_model_state_fingerprint':baseline,'strict_state_entries':entries,
          'strict_FP32_flags':STRICT_FLAGS,'source_repeat_bitexact':True,
          'source_unchanged':True,'files':files,'elapsed_seconds':time.perf_counter()-start}
    (a.output/'SEALED.json').write_text(json.dumps(seal,indent=2)+'\n')
    print('R10_SOURCE_SEALED',sha(a.output/'SEALED.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['plan','manifest','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True)
    main(p.parse_args())
