"""CPU-only official-source, sampling, posterior, pool and no-label audit.

No checkpoint model construction, RGB inference, optimizer or GPU operation.
Synthetic targets below test the upstream dead-read removal, not task quality.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional
import urllib.request

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R/'tools'), str(Path(__file__).parent)]
from full import atomic, sha
import cgstvg_interfaces as cg

VENDOR = R.parent/'outputs/maintenance/engineering_speed_v1/baseline_workspace/external/CGSTVG'
REVISION = 'f062d84b72aee8be4425a4a5ab20aa5b077ca142'
D = R/'outputs/paper_review_suite_v1'
P = R/'results/paper_review_suite_v1'


def functions(path, cls=None, names=()):
    tree = ast.parse(path.read_text())
    body = next(x.body for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls) if cls else tree.body
    return [x for x in body if isinstance(x, ast.FunctionDef) and x.name in names]


def main():
    import torch
    from torch import nn, Tensor
    import numpy as np
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    tests = []
    # Git blob IDs verify every archived upstream source against one official
    # immutable tree; there is no dependency on the moving main branch.
    url = f'https://api.github.com/repos/HengLan/CGSTVG/git/trees/{REVISION}?recursive=1'
    req = urllib.request.Request(url, headers={'User-Agent':'STVG-source-audit'})
    tree = json.load(urllib.request.urlopen(req, timeout=60))
    assert not tree.get('truncated')
    upstream = {e['path']:e for e in tree['tree'] if e['type']=='blob'}
    checked = {}
    for path in sorted(VENDOR.rglob('*')):
        if not path.is_file() or path.suffix not in {'.py','.yaml','.txt','.md'}:
            continue
        name = path.relative_to(VENDOR).as_posix()
        content = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert name in upstream and blob == upstream[name]['sha'], name
        checked[name] = dict(git_blob=blob, sha256=sha(path), bytes=len(content))
    missing = [n for n in upstream if Path(n).suffix in {'.py','.yaml','.txt','.md'} and n not in checked]
    assert not missing, missing
    tests.append('all_archived_official_code_and_config_bytes_match_pinned_upstream')
    weights = {}
    for name, expected in {
        'hcstvg2.pth':'020d77b1013e33d9ae58f01c5b60e77ba6eb3d05e04077a94fa8e4247d635a88',
        'vidstg.pth':'cbf3ee4036dab88de3314dc3a520ab0eb2b4d292a2796d12d0f3493c39eed9ef',
    }.items():
        path = R.parent/'checkpoints/cgstvg'/name
        assert path.stat().st_size == 3548827981 and sha(path) == expected
        weights[name] = dict(bytes=path.stat().st_size, sha256=expected)
    tests.append('both_official_cross_domain_checkpoint_file_hashes')

    # Exact original sampling functions run on synthetic metadata, no annotation
    # file and no RGB. All synthetic fields remain private CPU fixtures.
    from copy import copy
    from types import SimpleNamespace
    ns = dict(np=np, copy=copy)
    path = VENDOR/'datasets/data_utils.py'
    exec(compile(ast.Module(body=functions(path, names={'make_hcstvg_input_clip','make_vidstg_input_clip'}), type_ignores=[]), str(path), 'exec'), ns)
    cfg = SimpleNamespace(INPUT=SimpleNamespace(SAMPLE_FPS=3.2, TRAIN_SAMPLE_NUM=64))
    for ds in ['hc2','vidstg']:
        for n in [23,128,129,601,1000]:
            ids = list(range(n-1)) if ds=='hc2' else list(range(21,21+n))
            fixture = dict(item_id=0, vid='', width=2, height=2, description='', object=None,
                           bboxs=np.zeros((len(ids),4)), gt_temp_bound=[ids[0],ids[-1]],
                           frame_ids=ids, actioness=np.ones(len(ids)), start_heatmap=np.ones(len(ids)),
                           end_heatmap=np.ones(len(ids)), frame_count=n, qtype='', segment_bound=[ids[0],ids[-1]])
            native = ns['make_hcstvg_input_clip' if ds=='hc2' else 'make_vidstg_input_clip'](cfg, 'test', fixture)['frame_ids']
            meta = dict(frame_count=n, start_frame=ids[0], end_frame=ids[-1])
            assert cg.frame_grid(meta,ds) == native
    tests.append('10_official_test_frame_sampling_fixtures_exact_no_GT_dependency')

    # Original native postprocessor temporal readout, not TA hull readout.
    path = VENDOR/'models/post_processor.py'
    ns = dict(torch=torch, nn=nn, F=torch.nn.functional, box_cxcywh_to_xyxy=lambda x:torch.cat((x[...,:2]-x[...,2:]/2,x[...,:2]+x[...,2:]/2),-1))
    tree = ast.parse(path.read_text()); cls = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PostProcess')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),ns)
    torch.manual_seed(20261024)
    for n in [8,16,129]:
        ids = (np.arange(n)*3).tolist()
        z = torch.randn(1,n,2)
        _, interval = ns['PostProcess']()(dict(pred_sted=z,pred_boxes=torch.ones(n,4)*.5),torch.ones(n,2),[ids],[n])
        assert cg.native_map_interval(z,ids) == interval[0]
        q = cg.last_layer_posterior(z)
        v=z[0].double().numpy(); raw=np.exp(v[:,0,None]+v[None,:,1]); raw=np.triu(raw,1);raw/=raw.sum()
        np.testing.assert_allclose(q,raw,atol=2e-16,rtol=2e-14)
    tests.append('3_official_MAP_readouts_exact_and_single_grid_posterior_numeric_reference')

    from stvg.representation import extract
    x=torch.randn(10,7,3,5);valid=torch.ones(10,3,5,dtype=torch.bool);valid[:,-1,-1]=False
    box=np.tile([.5,.5,.4,.7],(10,1))
    old,_=extract(dict(frame_ids=list(range(10)),native_local_grid=[(x[::2].clone(),valid[::2].clone()),(x[1::2].clone(),valid[1::2].clone())]),box)
    assert np.array_equal(cg.pool_single_grid(x,valid,box),old)
    tests.append('single_grid_overlap_pool_byte_exact_to_existing_math_without_offset_assumption')

    # Execute the actual official PosDecoder.forward with synthetic operators;
    # real model/weights are deliberately NOT constructed in this audit.
    path=VENDOR/'models/grounding_model/query_decoder.py'
    ns=dict(torch=torch,Tensor=Tensor,Optional=Optional,gen_sineembed_for_position=lambda b:b.repeat(1,1,64))
    exec(compile(ast.Module(body=functions(path,'PosDecoder',{'forward','gt_info'}),type_ignores=[]),str(path),'exec'),ns)
    class Layer(nn.Module):
        def forward(self,query_tgt,**kwargs):return query_tgt+.1,None
    Fixture=type('SyntheticOfficialCG',(nn.Module,),dict(forward=ns['forward'],gt_info=ns['gt_info']))
    decoder=Fixture();decoder.layers=nn.ModuleList([Layer()]);decoder.ref_point_head=nn.Identity();decoder.query_scale=nn.Identity();decoder.bbox_embed=nn.Linear(256,4);decoder.conf=nn.Linear(256,1);decoder.conf2=nn.Linear(256,1);decoder.norm=nn.Identity();decoder.d_model=256;decoder.return_intermediate=True;decoder.return_weights=False;decoder.theta_s=.5;decoder.theta_t=.5
    decoder.generate_roi_feature=lambda feature,*args:feature.mean(0)
    decoder.get_context_index_by_our=lambda conf,*args:torch.arange(len(conf))
    decoder.generate_context=lambda a,b,index:torch.cat((a[index],b[index]))
    decoder.eval().requires_grad_(False)
    kw=dict(query_tgt=torch.zeros(4,1,256),pred_boxes=torch.ones(4,1,4)*.5,query_time=torch.zeros(4,1,256),query_mask=torch.zeros(1,4,dtype=torch.bool),encoded_feature=torch.ones(2,4,256),encoded_pos=torch.zeros(2,4,256),encoded_mask=torch.zeros(4,2,dtype=torch.bool),feature_size=(1,1,None),iteration_rate=-1,time_info=torch.ones(4),targets=[dict(eval=True,boxs=torch.zeros(4,4),durations=[4],actioness=torch.ones(4))])
    expected=decoder(**kw);original=type(decoder).forward;kw['targets']=cg.ForbidLabels()
    with cg.inference_only(decoder) as patch:
        actual=decoder(**kw)
        assert all(torch.equal(a,b) for a,b in zip(expected,actual))
        try:decoder(**{**kw,'iteration_rate':0})
        except AssertionError:pass
        else:raise AssertionError('Training gate failed')
    assert 'forward' not in decoder.__dict__ and type(decoder).forward is original
    try:decoder(**kw)
    except AssertionError:pass
    else:raise AssertionError('Original read was not restored')
    tests.append('official_dead_GT_read_bypass_exact_synthetic_outputs_eval_guard_and_restore')
    assert not torch.cuda.is_initialized()
    report=dict(status='CPU_interface_audit_passed_NOT_model_bridge',official_repository='https://github.com/HengLan/CGSTVG',paper='https://arxiv.org/abs/2401.01578',revision=REVISION,official_source_files=checked,checkpoints=weights,tests=tests,tests_count=len(tests),dead_GT_read_patch=patch,model_constructed=False,model_forward=0,expert_forward=0,GT_files_read=0,CG_GPU_started=False,real_checkpoint_noop='PENDING_until_all_TA_studies_complete',cross_domain_weights=dict(hc2_target='vidstg.pth',vidstg_target='hcstvg2.pth'),code_sha256={p.name:sha(p)for p in [Path(__file__),Path(cg.__file__)]})
    atomic(D/'cgstvg_cpu_interface_audit.json',report);atomic(P/'cgstvg_cpu_interface_audit.json',report)
    print(json.dumps(dict(status=report['status'],tests=len(tests),official_files=len(checked),GPU_started=False)))


if __name__=='__main__':main()
