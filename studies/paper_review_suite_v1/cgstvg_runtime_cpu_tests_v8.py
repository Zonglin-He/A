"""CPU-only runtime-contract tests; never construct a CG checkpoint model."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import ast
import copy
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R),str(R/'tools'),str(Path(__file__).parent)]
from full import atomic,sha
import cgstvg_runtime_v8 as runtime


def main():
    import numpy as np
    import torch
    from torchvision.transforms import ToTensor,ToPILImage,Resize
    from torchvision.transforms.functional import normalize
    from stvg.temporal import final_mbr
    from stvg.positions import observation_positions
    from cgstvg_interfaces import last_layer_posterior
    from cgstvg_score import native_official_metric
    from stvg.metrics import DenseTube
    torch.set_num_threads(1);tests=[]
    study=Path(__file__).with_name('cgstvg_study_v8.py').read_text()
    controller=next(n for n in ast.parse(study).body if isinstance(n,ast.FunctionDef)and n.name=='controller')
    constants={n.value for n in ast.walk(controller)if isinstance(n,ast.Constant)and isinstance(n.value,str)}
    assert {'cgstvg_v8_screen_sealed.json','cgstvg_v8_screen_score_receipt.json'}<=constants
    assert 'cgstvg_screen_sealed.json'not in constants and 'cgstvg_screen_score_receipt.json'not in constants
    tests.append('controller_confirm_reads_its_own_v3_screen_receipts')
    assert not torch.cuda.is_initialized()and 'models.pipeline'not in sys.modules
    import tempfile
    original_gate=runtime.ta_complete_gate
    with tempfile.TemporaryDirectory()as missing_receipts:
        runtime.ta_complete_gate=lambda:original_gate(missing_receipts)
        try:
            try:runtime.load_model('hc2')
            except RuntimeError as e:assert 'CG last-stage gate:'in str(e)
            else:raise AssertionError('CG loader crossed unfinished TA fixture gate')
        finally:runtime.ta_complete_gate=original_gate
    assert 'models.pipeline'not in sys.modules
    tests.append('loader_refuses_before_all_TA_complete_without_importing_or_constructing_CG')

    # Actual official NestedTensor class and dataset load_frames functions,
    # with only ffmpeg replaced by synthetic RGB bytes. No dataset constructor,
    # annotation, pretrained model, video or caption is used in this test.
    path=runtime.VENDOR/'utils/misc.py';tree=ast.parse(path.read_text());node=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='NestedTensor')
    ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    shim=ModuleType('utils.misc');shim.NestedTensor=ns['NestedTensor'];old=sys.modules.get('utils.misc');sys.modules['utils.misc']=shim
    rng=np.random.default_rng(20261024);frames=rng.integers(0,256,size=(9,13,19,3),dtype=np.uint8)
    class Pipe:
        def input(self,*args,**kwargs):return self
        def output(self,*args,**kwargs):return self
        def run(self,*args,**kwargs):return frames.tobytes(),b''
    try:
        ours,size=runtime.tensor_inputs(frames)
        for ds,name in [('hc2','HCSTVGDataset'),('vidstg','VidSTGDataset')]:
            path=runtime.VENDOR/'datasets'/('hcstvg.py'if ds=='hc2'else'vidstg.py');tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name==name);fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='load_frames')
            scope=dict(os=os,np=np,torch=torch,ffmpeg=Pipe(),Resize=Resize,ToTensor=ToTensor,ToPILImage=ToPILImage,Image=__import__('PIL.Image',fromlist=['Image']))
            exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
            fake=SimpleNamespace(data_dir='',cfg=SimpleNamespace(INPUT=SimpleNamespace(RESOLUTION=420)))
            native=scope['load_frames'](fake,dict(vid='synthetic',frame_ids=list(range(9)),height=13,width=19))
            native=normalize(native,[.485,.456,.406],[.229,.224,.225])
            assert torch.equal(native,ours.tensors)
            assert not ours.mask.any()and ours.durations==[9]and size==(420,588)
        tests.append('both_official_load_frames_resize_and_test_normalize_byte_exact_on_synthetic_RGB')
    finally:
        if old is None:del sys.modules['utils.misc']
        else:sys.modules['utils.misc']=old

    # Generic-feedback end-to-end CPU property on actual current math: no
    # feature distinction must leave the time decision unchanged. The same
    # source and eight expert boxes yield an exact zero shared correction.
    torch.manual_seed(20261024)
    n=12;ids=list(range(n));source=np.tile([.5,.5,.2,.4],(n,1));q0=last_layer_posterior(torch.randn(1,n,2))
    native_map=[0,n];native_mbr=final_mbr(ids,q0)['interval'];pos=observation_positions(ids,[0,n-1]);gh,gw=2,3
    cap=dict(Source=source,frame_ids=ids,native_map=native_map,native_mbr=native_mbr,q0=q0,grids=dict(projected=torch.ones(n,256,gh,gw),valid=torch.ones(n,gh,gw,dtype=torch.bool),spatial=torch.randn(n,gh*gw,256),spatial_valid=torch.ones(n,gh*gw,dtype=torch.bool),shape=(gh,gw)))
    teacher=dict(positions=pos,nested_mask_all_false=True,boxes=source[pos],native_mask_logits=torch.zeros(8,4,6),input_shape=[8,3,32,48],selected_query=0)
    for ds in ['hc2','vidstg']:
        arrays,details,cache=runtime.feedback(copy.deepcopy(cap),copy.deepcopy(teacher),ds)
        assert np.array_equal(arrays['Source'],source)and np.array_equal(arrays['SharedCorrection'],source)
        assert np.array_equal(arrays['Full_interval'],np.asarray(native_mbr))
        assert np.array_equal(arrays['Full'],arrays['Space'])
        assert not details['GT_read']and details['new_network_updates']==0
        assert np.max(np.abs(cache['q1']-q0))<2e-16
    tests.append('two_domain_zero_correction_and_no_information_feedback_leave_time_and_space_decisions_unchanged')

    degenerate=runtime.clip_native_boxes(np.array([[-.5,.5,.1,.2],[.5,-.5,.1,.2]]))
    assert np.array_equal(degenerate[:,2:],np.array([[0.,.2],[.1,0.]]))or np.allclose(degenerate[:,2:],np.array([[0.,.2],[.1,0.]]),atol=1e-16)
    assert np.isfinite(degenerate).all()and (degenerate[:,2:]>=0).all()
    tests.append('native_negative_xyxy_clamp_preserves_exact_zero_area_without_epsilon_or_dropping')
    from cgstvg_memory_stream import attention_source as prior_attention
    from cgstvg_memory_stream_v8 import attention_source as current_attention
    assert prior_attention()==current_attention()
    tests.append('streamed_attention_function_text_exactly_matches_previously_bitwise_tested_v7')
    checks=0;maxerr=0.
    for ds in ['hc2','vidstg']:
        for start,end in [(1,8),(0,11),(4,6)]:
            meta=dict(width=640,height=480)
            boxes=np.column_stack([rng.uniform(.2,.8,n),rng.uniform(.2,.8,n),rng.uniform(.1,.3,n),rng.uniform(.1,.3,n)])
            boxes=runtime.clip_native_boxes(boxes)
            truth={f:[110.,130.,230.,370.]for f in range(1,10)};span=[1,10];iv=[start,end]
            old=native_official_metric(boxes,ids,meta,truth,span,iv,ds)
            dense=DenseTube(boxes,dict(frame_ids=ids,input=meta),truth,span,clip=ds=='hc2').score(iv)
            err=max(abs(old[k]-dense[k])for k in ['v','t','s']);assert err<2e-12;maxerr=max(maxerr,err);checks+=1
    tests.append('six_pinned_official_CG_evaluator_vs_independent_dense_synthetic_checks')
    assert not torch.cuda.is_initialized()and 'models.pipeline'not in sys.modules
    report=dict(status='passed_no_CG_model',tests=tests,groups=len(tests),synthetic_metric_checks=checks,maximum_metric_error=maxerr,CG_model_constructed=False,CG_forward=0,expert_forward=0,GT_files_read=0,real_RGB_read=0,all_TA_completion_gate_tested=True,real_model_noop='PENDING_LAST_after_TA',code_sha256={p.name:sha(p)for p in [Path(__file__),Path(runtime.__file__),Path(__file__).with_name('cgstvg_study_v8.py'),Path(__file__).with_name('cgstvg_score.py')]})
    atomic(runtime.D/'cgstvg_v8_runtime_cpu_tests.json',report);atomic(R/'results/paper_review_suite_v1/cgstvg_v8_runtime_cpu_tests.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
