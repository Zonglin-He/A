"""Freeze replacement intent and run model-free official API/input contracts."""
from pathlib import Path
import sys,os,json,time,hashlib,argparse
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R),str(R/'studies/review_strengthening_v1')]
import review_common as parent
from review_common import read,sha,atomic,digest,selected,records,COUNTS,RUN,PROVIDER
D=R/'outputs/edgetam_comparison_v1';P=R/'results/edgetam_comparison_v1'
V=parent.D/'vendor/EdgeTAM'

def prepare():
    assert not(D/'preparation_plan.json').exists();parent.validate()
    code=[Path(__file__),S/'adapter.py',R/'studies/review_strengthening_v1/sam2_input_adapter_v1.py']
    inp=[parent.D/'intent.json',parent.D/'selection.json',D/'source_checkpoint_lock.json',D/'extension_reuse_receipt.json']
    for ds in COUNTS:inp += [selected.D/f'components_{ds}_sealed.json',selected.D/f'components_{ds}_reporting_receipt.json']
    p=dict(schema='EdgeTAM_replaces_SAM2_preparation_v1',authorization='2026-10-26 direct user: use EdgeTAM to replace SAM2; retain original interpolation plus a mature published propagation method, same eight expert boxes, observed frames keep input boxes, common current time branch, pilot correctness/cost before deciding full execution.',
        source_commit=read(D/'source_checkpoint_lock.json')['commit'],
        method=dict(name='EdgeTAM',config='configs/edgetam.yaml',checkpoint='official repository checkpoints/edgetam.pt',prompt='Exactly original eight normalized expert boxes converted to official pixel xyxy; no expert masks, no new expert. No new clipping, confidence selection or prompt replacement.',frames='Original native sampled RGB/time grid',propagation='All8 box prompts for one object before tracking; official propagate_in_video(start_frame_idx=0,reverse=False). Coverage must be exactly every native frame once; no reverse pass or custom fusion. This is an offline all-conditioning comparison, not causal online tracking.',output='Masks>0 to normalized half-open bounding boxes; empty unobserved output remains zero-area, no Source fallback. Eight observed outputs overwritten with exact original input boxes.',time='Reuse current audited q1 MAP interval; only spatial trajectory varies. tIoU must be identical by construction; main SourceNative and matched-time Source are distinct.',interpolation='Reuse already audited original center/logsize interpolation and its outside-anchor convention; no silent switch to a new xyxy-linear rule.',metrics='vIoU/R03/R05 plus matched tIoU/sIoU and unobserved fixed-support analysis; parentpairedCI, allsignedpositive-negative/severe t/v uniononce/rescue/shortlong.',cost='Actual video decode/preprocess, prompt encoding, propagation, mask-to-box/writeback, transfers, model-load/startup and peak VRAM, PID/start. Same8 is not same compute. Never apply paper FPS or SAM2 speed ratio to this machine.'),
        pilot=dict(cohorts={ds:len(read(parent.D/'selection.json')[ds])for ds in COUNTS},criteria='Correct input/weight/coverage/reset/mask-box contracts, finite outputs, actual time and VRAM/resource feasibility; no target accuracy-based checkpoint/rule selection.',full_release='Full1902/10303 only after root reviews fixed-pilot correctness and measured cost, records an explicit feasibility decision, and freezes a new independent execution plan. This preparation launches no model/GPU owner.'),
        dependencies='Do not interrupt current CG or gamma queue; corruption remains paused. SAM2 unexecuted preparation is retained read-only but is no longer an active predecessor requirement. New generic-TTA plan must refer to EdgeTAM, not demand SAM2 results.',
        checkpoint='Official EdgeTAM full state loads strictly. Official timm constructor requests redundant pretrained RepViT initialization; a scoped no-download initialization adapter is permitted only with zero missing/unexpected final checkpoint keys and every loaded state tensor equal to the pinned official checkpoint. No vendor file editing.',
        extension='Identical official connected-components CUDA source and exact torch2.7/cu128 binary artifact may be reused; CPU import is not an actual CUDA-kernel or propagation test. Missing/failed postprocessing is a hard failure, no silent fallback.',
        code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f.relative_to(R)):sha(f)for f in inp})
    p['fingerprint']=digest(p);atomic(D/'preparation_plan.json',p);atomic(P/'protocol.json',p)
    active=dict(method='EdgeTAM',supersedes_unexecuted='SAM2.1 Hiera Large',plan_sha256=sha(D/'preparation_plan.json'),fingerprint=p['fingerprint'],SAM2_execution_allowed=False,EdgeTAM_GPU_execution_released=False)
    atomic(parent.D/'active_external_comparator_v2.json',active);atomic(parent.P/'external_comparator_v2.json',active)
    print(p['fingerprint'])

def preflight():
    import torch,numpy as np,types,tempfile,importlib.metadata
    from PIL import Image
    from adapter import pixel_boxes,keep_observed,lossless_rgb_loader
    parent.gt_guard();torch.set_num_threads(1);p=read(D/'preparation_plan.json')
    assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for f,h in p[field].items():assert sha(R/f)==h,f
    lock=read(D/'source_checkpoint_lock.json')
    for f,h in lock['source_files_SHA256'].items():assert sha(V/f)==h,f
    sys.path.insert(0,str(V));import sam2
    assert Path(sam2.__file__).resolve().is_relative_to(V.resolve())
    sam2.__path__.append(str(D/'extensions/sam2'));import sam2._C as extension
    assert sha(Path(extension.__file__))==read(D/'extension_reuse_receipt.json')['module_sha256']
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from sam2.utils.misc import _load_img_as_tensor
    state=torch.load(V/'checkpoints/edgetam.pt',map_location='cpu',weights_only=True,mmap=True)['model']
    assert len(state)>0 and all(torch.isfinite(t).all()for t in state.values())
    assert any(k.startswith('image_encoder.trunk.body.')for k in state)and any(k.startswith('spatial_perceiver.')for k in state)
    # Exercise the actual pinned official generator with synthetic state and
    # non-model CPU callbacks: detect its default missing-prefix behavior and
    # verify the explicit frame0 API covers the full grid once without reverse.
    def contract(start):
        tensor=torch.zeros(1,1,2,2);pos=set(range(2,10));outputs={'cond_frame_outputs':{i:{'pred_masks':tensor}for i in pos},'non_cond_frame_outputs':{}}
        state={'output_dict':outputs,'consolidated_frame_inds':{'cond_frame_outputs':pos,'non_cond_frame_outputs':set()},'obj_ids':[1],'num_frames':12,'frames_already_tracked':{}}
        mock=types.SimpleNamespace(propagate_in_video_preflight=lambda s:None,_get_obj_num=lambda s:1,
            clear_non_cond_mem_around_input=False,clear_non_cond_mem_for_multi_obj=False,
            _run_single_frame_inference=lambda **kw:({'pred_masks':tensor},tensor),
            _add_output_per_object=lambda *args:None,_get_orig_video_res_output=lambda s,t:(None,t))
        return [i for i,_,_ in SAM2VideoPredictor.propagate_in_video(mock,state,start_frame_idx=start,reverse=False)]
    assert contract(None)==list(range(2,12));assert contract(0)==list(range(12))
    runtime=read(RUN/'plan.json')['provider_runtime_fingerprint'];roots=[PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',R.parent/'stvg-mainline/outputs/tastvg_paper_v1/teacher_cache'];proof=[]
    from stvg.decode import decode
    from stvg.teacher_bridge import evidence_key
    for ds in COUNTS:
        for row in records(ds)[:2]:
            c,original=selected.suite().load_parent(ds,row);frames,ids=decode(row['metadata'],ds);assert ids==c['frame_ids'];native=np.asarray(c['geometry']['native_boxes']);pos=list(c['positions8'])
            key,context=evidence_key(frames[pos],row['metadata']['caption'],positions=pos,frame_ids=ids,fingerprint=runtime,dataset=ds)
            hits=[root for root in roots if(root/(key+'.json')).exists()];assert hits
            tp=hits[0]/(key+'.pt');tm=read(tp.with_suffix('.json'));assert tm['key']==key and tm['context']==context and tm['sha256']==sha(tp)
            teacher=torch.load(tp,map_location='cpu',weights_only=False,mmap=True);assert np.array_equal(native,teacher['boxes'].numpy())
            h,w=frames.shape[1:3];xy=pixel_boxes(native,h,w);back=xy/np.array([w,h,w,h]);back=np.concatenate([(back[:,:2]+back[:,2:])/2,back[:,2:]-back[:,:2]],1);assert np.allclose(back,native,rtol=0,atol=1e-12)
            fake=np.zeros((len(ids),4));preserved=keep_observed(fake,native,pos);assert np.array_equal(preserved[pos],native)
            sample=frames[[0,-1]];actual,hh,ww=lossless_rgb_loader(sample,1024,True);expected=torch.zeros_like(actual)
            with tempfile.TemporaryDirectory(prefix='edgetam-CPU-RGB-')as tmp:
                for j,f in enumerate(sample):
                    path=Path(tmp)/f'{j}.png';Image.fromarray(f).save(path);t,eh,ew=_load_img_as_tensor(str(path),1024);assert(eh,ew)==(h,w);expected[j]=t
            expected-=torch.tensor([.485,.456,.406])[:,None,None];expected/=torch.tensor([.229,.224,.225])[:,None,None];assert torch.equal(actual,expected)
            proof.append(dict(dataset=ds,parent_capture_sha256=original['capture_sha256'],teacher_payload_sha256=sha(tp),teacher_receipt_sha256=sha(tp.with_suffix('.json')),native_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),same_eight_native_box_and_RGB_context_exact=True,box_pixel_roundtrip_exact=True,observed_output_input_boxes_exact=True,official_RGB_preprocessing_array_exact=True))
    assert not torch.cuda.is_initialized()
    receipt=dict(status='passed_CPU_source_checkpoint_input_and_API_contracts',cases=proof,source_commit=lock['commit'],checkpoint_sha256=lock['checkpoint']['sha256'],checkpoint_tensors=len(state),official_generator_mock_full_coverage_once=True,default_forward_missing_prefix_detected=True,synthetic_contract_not_actual_tracking=True,official_predictor_builder_imported_not_called=True,versions={n:importlib.metadata.version(n)for n in ['torch','torchvision','timm','hydra-core','iopath']},CUDA_initialized=False,new_model=0,new_Source=0,new_expert=0,new_GPU_forward=0,new_GT_scoring=0,EdgeTAM_actual_propagation=False,plan_sha256=sha(D/'preparation_plan.json'))
    atomic(D/'preflight_receipt.json',receipt);atomic(P/'preflight_receipt.json',receipt);print(receipt['status'])
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight']);globals()[a.parse_args().command]()
