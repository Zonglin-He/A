"""Freeze and verify only CPU input contracts; never construct a SAM 2 model."""
from review_common import *
import argparse


VENDOR = D/'vendor/sam2'
PLAN = D/'sam2_input_plan_v1.json'
PUBLIC = P/'SAM2_input_protocol_v1.json'


def prepare():
    assert not PLAN.exists()
    files = [Path(__file__), S/'sam2_input_adapter_v1.py', S/'review_common.py',
             R/'stvg/decode.py', R/'stvg/teacher_bridge.py', R/'stvg/teacher.py']
    inputs = [D/'intent.json', D/'selection.json', D/'sam2_source_lock.json',
              D/'vendor/sam2.1_hiera_large.download.json',
              D/'sam2_runtime_import_receipt.json']
    p = dict(schema='SAM2_CPU_input_contract_v1', scope='Four actual cached teacher contexts and native RGB grids, lossless official preprocessing and mask/box/API contracts only. No model creation/forward/optimizer/GT score/GPU or new owner.',
             source_commit=read(D/'sam2_source_lock.json')['commit'],
             checkpoint_sha256=read(D/'vendor/sam2.1_hiera_large.download.json')['sha256'],
             code_sha256={str(f.relative_to(R)):sha(f) for f in files},
             input_sha256={str(f.relative_to(R)):sha(f) for f in inputs},
             vendor_sha256={str((VENDOR/f).relative_to(R)):h for f,h in
                            read(D/'sam2_source_lock.json')['files_sha256'].items()
                            if f in ['sam2/sam2_video_predictor.py','sam2/utils/misc.py',
                                     'sam2/modeling/sam2_base.py','sam2/build_sam.py',
                                     'sam2/configs/sam2.1/sam2.1_hiera_l.yaml']},
             rules=dict(RGB='Pinned original ffmpeg/native physical sampled frame grid. Lossless ndarray adapter replaces only JPEG file decoding; exact official PIL square resize/float32-buffer/normalization is verified against official helper on PNG files.',
                        prompt='Same eight actual selected ReferDINO native logits, nested padding must be entirely false, native-logit bilinear resize directly to original H,W (align_corners=False), threshold >0. Aspect-preserving teacher resize verified. No new expert.',
                        boxes='Normalized cxcywh, half-open pixel bounds [min,max+1), empty mask zero-area [0,0,0,0], no source fallback.',
                        propagation='Fresh state, all eight initial conditioning masks inserted in physical temporal order before official forward from earliest and reverse prefix; duplicate earliest conditioning output must match exactly. No custom fusion or our DP.',
                        real_execution='NOT released by this CPU preparation plan. Requires separate frozen model execution, exclusive GPU ownership after CG/old768, official CUDA postprocessing support without silent extension fallback, actual first-query bridge and fixed pilot128parents/domain then full cohorts.'))
    p['fingerprint']=digest(p);atomic(PLAN,p);atomic(PUBLIC,p)


def run():
    import numpy as np, torch, tempfile
    from PIL import Image
    from sam2_input_adapter_v1 import selected_prompts, lossless_rgb_loader, mask_box
    from stvg.decode import decode
    from stvg.teacher_bridge import evidence_key
    gt_guard();torch.set_num_threads(1)
    p=read(PLAN);assert p['fingerprint']==digest({k:v for k,v in p.items() if k!='fingerprint'})
    for field in ['code_sha256','input_sha256','vendor_sha256']:
        for f,h in p[field].items():assert sha(R/f)==h,f
    sys.path.insert(0,str(VENDOR))
    from sam2.utils.misc import _load_img_as_tensor
    runtime=read(RUN/'plan.json')['provider_runtime_fingerprint']
    roots=[PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',
           R.parent/'stvg-mainline/outputs/tastvg_paper_v1/teacher_cache']
    proofs=[]
    for ds in COUNTS:
        for row in records(ds)[:2]:
            c,parent=selected.suite().load_parent(ds,row)
            frames,ids=decode(row['metadata'],ds);assert ids==c['frame_ids']
            pos=list(c['positions8']);assert len(set(pos))==8
            key,context=evidence_key(frames[pos],row['metadata']['caption'],positions=pos,
                                    frame_ids=ids,fingerprint=runtime,dataset=ds)
            hits=[root for root in roots if (root/(key+'.json')).exists()]
            assert hits,'Missing original exact teacher context; no expert fallback'
            root=hits[0];meta=read(root/(key+'.json'));tp=root/(key+'.pt')
            assert meta['key']==key and meta['context']==context and meta['sha256']==sha(tp)
            teacher=torch.load(tp,map_location='cpu',weights_only=False,mmap=True)
            assert teacher['native_same_track_all8'] is True
            assert np.array_equal(teacher['boxes'].numpy(),np.asarray(c['geometry']['native_boxes']))
            height,width=frames.shape[1:3]
            prompts=selected_prompts(teacher,height,width)
            assert np.array_equal(prompts,selected_prompts(teacher,height,width))
            assert prompts.shape==(8,height,width)
            # Compare numerical preprocessing with the actual pinned official
            # helper, using lossless PNGs solely as a CPU reference oracle.
            sample=frames[[0,-1]]
            actual,h,w=lossless_rgb_loader(sample,1024,True)
            assert (h,w)==(height,width)
            expected=torch.zeros_like(actual)
            with tempfile.TemporaryDirectory(prefix='sam2-lossless-CPU-') as tmp:
                for i,frame in enumerate(sample):
                    path=Path(tmp)/f'{i}.png';Image.fromarray(frame).save(path)
                    item,hh,ww=_load_img_as_tensor(str(path),1024)
                    assert (hh,ww)==(h,w);expected[i]=item
            expected-=torch.tensor([.485,.456,.406],dtype=torch.float32)[:,None,None]
            expected/=torch.tensor([.229,.224,.225],dtype=torch.float32)[:,None,None]
            assert torch.equal(actual,expected)
            assert np.array_equal(mask_box(np.zeros((h,w),dtype=bool)),np.zeros(4))
            assert np.array_equal(mask_box(np.ones((h,w),dtype=bool)),[.5,.5,1,1])
            proofs.append(dict(dataset=ds,parent_capture_sha256=parent['capture_sha256'],
                               teacher_payload_sha256=meta['sha256'],teacher_receipt_sha256=sha(root/(key+'.json')),
                               native_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                               exact_eight_RGB_context=True,native_expert_boxes_exact=True,
                               prompt_binary_sha256=hashlib.sha256(prompts.tobytes()).hexdigest(),
                               prompt_masks=8,teacher_padding_false=True,prompt_repeat_exact=True,
                               official_PIL_preprocessing_array_exact=True,
                               empty_prompt_count=int((~prompts.reshape(8,-1).any(1)).sum())))
    assert not torch.cuda.is_initialized()
    receipt=dict(status='passed_CPU_inputs_only',fingerprint=p['fingerprint'],cases=proofs,
                 original_RGB_not_JPEG_reencoded=True,empty_full_mask_box_contract=True,
                 new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0,
                 SAM2_actual_propagation=False,CUDA_initialized=False,worker=identity())
    atomic(D/'sam2_input_preflight_receipt_v1.json',receipt)
    atomic(P/'SAM2_input_preflight_receipt_v1.json',receipt)
    print(json.dumps(dict(status=receipt['status'],cases=len(proofs),SAM2_actual_propagation=False)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run'])
    globals()[a.parse_args().command]()
