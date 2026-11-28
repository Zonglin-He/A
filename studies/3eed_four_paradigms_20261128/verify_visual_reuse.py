"""Verify the patch hook can ride an ordinary frozen 2D decode unchanged."""
import torch

from prepare_blind import QWEN, RUN, digest, read, sealed_inputs, write
from vision_reads import image, patches, prototype_mask, visual_request


@torch.inference_mode()
def main():
    source,*_=sealed_inputs()
    from run_full_qwen import PROMPTS,load_model

    cases=read(RUN/'inputs-private.json')
    samples=[next(r for r in cases if r['target']==p and r['first_window'] and r['marks'])
             for p in ('waymo','drone','quad')]
    rows=[source[r['uid']]['frame'] for r in samples]
    processor,model,limit=load_model(QWEN)
    photos=[image(f['rgb']) for f in rows]
    data=visual_request(processor,photos,[PROMPTS['2d']+f['text'] for f in rows])
    net=model.model.visual
    unhooked=model.generate(**data,do_sample=False,max_new_tokens=limit)
    capture=[]
    hook=net.blocks[-1].register_forward_hook(lambda _m,_a,x:capture.append(x))
    try:
        hooked=model.generate(**data,do_sample=False,max_new_tokens=limit)
    finally:
        hook.remove()
    assert len(capture)==1 and torch.equal(unhooked,hooked)
    m=net.spatial_merge_size
    shapes=[(h,w) for t,h,w in data.image_grid_thw.tolist() if t==1]
    assert len(shapes)==len(samples)
    f=capture[0].split([h*w for h,w in shapes])
    embeddings=[x.reshape(h//m,w//m,m,m,-1).permute(0,2,1,3,4).reshape(h,w,-1)
                for x,(h,w) in zip(f,shapes,strict=True)]
    retrospective=patches(processor,model,photos)
    delta=[float((a.float()-b.float()).abs().max()) for a,b in zip(embeddings,retrospective,strict=True)]
    for f,r in zip(embeddings,samples,strict=True):
        assert prototype_mask(f,r['size'],r['window'],r['marks']) is not None
    stamp=dict(queries=len(samples),unchanged_native_tokens=len(samples),normal_forward_hook_calls=1,
               hook_added_forwards=0,validation_patch_only_extra_forward=True,
               retrospective_feature_max_abs_delta=max(delta),filters_applied=False,
               labels_read=False,code=digest(__file__))
    write(RUN/'visual-reuse-smoke.json',stamp)
    print(stamp,flush=True)


if __name__=='__main__':
    main()
