"""Joint original-frame VideoSwin context; unchanged offset Source query supports."""
from contextlib import contextmanager
import hashlib
import torch


def digest(x):
    x=x.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


@contextmanager
def motion_context(model, joint=True):
    from methods.decota_final_simplified_v1 import backbone
    original_batch=backbone.make_batch
    original_forward=model.vid.forward
    had_instance='forward' in model.vid.__dict__
    previous_instance=model.vid.__dict__.get('forward')
    assert not model.vid._forward_hooks and not model.vid._forward_pre_hooks
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    data={'calls':[], 'actual_encoder_calls':0, 'joint':joint}
    full=None;cached=None
    def batch(*args,**kwargs):
        nonlocal full
        assert full is None
        b=original_batch(*args,**kwargs)
        full=b['videos'].tensors.detach().cpu().clone()
        data.update(full_shape=list(full.shape),full_sha256=digest(full))
        return b
    def forward(samples,num_frames):
        nonlocal cached
        assert full is not None
        offset=len(data['calls']);assert offset in (0,1)
        assert num_frames==len(samples)==len(full[offset::2])
        assert torch.equal(samples.detach().cpu(),full[offset::2]),'physical offset inputs differ'
        if not joint:
            result=original_forward(samples,num_frames)
            data['actual_encoder_calls']+=1
        else:
            if cached is None:
                whole=full.to(samples.device)
                outputs=original_forward(whole,len(full))
                assert len(outputs['3'])==len(full)
                cached=outputs['3'].detach().cpu().contiguous()
                data.update(joint_feature_shape=list(cached.shape),joint_feature_sha256=digest(cached))
                data['actual_encoder_calls']+=1
                del whole,outputs
            selected=cached[offset::2].contiguous()
            result={'3':selected.to(samples.device)}
            assert torch.equal(result['3'].cpu(),cached[offset::2])
        data['calls'].append(dict(offset=offset,num_frames=num_frames,physical_indices=list(range(offset,len(full),2)),input_sha256=digest(samples),output_shape=list(result['3'].shape),output_sha256=digest(result['3'])))
        return result
    backbone.make_batch=batch;model.vid.forward=forward
    try:
        yield data
        assert len(data['calls'])==2
        assert data['actual_encoder_calls']==(1 if joint else 2)
    finally:
        backbone.make_batch=original_batch
        if had_instance:model.vid.forward=previous_instance
        else:del model.vid.forward
        assert backbone.make_batch is original_batch
        assert model.vid.forward==original_forward
        data['restored']=True


def properties():
    # For a frame-independent encoder, joint/split are exactly the same.
    x=torch.arange(65*3*2*2).reshape(65,3,2,2).float()
    f=lambda z:z.sum(1,keepdim=True)*2
    assert all(torch.equal(f(x)[i::2],f(x[i::2]))for i in (0,1))
    # Full input indices are never interpolated or reordered.
    assert sorted(list(range(0,65,2))+list(range(1,65,2)))==list(range(65))
    return dict(status='pass',frame_independent_equivalence=True,physical_rows_partition_exact=True,synthetic_not_model_effect=True)
