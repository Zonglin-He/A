"""Extract one ReferDINO pixel-decoder map during its original eight-frame call."""
import hashlib
import torch


def capture(teacher,frames,caption):
    maps=[];raw=[]
    handle=teacher.model.pixel_decoder.register_forward_hook(lambda module,args,out:maps.append(out.detach().float().cpu().clone()))
    original=teacher.model.infer
    def infer(*args,**kwargs):
        out=original(*args,**kwargs);raw.append(out['pred_boxes'][0].detach().float().cpu().clone());return out
    teacher.model.infer=infer
    try:
        torch.manual_seed(42);value=teacher(frames,caption)
    finally:
        teacher.model.infer=original;handle.remove()
    if len(maps)!=1 or len(raw)!=1 or maps[0].shape[0]!=8:raise ValueError('Expected one original eight-frame feature call')
    feature=maps[0];h,w=value['input_shape'][-2:]
    if feature.shape[-2:]!=((h+3)//4,(w+3)//4):raise ValueError('Unexpected finest pixel-decoder stride; do not guess ROI coordinates')
    ratio=torch.tensor([w/(4*feature.shape[-1]),h/(4*feature.shape[-2])]).repeat(8,1)
    others=[int(i)for i in value['native_query_scores'].argsort(descending=True)if int(i)!=value['selected_query']][:3]
    value.update(features=feature,valid_ratio=ratio,alternative_boxes=raw[0][:,others],
        feature_space='same ReferDINO.pixel_decoder output, stride4, float32, same eight-frame call',
        RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest())
    return value
