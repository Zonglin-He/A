"""TubeDETR-only implementation of ordinary features and affine parameter I/O."""
import math
import torch
from torch.func import functional_call
from stvg.tubedetr import native_forward
from tools.mask_reward import MaskAreaReward


def encode(model,frames,frame_ids,caption):
    # Last pre-text ResNet feature level with stride <=16 (layer3 here).
    choices=[m for n,m in model.backbone.named_modules()if n.endswith('body.layer3')]
    if len(choices)!=1:raise ValueError('Missing unique TubeDETR spatial visual interface')
    maps=[];shapes=[]
    def feature(module,args,out):
        if out.shape[0]==len(frame_ids):maps.append(out.detach())
    def canvas(module,args,out):
        if args[0].tensors.shape[0]==len(frame_ids):shapes.append(tuple(args[0].tensors.shape[-2:]))
    a=choices[0].register_forward_hook(feature);b=model.backbone.register_forward_hook(canvas)
    try:native=native_forward(model,frames,frame_ids,caption,224)
    finally:a.remove();b.remove()
    if len(maps)!=1 or len(shapes)!=1:raise ValueError('Expected one ordinary full-time visual map stream')
    feat=maps[0];H,W=shapes[0];T,C,h,w=feat.shape
    if (h,w)!=(math.ceil(H/16),math.ceil(W/16)):raise ValueError('Unexpected spatial visual grid stride')
    y,x=torch.meshgrid(torch.arange(h,device=feat.device),torch.arange(w,device=feat.device),indexing='ij')
    cells=torch.stack([x*16/W,y*16/H,((x+1)*16/W).clamp_max(1),((y+1)*16/H).clamp_max(1)],-1).reshape(-1,4).to(feat)
    return native,feat.flatten(2).transpose(1,2),cells,dict(feature='ordinary ResNet layer3 before text fusion',channels=C,spatial_stride=16,grid=[h,w],canvas=[H,W],coordinate_convention='stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries')


def teacher_seed_mass(masks,valid_ratio,cells):
    # Exact continuous area integration aligns same-image teacher evidence to
    # student cells despite different input resolutions and canvas padding.
    boxes=torch.cat([(cells[:,:2]+cells[:,2:])/2,cells[:,2:]-cells[:,:2]],-1)[None].expand(8,-1,-1)
    r=MaskAreaReward(masks,valid_ratio,torch.zeros(8,0,device=masks.device,dtype=torch.bool))
    return r.mass(boxes)[:,:,0].detach()


class ParameterInterface:
    def __init__(self,model,native):
        self.decoder=model.transformer.decoder;self.head=model.bbox_embed;self.args,self.kwargs=native['decoder_inputs'];names=[]
        for name,module in self.decoder.named_modules():
            if isinstance(module,torch.nn.LayerNorm)and module.elementwise_affine:
                names.extend([(name+'.'if name else'')+key for key in ['weight','bias']if getattr(module,key,None)is not None])
        allparams=dict(self.decoder.named_parameters());self.names=sorted(names);self.base={n:allparams[n].detach()for n in self.names};assert self.names
        self.sizes=[self.base[n].numel()for n in self.names];self.count=sum(self.sizes)
    def zero(self):return next(iter(self.base.values())).new_zeros(self.count)
    def execute(self,delta):
        if delta.shape!=(self.count,):raise ValueError('Parameter vector interface differs')
        updates={};offset=0
        for n,size in zip(self.names,self.sizes):updates[n]=self.base[n]+delta[offset:offset+size].reshape_as(self.base[n]);offset+=size
        hidden=functional_call(self.decoder,updates,self.args,self.kwargs,strict=False)
        if isinstance(hidden,tuple):hidden=hidden[0]
        return self.head(hidden.transpose(1,2).flatten(1,2)).sigmoid()[-1].float()
