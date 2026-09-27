"""Untouched official STCAT integration for the cross-backbone diagnosis only."""
import os,sys,types
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import ROOT,read,write,sha
from vg_tta.st_causal_audit_v2 import KeyIntervention,choose_layers
from vg_tta.expert_space_time_probe_v1 import tree


def load_model(g):
    repo=ROOT/'external/STCAT';sys.path.insert(0,str(repo))
    assert not any(n=='models' or n.startswith('models.') for n in sys.modules),'one architecture per process'
    # Optional, unused LSTM imports torchtext with incompatible binary ABI.
    stub=types.ModuleType('models.language_model.lstm')
    class NoLSTM:
        def __init__(self,*a,**kw):raise RuntimeError('LSTM not part of the released STCAT checkpoint')
    stub.RNNEncoder=NoLSTM;sys.modules['models.language_model.lstm']=stub
    torch.hub.set_dir(str(ROOT/'.cache/torch/hub'))
    from config.defaults import _C
    cfg=_C.clone();source='HC-STVG/e2e_STCAT_R101_HCSTVG.yaml' if g=='hc_to_vid' else 'VidSTG/e2e_STCAT_R101_VidSTG.yaml'
    cfg.merge_from_file(str(repo/'experiments'/source));cfg.INPUT.RESOLUTION=224
    snapshots=list((ROOT/'.cache/huggingface/hub/models--roberta-base/snapshots').iterdir());assert snapshots
    cfg.MODEL.TEXT_MODEL.NAME=str(snapshots[0]);cfg.freeze()
    from models.pipeline import STCATNet
    m=STCATNet(cfg)
    ck=ROOT/f"checkpoints/STCAT_{'hc' if g=='hc_to_vid' else 'vid'}_res416.pth";receipt=read(ck.with_suffix('.receipt.json'));assert sha(ck)==receipt['sha256']
    x=torch.load(ck,map_location='cpu',weights_only=False);key='model_ema' if 'model_ema' in x else 'model';state=x[key]
    # Modern transformers made two deterministic buffers non-persistent.
    target=m.state_dict();compat={}
    for n in list(state):
        if n not in target and (n.endswith('position_ids') or n.endswith('token_type_ids')):compat[n]=list(state.pop(n).shape)
    result=m.load_state_dict(state,strict=True)
    out=ROOT/'artifacts/st_causal_backbone_audit_v2'/f'stcat_loader_{g}.json'
    if not out.exists():write(out,dict(checkpoint=receipt,checkpoint_key=key,strict_missing=list(result.missing_keys),strict_unexpected=list(result.unexpected_keys),deterministic_compat_buffers=compat,
        eval_resolution=224,official_evaluation='two temporal offsets, no horizontal flip ensemble in matched diagnosis',
        source_note='HC-STVG v1 source, not HC-STVG2' if g=='hc_to_vid' else 'VidSTG source',runtime_sha256=sha(__file__)))
    return m.eval().requires_grad_(False).cuda()


class STCATReplay:
    def __init__(self,m,raw,ids,q):
        from utils.misc import NestedTensor
        self.m=m;self.ids=ids;self.offsets=[list(range(o,len(ids),2)) for o in [0,1]];self.caps=[];official=[]
        x=torch.from_numpy(np.ascontiguousarray(raw)).permute(0,3,1,2).float()/255.
        # Match the existing TA intervention input geometry. No GT crop/resampling.
        w=min(int(224*q['width']/q['height']),int(224*1.4))
        x=torch.nn.functional.interpolate(x,size=(224,w),mode='bilinear',align_corners=False,antialias=True)
        x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
        for pos in self.offsets:
            xx=x[pos].cuda();v=NestedTensor(xx,torch.zeros(len(xx),224,w,device='cuda',dtype=torch.bool),[len(xx)])
            h=m.ground_decoder.register_forward_pre_hook(lambda _,a,kw:self.caps.append(tree(kw)),with_kwargs=True)
            try:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):official.append(m(v,[q['caption']]))
            finally:h.remove()
        self.grids=[tuple(c['memory_cache']['fea_map_size']) for c in self.caps];self.base=self.run()
        merged=torch.stack([official[t%2]['pred_boxes'][t//2].float().cpu() for t in range(len(ids))]);assert torch.equal(merged,self.base['boxes'])
        z=torch.stack([official[t%2]['pred_sted'][0,t//2].float().cpu() for t in range(len(ids))]);assert torch.equal(z,self.base['logits'])
        from models.post_processor import PostProcess
        endpoints=[]
        for o,pos in zip(official,self.offsets):
            _,ts=PostProcess()(o,torch.tensor([[q['height'],q['width']]]*len(pos),device='cuda'),[[ids[i] for i in pos]],[len(pos)])
            endpoints.append(ts[0])
        assert self.base['indices']==[ids.index(min(t[0] for t in endpoints)),ids.index(max(t[1] for t in endpoints)-1)]

    def run(self,spec=None):
        from models.net_utils import inverse_sigmoid
        from vg_tta.decota_tastvg_episode_v1 import native_view_indices
        spec=spec or {};bs=[];zs=[];indices=[];audit=[]
        for off,(cap,pos,grid) in enumerate(zip(self.caps,self.offsets,self.grids)):
            cm=ExitStack()
            if 'allowed' in spec:
                allowed=np.asarray(spec['allowed'])[pos];ls=choose_layers(self.m.ground_decoder.decoder.layers,spec.get('stage','all'))
                if allowed.any():
                    h=cm.enter_context(KeyIntervention([l.self_attn for l in ls],allowed=allowed,mode=spec.get('mode','hard'),zero=spec.get('zero',False)));audit.append(h.audit)
            if 'roi' in spec:
                ls=choose_layers(self.m.ground_decoder.temp_decoder.layers,spec.get('stage','all'))
                h=cm.enter_context(KeyIntervention([l.cross_attn_image for l in ls],roi=spec['roi'][off],gain=spec.get('gain',4.),mode=spec.get('mode','hard'),zero=spec.get('zero',False)));audit.append(h.audit)
            try:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    (hh,reference),(ht,_)=self.m.ground_decoder(**cap)
                    delta=self.m.bbox_embed(hh);delta[...,:self.m.query_dim]+=inverse_sigmoid(reference)
                    bb=delta.sigmoid()[:,0];zz=self.m.temp_embed(ht)[:,0]
                ij=native_view_indices(zz[-1][None]);indices.append([pos[ij[0]],pos[ij[1]]]);bs.append(bb.float().cpu());zs.append(zz.float().cpu())
            finally:cm.close()
        merge=lambda v:torch.stack([v[t%2][:,t//2] for t in range(len(self.ids))],dim=1)
        lb,lz=merge(bs),merge(zs)
        return dict(boxes=lb[-1],logits=lz[-1],indices=[min(t[0] for t in indices),max(t[1] for t in indices)],layer_boxes=lb,layer_logits=lz,audit=audit)
