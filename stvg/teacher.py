"""Native ReferDINO with only its complete checkpoint, no detector pre-load."""
import inspect,textwrap,sys,time
from pathlib import Path
import torch
import hashlib
ROOT=Path(__file__).resolve().parents[1]
RT=ROOT/'runtime/referdino';EXT=ROOT/'external/ReferDINO'

class SingleRefer:
    # Exactly the existing native inference/readout, without the old constructor.
    @__import__('torch').no_grad()
    def __call__(self,frames,caption):
     import torch
     from PIL import Image
     from misc import nested_tensor_from_videos_list
     torch.cuda.synchronize();start=time.perf_counter();imgs=torch.stack([self.transform(Image.fromarray(f))for f in frames]).cuda();samples=nested_tensor_from_videos_list(imgs[None],size_divisibility=1);h,w=imgs.shape[-2:];target={'size':torch.as_tensor([h,w]).cuda()};exp=' '.join(caption.lower().split());torch.cuda.synchronize();p=time.perf_counter();out=self.model.infer(samples,[exp],[target]);torch.cuda.synchronize();f=time.perf_counter()
     logits=out['pred_logits'][0];scores=logits.sigmoid().mean(0).max(-1).values;idx=int(scores.argmax());b=out['pred_boxes'][0,:,idx].float().cpu();m=out['pred_masks'][0,:,idx];mstats=dict(shape=list(m.shape),finite=bool(torch.isfinite(m).all()),foreground_fraction_at_native_grid=float((m>0).float().mean()))
     assert torch.isfinite(b).all()and(b[:,2:]>0).all();torch.cuda.synchronize();end=time.perf_counter()
     return dict(boxes=b,native_mask_logits=m.detach().float().cpu(),nested_mask_all_false=bool(not samples.mask.any()),selected_query=idx,native_query_scores=scores.float().cpu(),selected_frame_scores=logits[:,idx].sigmoid().float().cpu(),mask_summary=mstats,text_sha256=hashlib.sha256(exp.encode()).hexdigest(),input_shape=list(imgs.shape),cost=dict(expert_seconds=end-start,preprocess_seconds=p-start,forward_seconds=f-p,postprocess_seconds=end-f),native_same_track_all8=True)
    def __init__(self):
        sys.path[:0]=[str(RT/'python'),str(RT/'cuda_build'),str(EXT)]
        from ruamel.yaml import YAML
        from easydict import EasyDict
        import models
        with open(EXT/'configs/mevis_swinb.yaml')as f:c=YAML(typ='safe',pure=True).load(f)
        args=EasyDict({k:v['value']for k,v in c.items()});args.device='cuda'
        args.GroundingDINO.pretrained_path=None
        args.GroundingDINO.text_encoder_type=str(RT/'bert-from-frozen-teacher')
        args.GroundingDINO.tracking_alpha=.1
        original=models.build_groundingdino;body=textwrap.dedent(inspect.getsource(original))
        removed=('    print("load pretrained GroundingDINO from {} ...".format(args.pretrained_path))\n'
                 '    checkpoint = torch.load(args.pretrained_path, map_location="cpu")\n'
                 '    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)\n')
        assert body.count(removed)==1
        changed=body.replace(removed,'');assert changed.replace('    return model\n',removed+'    return model\n')==body
        ns=dict(vars(sys.modules[original.__module__]));exec(compile(changed,__file__,'exec'),ns)
        models.build_groundingdino=ns[original.__name__]
        try:self.model,criterion,post=models.build_model(args);del criterion,post
        finally:models.build_groundingdino=original
        checkpoint=torch.load(ROOT/'checkpoints/referdino/ryt_mevis_swinb.pth',map_location='cpu',weights_only=False)
        sd=checkpoint['model_state_dict'];result=self.model.load_state_dict(sd,strict=False);learned=set(dict(self.model.named_parameters()))
        assert not(set(result.missing_keys)&learned),result
        assert all(k.endswith(('position_ids','token_type_ids'))for k in result.missing_keys),result
        assert not[k for k in result.unexpected_keys if not k.endswith('position_ids')],result
        self.load_audit=dict(missing=result.missing_keys,unexpected=result.unexpected_keys,
            all_learned_parameters_covered=True,all_state_except_deterministic_index_buffers_covered=True,independent_GroundingDINO_checkpoint_loaded=False,
            independent_detector_instances=0,expert_instances=1)
        self.model=self.model.cuda().eval().requires_grad_(False);self.params=sum(p.numel()for p in self.model.parameters())
        import torchvision.transforms as T
        self.transform=T.Compose([T.Resize(360),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
