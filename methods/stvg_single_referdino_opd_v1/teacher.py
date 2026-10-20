"""Native ReferDINO with only its complete checkpoint, no detector pre-load."""
import inspect,textwrap,sys,time
from pathlib import Path
import torch
from methods.stvg_referdino_opd_v1.experts import NativeRefer,RT,EXT,ROOT

class SingleRefer:
    # Exactly the existing native inference/readout, without the old constructor.
    __call__=NativeRefer.__call__
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
