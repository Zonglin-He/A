"""Native CG420 single-grid capture and the frozen SimplePath/Native2D core.

Importing this file never constructs a model. The loader refuses to run until
all TA GPU studies have completed and their owned processes have exited.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np
import cgstvg_interfaces as interface

R=Path(__file__).resolve().parents[2]
D=R/'outputs/paper_review_suite_v1'
VENDOR=R.parent/'outputs/maintenance/engineering_speed_v1/baseline_workspace/external/CGSTVG'
WEIGHTS={
    'hc2':('vidstg.pth','cbf3ee4036dab88de3314dc3a520ab0eb2b4d292a2796d12d0f3493c39eed9ef','vidstg.yaml'),
    'vidstg':('hcstvg2.pth','020d77b1013e33d9ae58f01c5b60e77ba6eb3d05e04077a94fa8e4247d635a88','hcstvg2.yaml'),
}


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def ta_complete_gate(directory=D):
    """No model imports/constructors/forward occur before this hard gate."""
    directory=Path(directory)
    names=['consistency_v2_score_receipt.json','budget_confirm_score_receipt.json','shifts_confirm_score_receipt.json']
    for name in names:
        path=directory/name
        if not path.is_file():raise RuntimeError('CG last-stage gate: missing '+name)
    consistency=json.loads((directory/names[0]).read_text())
    if consistency.get('status')!='scored' or consistency.get('checks')!=768 or not consistency.get('all384sealed_before_GT'):
        raise RuntimeError('CG gate: invalid TTA completion')
    for prefix in ['budget','shifts']:
        receipt=json.loads((directory/(prefix+'_confirm_score_receipt.json')).read_text())
        seal_path=directory/(prefix+'_confirm_sealed.json')
        seal=json.loads(seal_path.read_text())
        if receipt.get('seal_sha256')!=file_hash(seal_path)or seal.get('queries')!=256 or seal.get('GT_read')is not False:
            raise RuntimeError('CG gate: invalid '+prefix+' confirm seal/score binding')
    for name in ['budget_launch.json','budget_screen_predict_child.json','budget_confirm_predict_child.json','shifts_launch.json','shifts_screen_predict_child.json','shifts_confirm_predict_child.json']:
        path=directory/name
        if not path.exists():continue
        who=json.loads(path.read_text())
        try:
            fields=Path(f"/proc/{who['pid']}/stat").read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:continue
        if int(fields[19])==who['startticks']and fields[0]!='Z':raise RuntimeError('CG gate: TA owner still active: '+name)
    return {name:file_hash(directory/name)for name in names}


def verify_vendor():
    audit=json.loads((D/'cgstvg_cpu_interface_audit.json').read_text())
    assert audit['revision']=='f062d84b72aee8be4425a4a5ab20aa5b077ca142'
    for name,record in audit['official_source_files'].items():
        assert file_hash(VENDOR/name)==record['sha256'],name
    return audit


def prepare_asset_links():
    """Only new study runtime links; no writes to the frozen asset roots."""
    root=R.parent/'checkpoints'
    zoo=D/'cgstvg_runtime/model_zoo';zoo.mkdir(parents=True,exist_ok=True)
    hub=root/'huggingface/hub/models--roberta-base'
    revision=(hub/'refs/main').read_text().strip()
    links={'roberta-base':hub/'snapshots'/revision,'roberta':hub/'snapshots'/revision,
           'swin_tiny_patch244_window877_kinetics400_1k.pth':root/'tastvg_model_zoo/swin_tiny_patch244_window877_kinetics400_1k.pth'}
    for alias,src in links.items():
        src=src.resolve();assert src.exists(),src
        dst=zoo/alias
        if dst.is_symlink()or dst.exists():assert dst.resolve()==src
        else:dst.symlink_to(src,target_is_directory=src.is_dir())
    return zoo.parent


def load_model(target):
    ta_complete_gate();audit=verify_vendor()
    for name in ['models','config','utils','datasets']:
        module=sys.modules.get(name)
        if module is not None:
            path=getattr(module,'__file__',None)
            if not path or not Path(path).resolve().is_relative_to(VENDOR):
                raise RuntimeError('CG requires a fresh interpreter; foreign package '+name)
    import torch
    sys.path.insert(0,str(VENDOR))
    # Eager official LSTM import is unused by the native RoBERTa configuration.
    if 'torchtext'not in sys.modules:
        shim=ModuleType('torchtext');shim.__path__=[];sys.modules['torchtext']=shim
    from config import cfg
    from models import build_model
    filename,expected,yaml=WEIGHTS[target]
    checkpoint=R.parent/'checkpoints/cgstvg'/filename
    assert file_hash(checkpoint)==expected
    config=cfg.clone();config.merge_from_file(str(VENDOR/'experiments'/yaml))
    config.MODEL.DEVICE='cpu';config.INPUT.RESOLUTION=420;config.DATALOADER.NUM_WORKERS=0;config.freeze()
    runtime=prepare_asset_links();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',TORCH_HOME=str(R.parent/'checkpoints/torch'))
    cwd=Path.cwd()
    try:
        os.chdir(runtime)
        model,loss,weights=build_model(config)
        del loss,weights
    finally:os.chdir(cwd)
    raw=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    state=raw['model_ema'];expected_state=model.state_dict();extras=set(state)-set(expected_state)
    assert not(extras-{'text_encoder.body.embeddings.position_ids'})
    assert not(set(expected_state)-set(state))
    assert all(state[k].shape==v.shape for k,v in expected_state.items())
    model.load_state_dict({k:state[k]for k in expected_state},strict=True)
    loaded=dict(target=target,source='vidstg'if target=='hc2'else'hc2',checkpoint=filename,sha256=expected,model_key='model_ema',all_model_state_loaded=True,ignored_deterministic_extras=sorted(extras),precision='original_FP32_no_autocast',resolution=420,single_native_grid=True,source_config_yaml=yaml,original_source_thresholds_preserved=True)
    del raw,state;gc.collect()
    return model.eval().requires_grad_(False),loaded


def tensor_inputs(frames):
    """Official load_frames resize then test Normalize; no GT transforms."""
    import torch
    from torchvision.transforms import ToTensor,Resize
    from torchvision.transforms.functional import normalize
    from utils.misc import NestedTensor
    h,w=frames.shape[1:3];size=(420,min(int(420*w/h),int(420*1.4)))
    # Both upstream dataset loaders use torchvision's tensor resize. Explicit
    # antialias=True matches the installed torchvision's native default.
    resize=Resize(size,antialias=True)
    x=torch.stack([resize(ToTensor()(frame))for frame in frames])
    x=normalize(x,mean=[.485,.456,.406],std=[.229,.224,.225])
    return NestedTensor.from_tensor_list([x]),size


def clip_native_boxes(boxes):
    """CG postprocessor clamps negative xyxy for BOTH datasets, no upper clamp."""
    x=np.asarray(boxes,np.float64)
    original_lo=x[:,:2]-x[:,2:]/2;original_hi=x[:,:2]+x[:,2:]/2
    changed=(original_lo<0).any(1)|(original_hi<0).any(1)
    result=x.copy()
    lo=np.maximum(original_lo[changed],0);hi=np.maximum(original_hi[changed],0)
    result[changed]=np.concatenate(((lo+hi)/2,hi-lo),-1)
    assert np.isfinite(result).all()and(result[:,2:]>0).all()
    return result


def capture(model,frames,ids,caption,*,noop=False):
    import torch
    from models.post_processor import PostProcess
    from stvg.temporal import final_mbr
    ta_complete_gate()
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    batch,size=tensor_inputs(frames);batch=batch.to('cuda');model.cuda()
    h,w=frames.shape[1:3];n=len(ids);reference=None;extra_source=0
    start=time.perf_counter()
    with torch.no_grad():
        if noop:
            # Fake labels only exercise the upstream unused call. Their values
            # never come from annotations and cannot affect the eval branch.
            fake=[dict(eval=True,boxs=torch.zeros(n,4,device='cuda'),durations=[n],actioness=torch.ones(n,device='cuda'))]
            reference={k:(v.detach().cpu().clone()if torch.is_tensor(v)else None)for k,v in model(batch,[caption],fake).items()}
            extra_source=1
        grids={}
        def pre(module,args,kwargs):
            v=kwargs['videos'];grids['projected']=v.tensors.detach().cpu().clone();grids['valid']=(~v.mask).cpu().clone()
        def after(module,args,kwargs,out):
            gh,gw=out['fea_map_size'];a=out['encoded_feature'][:gh*gw]
            grids['spatial']=a.permute(1,0,2).detach().cpu().clone()
            grids['spatial_valid']=(~out['encoded_mask'][:,:gh*gw]).cpu().clone();grids['shape']=(gh,gw)
        hooks=[model.ground_encoder.register_forward_pre_hook(pre,with_kwargs=True),model.ground_encoder.register_forward_hook(after,with_kwargs=True)]
        try:
            with interface.inference_only(model.ground_decoder.decoder)as noGT:
                out=model(batch,[caption],interface.ForbidLabels())
        finally:
            for hook in hooks:hook.remove()
        if noop:
            for k,v in reference.items():
                if v is not None:assert torch.equal(v,out[k].detach().cpu()),('CG native observer/dead-read no-op',k)
        sizes=torch.tensor([[h,w]]*n,device='cuda')
        pixels,spans=PostProcess()(out,sizes,[ids],[n])
        native_map=interface.native_map_interval(out['pred_sted'],ids);assert native_map==spans[0]
        logits=out['pred_sted'].detach().cpu().clone()
        pixels=pixels.detach().cpu().numpy();norm=pixels.astype(np.float64)/np.array([w,h,w,h])
        source=np.concatenate(((norm[:,:2]+norm[:,2:])/2,norm[:,2:]-norm[:,:2]),-1)
        q0=interface.last_layer_posterior(logits);mbr=final_mbr(ids,q0)['interval']
        gh,gw=grids['shape'];assert grids['projected'].shape==(n,256,gh,gw)
        assert torch.equal(grids['valid'].reshape(n,-1),grids['spatial_valid'])
        if noop:
            # Pool serialization bridge on actual native feature tensors.
            a=interface.pool_single_grid(grids['projected'],grids['valid'],source)
            import io
            stream=io.BytesIO();torch.save(grids,stream);stream.seek(0);cold=torch.load(stream,map_location='cpu',weights_only=True)
            assert np.array_equal(a,interface.pool_single_grid(cold['projected'],cold['valid'],source))
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        heads={k:v.detach().cpu().clone()for k,v in out.items()if torch.is_tensor(v)}
    del batch,out,reference;model.cpu();gc.collect();torch.cuda.empty_cache()
    return dict(Source=source,Source_pixels=pixels,native_map=native_map,native_mbr=mbr,q0=q0,logits=logits,grids=grids,heads=heads,frame_ids=ids,GT_read=False,
                native_noop=bool(noop),source_forwards=1+extra_source,source_seconds=elapsed,precision='FP32',input_size=size,noGT_patch=noGT)


def feedback(cap,teacher,dataset):
    """Unchanged production equations; explicit CG feature/time adapters only."""
    from stvg.initialization import initialize
    from stvg.predictor import output_writeback
    from stvg.evidence import SourceEvidence
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.prototype import one_step
    from stvg.temporal import final_mbr
    from stvg.positions import observation_positions
    start=time.perf_counter();ids=cap['frame_ids'];iv=cap['native_map']
    pos=observation_positions(ids,[ids.index(iv[0]),ids.index(iv[1]-1)])
    assert list(teacher['positions'])==pos and teacher['nested_mask_all_false']
    cfg=json.loads((R/'configs/default.json').read_text())['datasets'][dataset]
    source=cap['Source'];reference=np.asarray(teacher['boxes'],np.float64)
    raw,init=initialize(source,pos,reference,cfg['E_calibration'])
    E=output_writeback(source,raw,cfg['initialization_gamma'])
    g=cap['grids'];ev=SourceEvidence.from_native_masks(g['spatial'],g['spatial_valid'],ids,pos,teacher['native_mask_logits'],teacher['input_shape'][-2:],g['shape'],selected_query=teacher['selected_query'])
    geometry=SpatialGeometry(ev,E,reference)
    aux=evidence_dp(geometry.for_interval([ids[0],ids[-1]+1]))['final'].numpy()
    features=interface.pool_single_grid(g['projected'],g['valid'],aux)
    q1,stats=one_step(features,cap['q0']);newiv=final_mbr(ids,q1)['interval']
    space=evidence_dp(geometry.for_interval(cap['native_mbr']))['final'].numpy()
    full=evidence_dp(geometry.for_interval(newiv))['final'].numpy()
    arrays=dict(SourceNative=source.copy(),Source=source.copy(),SharedCorrection=clip_native_boxes(E),Space=clip_native_boxes(space),Full=clip_native_boxes(full),SourceNative_interval=np.asarray(cap['native_map']),Source_interval=np.asarray(cap['native_mbr']),SharedCorrection_interval=np.asarray(cap['native_mbr']),Space_interval=np.asarray(cap['native_mbr']),Full_interval=np.asarray(newiv),positions8=np.asarray(pos),frame_ids=np.asarray(ids))
    details=dict(native_source_map_separate=True,same_MBR_for_Source_shared_space_full=True,calibration='TA source-training constants transferred unchanged to CG, NOT newly CG-calibrated',gamma='unchanged TA value; historical source-only selection provenance not established',initialization=init,feedback_seconds=time.perf_counter()-start,expert_instances=1,new_network_updates=0,GT_read=False)
    cache=dict(aux=aux,features=features,q0=cap['q0'],q1=q1,E=E,geometry=geometry.state_dict(),L=stats['L'])
    return arrays,details,cache
