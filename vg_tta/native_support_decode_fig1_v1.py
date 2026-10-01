"""Isolated frozen native hypotheses. No labels, expert, optimizer or update path."""
import numpy as np

def product_beam(logits, k=6):
    """Exact k-best product assignments for independent PTD masked-token slots.

    Stable descending score, then lexicographic token-value order. Truncating
    each independent prefix to k cannot discard a global top-k completion.
    """
    z=np.asarray(logits,np.float64)
    assert z.ndim==2 and np.isfinite(z).all() and z.shape[1]>=k
    beams=[(0.,())]
    for row in z:
        ids=np.argsort(-row,kind='stable')[:k]
        pool=[(score+float(row[j]),seq+(int(j),)) for score,seq in beams for j in ids]
        beams=sorted(pool,key=lambda p:(-p[0],p[1]))[:k]
    return np.array([b[1] for b in beams],dtype=np.int64),[b[0] for b in beams]

def include_native(native, beams):
    """Six slots including native; only its exact assignment removed once.

    No deduplication of decoded intervals/tubes, and no diversity resampling.
    """
    n=np.asarray(native);b=np.asarray(beams);assert len(b)==6
    rest=list(range(6));matches=[i for i in rest if np.array_equal(n,b[i])]
    if matches:rest.remove(matches[0])
    return np.stack([n]+[b[i] for i in rest[:5]])

def time_beam(logits, native, k=6):
    z=np.asarray(logits,np.float64);n=z.shape[1];assert z.shape==(2,n) and n>=3
    candidates=sorted([(float(z[0,s]+z[1,e]),s,e) for s in range(n) for e in range(s,n)],key=lambda x:(-x[0],x[1],x[2]))[:k]
    beams=np.array([[s,e] for _,s,e in candidates]);return include_native(native,beams)

def cx_to_xy(boxes):
    b=np.asarray(boxes);return np.concatenate([b[...,:2]-.5*b[...,2:],b[...,:2]+.5*b[...,2:]],-1)

class TAAdapter:
    def __init__(self):
        import torch
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        self.model=model_load('hcstvg1_test').eval().requires_grad_(False)
        self.torch=torch
    def predict(self,frames,row,subject,smoke=False):
        import types
        from scripts.run_tastvg_paper48_p5_online_v1 import source_capture
        from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
        from vg_tta.tastvg_causal_round2_v1 import forward
        from methods.tastvg_dual_evidence_j0_v1.method import combine_layers
        m=self.model;t=self.torch;boxes=[];logits=[];calls=[0]
        with t.no_grad():data=device_tree(source_capture(m,frames,row,subject),'cuda')
        def bh(_,a,o):
            calls[0]+=1
            if calls[0]%2==0:boxes.append(o[0].flatten(1,2).detach().cpu())
        def th(_,a,o):logits.append(o.detach().cpu())
        handles=[m.ground_decoder.register_forward_hook(bh),m.temp_embed.register_forward_hook(th)]
        try:
            with t.no_grad():_,_,native=forward(m,data,[v['H'] for v in data['views']])
        finally:
            for h in handles:h.remove()
        assert calls[0]==4
        layers=combine_layers(boxes,logits,data['records'],row['frame_ids'])
        assert t.equal(layers[-1]['boxes'],native['boxes']) and layers[-1]['indices']==native['indices']
        assert t.equal(native['boxes'],data['prediction']['boxes'].cpu()) and native['indices']==data['prediction']['indices']
        out=dict(boxes=cx_to_xy(np.stack([x['boxes'].numpy() for x in layers[::-1]])).tolist(),box_frame_ids=row['frame_ids'],intervals=[x['physical_interval'] for x in layers[::-1]],spatial_valid=[True]*6,temporal_valid=[True]*6,native_parity=True,format_valid=True,preprocess=dict(resolution=224,original_two_offsets=True,encoder_autocast='fp16',cached_suffix='fp32',layers=[5,4,3,2,1,0]))
        return out

class TubeAdapter:
    def __init__(self,checkpoint):
        import torch
        from vg_tta import tubedetr_runtime as rt
        from pathlib import Path
        self.repo=Path(__file__).resolve().parents[1]/'external/TubeDETR';self.rt=rt
        self.model,self.args=rt.build_model(self.repo,device='cuda',resolution=224,stride=2,video_max_len=200)
        r=rt.load_official_checkpoint(self.model,checkpoint);assert not r['missing_keys'] and not r['unexpected_keys'];self.model.eval().requires_grad_(False)
    def predict(self,frames,row,subject=None,smoke=False):
        import torch
        from datasets.video_transforms import make_video_transforms
        from models.postprocessors import PostProcessSTVG
        video,_=make_video_transforms('val',cautious=True,resolution=224)(frames,None)
        with torch.no_grad():out=self.rt.forward_video(self.model,video,row['input']['caption'],repo=self.repo,stride=2,device='cuda',use_bf16=True)
        layers=[out]+list(out['aux_outputs'])[::-1];assert len(layers)==6
        pp=PostProcessSTVG();intervals=[]
        for layer in layers:
            intervals.append(pp(layer,frames_id=[row['frame_ids']],video_ids=[0])[0])
        boxes=np.stack([x['pred_boxes'].detach().float().cpu().numpy() for x in layers]);assert boxes.shape==(6,len(frames),4)
        if smoke:
            with torch.no_grad():plain=self.rt.forward_video(self.model,video,row['input']['caption'],repo=self.repo,stride=2,device='cuda',use_bf16=True)
            assert torch.equal(out['pred_boxes'],plain['pred_boxes']) and torch.equal(out['pred_sted'],plain['pred_sted'])
        return dict(boxes=cx_to_xy(boxes).tolist(),box_frame_ids=row['frame_ids'],intervals=intervals,spatial_valid=[True]*6,temporal_valid=[True]*6,native_parity=True,format_valid=True,preprocess=dict(resolution=224,stride=2,fast=True,autocast='bf16',layers=[5,4,3,2,1,0],temporal_readout='unmodified official PostProcessSTVG including tie order'))

class PTDAdapter:
    def __init__(self):
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load
        self.processor=processor_load();self.model=model_load()
    def predict(self,frames,row,subject=None,smoke=False):
        import torch,hashlib
        from unittest.mock import patch
        import model.ptd_generation as pg
        from scripts.ptd_spatial_adapter_ab_v1 import inputs_for
        inputs,pre=inputs_for(row,self.processor,frames)
        tokenizer=self.processor.tokenizer;ids=pg.build_ptd_token_ids(tokenizer,max_time_tokens=len(frames))
        coord=torch.tensor([pg.get_token_id(tokenizer,f'<{i}>') for i in range(1001)],device='cuda')
        tids=torch.tensor(ids['ordered_time_tokens'],device='cuda');context={};cap={};sem=[]
        probe=pg._run_cached_ptd_probe;sample=pg.sample_token_ids;parse_sem=pg._parse_semantic_block
        def probe_hook(*a,**kw):
            q=kw['query_token_ids'].flatten().tolist();context['query']=q
            context['kind']='temporal' if q==[ids['ref_end']] else 'spatial' if all(x in ids['time_tokens'] for x in q) else 'semantic'
            if context['kind'] in ['temporal','spatial']:cap[context['kind']+'_prefix_sha256']=hashlib.sha256(a[1].detach().cpu().numpy().tobytes()).hexdigest()
            return probe(*a,**kw)
        def sampling(logits,**kw):
            native=sample(logits,**kw);kind=context.get('kind')
            if kind=='temporal':
                cap['time_logits']=logits[0,1:3][:,tids].float().cpu().numpy();cap['time_raw']=native.reshape(-1).tolist()
            elif kind=='spatial':
                n=len(context['query']);cap['space_logits']=logits.reshape(n,6,-1)[:,1:5][:,:,coord].float().cpu().numpy();cap['space_raw']=native.reshape(n,6).cpu().numpy();cap['positions']=[ids['time_token_indices'][x] for x in context['query']]
            return native
        def semantic(*a,**kw):
            result=parse_sem(*a,**kw);sem.extend(result[0]);return result
        with torch.inference_mode(),patch.object(pg,'_run_cached_ptd_probe',probe_hook),patch.object(pg,'sample_token_ids',sampling),patch.object(pg,'_parse_semantic_block',semantic):
            tokens,result=pg.generate_ptd(self.model,tokenizer,dict(inputs),max_new_tokens=1024,max_time_tokens=len(frames),temperature=0.,ptd_attn_implementation='sdpa')
        if smoke:
            with torch.inference_mode():plain,pres=pg.generate_ptd(self.model,tokenizer,dict(inputs),max_new_tokens=1024,max_time_tokens=len(frames),temperature=0.,ptd_attn_implementation='sdpa')
            assert torch.equal(tokens,plain) and result.stopped==pres.stopped
        intervals=[None]*6;tv=[False]*6;sv=[False]*6;boxids=[];boxes=np.zeros((6,0,4));temporal=None
        try:
            _,anchors=pg._parse_temporal_block(cap['time_raw'],ids,block_size=6)
            temporal=[ids['time_token_indices'][anchors[0]],ids['time_token_indices'][anchors[-1]]]
        except (KeyError,RuntimeError):pass
        if temporal is not None:
            ti=time_beam(cap['time_logits'],temporal);intervals=[[row['frame_ids'][s],row['frame_ids'][e]+1] for s,e in ti];tv=[True]*6
        # One official parallel spatial probe was conditioned on the identical
        # native semantic and time prefix for every alternative. No redecoding
        # of either branch, and no autoregressive dependence among masked slots.
        if temporal is not None and 'space_logits' in cap:
            raw=cap['space_raw'];native_coords=np.array([[ids['coord_id_to_value'].get(int(x),-1) for x in rr[1:5]] for rr in raw])
            beam,_=product_beam(cap['space_logits'].reshape(-1,1001))
            candidates=include_native(native_coords.flatten(),beam).reshape(6,-1,4)
            boxes=candidates/1000.;boxids=[row['frame_ids'][i] for i in cap['positions']]
            assert cap['positions']==list(range(temporal[0],temporal[1]+1))
            try:pg._validate_box_blocks(torch.tensor(raw),ids,expected_blocks=len(raw),block_size=6);sv[0]=True
            except RuntimeError:pass
            sv[1:]=[True]*5
            # Grammar validity is tube-level; degenerate geometry is scored
            # as zero overlap per frame, never used to discard an entire tube.
        pre.update(max_new_tokens=1024,temperature=0,precision='bf16',attn='sdpa',candidate_rule='native + top product-beam assignments; width6; fixed native semantic/time spatial prefix',spatial_coverage='native time anchors only; linear inside anchor hull; zero IoU outside; never GT extrapolation')
        return dict(boxes=boxes.tolist(),box_frame_ids=boxids,intervals=intervals,spatial_valid=sv,box_geometry_valid=((boxes[:,:,2:]>boxes[:,:,:2]).all(-1)&np.isfinite(boxes).all(-1)).tolist(),temporal_valid=tv,native_parity=True,format_valid=bool(result.stopped),native_semantic_sha256=hashlib.sha256(np.asarray(sem,dtype=np.int64).tobytes()).hexdigest(),prefix_hashes={k:v for k,v in cap.items() if k.endswith('sha256')},spatial_native_time_fixed=True,preprocess=pre,native_completion_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest())
