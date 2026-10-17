"""Actual frozen text re-grounding and integrated native512 head readout."""
from pathlib import Path
from contextlib import contextmanager
import collections,copy,gc,hashlib,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_semantic_view32_v1 import BASE,OLD,read,write,sha,verify,guard

def move(x,device):
    import torch
    if torch.is_tensor(x):return x.detach().to(device).clone()
    if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
    if isinstance(x,list):return [move(v,device) for v in x]
    if isinstance(x,tuple):return tuple(move(v,device) for v in x)
    if hasattr(x,'tensors') and hasattr(x,'mask') and hasattr(x,'durations'):return type(x)(move(x.tensors,device),move(x.mask,device),list(x.durations))
    return copy.deepcopy(x)

@contextmanager
def visual_cache(model):
    """Only raw ResNet+Swin outputs before any text fusion; per-Query lifetime."""
    import torch
    caches={};counts=dict(vis_new=0,vid_new=0,vis_reuse=0,vid_reuse=0)
    saved={n:getattr(model,n).forward for n in ['vis_encoder','vid']}
    def wrapper(name,original):
        def call(*args,**kwargs):
            tensor=args[0].tensors if hasattr(args[0],'tensors') else args[0];raw=tensor.detach().cpu().contiguous();key=(name,str(raw.dtype),tuple(raw.shape),hashlib.sha256(raw.numpy().tobytes()).hexdigest())
            if hasattr(args[0],'mask') and args[0].mask is not None:key+= (hashlib.sha256(args[0].mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),tuple(args[0].durations))
            device=tensor.device
            if key in caches:counts['vis_reuse' if name=='vis_encoder' else 'vid_reuse']+=1;return move(caches[key],device)
            out=original(*args,**kwargs);caches[key]=move(out,'cpu');counts['vis_new' if name=='vis_encoder' else 'vid_new']+=1;return out
        return call
    try:
        for n,f in saved.items():getattr(model,n).forward=wrapper(n,f)
        yield counts
    finally:
        for n,f in saved.items():getattr(model,n).forward=f
        caches.clear()

def row_for(rec):
    from scripts.run_stvg_table1_source_v1 import _validate_roster_record
    return _validate_roster_record(rec)[1]

def fresh(model,frames,row):
    import torch
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    torch.set_num_threads(2)
    _,records,base=frozen_forward(model,frames,row)
    out=dict(logits=[z.detach().cpu().clone() for z in base.zero['logits']],hidden=[h[-1,0].detach().cpu().clone() for h in base.temporal_inputs],boxes=base.zero['boxes'].detach().cpu().clone(),records=records)
    del base;gc.collect();torch.cuda.empty_cache();return out

def exact(a,b):
    import torch
    return a['records']==b['records'] and torch.equal(a['boxes'],b['boxes']) and all(torch.equal(x,y) for k in ['logits','hidden'] for x,y in zip(a[k],b[k]))

def cached_source(model,c):
    import torch
    from scripts.stvg_pipeline_speed_v1 import BASE as S,to_device
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    p=S/'source_features'/(c['case_id']+'.pt');r=read(p.with_suffix('.receipt.json'));assert sha(p)==r['sha256'];bind=r['binding']
    assert bind['source_case_sha256']==c['source_case_sha256'] and bind['checkpoint_sha256']==c['source_checkpoint_sha256'] and bind['pixel_sha256']==c['pixel_sha256'] and bind['frame_ids']==list(map(int,c['frame_ids']))
    d=torch.load(p,map_location='cpu',weights_only=False);assert d['binding']==bind and model._fixed_full_state_hash==c['source_model_state_sha256'];views=to_device(d['views'],next(model.parameters()).device);base=SpatialReplay(model,views,len(c['frame_ids']));assert torch.equal(base.zero['boxes'].cpu(),c['zero']['boxes'])
    assert all(torch.equal(z.cpu(),v) for z,v in zip(base.zero['logits'],c['zero']['logits']));assert all(torch.equal(z[-1,0].cpu(),v) for z,v in zip(base.temporal_inputs,c['temporal_hidden']));return base

def targets():
    import torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts.run_decota_paper_main_v1 import model_for
    from methods.decota_final_simplified_v1.observations import QuerySubjectParser
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts import stvg_temporal_semantic_view_fit32_v1 as fit
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from methods.decota_final_simplified_v1.objectives import prediction
    verify();assert read(BASE/'TEACHER_ROOT_READBACK.json')['status']=='pass';sys.addaudithook(guard);torch.set_num_threads(2);parser=QuerySubjectParser(ROOT/'.cache/stanza');model=None;source=None;qualified=set();viewqualified=set();files=[];actual=0;traces=[]
    jobmap={r['case_id']:r for r in read(BASE/'INPUT_BARRIER.json')['jobs']}
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        verify();c,_=load_case(rec);cid=rec['probe_alias'];row=row_for(rec);src='vidstg' if rec['dataset']=='hc2' else 'hcstvg2'
        if src!=source:
            if model is not None:assert state_hash(model.state_dict())==model._fixed_full_state_hash;del model;gc.collect();torch.cuda.empty_cache()
            model=model_for('vidstg' if rec['dataset']=='hc2' else 'hcstvg2');source=src
        assert model._fixed_full_state_hash==c['source_model_state_sha256'];j=read(ROOT/jobmap[cid]['path']);assert sha(ROOT/j['source_grid_path'])==j['source_grid_sha256'];frames=np.load(ROOT/j['source_grid_path'],allow_pickle=False);assert hashlib.sha256(frames.tobytes()).hexdigest()==c['pixel_sha256'];teacher=read(BASE/'teacher'/(cid+'.json'));off=fit.offsets(c,_hidden_from_source_case(c));p0=[fit.original._prob(o['base'])[0] for o in off];result={};intervals={};admitted=teacher['admitted'];texts={'SEM':teacher['caption']['raw'],'PARA':teacher['paraphrase']['raw']};callcount=0
        with visual_cache(model) as counts:
            if rec['dataset'] not in qualified:
                z=fresh(model,frames,row);actual+=1;callcount+=1;assert z['records']==c['records'] and torch.equal(z['boxes'],c['zero']['boxes']) and all(torch.equal(a,b) for a,b in zip(z['logits'],c['zero']['logits']));z2=fresh(model,frames,row);actual+=1;callcount+=1;assert exact(z,z2);qualified.add(rec['dataset'])
                write(BASE/'qualification'/(rec['dataset']+'_Source_cache.json'),dict(status='pass',count=1,actual_fresh_and_visual_only_cached_Source_logits_hidden_boxes_exact=True,Source_matches_sealed_case=True,GT_read=False));del z,z2
            for arm in ['SEM','PARA']:
                if not admitted[arm]:result[arm]=[q.copy() for q in p0];intervals[arm]=list(c['Native_interval']);continue
                text=texts[arm]
                if text.strip()==j['query_text'].strip():result[arm]=[q.copy() for q in p0];intervals[arm]=list(c['Native_interval']);continue
                newrow=copy.deepcopy(row);newrow['input']['caption']=text;newrow['parses']=parser(text);z=fresh(model,frames,newrow);actual+=1;callcount+=1;assert z['records']==c['records'];result[arm]=fit.distribution_targets(z['logits'],off)
                v=prediction(z['logits'],z['boxes'],z['records'],list(map(int,c['frame_ids'])));intervals[arm]=v['physical_interval']
                if arm=='SEM' and rec['dataset'] not in viewqualified:
                    # Fully fresh RGB/caption route, outside the raw-vision cache.
                    wrappers={n:getattr(model,n).forward for n in ['vis_encoder','vid']}
                    for n in wrappers:getattr(model,n).forward=type(getattr(model,n)).forward.__get__(getattr(model,n),type(getattr(model,n)))
                    try:z2=fresh(model,frames,newrow);actual+=1;callcount+=1;assert exact(z,z2)
                    finally:
                        for n,f in wrappers.items():getattr(model,n).forward=f
                        counts['qualification_uncached_two_offset_views']=2
                    viewqualified.add(rec['dataset']);write(BASE/'qualification'/(rec['dataset']+'_new_text.json'),dict(status='pass',count=1,caption_cached_and_uncached_complete_outputs_exact=True,text_fusion_both_decoders_native_gates_recomputed=True,GT_read=False));del z2
                del z
            result['DIRECT']=fit.boundary_targets(off,teacher['Direct_boundary']['parsed']['interval']) if admitted['DIRECT'] else [q.copy() for q in p0]
            # Parameter-free Direct uses the same native legal grid target MAP.
            if admitted['DIRECT']:
                iv=[]
                for o,q in zip(off,result['DIRECT']):
                    s,e=o['ij'][:,int(q.argmax())];iv.append([o['ids'][s],o['ids'][e]+1])
                intervals['DIRECT']=[min(x[0] for x in iv),max(x[1] for x in iv)]
            else:intervals['DIRECT']=list(c['Native_interval'])
            trace=dict(case_id=cid,dataset=rec['dataset'],frozen_text_conditioned_forward_views=callcount,visual_only_cache_counts=dict(counts),original_query_fusion_never_reused_for_new_text=True,GT_read=False);traces.append(trace)
        path=BASE/'targets'/(cid+'.pt');r=save_pt(path,dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=c['source_case_sha256'],targets=result,admitted=admitted,view_intervals=intervals,Teacher_to_Native_KL={a:sum(float(np.sum(q[q>0]*(np.log(q[q>0])-fit.original._prob(o['base'])[1][q>0]))) for q,o in zip(result[a],off)) for a in ['SEM','DIRECT','PARA']},trace=trace,GT_read=False));r['case_id']=cid;files.append(r);write(BASE/'TARGET_STATUS.json',dict(status='running',done=len(files),total=32,count=32,GT_read=False),mutable=True);del c,frames,result;gc.collect();torch.cuda.empty_cache()
    assert state_hash(model.state_dict())==model._fixed_full_state_hash and qualified=={'hc2','vidstg'}
    write(BASE/'TARGET_BARRIER.json',dict(status='complete',count=32,files=files,actual_frozen_text_view_forwards=actual,actual_two_offset_calls=actual*2,cache_traces=traces,GT_read=False));write(BASE/'TARGET_ROOT_READBACK.json',dict(status='pass',count=32,zero_query_and_new_text_actual_GPU_cache_qualification=True,view_qualified_domains=sorted(viewqualified),no_text_conditioned_feature_reuse=True,GT_read=False))

class_placeholder=None

def readout():
    import torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts.run_decota_paper_main_v1 import model_for
    from scripts import stvg_temporal_semantic_view_fit32_v1 as fit
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from methods.decota_final_simplified_v1.objectives import prediction
    verify();sys.addaudithook(guard);torch.set_num_threads(2);files=[];model=None;src=None;fits={f['case_id']:f for f in read(BASE/'FIT_BARRIER.json')['files']};targets={f['case_id']:f for f in read(BASE/'TARGET_BARRIER.json')['files']}
    class Affine(torch.nn.Module):
        def __init__(self,head,theta):super().__init__();self.head=head;self.weight=torch.nn.Parameter(torch.tensor(theta,dtype=torch.float64));assert self.weight.numel()==512
        def forward(self,x):
            y=self.head(x);a=x.detach().cpu().double().numpy()@self.weight.detach().cpu().numpy();return y+torch.from_numpy(a).to(device=y.device,dtype=y.dtype)
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        verify();c,_=load_case(rec);cid=rec['probe_alias'];source='vidstg' if rec['dataset']=='hc2' else 'hcstvg2'
        if source!=src:
            if model is not None:del model;gc.collect();torch.cuda.empty_cache()
            model=model_for(source);src=source
        base=cached_source(model,c);f=fits[cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);t=targets[cid];assert sha(ROOT/t['path'])==t['sha256'];td=torch.load(ROOT/t['path'],map_location='cpu',weights_only=False);off=fit.offsets(c,_hidden_from_source_case(c));head=base.head;outputs={}
        for arm in ['SEM','DIRECT','PARA']:
            ff=d['fits'][arm];base.head=Affine(head,ff['theta'])
            with torch.inference_mode():v=base.reference_values(capture=False)
            assert torch.equal(v['boxes'].cpu(),c['zero']['boxes']);expected=ff['final_logits'];assert all(torch.equal(a.cpu(),b) for a,b in zip(v['logits'],expected)), 'actual inserted affine head must match optimized full-grid logits exactly'
            native=prediction(v['logits'],v['boxes'],c['records'],list(map(int,c['frame_ids'])));assert native['physical_interval']==ff['final_interval'];outputs[arm+'_STUDENT']=dict(boxes=np.asarray(c['native_boxes_xyxy_normalized']).copy(),interval=ff['final_interval']);outputs[arm+'_VIEW']=dict(boxes=np.asarray(c['native_boxes_xyxy_normalized']).copy(),interval=td['view_intervals'][arm])
        base.head=head;outputs['Native']=dict(boxes=np.asarray(c['native_boxes_xyxy_normalized']).copy(),interval=list(c['Native_interval']));r=save_pt(BASE/'predictions'/(cid+'.pt'),dict(case_id=cid,dataset=rec['dataset'],arms=outputs,source_case_sha256=c['source_case_sha256'],fit_sha256=f['sha256'],target_sha256=t['sha256'],actual_full_native_prefix_both_decoder_temporal_head_readout=True,frozen_spatial_boxes_exact=True,GT_read=False));r['case_id']=cid;files.append(r);write(BASE/'READOUT_STATUS.json',dict(status='running',done=len(files),total=32,count=32,GT_read=False),mutable=True);del c,base,d,td;gc.collect();torch.cuda.empty_cache()
    write(BASE/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False));write(BASE/'PRIMARY_ROOT_READBACK.json',dict(status='pass',count=32,all_original_query_integrated_head_outputs_exact=True,all_spatial_boxes_unchanged=True,actual_all_output_frames=True,GT_read=False))
if __name__=='__main__':globals()[sys.argv[1]]()
