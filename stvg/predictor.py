"""Single ReferDINO ERef initialization and complete-box fixed inference.

Strong remains the known temporal reference. Native224 and one tube-guided
cached temporal replay are explicit modes; no parameter adaptation occurs.
"""
import gc,json,time
import numpy as np
import torch
from .runtime import ROOT,load_source
from .native import frozen_native_forward
from .tensors import detached
from .readout import prediction
from .temporal import probabilities,mass_table,final_mbr
from .positions import observation_positions
from .initialization import initialize
from .teacher_bridge import refer_native
from .evidence import SourceEvidence
from .spatial import build_geometry,evidence_dp


def output_writeback(E,raw,alpha):
    """Retained numeric interpolation for ERef strength and legacy callers."""
    E=np.asarray(E,dtype=np.float64);raw=np.asarray(raw,dtype=np.float64);alpha=float(alpha)
    assert np.isfinite(E).all()and np.isfinite(raw).all()and(E[:,2:]>0).all()and(raw[:,2:]>0).all()
    if alpha==0:return E.copy()
    if alpha==1:return raw.copy()
    return np.concatenate((E[:,:2]+alpha*(raw[:,:2]-E[:,:2]),E[:,2:]*np.exp(alpha*np.log(raw[:,2:]/E[:,2:]))),-1)


class BoxFixedPredictor:
    def __init__(self,dataset='hc2',config=None,source=None,check=lambda:None):
        self.config=dict(json.loads((ROOT/'configs/default.json').read_text())['datasets'][dataset])if config is None else dict(config)
        self.source=source;self.check=check;self.parser=None;self.teacher_service=None
    def initialize_cached(self,cache):
        raw,meta=initialize(np.asarray(cache['Source'],dtype=np.float64),cache['planned_positions'],np.asarray(cache['teacher_native']['boxes'],dtype=np.float64),self.config['E_calibration'])
        gamma=float(self.config['initialization_gamma']);E=output_writeback(cache['Source'],raw,gamma)
        meta.update(initialization_gamma=gamma,unscaled_delta=meta['delta'],delta=(gamma*np.array(meta['delta'])).tolist())
        return {**cache,'E':E,'initialization':meta}
    @torch.no_grad()
    def capture(self,frames,frame_ids,metadata,parses=None,*,temporal_mode=None,export_replay=False,teacher_override=None):
        mode=temporal_mode or self.config.get('temporal_mode','strong')
        if mode not in ('strong','native224','tube_replay'):raise ValueError('Unknown temporal mode')
        self.check();begin=time.perf_counter();tick=begin
        if self.source is None:self.source=load_source(self.config)
        source_load=time.perf_counter()-tick;tick=time.perf_counter()
        if parses is None:
            from .query import QuerySubjectParser
            if self.parser is None:self.parser=QuerySubjectParser(ROOT/'checkpoints/stanza')
            parses={'subject':self.parser(metadata['caption'])['subject']}
        parse_seconds=time.perf_counter()-tick;tick=time.perf_counter();model=self.source.cuda();torch.cuda.synchronize();source_to_GPU=time.perf_counter()-tick
        row=dict(input=metadata,frame_ids=list(frame_ids),parses=parses)
        costs=dict(parse_seconds=parse_seconds,Source_load_seconds=source_load,Source_to_GPU_seconds=source_to_GPU,Source448_seconds=0.,Source448_calls=0,Source224_offsets=2)
        tick=time.perf_counter();calls={'time':0,'spatial':0}
        def count_time(m,a,kw):calls['time']+=1
        def count_spatial(m,a,kw):calls['spatial']+=1
        hooks=[model.ground_decoder.time_decoder.register_forward_pre_hook(count_time,with_kwargs=True),model.ground_decoder.decoder.register_forward_pre_hook(count_spatial,with_kwargs=True)]
        try:batch,recs,base=frozen_native_forward(model,frames,row)
        finally:
            for hook in hooks:hook.remove()
        costs.update(native_time_decoder_calls=calls['time'],native_position_decoder_calls=calls['spatial'])
        native=prediction(base.zero['logits'],base.zero['boxes'],recs,frame_ids)
        torch.cuda.synchronize();costs['Source224_seconds']=time.perf_counter()-tick
        spatial=detached(base.spatial_inputs,'cpu');src=native['boxes'].cpu();logits=detached(base.zero['logits'],'cpu')
        tick=time.perf_counter();U,q=mass_table(probabilities(recs,logits));assert U==list(frame_ids)
        native_interval=final_mbr(U,q)['interval'];positions=observation_positions(U,native['indices'])
        intervals={'Native':native_interval};posterior={'Native':q};replay=None
        if mode=='tube_replay'or export_replay:
            from .tube_replay import cache_temporal
            replay=cache_temporal(model,base,recs,U)
        layers=None
        if mode=='strong':
            layers=[model.temp_embed(h).detach().cpu()for h in base.temporal_inputs]
            assert all(torch.equal(x[-1],z)for x,z in zip(layers,logits))
            lq=[mass_table(probabilities(recs,[x[k]for x in layers]))[1]for k in range(6)]
        costs['native_time_postprocess_seconds']=time.perf_counter()-tick
        del batch,base;gc.collect();torch.cuda.empty_cache()
        if mode=='strong':
            from .temporal_capture import frozen_temporal_forward
            from .inputs import resolution
            tick=time.perf_counter()
            with resolution(448):batch,hrrecs,hr=frozen_temporal_forward(model,frames,row,audit=False)
            hu,hq=mass_table(probabilities(hrrecs,detached(hr.zero['logits'],'cpu')));assert hu==U
            strong=.5*hq+.5*np.mean(lq,axis=0);intervals['Strong']=final_mbr(U,strong)['interval'];posterior['Strong']=strong
            torch.cuda.synchronize();costs['Source448_seconds']=time.perf_counter()-tick;costs['Source448_calls']=2
            del batch,hr,layers
        tick=time.perf_counter();model.cpu();gc.collect();torch.cuda.empty_cache();costs['Source_to_CPU_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        if teacher_override is not None:
            teacher=teacher_override
            assert teacher.get('evidence_binding_verified')is True and teacher.get('native_mask_logits')is not None
            if list(teacher['positions8'])!=positions or list(teacher['frame_ids'])!=U:raise ValueError('Cached expert observation grid differs')
        elif self.teacher_service is None:teacher=refer_native(frames[positions],metadata['caption'])
        else:teacher=self.teacher_service(frames[positions],metadata['caption'],positions=positions,frame_ids=U,dataset=self.config['dataset'])
        costs['expert_current_wall_seconds']=time.perf_counter()-tick
        if teacher.get('native_mask_logits')is None:raise ValueError('Teacher cache lacks selected native masks; regenerate this evidence, do not add a second expert')
        if not teacher.get('nested_mask_all_false'):raise ValueError('Teacher padding geometry requires an explicit crop mapping')
        costs.update(ReferDINO=teacher.get('cost',{}),ReferDINO_load_seconds=teacher.get('model_load_seconds',0.),ReferDINO_query_forwards=teacher.get('query_forwards',1),ReferDINO_cache_hit=teacher.get('cache_hit',False),expert_instances=1)
        cache=dict(Source=src.numpy(),frame_ids=U,planned_positions=positions,spatial_inputs=spatial,teacher_native=teacher,intervals=intervals,posterior=posterior,
            temporal_cache=replay,records=recs,cost=costs,GT_read=False)
        tick=time.perf_counter();cache=self.initialize_cached(cache);costs['ERef_seconds']=time.perf_counter()-tick
        tick=time.perf_counter();evidence=SourceEvidence.from_capture(spatial,U,positions,teacher['native_mask_logits'],teacher['input_shape'][-2:],selected_query=teacher['selected_query'])
        cache['evidence']=evidence;cache['geometry']=build_geometry(evidence,cache['E'],teacher['boxes']);costs['evidence_seconds']=time.perf_counter()-tick
        costs['capture_wall_seconds']=time.perf_counter()-begin;return cache
    @torch.no_grad()
    def finish(self,cache,temporal_mode=None,*,permuted=False):
        mode=temporal_mode or self.config.get('temporal_mode','strong');geometry=cache['geometry'];cost={};begin=time.perf_counter();aux=None;replay=None
        if mode=='strong':interval=cache['intervals']['Strong'];posterior=cache['posterior']['Strong']
        elif mode=='native224':interval=cache['intervals']['Native'];posterior=cache['posterior']['Native']
        elif mode=='tube_replay':
            if cache['temporal_cache']is None:raise ValueError('tube_replay requires actual native temporal inputs captured during this forward')
            from .tube_replay import replay_temporal
            tick=time.perf_counter();aux=evidence_dp(geometry.for_full_timeline());cost['auxiliary_DP_seconds']=time.perf_counter()-tick
            replay=replay_temporal(self.source,cache['temporal_cache'],aux['final'],geometry.temporal_reliability(),alpha=.5,permuted=permuted)
            cost.update(replay['cost']);interval=replay['interval'];posterior=replay['posterior'];cost['auxiliary_DP_score']=aux['score']
        else:raise ValueError('Unknown temporal mode')
        tick=time.perf_counter();spatial=evidence_dp(geometry.for_interval(interval));cost['final_DP_seconds']=time.perf_counter()-tick
        cost.update(finish_wall_seconds=time.perf_counter()-begin,actual_updates=0,extra_decoder_calls=cost.get('extra_decoder_calls',0))
        return dict(Source=np.asarray(cache['Source']),E_Ref=np.asarray(cache['E']),BoxFixed=spatial['final'].numpy(),boxes=spatial['final'].numpy(),interval=list(interval),
            intervals={**cache['intervals'],'Selected':list(interval)},posterior=posterior,spatial=spatial,temporal_mode=mode,permuted=permuted,
            auxiliary=aux,temporal_audit=replay,positions8=list(cache['planned_positions']),cost=cost,capture_cost=cache['cost'],initialization=cache['initialization'],GT_read=False)
    def predict(self,frames,frame_ids,metadata,key=None,*,temporal_mode=None):
        mode=temporal_mode or self.config.get('temporal_mode','strong')
        cache=self.capture(frames,frame_ids,metadata,temporal_mode=mode,export_replay=mode=='tube_replay')
        return self.finish(cache,mode)

Predictor=BoxFixedPredictor
