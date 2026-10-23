"""Single main method: native temporal prior + one-round Native2D feedback + BoxFixed.

Original native224 Source computation is unchanged. Its existing final-layer
logits supply both prototype seeds and the posterior prior; no Global6 head readout. No network-weight update, high-resolution branch or alternative profile is reachable.
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
    def capture(self,frames,frame_ids,metadata,parses=None,*,teacher_override=None):
        if self.config.get('temporal_mode','native2d')!='native2d':raise ValueError('Only the frozen region-kernel main method is supported')
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
        try:
            from .representation import observe_native_grid
            with observe_native_grid(model) as native_grid:batch,recs,base=frozen_native_forward(model,frames,row)
        finally:
            for hook in hooks:hook.remove()
        costs.update(native_time_decoder_calls=calls['time'],native_position_decoder_calls=calls['spatial'])
        native=prediction(base.zero['logits'],base.zero['boxes'],recs,frame_ids)
        torch.cuda.synchronize();costs['Source224_seconds']=time.perf_counter()-tick
        spatial=detached(base.spatial_inputs,'cpu');src=native['boxes'].cpu();logits=detached(base.zero['logits'],'cpu')
        tick=time.perf_counter();U,q=mass_table(probabilities(recs,logits));assert U==list(frame_ids)
        native_interval=final_mbr(U,q)['interval'];positions=observation_positions(U,native['indices'])
        intervals={'Native':native_interval};posterior={'Native':q};replay=None
        from .temporal_hidden import cache_temporal
        replay=cache_temporal(model,base,recs,U)
        costs['native_time_postprocess_seconds']=time.perf_counter()-tick
        del batch,base;gc.collect();torch.cuda.empty_cache()
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
            native_hidden=replay,native_logits=logits,native_local_grid=native_grid,records=recs,cost=costs,GT_read=False)
        tick=time.perf_counter();cache=self.initialize_cached(cache);costs['ERef_seconds']=time.perf_counter()-tick
        tick=time.perf_counter();evidence=SourceEvidence.from_capture(spatial,U,positions,teacher['native_mask_logits'],teacher['input_shape'][-2:],selected_query=teacher['selected_query'])
        cache['evidence']=evidence;cache['geometry']=build_geometry(evidence,cache['E'],teacher['boxes']);costs['evidence_seconds']=time.perf_counter()-tick
        costs['capture_wall_seconds']=time.perf_counter()-begin;return cache
    @torch.no_grad()
    def finish(self,cache):
        from .prior import native_prior
        begin=time.perf_counter()
        q0=native_prior(cache['records'],cache['native_logits'],cache['frame_ids'])
        assert np.array_equal(q0,cache['posterior']['Native'])
        aux=evidence_dp(cache['geometry'].for_interval([cache['frame_ids'][0],cache['frame_ids'][-1]+1]))['final']
        from .representation import extract
        from .prototype import one_step
        tick=time.perf_counter();features,support=extract(cache,aux.numpy())
        pooling_seconds=time.perf_counter()-tick;tick=time.perf_counter();q,prototype=one_step(features,q0);prototype_seconds=time.perf_counter()-tick
        interval=final_mbr(cache['frame_ids'],q)['interval'];posterior_seconds=time.perf_counter()-tick
        tick=time.perf_counter();spatial=evidence_dp(cache['geometry'].for_interval(interval));dp_seconds=time.perf_counter()-tick
        cost=dict(region_pooling_seconds=pooling_seconds,prototype_seconds=prototype_seconds,additional_classifier_forwards=0,finish_wall_seconds=time.perf_counter()-begin,additional_time_head_forwards=0,Global6_aggregation=False,posterior_seconds=posterior_seconds,final_DP_seconds=dp_seconds,actual_updates=0,extra_decoder_calls=0)
        return dict(Source=np.asarray(cache['Source']),E_Ref=np.asarray(cache['E']),BoxFixed=spatial['final'].numpy(),boxes=spatial['final'].numpy(),interval=list(interval),
            intervals={**cache['intervals'],'Selected':list(interval)},posterior=q,content_features=features,prototype=prototype,spatial=spatial,temporal_mode='native2d',positions8=list(cache['planned_positions']),cost=cost,capture_cost=cache['cost'],initialization=cache['initialization'],GT_read=False)
    def predict(self,frames,frame_ids,metadata,key=None,*,temporal_mode=None):
        if temporal_mode not in (None,'native2d'):raise ValueError('Only the frozen region-kernel main method is supported')
        return self.finish(self.capture(frames,frame_ids,metadata))

Predictor=BoxFixedPredictor
