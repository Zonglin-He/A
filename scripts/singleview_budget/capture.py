"""Isolated research capture copied from the frozen Predictor.capture.

Only temporal branch selection/readout changes. Every measured arm executes its
own native224, actual eight-frame ReferDINO and full BoxFixed pipeline. Original
stvg sources remain immutable; production Strong remains the unmodified reference.
"""
import gc,json,time
import numpy as np
import torch
from stvg.runtime import ROOT,load_source
from stvg.native import frozen_native_forward
from stvg.tensors import detached
from stvg.readout import prediction
from stvg.temporal import probabilities,mass_table,final_mbr
from stvg.positions import observation_positions
from stvg.initialization import initialize
from stvg.teacher_bridge import refer_native
from stvg.evidence import SourceEvidence
from stvg.spatial import build_geometry,evidence_dp


@torch.no_grad()
def capture(self,frames,frame_ids,metadata,parses=None,*,profile='Strong',export_replay=False,teacher_override=None):
    mode=profile
    if mode not in ('Strong','Global6','Single448_0','Single448_1','Single320'):raise ValueError('Unknown temporal mode')
    self.check();begin=time.perf_counter();tick=begin
    if self.source is None:self.source=load_source(self.config)
    source_load=time.perf_counter()-tick;tick=time.perf_counter()
    if parses is None:
        from stvg.query import QuerySubjectParser
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
    if export_replay:
        from stvg.tube_replay import cache_temporal
        replay=cache_temporal(model,base,recs,U)
    layers=None
    layers=[model.temp_embed(h).detach().cpu()for h in base.temporal_inputs]
    assert all(torch.equal(x[-1],z)for x,z in zip(layers,logits))
    lq=[mass_table(probabilities(recs,[x[k]for x in layers]))[1]for k in range(6)]
    qG=np.mean(lq,axis=0)
    costs['native_time_postprocess_seconds']=time.perf_counter()-tick
    del batch,base,layers;gc.collect();torch.cuda.empty_cache()
    extra=None;selected=qG
    if mode!='Global6':
        from stvg.temporal_capture import frozen_temporal_forward
        from stvg.inputs import resolution
        tick=time.perf_counter()
        if mode=='Strong':
            with resolution(448):batch,hrrecs,hr=frozen_temporal_forward(model,frames,row,audit=False)
            hu,hq=mass_table(probabilities(hrrecs,detached(hr.zero['logits'],'cpu')));assert hu==U
            del batch,hr
            nviews=2;size=448
        else:
            from scripts.singleview_budget.model import forward,posterior as single_posterior
            size=320 if mode=='Single320'else 448;offset=1 if mode=='Single448_1'else 0
            with resolution(size):extra=forward(model,frames,row,(offset,))
            hq=single_posterior(extra,U);nviews=1
        selected=.5*hq+.5*qG
        torch.cuda.synchronize();costs['extra_temporal_branch_seconds']=time.perf_counter()-tick
        costs['Source448_seconds']=costs['extra_temporal_branch_seconds']if size==448 else 0.
        costs['Source448_calls']=nviews if size==448 else 0
        costs['Source320_calls']=nviews if size==320 else 0
        costs['actual_extra_views']=nviews;costs['extra_resolution']=size
    intervals['Strong']=final_mbr(U,selected)['interval'];posterior.update(Global6=qG,Strong=selected)
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
        temporal_cache=replay,records=recs,cost=costs,GT_read=False,experimental_profile=mode,extra=extra)
    tick=time.perf_counter();cache=self.initialize_cached(cache);costs['ERef_seconds']=time.perf_counter()-tick
    tick=time.perf_counter();evidence=SourceEvidence.from_capture(spatial,U,positions,teacher['native_mask_logits'],teacher['input_shape'][-2:],selected_query=teacher['selected_query'])
    cache['evidence']=evidence;cache['geometry']=build_geometry(evidence,cache['E'],teacher['boxes']);costs['evidence_seconds']=time.perf_counter()-tick
    costs['capture_wall_seconds']=time.perf_counter()-begin;return cache
