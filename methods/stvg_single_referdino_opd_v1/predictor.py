"""Single ReferDINO initialization+OPD and TA-STVG-only Strong time branch."""
import json,time,gc
from pathlib import Path
import numpy as np,torch
from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
from methods.decota_final_simplified_v1.tensors import detached
from methods.decota_final_simplified_v1.objectives import prediction
from methods.stvg_referdino_opd_v1.temporal import probabilities,mass_table,final_mbr
from methods.stvg_referdino_opd_v1.state import Replay
from methods.stvg_referdino_opd_v1.policy import fit
from vg_tta.stvg_temporal_hires128_v1 import resolution
from .initialization import initialize,evidence_hash
from .positions import observation_positions
from .runtime import ROOT,load_source
from .teacher_bridge import refer_native

class SingleReferDINOOPDPredictor:
    def __init__(self,dataset='hc2',config=None,source=None,check=lambda:None):
        manifest=json.loads(Path(__file__).with_name('config.json').read_text())
        self.config=dict(manifest['datasets'][dataset])if config is None else dict(config)
        self.source=source;self.check=check
    def capture(self,frames,frame_ids,metadata,parses=None):
        start=time.perf_counter();self.check()
        load_start=time.perf_counter()
        if self.source is None:self.source=load_source(self.config)
        source_load=time.perf_counter()-load_start
        parse_start=time.perf_counter()
        if parses is None:
            from methods.decota_final_simplified_v1.observations import QuerySubjectParser
            parser=QuerySubjectParser(ROOT/'.cache/stanza');parses=dict(subject=parser(metadata['caption'])['subject']);del parser
        model=self.source.cuda();row=dict(input=metadata,frame_ids=list(frame_ids),parses=parses);cost=dict(parse_seconds=time.perf_counter()-parse_start,Source_load_seconds=source_load);tick=time.perf_counter()
        batch,recs,base=frozen_forward(model,frames,row);native=prediction(base.zero['logits'],base.zero['boxes'],recs,frame_ids);torch.cuda.synchronize();cost['Source224_seconds']=time.perf_counter()-tick
        spatial=detached(base.spatial_inputs,'cpu');src=native['boxes'].detach().cpu();logits=detached(base.zero['logits'],'cpu')
        with torch.no_grad(),torch.autocast('cuda',enabled=False):layers=[model.temp_embed(h).detach().cpu()for h in base.temporal_inputs]
        assert all(torch.equal(x[-1],z)for x,z in zip(layers,logits))
        U,q=mass_table(probabilities(recs,logits));assert U==list(frame_ids)
        lq=[mass_table(probabilities(recs,[x[k]for x in layers]))[1]for k in range(6)]
        native_interval=final_mbr(U,q)['interval'];positions=observation_positions(U,native['indices'])
        del batch,base;gc.collect();torch.cuda.empty_cache();tick=time.perf_counter()
        with resolution(448):batch,hrrecs,hr=frozen_forward(model,frames,row)
        hu,hq=mass_table(probabilities(hrrecs,detached(hr.zero['logits'],'cpu')));assert hu==U
        strong=.5*hq+.5*np.mean(lq,axis=0);strong_interval=final_mbr(U,strong)['interval'];torch.cuda.synchronize();cost['Source448_seconds']=time.perf_counter()-tick
        del batch,hr;model.cpu();gc.collect();torch.cuda.empty_cache()
        teacher=refer_native(frames[positions],metadata['caption']);tick=time.perf_counter()
        E,init=initialize(src.double().numpy(),positions,teacher['boxes'].numpy(),self.config['E_calibration']);cost['ERef_seconds']=time.perf_counter()-tick
        cost.update(ReferDINO=teacher['cost'],ReferDINO_load_seconds=teacher['model_load_seconds'],ReferDINO_process_seconds=teacher['process_wall_seconds'],ReferDINO_query_forwards=1,ReferDINO_frames=8,independent_GroundingDINO_forwards=0,expert_instances=1,peak_GPU_bytes=max(torch.cuda.max_memory_allocated(),teacher['peak_GPU_bytes']),capture_wall_seconds=time.perf_counter()-start)
        return dict(Source=src,E=E,frame_ids=U,planned_positions=positions,spatial_inputs=spatial,
            teacher_native=teacher,initialization=init,intervals=dict(Native=native_interval,Strong=strong_interval),
            posterior=dict(Native=q,Strong=strong),cost=cost,GT_read=False)
    def initialize_cached(self,cache):
        # Deliberately recompute initialization from Source + ReferDINO only.
        E,init=initialize(np.asarray(cache['Source'],dtype=np.float64),cache['planned_positions'],np.asarray(cache['teacher_native']['boxes'],dtype=np.float64),self.config['E_calibration'])
        return {**cache,'E':E,'initialization':init}
    def adapt(self,cache,key):
        if self.source is None:self.source=load_source(self.config)
        self.source.ground_decoder.decoder.cuda();cache=self.initialize_cached(cache)
        rp=Replay(self.source,cache,cache['Source'],cache['E'],cache['planned_positions'],self.check)
        try:
            out=fit(rp,cache['teacher_native']['boxes'],key,'Live',41,self.config)
            assert evidence_hash(out['target'].numpy())==cache['initialization']['evidence_sha256']
            assert torch.equal(out['path'][0]['boxes'],torch.as_tensor(cache['E'],dtype=torch.float64))
            return dict(fit=out,Source=np.asarray(cache['Source']),E_Ref=cache['E'],Live=out['final'].numpy(),initialization=cache['initialization'],intervals=cache['intervals'],posterior=cache['posterior'],cost=dict(actual_updates=out['actual_updates'],fit_seconds=out['seconds'],decoder_calls=rp.calls),GT_read=False)
        finally:del rp;gc.collect();torch.cuda.empty_cache()
    def predict(self,frames,frame_ids,metadata,key):
        cache=self.capture(frames,frame_ids,metadata);out=self.adapt(cache,key)
        return {**out,'boxes':out['Live'],'interval':cache['intervals']['Strong'],'capture_cost':cache['cost']}
