"""Native-WHEN, single frozen DINO, online spatial Gaussian-policy adaptation."""
import hashlib,json
from pathlib import Path
import torch
from vg_tta.decota_spatial_opd_tunable_v1 import fit,CONFIG
from vg_tta.spatial_online_state_v1 import arrival
from methods.decota_final_simplified_v1.tensors import detached

class SpatialOPDPredictor:
    def __init__(self,student,expert,*,dataset,config=None):
        if dataset not in ('vidstg','hc2'):raise ValueError(dataset)
        self.student=student.eval().requires_grad_(False)
        self.expert=expert;self.expert.model.eval().requires_grad_(False)
        self.dataset=dataset;self.previous=None
        if config is None:
            saved=json.loads((Path(__file__).parent/'configs.json').read_text())
            config=saved['datasets'][dataset]['config']
        self.config=dict(config)

    def reset_stream(self):self.previous=None

    def predict(self,frames,frame_ids,metadata,*,parses,key=None):
        # Parses are the original deterministic text-only subject/context cache.
        from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
        from methods.decota_final_simplified_v1.observations import observations
        from methods.decota_final_simplified_v1.objectives import prediction
        key=key or hashlib.sha256(json.dumps([metadata['caption'],frame_ids],ensure_ascii=False).encode()).hexdigest()
        row=dict(input=metadata,frame_ids=list(frame_ids),parses=parses,key=key)
        batch,records,base=frozen_forward(self.student,frames,row)
        native=prediction(base.zero['logits'],base.zero['boxes'],records,frame_ids)
        evidence=observations(self.expert,parses,frames,frame_ids,native['indices'],audit=True)
        initial=arrival(base.initial,self.previous,'O-split')
        result=fit(base,initial,evidence,frame_ids,key,'on_policy',config=self.config)
        alpha=self.config['writeback']
        self.previous={n:(torch.zeros_like(v.cpu()) if n=='spatial.query_residual'
                          else v.cpu()+(result['state'][n]-v.cpu())*alpha) for n,v in initial.items()}
        return dict(boxes=result['final'],indices=native['indices'],physical_interval=native['physical_interval'],
                    frame_ids=list(frame_ids),spatial_audit=result,expert=evidence,
                    temporal_frozen=True,GT_used=False,active_parameters=1792)
