"""Production promotion proof against sealed K16 on all256 cached official inputs."""
import sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from full import OLD,RUN,PUBLIC,sha,read,atomic,check,identity
import numpy as np,torch
from stvg.prototype import one_step
from stvg.paper_panel import derive
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.semantic import target_semantics
from stvg.runtime import setup,lease,load_source
from stvg.decode import decode
from stvg.native import frozen_native_forward
from stvg.temporal_hidden import cache_temporal
from stvg.global6 import read_logits,posterior
from stvg.query import QuerySubjectParser

def main():
    p=check(True);setup();torch.set_num_threads(1);lock=lease();begin=time.perf_counter()
    A=OLD/'outputs/target_semantic128_collect_v3';B=OLD/'outputs/target_semantic128_shrink_v1'
    ap=read(A/'plan.json');aseal=read(A/'sealed.json');bseal=read(B/'sealed.json');counts=dict(CPU_math=0,ASA_head_forwards=0,Source_queries=0,new_expert=0);parser=None
    atomic(RUN/'regression_worker.json',identity())
    try:
        for ds in ['hc2','vidstg']:
            model=load_source(read(ROOT/'configs/default.json')['datasets'][ds]);model.vid.cpu();model.vis_encoder.cpu();fresh=False
            for item in [x for x in ap['items'] if x['dataset']==ds]:
                key=item['alias'];ae=aseal['files'][key];be=bseal['files'][key]
                assert sha(A/ae['path'])==ae['sha256'] and sha(A/ae['capture_path'])==ae['capture_sha256'] and sha(B/be['trace_path'])==be['trace_sha256'] and sha(B/be['path'])==be['sha256']
                x=torch.load(A/ae['path'],map_location='cpu',weights_only=False);cap=torch.load(A/ae['capture_path'],map_location='cpu',weights_only=False);prior=torch.load(B/be['trace_path'],map_location='cpu',weights_only=False)
                q,step=one_step(x['semantic_p']['TargetContext'],x['q0']);assert np.array_equal(q,prior['q']['ShrinkK16'])
                cap['semantic_p']=x['semantic_p']['TargetContext'];arr,_=derive(cap)
                with np.load(B/be['path'],allow_pickle=False) as z:
                    assert np.array_equal(arr['Full'],z['ShrinkK16']) and np.array_equal(arr['Full_interval'],z['ShrinkK16_interval'])
                    assert np.array_equal(arr['Global6'],z['NativeG6']) and np.array_equal(arr['Source'],z['Source'])
                g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids'];aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final']
                model.t_spatial_clas.cuda();prob,u,_,_=target_semantics(model,cap,aux);model.t_spatial_clas.cpu();counts['ASA_head_forwards']+=2
                assert np.array_equal(prob,x['semantic_p']['TargetContext']) and np.array_equal(u,x['target_u']),(key,'ASA target bitwise')
                if not fresh:
                    row=item['row'];frames,physical=decode(row['metadata'],ds);model.cuda()
                    if parser is None:parser=QuerySubjectParser(ROOT/'checkpoints/stanza')
                    rr=dict(input=row['metadata'],frame_ids=physical,parses={'subject':parser(row['metadata']['caption'])['subject']})
                    batch,recs,base=frozen_native_forward(model,frames,rr)
                    assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
                    tc=cache_temporal(model,base,recs,physical);logs,_=read_logits(model.temp_embed,[o['hidden'] for o in tc['offsets']],[o['native_logits'] for o in tc['offsets']]);assert np.array_equal(posterior(recs,logs,physical),x['q0'])
                    prob,_,_,_=target_semantics(model,dict(temporal_cache=tc,frame_ids=physical),aux);assert np.array_equal(prob,x['semantic_p']['TargetContext'])
                    counts['ASA_head_forwards']+=2;counts['Source_queries']+=1;fresh=True;del batch,base,tc,frames,logs;model.cpu();gc.collect();torch.cuda.empty_cache()
                counts['CPU_math']+=1;atomic(RUN/'regression_progress.json',dict(completed=counts['CPU_math'],total=256,counts=counts,seconds=time.perf_counter()-begin))
            del model;gc.collect();torch.cuda.empty_cache()
        assert counts['CPU_math']==256 and counts['Source_queries']==2
        out=dict(status='passed',queries=256,K16_q_interval_finalboxes_bitwise=True,Global6_Source_aliases_bitwise=True,actual_target_p_u_bitwise256=True,actual_Source_rebuilds_bitwise=2,counts=counts,seconds=time.perf_counter()-begin,GT_read=False,plan_sha256=sha(RUN/'plan.json'),code_sha256=sha(Path(__file__)))
        atomic(RUN/'regression.json',out);atomic(PUBLIC/'REGRESSION.json',out);print(out)
    finally:lock.close()

if __name__=='__main__':main()
