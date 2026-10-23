"""Read-only cached K8 bridge for generalized constructor; no model/expert/GT."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools'),str(Path(__file__).parent)]
import torch,numpy as np
from full import RUN,PROVIDER,read,sha,atomic
from budget import variable_geometry,evidence,positions
from consensus_exact import estimate
from stvg.consensus import encode,apply
from stvg.decode import decode
from stvg.teacher_bridge import evidence_key
from stvg.predictor import output_writeback
from stvg.representation import extract
from stvg.spatial import evidence_dp
D=R/'outputs/paper_review_suite_v1';torch.set_num_threads(1);klass=variable_geometry();rows=read(D/'selection.json')['hc2']['screen'][:4];proof=[]
for row in rows:
 key=row['id'];rc=read(D/'supports/hc2'/(key+'.json'));shared=torch.load(D/rc['shared_path'],map_location='cpu',weights_only=False);cap=torch.load(shared['parent_capture'],map_location='cpu',weights_only=False);frames,ids=decode(row['metadata'],'hc2');pos=list(map(int,cap['positions8']));fp=read(RUN/'plan.json')['provider_runtime_fingerprint'];k,context=evidence_key(frames[pos],row['metadata']['caption'],positions=pos,frame_ids=ids,fingerprint=fp,dataset='hc2');roots=[PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',R.parent/'stvg-mainline/outputs/tastvg_paper_v1/teacher_cache'];folder=next(x for x in roots if(x/(k+'.json')).exists());meta=read(folder/(k+'.json'));assert meta['context']==context and sha(folder/(k+'.pt'))==meta['sha256'];teacher=torch.load(folder/(k+'.pt'),map_location='cpu',weights_only=False);ref=np.asarray(teacher['boxes'],np.float64);cfg=read(R/'configs/default.json')['datasets']['hc2'];cal=cfg['E_calibration'];est=estimate(encode(ref)-encode(cap['Source'])[pos],cal['scale'],cal['truncation']);E=output_writeback(cap['Source'],apply(cap['Source'],est['delta']),cfg['initialization_gamma']);assert np.array_equal(E,cap['E']);ev=evidence(shared['spatial_features'],shared['spatial_valid'],ids,pos,teacher['native_mask_logits'],shared['spatial_grid_shape'],teacher['input_shape'][-2:]);g=klass(ev,E,ref)
 for field in ['source_anchors','side_available','side_reference_boxes','side_prototypes','seed_features','seed_valid','boxes','features','association_scores','valid','u','supported','link']:assert torch.equal(getattr(g,field),cap['geometry'][field]),field
 aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();f,_=extract(dict(frame_ids=ids,native_local_grid=shared['native_local_grid']),aux);assert np.array_equal(f,cap['content_features']);proof.append(dict(query=len(proof)+1,geometry_fields_bitwise=14,pool_bitwise=True,initialization_bitwise=True,expert_calls=0))
atomic(D/'budget_K8_bridge.json',dict(status='passed',queries=4,proof=proof,new_model=0,new_expert=0,labels_read=False));print('Budget K8 geometry/pool bridge4 passed')
