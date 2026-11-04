import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import sys,json,time,hashlib,gc
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from model_utils import cpu_source,spatial_parameters,spatial_modules
from tools.full import atomic,sha,identity
import torch
D=R/'outputs/expert_guided_ttft_v1';P=R/'results/expert_guided_ttft_v1'
if __name__=='__main__':
 torch.set_num_threads(1);start=time.perf_counter();out={}
 for ds in ['hc2','vidstg']:
  assert not(D/(ds+'_CPU_doctor.json')).exists();tick=time.perf_counter();model,metadata=cpu_source(ds);named=spatial_parameters(model);params=[dict(name=n,shape=list(p.shape),scalars=p.numel(),dtype=str(p.dtype))for n,p in named];assert not any(p.requires_grad for p in model.parameters());assert all(p.device.type=='cpu'for p in model.parameters());w=dict(dataset=ds,source_dataset=str(model.cfg.DATASET.NAME),spatial_scope='ground_decoder.pos_fc + full ground_decoder.decoder including aliased original bbox_embed; temporal decoder/time_fc/time_embed/encoders frozen',parameters=params,unique_parameter_tensors=len(named),unique_parameter_scalars=sum(p.numel()for n,p in named),decoder_layers=len(model.ground_decoder.decoder.layers),source_training_loss_weights=dict(L1=float(model.cfg.SOLVER.BBOX_COEF),GIoU=float(model.cfg.SOLVER.GIOU_COEF)),source_training_max_grad_norm=float(model.cfg.SOLVER.MAX_GRAD_NORM),checkpoint_load=metadata.get('checkpoint_load'),worker=identity(),model_CPU_constructions=1,Source_forward_calls=0,GPU_calls=0,optimizer_steps=0,actual_seconds=time.perf_counter()-tick);atomic(D/(ds+'_CPU_doctor.json'),w);atomic(P/(ds+'_spatial_parameter_scope.json'),{k:v for k,v in w.items()if k!='checkpoint_load'});out[ds]=w;del model,named;gc.collect()
 atomic(D/'CPU_doctor_complete.json',dict(status='both_source_checkpoints_strict_CPU_loaded_spatial_scope_verified',worker=identity(),new_Source_forwards=0,new_expert=0,GPU=0,optimizer_steps=0,model_CPU_constructions=2,elapsed_wall_seconds=time.perf_counter()-start));print(json.dumps({ds:{k:v[k]for k in ['unique_parameter_tensors','unique_parameter_scalars','source_training_loss_weights','source_training_max_grad_norm']}for ds,v in out.items()}),flush=True)
