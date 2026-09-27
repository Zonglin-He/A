"""One fixed source-train task/aux gradient decomposition, no optimizer step."""
from pathlib import Path
import math
import torch
from scripts.desta3d_v2_p0 import read,write,adapter_sha256
from vg_tta.desta3d_v2_ptd import branch_injection
from vg_tta.desta3d_v2_source import split_source_loss_masks
from vg_tta.desta3d_v2_training import source_evidence_losses


def audit_source_gradient(model,processor,adapter,row,fields,output_dir):
    from scripts.desta3d_source_fit_v1 import _training_inputs
    from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss
    root=Path(__file__).resolve().parents[1]
    assert row['split']=='train'
    records=read(root/'artifacts/desta3d_v1/source_fit/SOURCE_TRAIN_RECORDS.json')
    record=next(x for x in records if x['key']==row['key'])
    if not record['response_eligible']:
        write(output_dir/'SOURCE_GRADIENT_DECOMPOSITION.json',{'key':row['key'],'status':'fixed query lacks legal CE; retained, no replacement','GT_scope':'source training only'})
        return
    start_hash=adapter_sha256(adapter);flags={n:p.requires_grad for n,p in adapter.named_parameters()}
    adapter.set_train_stage('B integration');adapter.train()
    model.train();model.model.visual.eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    data,_,_=_training_inputs(processor,model,row,record);masks=split_source_loss_masks(data,processor.tokenizer)
    params=[p for n,p in adapter.named_parameters() if n.startswith('shared_stem.')]
    assert params
    vectors={};losses={}
    for branch in ('event','spatial'):
        adapter.zero_grad(set_to_none=True)
        d=dict(data);d['labels']=data['labels'].clone();d['labels'][~masks[branch].to(d['labels'].device)]=-100
        with branch_injection(model,adapter,data,fields,branch):
            loss,stats=joint_loss(model,d)
            loss.backward()
        vectors[branch]=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().reshape(-1).cpu() for p in params])
        losses[branch]={'loss':float(loss.detach()),'stats':stats,'tokens':int(masks[branch].sum())}
        del loss,d
    adapter.zero_grad(set_to_none=True)
    out=adapter(fields['visual_grid'],fields['query_tokens'],query_mask=fields['query_mask'],frame_times=fields['frame_times'])
    aux=source_evidence_losses(out,record);loss=aux['ref']+aux['event'];loss.backward()
    vectors['aux_unweighted']=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().reshape(-1).cpu() for p in params])
    losses['aux_unweighted']={'ref':float(aux['ref'].detach()),'event':float(aux['event'].detach()),'loss':float(loss.detach())}
    vectors['task_sum']=vectors['event']+vectors['spatial'];vectors['aux_at_B_weight']=vectors['aux_unweighted']*.1
    assert all(torch.isfinite(x).all() for x in vectors.values())
    norms={n:float(x.norm()) for n,x in vectors.items()};angles={}
    for a,b in [('event','spatial'),('event','aux_unweighted'),('spatial','aux_unweighted'),('task_sum','aux_unweighted')]:
        denom=norms[a]*norms[b]
        cosine=float(torch.dot(vectors[a],vectors[b]))/denom if denom>0 else None
        cosine=max(-1.,min(1.,cosine)) if cosine is not None else None
        angles[a+'__'+b]={'cosine':cosine,'degrees':math.degrees(math.acos(cosine)) if cosine is not None else None,'zero_gradient_undefined':denom==0}
    assert all(p.grad is None for p in model.parameters())
    adapter.zero_grad(set_to_none=True);adapter.eval();model.eval()
    for n,p in adapter.named_parameters():p.requires_grad_(flags[n])
    assert adapter_sha256(adapter)==start_hash
    write(output_dir/'SOURCE_GRADIENT_DECOMPOSITION.json',{'key':row['key'],'source':row['source'],'parameter_scope':'shared_stem only',
      'parameters':sum(p.numel() for p in params),'norms':norms,'angles':angles,'losses':losses,
      'B_weighted_aux_to_sum_task_norm_ratio':norms['aux_at_B_weight']/norms['task_sum'] if norms['task_sum'] else None,
      'query_selection':'fixed metadata first source train query, no GT outcome selection','source_GT_used':True,'target_GT_read':False,'validation_GT_read':False,
      'backbone_grad_none':True,'adapter_unchanged':True,'optimizer_steps':0,'scope':'one-query mechanism diagnostic; cannot establish sustained conflict or justify PCGrad'})
