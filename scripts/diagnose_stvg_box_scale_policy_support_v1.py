"""Fixed local Gaussian reward observability only; no fit, optimizer or GT reward."""
import sys,hashlib,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from scripts.stvg_temporal_caption_query_v1 import ROOT,roster,catalog,read,write
from scripts.stvg_target_teacher_student_v1 import load_original_opd_config
from scripts.stvg_box_scale_grid_v1 import iou_cxcywh_numpy,diou_reward_cxcywh_numpy
B=ROOT/'artifacts/stvg_box_scale_global_evidence_v1';torch.set_num_threads(2);controls={r['case']:r for r in read(B/'source_position_control001/ROOT_READBACK.json')['rows']};grid={r['case_id']:r for r in read(B/'GRID_INPUT_BARRIER.json')['records']};prior={r['case']:r for r in read(B/'ROOT_PRELIMINARY_POSTSEAL_METRICS.json')['rows']};rows=[]
for r in roster():
 cid=r['case_id'];alias=('HC'if cid.startswith('hc2')else 'V')+f"{int(cid.split('_')[-1])+1:02d}";c,_=catalog(r);g=grid[cid];box=torch.tensor(g['grid']['student_baseline']['cxcywh'],dtype=torch.float32);cfg=load_original_opd_config(r['dataset']);gen=torch.Generator().manual_seed(int(hashlib.sha256(('box-scale-policy-diagnostic|'+cid).encode()).hexdigest()[:15],16));noise=torch.randn(16,4,generator=gen);noise=torch.cat((noise,-noise),0);samples=torch.sigmoid(torch.logit(box)[None,:]+cfg['sigma']*noise).numpy()
 for kind in ['mask','marked']:
  keep=controls[alias][kind+'_grid_plus_source_keeps_source'];id=prior[alias]['mask_selected_window_id'if kind=='mask'else 'marked_selected_window_id'];w=next(w for w in g['grid']['windows']if w['window_id']==id);target=box.numpy()if keep else np.array([w['cx'],w['cy'],w['w'],w['h']]);ir=iou_cxcywh_numpy(samples,target);dr=diou_reward_cxcywh_numpy(samples,target);rows.append(dict(case=alias,dataset=r['dataset'],reference_protocol=kind,reference_from_source=keep,samples=32,sigma=cfg['sigma'],tau=cfg['tau'],IoU_all_exact_zero=bool((ir==0).all()),IoU_reward_variance=float(ir.var()),DIoU_reward_variance=float(dr.var()),DIoU_all_equal=bool((dr==dr[0]).all()),IoU_reward_min=float(ir.min()),IoU_reward_max=float(ir.max()),DIoU_reward_min=float(dr.min()),DIoU_reward_max=float(dr.max()),DIoU_lambda_center=1,GT_used_for_sampling_or_reward=False))
write(B/'ROOT_LOCAL_POLICY_REWARD_DIAGNOSTIC.json',dict(status='pass',scope='CPU32 fixed antithetic Gaussian samples per source, same samples IoU vs DIoU; no parameter fit/no efficacy claim',count=16,rows=rows,new_student_fits=0,optimizer_steps=0,GT_read=False,CPU_only=True))
print(json.dumps(dict(IoU_exact_zero_case_protocols=sum(r['IoU_all_exact_zero']for r in rows),case_protocols=16,DIoU_flat_case_protocols=sum(r['DIoU_all_equal']for r in rows))))
