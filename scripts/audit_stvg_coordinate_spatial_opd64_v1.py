"""Independent geometric gradient, lineage, original A and official metric audit."""
import os,sys,time,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_coordinate_spatial_opd64_v1 import D,P,ARMS,ENDPOINTS,CONTRASTS,verify,read,load,put,sha

def main():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
 from vg_tta.stvg_discriminative_dense32_v1 import candidates,student_logits
 from vg_tta.stvg_coordinate_spatial_opd64_v1 import kernels,spatial_loss
 from vg_tta.stvg_roi_layout_opd32_v1 import build,feedback
 torch.set_num_threads(4);start=time.perf_counter();p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64 and bar['all_outputs_before_GT'];res=read(P/'RESULTS_PUBLIC.json');checks=collections.Counter();errors=collections.Counter();outs=collections.Counter();teacher_audit_seconds=0.
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;x=load(D/'predictions'/(r['alias']+'.pt'));src=load(ROOT/r['path']);features=load(ROOT/r['feature_path']);E=torch.as_tensor(x['outputs']['E_MBR']);allowed=x['allowed'];cfg=x['configuration'];rec=next(z for z in res['rows']if z['alias']==r['alias']);assert torch.equal(x['fits']['A_Live']['final'],src['fits'][r['arm']]['final'])
  ctx=build(features['maps'],features['valid'],features['positions'],torch.stack([torch.as_tensor(b)for b in features['boxes']])if len(features['boxes'])else[]);first=x['fits']['A_Live']['rounds'][0];tick=time.perf_counter();f=feedback(ctx,first['candidates'],x['noise'][0],.1);teacher_audit_seconds+=time.perf_counter()-tick
  for key in ['prob','prior','rewards','valid','informative']:
   if f[key].dtype==torch.bool:assert torch.equal(f[key],first['feedback'][key])
   else:
    err=float((f[key]-first['feedback'][key]).abs().max());errors['frozen_teacher_replay_float64']=max(errors['frozen_teacher_replay_float64'],err);assert err<1e-12
   checks['teacher_replay_fields']+=1
  for arm,fit in x['fits'].items():
   assert len(fit['rounds'])==2 and torch.equal(fit['path'][0]['eta'],torch.zeros_like(fit['path'][0]['eta']))and torch.equal(fit['path'][0]['boxes'],E)
   for k,t in enumerate(fit['rounds']):
    center=fit['path'][k]['eta']if arm.endswith('Live')else torch.zeros_like(fit['path'][k]['eta']);assert torch.equal(center,t['sampling_center']);d=center[:,None]+x['noise'][k];assert torch.equal(d,t['standardized_candidates']);assert torch.equal(candidates(E,center,x['noise'][k],.25),t['candidates']);assert torch.equal(fit['path'][k]['eta'],t['eta_before']);active=allowed&t['feedback']['informative'];lp=student_logits(d,t['eta_before'],t['feedback']['valid']);sp=lp.softmax(-1);q=t['feedback']['prob'];assert torch.equal(sp,t['student_prob']);g=((sp-q)[...,None]*d).sum(1)
    if arm.startswith('B'):
     K,enc=kernels(t['candidates']);assert torch.equal(enc['x'],t['spatial_encoding']['x'])and torch.equal(enc['y'],t['spatial_encoding']['y'])and torch.equal(enc['outside'],t['spatial_encoding']['outside']);Q=torch.einsum('nk,nkg->ng',q,K);Ps=torch.einsum('nk,nkg->ng',sp,K);assert torch.allclose(Q,t['spatial_target'],atol=1e-13,rtol=0)and torch.allclose(Ps,t['spatial_student'],atol=1e-13,rtol=0)
     ratio=torch.zeros_like(Ps);pos=Ps>0;ratio[pos]=Q[pos]/Ps[pos];gl=sp*(1-torch.einsum('nkg,ng->nk',K,ratio));g=(gl[...,None]*(d-t['eta_before'][:,None])).sum(1);err=float((g-t['raw_unmasked_gradient']).abs().max());errors['B_independent_analytic_gradient']=max(errors['B_independent_analytic_gradient'],err);assert err<2e-12
     # Independent area formula, including small and partially off-image boxes.
     nodes=active.nonzero().flatten().tolist();j=nodes[0]if nodes else 0
     for c in [0,1,16]:
      cx,cy,w,h=t['candidates'][j,c].tolist();l,rr=max(0.,cx-w/2),min(1.,cx+w/2);u,v=max(0.,cy-h/2),min(1.,cy+h/2);visible=max(0.,rr-l)*max(0.,v-u)
      if visible==0:assert K[j,c,-1]==1
      else:
       for iy,ix in [(0,0),(31,31),(63,63),(min(63,int((u+v)/2*64)),min(63,int((l+rr)/2*64)))]:
        val=max(0.,min(rr,(ix+1)/64)-max(l,ix/64))*max(0.,min(v,(iy+1)/64)-max(u,iy/64))/visible;err=abs(val-float(K[j,c,iy*64+ix]));assert err<1e-12;checks['scalar_cell_intersections']+=1
     nz=q>0;akl=torch.zeros_like(q);akl[nz]=q[nz]*(q[nz].log()-lp.log_softmax(-1)[nz]);assert (t['spatial_KL_per_node']-akl.sum(-1)).max()<2e-12;checks['KL_data_processing_rows']+=len(q)
     if r['alias']in ['NH001','NV001','CH001','CV001']and nodes:
      vv=t['eta_before'][j:j+1].clone().requires_grad_(True);dd=d[j:j+1].detach();qq=q[j:j+1].detach();kk=K[j:j+1].detach();valid=t['feedback']['valid'][j:j+1]
      def target(vv):return spatial_loss(student_logits(dd,vv,valid),qq,kk)[0].sum()
      gg=torch.autograd.grad(target(vv),vv)[0];fd=torch.zeros_like(vv)
      for e in range(2):
       delta=torch.zeros_like(vv);delta[0,e]=1e-5;fd[0,e]=(target(vv.detach()+delta)-target(vv.detach()-delta))/(2e-5)
      err=float((gg-fd).abs().max());assert err<1e-8;errors['real_support_finite_difference']=max(errors['real_support_finite_difference'],err);checks['real_support_gradient_FD']+=1
     outs['B_outside_valid']+=t['outside_valid_candidates'];outs['B_outside_active']+=t['outside_active_candidates']
    g[~active]=0;assert torch.allclose(g,t['gradient'],atol=2e-12,rtol=0);assert torch.equal(t['eta_before']-r['step']*t['gradient'],t['eta_after']);out=candidates(E,t['eta_after'],torch.zeros(len(E),1,2),.25)[:,0];out[~allowed]=E[~allowed];assert torch.equal(out,fit['path'][k+1]['boxes']);assert torch.equal(t['eta_after'][~active],t['eta_before'][~active]);checks['actual_round_updates_and_lineage']+=1
   assert torch.equal(fit['final'],fit['path'][-1]['boxes'])and np.array_equal(x['outputs'][arm],fit['final'].numpy());checks['final_student_readouts']+=1
  for route in ['A','B']:assert torch.equal(x['fits'][route+'_Fixed']['path'][1]['boxes'],x['fits'][route+'_Live']['path'][1]['boxes']);checks['Fixed_Live_first_round_exact']+=1
  for fit in x['fits'].values():
   for k in ['candidates','student_prob']:assert torch.equal(fit['rounds'][0][k],first[k]);checks['first_round_A_B_shared_support']+=1
   assert torch.equal(fit['rounds'][0]['feedback']['prob'],first['feedback']['prob'])
  meta=plans[r['dataset']][r['query_ordinal']];truth,span=gt[r['dataset']][0][r['query_ordinal']],gt[r['dataset']][1][r['query_ordinal']];ids=np.asarray(x['frame_ids']);known=np.array([int(f)in truth for f in ids]);obs=np.isin(np.arange(len(ids)),x['planned_positions']);assert int((known&~obs).sum())==rec['known_unobserved_actual']
  for arm,z in x['outputs'].items():
   assert np.array_equal(z[~allowed.numpy()],E.numpy()[~allowed.numpy()])and np.array_equal(z[:,2:],E.numpy()[:,2:]);vals=official(z,meta,truth,span,x['interval'],r['dataset']);tube=DenseTube(z,meta,truth,span,clip=r['dataset']=='hc2');si=np.array([iou_gt(z[i],truth[int(f)],meta,r['dataset'])if known[i]else np.nan for i,f in enumerate(ids)]);interp=~np.isin(tube.fids,ids);vals.update(unobserved_actual_s=float(si[known&~obs].mean())if(known&~obs).any()else None,observed_actual_s=float(si[known&obs].mean())if(known&obs).any()else None,benchmark_interpolated_s=float(tube.iou[interp].mean())if interp.any()else None)
   for e,a in vals.items():
    b=rec['values'][arm][e];assert a is None and b is None or a is not None and b is not None and abs(a-b)<1e-12;checks['independent_metrics']+=1
  checks['parents']+=1
 for pan,g in res['groups'].items():
  for ds,s in g.items():
   rr=[r for r in res['rows']if r['dataset']==ds and(pan=='combined'or r['panel']==pan)]
   for a,b in CONTRASTS:
    for ep in ENDPOINTS:
     v=np.array([100*(r['values'][a][ep]-r['values'][b][ep])for r in rr if r['values'][a][ep]is not None and r['values'][b][ep]is not None]);z=s['paired'][a+'_minus_'+b][ep];assert len(v)==z['parents']
     if len(v):
      ci=np.quantile(np.random.default_rng(20261018).choice(v,size=(10000,len(v)),replace=True).mean(1),[.025,.975]);assert abs(float(v.mean())-z['mean'])<1e-12 and np.allclose(ci,z['CI95'],rtol=0,atol=1e-12)
     checks['independent_parent_bootstrap']+=1
 verify(True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=dict(checks),max_errors=dict(errors),outside_counts=dict(outs),CPU_seconds=time.perf_counter()-start,teacher_audit_feedback_seconds=teacher_audit_seconds,audit_teacher_ROI_calls=64,audit_teacher_candidate_evaluations=sum(len(load(D/'predictions'/(r['alias']+'.pt'))['frame_ids'])*17 for r in p['records']),new_model_GPU_DINO_training=0,all64_sealed_before_GT=True,formal64_and_registries_unchanged=True));print('root independent audit passed',dict(checks),dict(errors),flush=True)
if __name__=='__main__':main()
