"""Post-seal CPU derivative information audit; no fit, taskGT or parameter update."""
from audit116 import *
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;rows=[];registry=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];torch.set_num_threads(4)
 for src in ['waymo','drone','quad']:
  with np.load(R/registry[src]['state_path']) as zz:S={k:zz[k] for k in zz.files}
  m=Fusion91(nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64))).cpu().eval();m.load_state_dict({k:torch.from_numpy(np.asarray(v).copy()) for k,v in S.items()},strict=True);m.requires_grad_(False);ps=[p for x in m.rgb.modules() if isinstance(x,nn.GroupNorm) for p in x.parameters(recurse=False)];assert sum(p.numel() for p in ps)==224
  for p in ps:p.requires_grad_(True)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt
   for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
    sid=r['id'];meta=rd(B/'formal'/name/'records_PRIVATE'/(sid+'.json'))
    if meta['protected_alias_fallback']:continue
    with np.load(R/'crossplatform/round111'/name/'prefix_PRIVATE'/(sid+'.npz')) as zf:xyz=zf['xyz'];ix=zf['point_indices']
    key=hashlib.sha256(xyz.tobytes()+ix.tobytes()).hexdigest()
    with np.load(R/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+key[:20]+'.npz')) as zf:geom={k:zf[k] for k in zf.files}
    d=tensor_data(geom,r,device='cpu');tr,ho,wrong=split(r,d);correct=d['depth_target'];im=[masks(r,d,t)[1] for t in [76001,76002]];pred=[]
    for mask in im:
     mm=mask.reshape(1,1,32,48).repeat_interleave(8,2).repeat_interleave(8,3);x=d['image'].masked_fill(mm,.5);I=m.image_adapter(m.rgb((x-.5)*2).flatten(2).transpose(1,2)[0]);pred.append(m.pred_depth(I).squeeze(-1)[tr])
    residual=torch.stack(pred)-correct[tr];rw=torch.stack(pred)-wrong[tr];grads={}
    for j,(lossname,loss) in enumerate([('hub_correct',F.smooth_l1_loss(torch.stack(pred),correct[tr].expand(2,-1))),('hub_wrong',F.smooth_l1_loss(torch.stack(pred),wrong[tr].expand(2,-1))),('mse_correct',residual.square().mean()/2),('mse_wrong',rw.square().mean()/2)]):
     gs=torch.autograd.grad(loss,ps,retain_graph=j<3);grads[lossname]=torch.cat([g.detach().flatten() for g in gs]).double()
    cos=lambda a,b:float(F.cosine_similarity(grads[a][None],grads[b][None])[0]);rows.append({'source':src,'target':tgt,'id_PRIVATE':sid,'cells':len(tr),'residual_abs_ge1_fraction':float((abs(residual.detach())>=1).float().mean()),'correct_wrong_huber_derivative_equal_fraction':float((residual.detach().clamp(-1,1)==rw.detach().clamp(-1,1)).float().mean()),'Huber_correct_wrong_gradient_cosine':cos('hub_correct','hub_wrong'),'MSE_correct_wrong_gradient_cosine':cos('mse_correct','mse_wrong'),'Huber_MSE_correct_gradient_cosine':cos('hub_correct','mse_correct'),'Huber_gradient_L2':float(grads['hub_correct'].norm()),'MSE_gradient_L2':float(grads['mse_correct'].norm()),'GT_used':False})
 groups=[]
 for src in ['all','waymo','drone','quad']:
  rr=[r for r in rows if src=='all' or r['source']==src];groups.append({'source':src,'n':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['residual_abs_ge1_fraction','correct_wrong_huber_derivative_equal_fraction','Huber_correct_wrong_gradient_cosine','MSE_correct_wrong_gradient_cosine','Huber_MSE_correct_gradient_cosine','Huber_gradient_L2','MSE_gradient_L2']}})
 out={'postseal_no_new_GT_fits_or_parameter_updates':True,'rows':groups,'scope':224,'FINAL_access':False};(H/'DEPTH_SIGNAL_INFORMATION_PRIVATE.json').write_text(json.dumps(rows,indent=2)+'\n');(H/'DEPTH_SIGNAL_INFORMATION_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
