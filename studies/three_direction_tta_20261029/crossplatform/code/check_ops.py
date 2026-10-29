import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'upstream'))
from pointnet2 import pointnet2_utils as a
from ops.teed_pointnet.pointnet2_batch import pointnet2_utils as b
from ops.teed_pointnet.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu,points_in_boxes_cpu
report={}
for name,mod in [('pointnet2',a),('teed_pointnet',b)]:
 x=torch.arange(24,dtype=torch.float32,device='cuda').reshape(1,3,8).requires_grad_();idx=torch.tensor([[1,4,1]],dtype=torch.int32,device='cuda')
 y=mod.gather_operation(x,idx);ref=torch.gather(x,2,idx.long()[:,None].expand(-1,3,-1));assert torch.equal(y,ref)
 y.sum().backward();grad=x.grad.clone();expected=torch.zeros_like(x);expected[:,:,1]=2;expected[:,:,4]=1;assert torch.equal(grad,expected)
 known=torch.tensor([[[0.,0,0],[1.,0,0],[3.,0,0],[6.,0,0]]],device='cuda');q=torch.tensor([[[.2,0,0],[2.2,0,0]]],device='cuda')
 dist,inds=mod.three_nn(q,known);d=torch.cdist(q,known).sort(-1).values[:,:,:3];assert torch.allclose(dist,d,atol=1e-6)
 w=torch.full((1,2,3),1/3.,device='cuda');f=torch.arange(4,dtype=torch.float32,device='cuda').view(1,1,4).requires_grad_();v=mod.three_interpolate(f,inds,w);ref=(f[0,:,inds.long()[0]]*w[0,None]).sum(-1)[None];assert torch.allclose(v,ref);v.sum().backward();assert torch.isfinite(f.grad).all()
 fps=mod.furthest_point_sample(known,3);assert fps.shape==(1,3) and fps.min()>=0 and fps.max()<4
 report[name]={'gather_forward_max_error':float((y-ref).abs().max()) if y.shape==ref.shape else 0.,'gather_backward_exact':True,'three_nn_max_error':float((dist-d).abs().max()),'interpolation_backward_finite':True,'fps_range_valid':True}
pts=torch.tensor([[[0.,0,0],[.9,.9,.9],[2.,0,0]]],device='cuda');box=torch.tensor([[[0.,0,0,2.,2,2,0]]],device='cuda');gpu=points_in_boxes_gpu(pts,box);cpu=points_in_boxes_cpu(pts[0].cpu(),box[0].cpu());assert gpu.tolist()==[[0,0,-1]];assert cpu.tolist()==[[1,1,0]];report['roi']={'gpu_expected':[0,0,-1],'cpu_expected':[1,1,0]}
torch.cuda.synchronize();(ROOT/'ops_checked.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
