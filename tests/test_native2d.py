"""Native input observer does not change Source values; frame/grid alignment is explicit."""
import numpy as np,torch,pytest
from stvg.representation import observe_native_grid,extract
class Encoder(torch.nn.Module):
 def forward(self,*,videos):return videos.tensors*2
class Video:
 def __init__(self,x):self.tensors=x;self.mask=torch.zeros(x.shape[0],*x.shape[2:],dtype=torch.bool)
class Model:
 def __init__(self):self.ground_encoder=Encoder()
def test_observer_is_readonly_and_removed_on_error():
 m=Model();x=torch.arange(24.).reshape(2,3,2,2);v=Video(x);expected=m.ground_encoder(videos=v)
 with observe_native_grid(m)as obs:actual=m.ground_encoder(videos=v)
 assert torch.equal(actual,expected)and torch.equal(x,v.tensors)and len(obs)==1 and not m.ground_encoder._forward_pre_hooks
 with pytest.raises(RuntimeError):
  with observe_native_grid(m):raise RuntimeError('deliberate')
 assert not m.ground_encoder._forward_pre_hooks
@pytest.mark.parametrize('T',[16,65,200])
def test_interleaved_native_grid_has_correct_physical_order_and_arbitrary_dimension(T):
 D=7;grid=[]
 for off in [0,1]:
  values=torch.tensor(np.broadcast_to(np.arange(off,T,2)[:,None,None,None],(len(range(off,T,2)),D,3,4)).copy(),dtype=torch.float32);grid.append((values,torch.ones(values.shape[0],3,4,dtype=torch.bool)))
 cap=dict(frame_ids=list(range(10,10+T)),native_local_grid=grid);z,stats=extract(cap,np.tile([.5,.5,.6,.6],(T,1)));np.testing.assert_allclose(z,np.broadcast_to(np.arange(T)[:,None],(T,D)),rtol=0,atol=1e-12);assert len(stats)==T
