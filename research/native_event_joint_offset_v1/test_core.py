import unittest,torch,numpy as np
from research.native_event_joint_offset_v1.core import interleave,AXES,parity_block,single_posterior,independent_marginal
class Layer(torch.nn.Module):
 def __init__(self):super().__init__();self.self_attn=torch.nn.MultiheadAttention(4,1,dropout=0.)
class Dec(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([Layer()for _ in range(2)])
class Tests(unittest.TestCase):
 def test_actual_axes_and_padding(self):
  xs=[]
  for o,n in enumerate([4,3]):
   x={}
   for k,a in AXES.items():
    shape=[n,1,4]if k.startswith('query_')else[5,n,4]
    if k=='query_mask':shape=[1,n]
    if k=='encoded_mask':shape=[n,5]
    t=torch.arange(int(np.prod(shape))).reshape(shape)+100*o;x[k]=(t%2==0)if'mask'in k else t.float()
   xs.append(x)
  z=interleave(xs)
  self.assertEqual(z['encoded_mask'].shape,(7,5));self.assertTrue(torch.equal(z['encoded_feature'][:,1],xs[1]['encoded_feature'][:,0]))
 def test_blocks_only_other_parity_and_cleans_up(self):
  d=Dec().eval();x=torch.randn(5,1,4)
  with parity_block(d,5)as calls:
   for l in d.layers:
    _,w=l.self_attn(x,x,value=x,key_padding_mask=torch.zeros(1,5,dtype=torch.bool));self.assertTrue(torch.equal(w[0][torch.arange(5)[:,None]%2!=torch.arange(5)[None,:]%2],torch.zeros(12)))
  self.assertEqual(len(calls),2);self.assertTrue(all(not l.self_attn._forward_pre_hooks for l in d.layers))
  with self.assertRaises(RuntimeError):
   with parity_block(d,5):raise RuntimeError('test cleanup')
  self.assertTrue(all(not l.self_attn._forward_pre_hooks for l in d.layers))
 def test_single_uniform_not_hull_and_physical_parity(self):
  a=torch.zeros(6,1,3,2);b=torch.zeros(6,1,2,2);q=single_posterior(a);self.assertTrue(np.allclose(q[np.triu_indices(3,1)],1/3));m=independent_marginal([a,b]);self.assertEqual(m.shape,(5,5));self.assertAlmostEqual(m.sum(),1);self.assertEqual(m[0,1],0);self.assertAlmostEqual(m[1,3],.5)
if __name__=='__main__':unittest.main()
