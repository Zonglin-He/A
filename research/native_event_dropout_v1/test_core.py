import unittest,torch,numpy as np
from research.native_event_dropout_v1.core import stochastic_time,input_seed,gaussian_delta,wrong_delta
class CoreTests(unittest.TestCase):
 def toy(self):return torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.Dropout(.1),torch.nn.MultiheadAttention(8,2,dropout=.1)).eval()
 def test_scoped_exception_restore(self):
  m=self.toy();flags=[x.training for x in m.modules()]
  with self.assertRaises(RuntimeError):
   with stochastic_time(m):
    self.assertFalse(m.training);self.assertFalse(m[0].training);self.assertTrue(m[1].training);self.assertTrue(m[2].training);raise RuntimeError()
  self.assertEqual(flags,[x.training for x in m.modules()]);self.assertEqual(m[1].p,.1);self.assertEqual(m[2].dropout,.1)
 def test_zero_and_seed(self):
  m=self.toy();x=torch.randn(5,1,8)
  def run():
   a=m[1](m[0](x));return m[2](a,a,a)[0]
  y=run()
  with stochastic_time(m,zero=True):self.assertTrue(torch.equal(y,run()))
  a=[]
  for _ in range(2):
   with torch.random.fork_rng():
    torch.manual_seed(input_seed({'test':1}))
    with stochastic_time(m):a.append(run())
  self.assertTrue(torch.equal(*a));self.assertFalse(torch.equal(y,a[0]))
 def test_moments(self):
  a=np.random.default_rng(5).normal(size=(4,6,1,33,2)).astype(np.float32);a[:,:,:,0]=3.;b=gaussian_delta(a,61)
  self.assertLess(abs(a.astype(float).mean(0)-b.astype(float).mean(0)).max(),2e-7);self.assertLess(abs(a.astype(float).std(0)-b.astype(float).std(0)).max(),2e-7);self.assertTrue(np.all(b[:,:,:,0]==3.))
 def test_time_shift(self):
  a=np.random.default_rng(5).normal(size=(4,6,1,33,2)).astype(np.float32);b=wrong_delta(a);self.assertTrue(np.array_equal(np.sort(a,axis=-2),np.sort(b,axis=-2)));self.assertTrue(np.array_equal(np.roll(b,-16,axis=-2),a))
 def test_probability_mixture_and_seed_binding(self):
  q=np.triu(np.ones((65,65)),1);q/=q.sum();a=np.stack([q]*4).mean(0);self.assertTrue(np.array_equal(q,a));self.assertAlmostEqual(a.sum(),1.);self.assertNotEqual(input_seed({'a':1}),input_seed({'a':2}))
if __name__=='__main__':unittest.main()
