import unittest
import torch
from vg_tta.desta3d_v3_joint_mixer import JointCorrectionMixer,source_evidence,native_supervision
from vg_tta.optimizer_checkpoint import cpu_clone,restore_optimizer

class Contracts(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(12)
        q=torch.linalg.qr(torch.randn(20,8)).Q
        m=JointCorrectionMixer(q,hidden=4)
        args=(torch.randn(1,3,2,2,4),torch.randn(1,4),torch.randn(1,4),torch.rand(1,3,2,2,8),torch.randn(1,3,2,2,20))
        return m,args
    def test_zero_scope_and_union(self):
        m,a=self.inputs();self.assertTrue(torch.equal(m(*a),torch.zeros_like(a[-1])))
        m(*a).sum().backward();self.assertIsNone(m.basis.grad);self.assertGreater(float(m.output.weight.grad.norm()),0)
        with torch.no_grad():m.output.weight.normal_();m.output.bias.normal_()
        d=m(*a);self.assertLessEqual(float(d.norm()/a[-1].norm()),m.radius+1e-6)
        self.assertTrue(torch.allclose(d,d@m.basis@m.basis.T,atol=2e-6))
    def test_frozen_inputs_and_restore(self):
        m,a=self.inputs();o=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0)
        for _ in range(2):o.zero_grad();m(*a).square().sum().backward();o.step()
        state=cpu_clone(m.state_dict());opt=cpu_clone(o.state_dict());n,b=self.inputs();n.load_state_dict(state)
        p=torch.optim.AdamW(n.parameters(),lr=1e-3,weight_decay=0);restore_optimizer(p,opt)
        for mm,oo in ((m,o),(n,p)):oo.zero_grad();mm(*a).sum().backward();oo.step()
        for x,y in zip(m.parameters(),n.parameters()):self.assertTrue(torch.equal(x,y))
        self.assertTrue(all(x.grad is None for x in a));self.assertTrue(all(isinstance(k,int) for k in opt['state']))
    def test_unknown_box_and_physical_time(self):
        lab=dict(frame_ids=[10,11,90],event_interval=dict(begin_fid=11,end_fid=91),event_active=[False,True,True],box_valid=[True,False,True],boxes_xyxy=[[0,0,.5,.5],[0,0,0,0],[.5,.5,1,1]])
        e,p=source_evidence(lab,lab['frame_ids'],2,2)
        self.assertEqual(p.tolist(),[1,2]);self.assertEqual(float(e[0,1,:,:,4].sum()),0)
        self.assertEqual(float(e[0,1,:,:,3].sum()),0);self.assertAlmostEqual(float(e[0,1,0,0,5]),1/80)
        self.assertEqual(native_supervision(lab,{'branches':[]},'event')[1],'missing_native_event_support')
    def test_full_vocab_denominator(self):
        from vg_tta.desta3d_v3_free_actuation import native_ce
        logits=torch.tensor([[[0.,2.,1.,4.,-1.]]],requires_grad=True)
        y=torch.tensor([[1]]);valid=torch.ones_like(y,dtype=torch.bool)
        loss=native_ce(logits,y,valid)
        self.assertAlmostEqual(float(loss.detach()),float(torch.logsumexp(logits[0,0],0)-logits[0,0,1]),places=6)
        loss.backward();self.assertGreater(float(logits.grad[0,0,3]),0)

if __name__=='__main__':unittest.main()
