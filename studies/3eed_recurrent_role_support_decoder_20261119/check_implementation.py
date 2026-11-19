"""Check finite gradients, actual support feedback and intervention boundaries."""
import torch

from decoder import Decoder, objective


def run():
    torch.set_num_threads(1)
    torch.manual_seed(174)
    b, n, l = 3, 96, 20
    rand = torch.randn
    last = torch.rand(b, 256, 3) + .25
    # The first row has no projected evidence and remains a valid training input.
    visible = (torch.arange(b)[:, None] > 0).expand(-1, n)
    feat = dict(points=rand(b, n, 288), rgb=rand(b, n, 64), xyz=rand(b, n, 3),
                uv=torch.rand(b, n, 2), visible=visible, in_image=visible,
                text=rand(b, l, 288), valid=torch.ones(b, l, dtype=torch.bool),
                role_mask=torch.stack([torch.arange(l)//6 == k for k in range(3)])[None].expand(b, -1, -1).clone(),
                has_reference=torch.tensor([True, True, False]), prior=torch.cat((rand(b, 256, 3), last), -1),
                prior_scores=rand(b, 2, 256), anchors=torch.arange(8)[None].expand(b, -1).reshape(b, 2, 4),
                gt=torch.tensor([0., 0., 0., 1., .8, 1., .5])[None, None].expand(b, 2, -1),
                known_ref=torch.tensor([True, True, False]))
    joint = Decoder('joint')
    no_feedback = Decoder('without_feedback')
    no_rgb = Decoder('without_rgb')
    for model in (no_feedback, no_rgb):
        model.load_state_dict(joint.state_dict())
    out = joint(feat)
    without = no_feedback(feat)
    torch.testing.assert_close(out['passes'][0][2], without['passes'][0][2], atol=0., rtol=0.)
    assert (out['passes'][1][2] - without['passes'][1][2]).abs().max() > 1e-6
    shifted = feat['role_mask'].clone()
    shifted[:, 1] = torch.roll(shifted[:, 1], 4, dims=-1)
    changed = dict(feat, role_mask=shifted)
    torch.testing.assert_close(no_feedback(changed)['passes'][-1][2][:, 0],
                               without['passes'][-1][2][:, 0], atol=0., rtol=0.)
    assert (joint(changed)['passes'][-1][2][:, 0] - out['passes'][-1][2][:, 0]).abs().max() > 1e-7
    torch.testing.assert_close(no_rgb(feat)['boxes'], no_rgb(dict(feat, rgb=feat['rgb']*100))['boxes'],
                               atol=0., rtol=0.)
    for m in (joint, no_feedback, no_rgb):
        pred = m(feat)
        loss = objective(feat, pred)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        assert pred['boxes'].shape == (b, 2, 7) and (pred['boxes'][..., 3:6] > 0).all()
    assert joint.pair[0].weight.grad.abs().sum() > 0
    assert joint.spatial_gate.grad.abs() > 0
    assert joint.box[0].weight.grad is not None and joint.box[-1].weight.grad.abs().sum() > 0
    # The BF16 training trace reached the exact zero double-angle vector.
    singular = Decoder('joint')
    with torch.no_grad():
        singular.box[-1].bias[7] = -1.
    out = singular(feat)
    objective(feat, out).backward()
    assert all(torch.isfinite(p.grad).all() for p in singular.parameters() if p.grad is not None)
    print('First-read equivalence, recurrent support coupling, role intervention, RGB ablation and finite gradients verified.')


if __name__ == '__main__':
    run()
