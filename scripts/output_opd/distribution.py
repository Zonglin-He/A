"""Fixed full-rank Gaussian policy on complete output trajectories.

The distribution lives in a chart relative to a fixed E trajectory. It has
unbounded centers/log-sizes and fixed, physical-time AR(1) covariance. No model,
annotations, trainable covariance, clipping, or reward evaluation lives here.
Score-function optimization of this policy has the usual policy-gradient/RL
connection; this module does not establish an advantage over pathwise Direct.
"""
from __future__ import annotations

import hashlib
import math

import torch


SEED = 41
SAMPLES = 4


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class JointTrajectoryGaussian:
    """N(mu, Sigma) in a fixed E chart, with four independent coordinate chains.

    ``reference_boxes`` is [T,4] in center/positive-width/positive-height form.
    ``frame_ids`` contains its strictly increasing physical frame IDs. The eight
    distinct ``anchor_frame_ids`` must be members of that grid; their sorted
    adjacent gaps define the fixed correlation length by the median gap.

    Distribution arithmetic is FP64 on the reference device. Casting student
    outputs to that precision preserves their autograd connection. E, physical
    time, and covariance are detached constants. q0 is this same distribution
    with zero mean, so KL(q_mu || q0) is available exactly in O(T).
    """

    def __init__(self, reference_boxes, frame_ids, anchor_frame_ids, sigma=.1):
        reference = torch.as_tensor(reference_boxes)
        _require(reference.ndim == 2 and reference.shape[1] == 4,
                 'Reference must have shape [T,4]')
        _require(reference.shape[0] >= 8, 'Eight distinct anchors require T >= 8')
        self.reference = reference.detach().to(dtype=torch.float64).clone()
        self.frames = len(self.reference)
        _require(bool(torch.isfinite(self.reference).all()) and
                 bool((self.reference[:, 2:] > 0).all()),
                 'Reference boxes must be finite with positive dimensions')
        self.frame_ids = torch.as_tensor(frame_ids, dtype=torch.float64,
            device=self.reference.device).detach().clone()
        anchors = torch.as_tensor(anchor_frame_ids, dtype=torch.float64,
            device=self.reference.device).detach().clone()
        _require(self.frame_ids.shape == (self.frames,) and
                 bool(torch.isfinite(self.frame_ids).all()),
                 'Physical frame IDs must be a finite length-T vector')
        gaps = self.frame_ids[1:] - self.frame_ids[:-1]
        _require(bool((gaps > 0).all()), 'Physical frame IDs must strictly increase')
        _require(anchors.shape == (8,) and bool(torch.isfinite(anchors).all()),
                 'Exactly eight finite anchor frame IDs are required')
        self.anchor_frame_ids = anchors.sort().values
        anchor_gaps = self.anchor_frame_ids[1:] - self.anchor_frame_ids[:-1]
        _require(bool((anchor_gaps > 0).all()), 'Anchor frame IDs must be distinct')
        _require(bool(torch.isin(self.anchor_frame_ids, self.frame_ids).all()),
                 'Each anchor must belong to the physical frame grid')
        self.anchor_median_gap = anchor_gaps.median()
        self.sigma = float(sigma)
        _require(math.isfinite(self.sigma) and self.sigma > 0,
                 'Sigma must be finite and positive')
        ratio = gaps / self.anchor_median_gap
        self.rho = torch.exp(-ratio)
        # -expm1(-2r) avoids cancellation when the correlation is close to one.
        conditional_std = self.sigma * torch.sqrt(-torch.expm1(-2*ratio))
        self.innovation_std = torch.cat((self.reference.new_tensor([self.sigma]),
                                        conditional_std))
        _require(bool(torch.isfinite(self.innovation_std).all()) and
                 bool((self.innovation_std > 0).all()),
                 'The fixed covariance must remain full rank numerically')
        self._reference_log_size = self.reference[:, 2:].log()

    def _trajectory(self, value, name, batched=True):
        value = torch.as_tensor(value, dtype=torch.float64,
                                device=self.reference.device)
        allowed = (2, 3) if batched else (2,)
        _require(value.ndim in allowed and value.shape[-2:] == (self.frames, 4),
                 f'{name} must have shape [T,4]' + (' or [K,T,4]' if batched else ''))
        return value

    def chart(self, boxes):
        """Center offsets scaled by E dimensions, followed by log-size ratios."""
        boxes = self._trajectory(boxes, 'Boxes')
        _require(bool(torch.isfinite(boxes).all()) and bool((boxes[..., 2:] > 0).all()),
                 'Boxes must be finite with strictly positive dimensions')
        center = (boxes[..., :2] - self.reference[:, :2]) / self.reference[:, 2:]
        # log(w)-log(Ew) avoids under/overflow in the intermediate ratio w/Ew.
        size = boxes[..., 2:].log() - self._reference_log_size
        return torch.cat((center, size), dim=-1)

    def inverse(self, chart):
        """Invert the chart without clipping or truncating any coordinate.

        Unrepresentable floating-point sizes fail explicitly, instead of being
        silently clamped into a different distribution.
        """
        chart = self._trajectory(chart, 'Chart')
        center = self.reference[:, :2] + self.reference[:, 2:] * chart[..., :2]
        size = (self._reference_log_size + chart[..., 2:]).exp()
        boxes = torch.cat((center, size), dim=-1)
        if not bool(torch.isfinite(boxes).all()) or not bool((size > 0).all()):
            raise FloatingPointError('Unbounded chart is outside representable box coordinates')
        return boxes

    def sample(self, mu, eps, detach_mean=False):
        """Apply the fixed AR(1) reparameterization to independent eps[K,T,4].

        The recurrent residual is independent of mu. This is algebraically
        X_t = mu_t + rho_t*(X_(t-1)-mu_(t-1)) + innovation_std_t*eps_t.
        Detaching mu is explicit for score-function samples. The default keeps
        the full pathwise mean gradient used by the matched DirectMC control.
        """
        mu = self._trajectory(mu, 'Mean', batched=False)
        eps = self._trajectory(eps, 'Noise')
        _require(eps.ndim == 3, 'Noise must have shape [K,T,4]')
        mean = mu.detach() if detach_mean else mu
        residuals = [self.innovation_std[0]*eps[:, 0]]
        for t in range(1, self.frames):
            residuals.append(self.rho[t-1]*residuals[-1] +
                             self.innovation_std[t]*eps[:, t])
        return mean[None] + torch.stack(residuals, dim=1)

    def innovations(self, trajectory):
        """Apply the covariance-whitening innovation map to [T,4]/[K,T,4]."""
        trajectory = self._trajectory(trajectory, 'Trajectory')
        first = trajectory[..., :1, :]
        following = trajectory[..., 1:, :] - self.rho[:, None]*trajectory[..., :-1, :]
        return torch.cat((first, following), dim=-2) / self.innovation_std[:, None]

    def log_prob_terms(self, X, mu):
        """Return conditional log-density factors [K,T], summing four coordinates.

        X is treated exactly as provided: callers must detach sampled X when
        requesting a score-function derivative. This function does not detach
        it implicitly, so its ordinary density gradients remain testable.
        """
        X = self._trajectory(X, 'Samples')
        _require(X.ndim == 3, 'Samples must have shape [K,T,4]')
        mu = self._trajectory(mu, 'Mean', batched=False)
        whitened = self.innovations(X-mu[None])
        return (-.5*whitened.square().sum(-1) - 2*math.log(2*math.pi)
                - 4*self.innovation_std.log())

    def kl_reference_terms(self, mu):
        """Exact per-frame terms of KL(N(mu,Sigma) || N(0,Sigma))."""
        mu = self._trajectory(mu, 'Mean', batched=False)
        return .5*self.innovations(mu).square().sum(-1)

    def kl_reference(self, mu):
        return self.kl_reference_terms(mu).sum()


def antithetic_noise(key, round_index, n_frames, *, device=None, dtype=torch.float64):
    """Two independent full-trajectory noises and their negatives, seed 41.

    Ordering is exactly [e0,e1,-e0,-e1]. A private CPU generator avoids changing
    global CPU/CUDA RNG state; transfer/casting happens only after generation.
    """
    _require(isinstance(n_frames, int) and n_frames > 0, 'n_frames must be positive')
    _require(isinstance(round_index, int) and round_index >= 0,
             'round_index must be a nonnegative integer')
    token = f'output-opd-v1|{SEED}|{key}|{round_index}'
    seed = int(hashlib.sha256(token.encode()).hexdigest()[:15], 16)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(2, n_frames, 4, generator=generator, dtype=torch.float64)
    return torch.cat((noise, -noise), dim=0).to(device=device, dtype=dtype)


def independent_pair_baseline(returns):
    """Use only the other independent noise pair for each sample's baseline.

    With [e0,e1,-e0,-e1] ordering, samples 0/2 use mean(R1,R3), while samples
    1/3 use mean(R0,R2). A usual all-other-sample baseline would include the
    dependent antithetic partner and need not be unbiased. The baseline is
    detached; callers must separately detach their sampled score advantages.
    Trailing dimensions (e.g. per-factor return-to-go) are preserved.
    """
    returns = torch.as_tensor(returns)
    _require(returns.ndim >= 1 and returns.shape[0] == SAMPLES,
             'Returns must have four samples ordered [e0,e1,-e0,-e1]')
    values = returns.detach()
    other_for_even = .5*(values[1]+values[3])
    other_for_odd = .5*(values[0]+values[2])
    return torch.stack((other_for_even, other_for_odd,
                        other_for_even, other_for_odd), dim=0)
