"""Cached A0.1: separate immutable feature width from internal mixer capacity."""
import hashlib
from torch import nn
from vg_tta.desta3d_v3_gap_candidates import StateAwareDirectionMixer, STATE_DIM


class FitabilityDirectionMixer(StateAwareDirectionMixer):
    def __init__(self, basis, feature_dim=128, hidden_dim=128, radius=.13545580427763146):
        # Same construction order and initialization as A0 at width128.
        nn.Module.__init__(self)
        self.radius = radius
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.register_buffer('basis', basis.detach().float().clone())
        self.input = nn.Linear(3 * feature_dim + 8 + STATE_DIM, hidden_dim)
        self.local = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.mix = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, basis.shape[1])
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)


def q1_index(rows):
    return min(range(len(rows)), key=lambda i: hashlib.sha256(
        ('DESTA-A01-v1|q1|' + str(rows[i]['key'])).encode('utf-8')).hexdigest())


def route(width, longer, q1):
    """Predeclared priority if both scientific arms pass: width first, no score ranking."""
    def passes(r):
        return r['train']['median'] >= .3 and r['dev']['median'] >= .1
    if passes(width):
        return 'capacity_candidate_next_dev64_native'
    if passes(longer):
        return 'optimization_budget_candidate_next_dev64_native'
    if q1 >= .9:
        return 'stop_width_steps_next_global_context_conditioning'
    return 'stop_mixer_capacity_tuning_next_cached_oracle_structure_audit'
