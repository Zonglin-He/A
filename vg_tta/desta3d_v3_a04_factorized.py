"""A0.4: original local H128, fixed train-only R16 output parameterization."""
import torch
from torch import nn
from vg_tta.desta3d_v3_a0_screen import coefficients

class FactorizedDirectionMixer(nn.Module):
    def __init__(self, union, channel_basis, radius=.13545580427763146):
        super().__init__()
        if union.shape != (2560,256) or channel_basis.shape != (256,16):
            raise ValueError('Fixed union256/shared R16 support')
        self.radius=radius
        self.register_buffer('union',union.detach().float().clone())
        self.register_buffer('channel_basis',channel_basis.detach().double().clone())
        self.register_buffer('basis',(union.detach().double()@channel_basis.detach().double()).float())
        # Construction order and hidden initialization identical to original A0.
        self.input=nn.Linear(425,128)
        self.local=nn.Conv3d(128,128,3,padding=1,groups=128)
        self.mix=nn.Linear(128,128)
        self.output=nn.Linear(128,16)
        nn.init.normal_(self.output.weight,std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self,cache):
        return coefficients(self,cache)


def route(train_median,dev_median,steps):
    if train_median is None or dev_median is None:
        return 'undefined_direction_stop'
    if train_median>=.3 and dev_median>=.1:
        return 'dev64_native'
    if train_median>=.3:
        return 'conditioning_generalization_stop'
    if steps==200:
        return 'continue_same_trajectory_to_2000'
    return 'stop_offline_predictor_tuning_next_R16_optimization_feasibility'
