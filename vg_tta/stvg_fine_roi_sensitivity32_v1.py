"""Four predeclared local sensitivity arms, original Center/ROI teacher."""
from vg_tta.stvg_roi_layout_opd32_v1 import build,feedback,patches
from vg_tta.stvg_fine_roi_scoped_opd32_v1 import scope,scoped_fit
ARMS=['E_MBR','Center_live','Tau_low','Tau_high','Step_low','Step_high']
CONFIGS={'Tau_low':dict(temperature=.05,step_size=.5),'Tau_high':dict(temperature=.2,step_size=.5),'Step_low':dict(temperature=.1,step_size=.25),'Step_high':dict(temperature=.1,step_size=1.)}
def fit(E,context,eps,cfg,allowed,mode,permutation):
 c=dict(cfg,**CONFIGS[mode]);f=scoped_fit(E,context,eps,c,[],'On_policy_dense',permutation,allowed);f.update(actual_configuration=c,mode=mode);return f

def properties():
 from vg_tta.stvg_fine_roi_scoped_opd32_v1 import properties as base
 x=base();x.update(original_scorer_and_update_unmodified=True,only_four_predeclared_tau_or_step_changes=True,no_noise_scale_or_round_change=True);return x
