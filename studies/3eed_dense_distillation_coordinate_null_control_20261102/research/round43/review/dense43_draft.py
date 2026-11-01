"""PROPOSAL ONLY: loss algebra, no reader/model/optimizer/training entry point.

The original full-probability control must replay R24 before reading effects.
The null teacher is always source probability coordinates in BOTH arms.
No new source anchor, hinge coordinates, loss weights, teacher, or readout.
"""

def proposed_dense_objective(native_probability,token_logodds,raw_teacher_F,
                             frozen_source_native_probability,parameter_anchor,
                             student_teacher_coordinate='probability',
                             teacher_mode='full'):
 import torch
 F=torch.nn.functional
 assert native_probability.shape==token_logodds.shape==raw_teacher_F.shape==frozen_source_native_probability.shape
 assert native_probability.shape[-1]==256
 if student_teacher_coordinate not in ('probability','token_logodds'):raise ValueError('fixed two-coordinate comparison only')
 if teacher_mode not in ('full','null_deltaG'):raise ValueError('full or same-information zero-evidence control only')
 # Do not silently replace the null target by source-z: that would remove the
 # diagnosed coordinate calibration effect by changing the teacher as well.
 teacher=(raw_teacher_F if teacher_mode=='full' else frozen_source_native_probability).detach()
 teacher_distribution=(teacher/.25).softmax(-1)
 source_distribution=(frozen_source_native_probability.detach()/.25).softmax(-1)
 native_log_distribution=(native_probability/.25).log_softmax(-1)
 kd_input=native_probability if student_teacher_coordinate=='probability' else token_logodds
 teacher_KL=F.kl_div((kd_input/.25).log_softmax(-1),teacher_distribution,reduction='batchmean')
 # These three terms are exactly the original R24 coordinates and weights.
 source_KL=F.kl_div(native_log_distribution,source_distribution,reduction='batchmean')
 target=teacher.argmax(-1);chosen=native_probability.gather(1,target[:,None]).squeeze(1)
 other=native_probability.clone().scatter(1,target[:,None],-torch.inf).max(-1).values
 hinge=F.relu(.05+other-chosen).mean()
 total=teacher_KL+.2*source_KL+.5*hinge+parameter_anchor
 return total,{'teacher_KL':teacher_KL,'source_KL_native_probability':source_KL,
               'hinge_native_probability':hinge,'parameter_anchor_original':parameter_anchor,
               'teacher_distribution':teacher_distribution.detach(),
               'teacher_target':target.detach()}

if __name__=='__main__':
 raise SystemExit('PROPOSAL ONLY: no authorized training runner; root approval required')
