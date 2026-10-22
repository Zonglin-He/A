"""Separate full-scene appearance from target-context motion, with matched controls."""
import torch
from research.native_event_support_view_v1.core import keep_region,inside_fill,reflected

def focus(appearance,motion,keep,valid,kind):
 assert appearance.shape==motion.shape
 if kind=='Full':return appearance,motion
 if kind in ['MotionTarget','MotionReflected']:return appearance,inside_fill(motion,keep,valid)
 if kind in ['AppearanceTarget','AppearanceReflected']:return inside_fill(appearance,keep,valid),motion
 assert kind=='WrongTimeMotion'
 # Original cached clip features are only reassigned in time, not re-encoded backward.
 moved=torch.where(valid[:,None],motion.flip(0),motion)
 return appearance,inside_fill(moved,keep,valid)
