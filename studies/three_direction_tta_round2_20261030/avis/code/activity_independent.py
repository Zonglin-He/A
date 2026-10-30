"""Single-factor supporting-condition correction: remove class competition."""
import pathlib,sys,numpy as np
ORIGINAL=pathlib.Path(__file__).resolve().parents[2]/'audio_activity_v1/scripts'
sys.path.insert(0,str(ORIGINAL))
from activity_adapter import visibility,optimize,apply_gates,direct

def evidence(cosine,margin=.04,temperature=.02):
    # Same API/temperature, but margin is deliberately unused. Independent
    # background-relative multi-label evidence; still not calibrated probability.
    cls=cosine[:,:26];reference=cosine[:,26:].max(-1,keepdims=True)
    z=np.clip((cls-reference)/temperature,-40,40)
    return (1/(1+np.exp(-z))).astype('float32')
