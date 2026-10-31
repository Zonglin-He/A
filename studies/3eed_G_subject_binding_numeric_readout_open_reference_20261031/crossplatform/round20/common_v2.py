from common import *
from common import feature as previous_feature
def feature(r):
 if r['stage']!='Z':return previous_feature(r)
 with np.load(ROOT/'crossplatform/round20/sourceZ64_attempt001/samples_PRIVATE'/(r['id']+'.npz')) as z:return {k:z[k].copy() for k in z.files}
