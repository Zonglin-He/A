from common import *
def feature(r):
 folder='round17/features96_attempt001' if r['stage']=='A' else 'round19/sourceV16_attempt001' if r['stage']=='V' else 'round19/sourceH32_attempt001'
 with np.load(ROOT/'crossplatform'/folder/'samples_PRIVATE'/(r['id']+'.npz')) as z:return {k:z[k].copy() for k in z.files}
