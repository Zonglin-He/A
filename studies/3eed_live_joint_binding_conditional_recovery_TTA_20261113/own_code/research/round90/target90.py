"""Target-only locked protocol after source-only preparation and source audits."""
from adapt90 import *
BASE90=['source','frozen_bi','frozen_ABC','R59','R89_candidate_frozen','R89_candidate_tta1','R89_continue_tta1']
VARIANTS90=['no_ref','soft_pool','wrong_space']
ARMS90=BASE90+[b+'_'+s for b in BRANCHES90 for s in ['frozen','mean3','feature3','tta1']]+['candidate_joint_tta3']+[v+'_'+s for v in VARIANTS90 for s in ['frozen','tta1']]+['candidate_joint_'+v+'1' for v in ['bypass','physical','shuffle']]
def specs90():return [(b,b,'full','strict','full') for b in BRANCHES90]+[(v,'candidate_joint','no_ref' if v=='no_ref' else 'wrong_space' if v=='wrong_space' else 'full','soft' if v=='soft_pool' else 'strict','full') for v in VARIANTS90]+[('candidate_joint_'+k,'candidate_joint','full','strict',k) for k in ['bypass','physical','shuffle']]
def arm90(name,step):return name+str(step) if name.startswith('candidate_joint_') and name not in BRANCHES90 else name+'_tta'+str(step)
