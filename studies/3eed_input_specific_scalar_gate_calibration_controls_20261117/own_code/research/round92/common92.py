"""R91 architecture/weights fixed. Change only how adapted base and residual are read."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round91'))
from targets91 import *
H92=ROOT/'research/round92';B92=ROOT/'crossplatform/round92';P91=B91/'prediction336_attempt001';NAME91='R76_full_disc'
REF_MAP92={'source':'source','R76_frozen':'R76_frozen','ABC':'ABC','R59':'R59','R80_raw':'R80_raw','R82_bt':'R82_bt','R91_frozen':NAME91+'_frozen','R91_mean3':NAME91+'_mean3','R91_feature3':NAME91+'_feature3','R91_raw_dynamic':NAME91+'_physical_raw','R91_bt_dynamic':NAME91+'_physical_bt','R91_wrong_raw_dynamic':NAME91+'_wrong_physical_raw','R91_wrong_bt_dynamic':NAME91+'_wrong_physical_bt'}
NEW92=[('same_state_'+kind+'_'+part) for kind in ['raw','bt','wrong_raw','wrong_bt'] for part in ['static_residual','residual_only']]+['static_residual_formal_bt','static_residual_formal_wrong_bt']
ARMS92=list(REF_MAP92)+NEW92
def source92():return load(ROOT/read(B91/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json')['models'][NAME91]['state_path'])
def model92():
 p=nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64));m=Fusion91(p).cuda().eval().requires_grad_(False);s=source92();restore(m,s);freeze91(m);return m,s
def feature_to92(a):return {k:torch.as_tensor(v,device='cuda') for k,v in a.items()}
def parts92(m,f,route,s):
 p,e,_=scorer91(m,f,route,s,'R76','full_disc');return e['native_scores'].detach().double(),e['residual'].detach().double(),e['projected_query'].detach(),p
def readout92(b,r):return (b.double()+r.double()).softmax(0)
def same_parts92(b0,r0,b1,r1):return {'static_residual':readout92(b1,r0),'residual_only':readout92(b0,r1),'dynamic':readout92(b1,r1)}
