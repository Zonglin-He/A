"""Frozen native soft-token head prerequisite, no external model or labels."""
from pathlib import Path
import sys
R108=Path(__file__).resolve().parents[2];H108=R108/'research/round108';B108=R108/'crossplatform/round108';sys.path.insert(0,str(R108/'research/round107'));from core107 import *
from models.modules import ThreeLayerMLP
VIEWS108=['A','B','C','fusedABC'];POLICIES108=['softtoken','softtoken_geometric','softtoken_arithmetic','softtoken_shuffled_geometric'];PRIMARY108='softtoken_geometric_fusedABC'
REF108=['original_source','prepared_frozen','real_frozen_B','real_frozen_C','real_frozen_ensemble','real_frozen_fusedABC','real_TTA_C','real_TTA_fusedABC','randomA_frozen_fusedABC','randomA_TTA_fusedABC','wrong_frozen_fusedABC','wrong_TTA_fusedABC','teacher_shuffle_fusedABC']
ARMS108=REF108+[p+'_'+v for p in POLICIES108 for v in VIEWS108]
def ce108(head,h,mask):
 logits=head(h.T[None])[0].T;positive=torch.zeros(256,device=mask.device,dtype=torch.bool);assert len(mask)<=255;positive[:len(mask)]=mask.bool();assert positive.any();mass=logits.double().log_softmax(-1).masked_fill(~positive[None],-torch.inf).logsumexp(-1);return mass.softmax(0),logits

def pool108(p,q,r):
 out={'softtoken':q,'softtoken_geometric':((p.log()+q.log())/2).softmax(0),'softtoken_arithmetic':(p+q)/2};qq=q[torch.as_tensor(qperm94(r),device=p.device)];out['softtoken_shuffled_geometric']=((p.log()+qq.log())/2).softmax(0);return out

def statece108(head):return {k:x.detach().cpu().numpy().copy() for k,x in head.state_dict().items()}
def loadce108(src):
 c=source97(src);assert sha(c['checkpoint_path'])==c['checkpoint_sha256'];v=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True);assert v['epoch']==100 and v['trained_platforms']==[src];prefix='prediction_heads.5.sem_cls_scores_head.';s={k[len(prefix):]:x for k,x in v['model'].items() if k.startswith(prefix)};assert len(s)==14;head=ThreeLayerMLP(288,256).cuda().eval().requires_grad_(False);head.load_state_dict(s,strict=True);return head
