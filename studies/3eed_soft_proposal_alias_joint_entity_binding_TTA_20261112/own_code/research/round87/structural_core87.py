"""One separately locked role-inventory correction; original32 arms unchanged."""
from core87 import *
MODES87=['full','hard_alias','no_alias','independent','no_subject']
BASE87=BASE87+['original_full_direct','original_full_tta1','original_hard_alias_tta1','original_full_half']
ARMS87=BASE87+[v+'_'+r for v in MODES87 for r in ['direct','half','tta1']]+['wrong_direct','wrong_half','wrong_tta1','candidate_only1']
def prepared87(r):
 ev=load(B85/'observations366_attempt001/samples_PRIVATE'/(r['id']+'.npz'));c=read(B87/'role_inventory336_attempt003/cases_PRIVATE'/(r['id']+'.json'));a=load(B87/'structural_evidence336_attempt002/samples_PRIVATE'/(r['id']+'.npz'));return ev,c['graph'],a['source_entity_scores']
