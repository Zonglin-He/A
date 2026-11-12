from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round86'))
from core86 import *
from evidence87 import *
H87=ROOT/'research/round87';B87=ROOT/'crossplatform/round87'
MODES87=['full','hard_alias','no_alias','independent','no_subject','no_relation','no_RGB','no_null','no_exclusivity']
BASE87=['source','frozen_bi','frozen_ABC','R59','R86_direct','R86_TTA','R86_raw','frozen_no_reverse','frozen_gray_RGB']
ARMS87=BASE87+[v+'_'+r for v in MODES87 for r in ['direct','tta1']]+['full_tta3','wrong_direct','wrong_tta1','wrong_tta3','candidate_only1']
def prepared87(r):
 sid=r['id'];ev=load(B85/'observations366_attempt001/samples_PRIVATE'/(sid+'.npz'));meta=read(B86/'evidence366_attempt002/cases_PRIVATE'/(sid+'.json'));old=load(B85/'prediction366_attempt002/samples_PRIVATE'/(sid+'.npz'));scores=old['entity_scores'][[e['source_entity_index'] for e in meta['graph']['entities']]];return ev,meta['graph'],scores
