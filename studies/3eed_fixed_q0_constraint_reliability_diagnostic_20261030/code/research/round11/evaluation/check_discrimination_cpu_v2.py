"""Synthetic only: tied/oneclass/NA score/ceil coverage correctness, no data."""
from pathlib import Path
import json,numpy as np
from discrimination import *
import hashlib
_expected=sorted(['a','b','c'],key=lambda s:hashlib.sha256(('synthetic|'+s).encode()).hexdigest())
_assert=deterministic_select(['a','b','c'],1,{x:0 for x in 'abc'},{x:True for x in 'abc'},salt='synthetic',platform='d')
assert _assert['ordered']==_expected
assert weighted_auc([1,1],[True,False],[1,1])==.5
assert weighted_ap([1,1],[True,False],[1,1])==.5
assert weighted_auc([2,1],[True,False],[1,1])==1
assert weighted_ap([2,1],[True,False],[1,1])==1
u=[dict(case_id='a',candidate=True,score_defined=True,score=.8,effect=1),dict(case_id='b',candidate=True,score_defined=True,score=.1,effect=-1),dict(case_id='c',candidate=True,score_defined=False,score=0,effect=-2),dict(case_id='d',candidate=True,score_defined=True,score=.4,effect=0),dict(case_id='e',candidate=False,score_defined=False,score=0,effect=0)]
a=assay(u);assert a['ranking']['helpful_positive_AUROC_casebalanced']==1 and a['n_score_undefined']==1 and a['effect_counts_undefined']['harmful']==1 and a['effect_counts_all_candidate']['zero_effect']==1
one=assay([u[0],u[3]]);assert one['ranking']['helpful_positive_AP_casebalanced'] is None and not one['ranking']['both_classes']
allzero=assay([u[3]]);assert allzero['ranking']['helpful_positive_AUROC_casebalanced'] is None
assert spearman([0,0,0],[1,2,3])['reason']=='constant_score'
ids=['a','b','c'];p=deterministic_select(ids,.5,dict(a=1,b=0,c=9),dict(a=True,b=False,c=False),salt='synthetic',platform='d');assert p['K']==2 and p['selected'][0]=='a' and p['n_selected_undefined']==1
assert deterministic_select(ids,0,dict(a=1,b=0,c=9),dict(a=True,b=False,c=False),salt='synthetic',platform='d')['K']==0
assert risk_summary([],3)['allreject_reference_not_valid_success']
# Weights: two candidate units in one case cannot silently double itscase mass.
b=assay([dict(case_id='a',candidate=True,score_defined=True,score=2,effect=1),dict(case_id='a',candidate=True,score_defined=True,score=2,effect=1),dict(case_id='b',candidate=True,score_defined=True,score=1,effect=-1)]);assert b['ranking']['helpful_prevalence_casebalanced']==.5
out={'state':'synthetic_R11_discrimination_and_matchedcoverage_primitives_pass','tied_AUC_AP_baseline':True,'missingclass_metrics_NA_not0_5_or1':True,'zeroeffect_not_harm_padding':True,'undefined_harms_counted':True,'NA_last_exactceilcoverage':True,'casebalancedunitweights':True,'allreject_not_success':True,'no_model_GPU_data_GT_annotation_access':True};f=Path(__file__).with_name('CPU_DISCRIMINATION_FIXTURE_v2.json');assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
