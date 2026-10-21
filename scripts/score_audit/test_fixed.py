"""Synthetic fixed-score audit tests; no assets or real labels."""
import numpy as np
import torch
import torch.nn.functional as F

from scripts.score_audit.fixed import single_replacement_delta, path_score, relative_feedback, outcome_summary


def test_single_replacement_includes_incoming_and_outgoing_at_boundaries():
    unary=np.arange(12,dtype=float).reshape(4,3)/17
    pair=np.arange(27,dtype=float).reshape(3,3,3)/13
    context=np.array([1,2,0,1])
    for node in range(4):
        result=single_replacement_delta(unary,pair,context,node,2)
        a,b=context.copy(),context.copy();a[node]=2;b[node]=0
        assert abs(result['total']-(path_score(unary,pair,a)-path_score(unary,pair,b)))<1e-12
        if node==0:assert result['incoming']==0
        if node==3:assert result['outgoing']==0
    assert single_replacement_delta(unary,pair,context,1,2)['outgoing']!=0


def test_same_state_feedback_exact_component_decomposition():
    features=F.normalize(torch.arange(24,dtype=torch.float64).reshape(2,3,4)+1,dim=-1)
    bank=dict(features=features,projection=torch.tensor([[.2,.8,.4],[.5,.3,.9]],dtype=torch.float64),
              guard=torch.tensor([[0,.2,.3],[0,.5,.4]],dtype=torch.float64),
              link=torch.arange(9,dtype=torch.float64).reshape(1,3,3)/10)
    memory=F.normalize(torch.tensor([[1.,2,1,3],[3.,2,1,1]],dtype=torch.float64),dim=-1)
    previous=torch.tensor([[1.,0,0],[0.,1,0]],dtype=torch.float64)
    r=[]
    for node in range(2):
        link=previous[node]@bank['link'][node-1] if node else torch.zeros(3,dtype=torch.float64)
        r.append(bank['projection'][node]+.25*link+.1*(features[node]@memory[node])-.1*bank['guard'][node])
    path=dict(memory_before=memory,previous_action_probs=previous,r=torch.stack(r))
    for node in range(2):
        for slot in (1,2):
            out=relative_feedback(bank,path,node,slot)
            assert abs(sum(out['terms'].values())-float(path['r'][node,slot]-path['r'][node,0]))<1e-12


def test_harm_denominator_is_replacements_not_all_fixed_nodes():
    rows=[dict(query_index=0,delta_IoU_pp=-10.),dict(query_index=0,delta_IoU_pp=3.),dict(query_index=1,delta_IoU_pp=0.)]
    result=outcome_summary(rows,query_count=3)
    assert result['replacements_with_GT']==3 and result['harmed']==1
    assert result['harm_fraction']==1/3 and result['severe_drop_over5pp']==1
    assert result['mean_loss_among_harmed_pp']==-10
