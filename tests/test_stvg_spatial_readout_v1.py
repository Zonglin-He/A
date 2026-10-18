import torch
from vg_tta.stvg_spatial_readout_v1 import context_probe,select_layer

def test_final_score_can_be_lost_before_top3():
 d=dict(all_boxes=torch.tensor([[.1,.1,.1,.1],[.3,.3,.1,.1],[.5,.5,.1,.1],[.8,.8,.1,.1]]),all_phrase_scores=torch.tensor([.8,.7,.6,.1]),target_all_scores=torch.tensor([.8,.7,.6,.95]))
 old,_=context_probe(d,0,0,'mixed');new,_=context_probe(d,0,0,'target');assert 3 not in old['candidate_ids']and 3 in new['candidate_ids'];assert int(torch.as_tensor(new['target_scores']).argmax())==new['candidate_ids'].index(3)

def test_one_layer_for_whole_trajectory_and_empty_fallback():
 layers=torch.ones(6,4,4)*.2;layers[:,:,0]=.8;layers[2,:,0]=.2;ref=[dict(position=0,box=[.2,.2,.2,.2]),dict(position=3,box=[.2,.2,.2,.2])]
 sel=select_layer(layers,ref);assert sel['layer']==2 and torch.equal(sel['boxes'],layers[2]);assert select_layer(layers,[])['layer']==5
