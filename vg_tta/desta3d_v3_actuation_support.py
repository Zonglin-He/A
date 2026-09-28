"""Actual native box-probe anchors survive parser format failure."""
import torch

def spatial_positions(trace):
    times=[int(x) for x in trace['token_ids']['ordered_time_tokens']]
    selected=[]
    for row in trace['branches'][1]['probes']:
        query=torch.as_tensor(row['kwargs']['query_token_ids']).flatten().tolist()
        if query and all(int(q) in times for q in query):selected.append([times.index(int(q)) for q in query])
    if len(selected)!=1 or len(set(selected[0]))!=len(selected[0]):raise ValueError('Ambiguous actual box probe support')
    pos=selected[0]
    if len(pos)!=trace['branches'][1]['logits']['coordinate'].shape[0]:raise ValueError('Anchor/logit support mismatch')
    return pos
