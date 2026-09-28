import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v3_source import configure_branch,detached_referent_pool,structured_support,structured_ce,source_branch_objective


class Tokenizer:
    unk_token_id=None
    pad_token_id=0
    def __init__(self):
        tokens=['<|object_ref_start|>','<|object_ref_end|>','<|time_start|>','<|time_end|>',
                '<|box_start|>','<|box_end|>','<|im_end|>','<null>','<text_mask>','\n','prompt','person']
        tokens += [f'<t{i}>' for i in range(1,101)]+[f'<{i}>' for i in range(1001)]
        self.ids={s:i+1 for i,s in enumerate(tokens)}
    def convert_tokens_to_ids(self,s):return self.ids[s]
    def encode(self,s,**kw):return [self.ids[s]]


def official_sample(T=7):
    tok=Tokenizer(); ids=tok.ids
    seq=['prompt','<|object_ref_start|>','person','<|object_ref_end|>','\n',
         '<|time_start|>','<t2>','<t3>','<|time_end|>','\n']
    for j in [2,3]:seq += [f'<t{j}>','<|box_start|>','<10>','<20>','<100>','<200>','<|box_end|>']+(['\n'] if j==2 else [])
    seq+=['<|im_end|>','\n'];inp=torch.tensor([ids[s] for s in seq])
    data=dict(input_ids=inp,labels=inp.clone(),mm_token_type_ids=torch.zeros_like(inp),video_grid_thw=torch.tensor([[T,2,2]]))
    data['labels'][0]=-100
    path=Path('external/ParallelTubeDecoding/src/dataset/sft_dataset.py')
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SupervisedDataset')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_append_ptd_targets')
    ns=dict(torch=torch,PTD_BLOCK_SIZE=6,IGNORE_INDEX=-100,DEFAULT_IM_END_TOKEN='<|im_end|>')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    obj=SimpleNamespace(processor=SimpleNamespace(tokenizer=tok),ptd_token_ids=ids,newline_id=ids['\n'],
        time_id_to_index={ids[f'<t{i}>']:i for i in range(1,101)},time_index_to_id={i:ids[f'<t{i}>'] for i in range(1,101)},
        coordinate_id_to_value={ids[f'<{i}>']:i for i in range(1001)})
    d=ns['_append_ptd_targets'](obj,data,1);d['ptd_prefix_lengths']=d.pop('ptd_prefix_length').reshape(1)
    for k in ['input_ids','labels','ptd_position_ids','ptd_context_limits','attention_mask','mm_token_type_ids']:d[k]=d[k][None]
    return d,tok


@pytest.mark.parametrize('T',[7,31,32])
def test_official_MTP_endpoint_coordinate_support(T):
    d,t=official_sample(T);before={k:v.clone() for k,v in d.items()};s=structured_support(d,t)
    assert s['targets']=={'event':[1,2],'spatial':[10,20,100,200]*2}
    assert len(s['time_ids'])==T and len(s['coordinate_ids'])==1001
    assert all(torch.equal(v,d[k]) for k,v in before.items())
    assert all(p>=int(d['ptd_prefix_lengths'][0]) for p in s['positions']['event']+s['positions']['spatial'])


def test_full_support_reductions_and_gradient():
    for branch,n,c in [('event',2,7),('spatial',8,1001)]:
        x=torch.zeros(n,c,requires_grad=True);y=torch.zeros(n,dtype=torch.long)
        loss=structured_ce(x,y,branch);want=torch.tensor(float((2 if branch=='event' else 1)*__import__('math').log(c)))
        torch.testing.assert_close(loss,want);loss.backward();assert torch.isfinite(x.grad).all() and x.grad.norm()>0


def test_branch_scope_no_shared_LN_and_live_gradient():
    torch.manual_seed(7);m=Desta3DAdapterV2(in_channels=8,query_dim=8,hidden_dim=128,architecture='dual3d')
    names=configure_branch(m,'event',0);assert sum(p.numel() for n,p in names)==33280
    assert all(n.startswith(('film_event.','norm_event.')) for n,p in names)
    q=torch.randn(1,3,8);v=torch.randn(1,3,2,2,8)
    o=m(v,q,torch.ones(1,3,dtype=torch.bool),frame_times=torch.tensor([[0.,1.,2.]]))
    o['updated_tokens_event'].square().sum().backward()
    assert any(p.grad is not None and p.grad.norm()>0 for _,p in names)
    assert all(p.grad is None for n,p in m.named_parameters() if not p.requires_grad)
    for level in (1,2,3):
        selected=configure_branch(m,'spatial',level)
        assert all(not n.startswith(('film_event.','norm_stem.','out_proj_event.','event_reader.')) for n,p in selected)


def test_referent_stopgrad_and_uniform_control():
    features=torch.randn(2,3,2,4,5,requires_grad=True);mask=torch.zeros(2,3,2,4,requires_grad=True)
    pooled,w=detached_referent_pool(features,mask);torch.testing.assert_close(pooled,features.mean((2,3)))
    pooled.square().sum().backward();assert mask.grad is None and features.grad.norm()>0
    torch.testing.assert_close(w.sum((2,3)),torch.ones(2,3))


def test_objective_matches_manual_original_denominator_and_checkpoint_gradient():
    d,t=official_sample();s=structured_support(d,t);torch.manual_seed(9)
    h=torch.nn.Parameter(torch.randn(d['input_ids'].shape[1],5));head=torch.nn.Linear(5,len(t.ids)+1,bias=False).requires_grad_(False)
    model=SimpleNamespace(model=lambda **kw:SimpleNamespace(last_hidden_state=h[None]),lm_head=head)
    for branch in ['event','spatial']:
        loss,info=source_branch_objective(model,d,s,branch)
        logits=head(h);classes=s['time_ids' if branch=='event' else 'coordinate_ids'];p=s['positions'][branch]
        st=structured_ce(logits[p][:,classes],torch.tensor(s['targets'][branch]),branch)
        targetpos=torch.nonzero(s['original'][branch][0]).flatten()
        regularizer=F.cross_entropy(logits[targetpos-1],d['labels'][0,targetpos])
        expected=st+.05*regularizer
        torch.testing.assert_close(loss,expected)
        g=torch.autograd.grad(loss,h)[0];g2=torch.autograd.grad(expected,h)[0]
        torch.testing.assert_close(g,g2);assert head.weight.grad is None


def test_missing_support_rejected():
    d,t=official_sample();s=structured_support(d,t);d['labels'][0,s['positions']['event'][0]+1]=-100
    with pytest.raises(ValueError,match='missing structured'):structured_support(d,t)
