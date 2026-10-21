"""CPU-only actual-TA decoder tests; no weights, RGB, query data or GT."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from stvg import tube_replay as tube

torch.set_num_threads(1)


def small_native():
    # Compile the original class AST only, with tiny synthetic dimensions.
    # This executes no external import, loader, network or model checkpoint.
    path=Path(__file__).resolve().parents[1]/'external/TA-STVG/models/grounding_model/query_decoder.py'
    tree=ast.parse(path.read_text())
    names={'TimeDecoder','TimeDecoderLayer','_get_activation_fn'}
    tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names]
    ns=dict(torch=torch,nn=torch.nn,Tensor=torch.Tensor,Optional=Optional,F=F)
    exec(compile(tree,str(path),'exec'),ns)
    cfg=SimpleNamespace(MODEL=SimpleNamespace(TASTVG=SimpleNamespace(HIDDEN=8,HEADS=2,FFN_DIM=16,DROPOUT=0.)))
    torch.manual_seed(19)
    decoder=ns['TimeDecoder'](cfg,2,return_intermediate=True,return_weights=True,d_model=8).eval().requires_grad_(False)
    head=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.ReLU(),torch.nn.Linear(8,2)).eval().requires_grad_(False)
    return SimpleNamespace(ground_decoder=SimpleNamespace(time_decoder=decoder),temp_embed=head)


def cache_fixture():
    model=small_native();ids=[2,5,11,18,25,40];records=[];inputs=[];hidden=[];logits=[]
    generator=torch.Generator().manual_seed(23)
    for offset in (0,1):
        N=3;S=6;C=8
        memory=torch.randn(S+4,N,C,generator=generator)[4:]
        kw=dict(query_tgt=torch.randn(1,1,C,generator=generator).expand(N,1,C),
            query_content=torch.randn(1,1,C,generator=generator).expand(N,1,C),
            query_time=torch.randn(N,1,C,generator=generator),query_mask=torch.zeros(1,N,dtype=torch.bool),
            encoded_feature=memory,encoded_pos=torch.randn(S,N,C,generator=generator),
            encoded_mask=torch.zeros(N,S+4,dtype=torch.bool)[:,:S])
        # Keep this exact historical mask, including arbitrary prefix/suffix
        # padding; the adapter must not "correct" it from token semantics.
        kw['encoded_mask'][0,1]=True;kw['encoded_mask'][1,-1]=True
        with torch.no_grad():h=model.ground_decoder.time_decoder(**kw);z=model.temp_embed(h)[-1]
        records.append(dict(offset=offset,flip=False,frame_ids=ids[offset::2]));inputs.append(kw);hidden.append(h);logits.append(z)
    layouts=[{k:dict(shape=list(v.shape),stride=list(v.stride()),storage_offset=v.storage_offset())
              for k,v in item.items()}for item in inputs]
    base=SimpleNamespace(time_inputs=inputs,time_input_layouts=layouts,temporal_inputs=hidden,zero={'logits':logits},
        views=[{'info':{'fea_map_size':(2,2)}}for _ in range(2)])
    cache=tube.cache_temporal(model,base,records,ids)
    boxes=torch.tensor([[.35,.4,.3,.3],[.5,.5,.25,.25],[.6,.4,.2,.4],
                        [.3,.6,.35,.2],[.7,.6,.2,.2],[.5,.3,.3,.25]],dtype=torch.float64)
    return model,cache,boxes,base,records


def test_per_head_query_visual_mass_text_padding_and_zero_reliability():
    N,H,Q,L,V=2,3,4,2,4
    generator=torch.Generator().manual_seed(5)
    A=torch.randn(N*H,Q,L+V,dtype=torch.float64,generator=generator).softmax(-1)
    legal=torch.tensor([[True,True,False,True],[True,False,True,True]])
    padded=torch.cat((torch.ones(N,L,dtype=torch.bool),legal),-1)[:,None,None,:]
    A=A.reshape(N,H,Q,L+V)*padded;A=A/A.sum(-1,keepdim=True);A=A.reshape(N*H,Q,L+V)
    M=torch.tensor([[.1,.9,0.,.4],[.3,0.,.7,.5]],dtype=torch.float64)
    r=torch.tensor([.8,0.],dtype=torch.float64)
    new=tube.redistribute_attention(A,M,r,legal,H,.5)
    assert torch.equal(new[...,:L],A[...,:L])
    old_visual=A[...,-V:].reshape(N,H,Q,V);new_visual=new[...,-V:].reshape(N,H,Q,V)
    torch.testing.assert_close(new_visual.sum(-1),old_visual.sum(-1),atol=2e-16,rtol=0)
    assert torch.equal(new_visual.masked_select(~legal[:,None,None,:]),old_visual.masked_select(~legal[:,None,None,:]))
    assert torch.equal(new.reshape(N,H,Q,L+V)[1],A.reshape(N,H,Q,L+V)[1])
    assert not torch.equal(new.reshape(N,H,Q,L+V)[0],A.reshape(N,H,Q,L+V)[0])
    assert tube.redistribute_attention(A,M,r,legal,H,0)is A


def test_empty_visual_mass_is_finite_and_preserves_text_and_padding():
    A=torch.tensor([[[.4,.6,0.,0.,0.,0.]]],dtype=torch.float64)
    legal=torch.zeros(1,4,dtype=torch.bool)
    new=tube.redistribute_attention(A,torch.zeros(1,4),torch.ones(1),legal,1,.5)
    assert torch.equal(new,A) and torch.isfinite(new).all()


def test_continuous_context_occupancy_and_permutation_preserve_each_legal_distribution():
    boxes=torch.tensor([[.5,.5,.4,.2],[.1,.1,.3,.3],[1.5,.5,.2,.2]],dtype=torch.float64)
    original=boxes.clone();legal=torch.ones(3,12,dtype=torch.bool);legal[1,3]=False
    M,meta=tube.occupancy_prior(boxes,(3,4),legal)
    D,dmeta=tube.occupancy_prior(boxes,(3,4),legal,permuted=True)
    assert torch.equal(boxes,original)
    assert abs(float(M[0].sum())-.6*.3*12)<1e-14
    assert M[2].eq(0).all() and M[1,3]==D[1,3]==0
    for t in range(3):assert torch.equal(M[t,legal[t]].sort().values,D[t,legal[t]].sort().values)
    assert dmeta['legal_counts']==meta['legal_counts']
    assert dmeta['shifts']==[6,5,6]


def test_capture_keeps_actual_kwargs_physical_offsets_and_alpha0_recovers_bitwise():
    model,cache,boxes,base,records=cache_fixture()
    assert tube.verify_decoder(model.ground_decoder.time_decoder)['layers']==2
    for offset in (0,1):
        for key,value in base.time_inputs[offset].items():
            assert torch.equal(cache['offsets'][offset]['inputs'][key],value)
            restored=tube.restore_layout(cache['offsets'][offset]['inputs'][key],cache['offsets'][offset]['input_layouts'][key],'cpu')
            assert restored.stride()==value.stride() and restored.storage_offset()==value.storage_offset()
            assert torch.equal(restored,value)
        assert cache['offsets'][offset]['frame_ids']==records[offset]['frame_ids']
    original_softmax=F.softmax;original_forward=torch.nn.MultiheadAttention.forward
    result=tube.replay_temporal(model,cache,boxes,torch.ones(6),alpha=0,device='cpu')
    for actual,record in zip(result['logits'],cache['offsets']):assert torch.equal(actual,record['native_logits'])
    assert result['frame_ids']==cache['frame_ids']
    assert result['cost']['extra_decoder_calls']==2 and result['cost']['offset_time_decoder_calls']==[1,1]
    assert result['attention_audit']==[]
    assert F.softmax is original_softmax and torch.nn.MultiheadAttention.forward is original_forward
    assert not torch.cuda.is_initialized()


def test_real_native_MHA_reweighting_is_per_head_and_C_D_have_matched_budget_and_priors():
    model,cache,boxes,base,_=cache_fixture();r=torch.tensor([1.,.8,.6,.4,.2,0.],dtype=torch.float64)
    params=[p.detach().clone()for p in model.ground_decoder.time_decoder.parameters()]
    C=tube.replay_temporal(model,cache,boxes,r,alpha=.5,device='cpu')
    D=tube.replay_temporal(model,cache,boxes,r,alpha=.5,permuted=True,device='cpu')
    assert C['cost']['extra_decoder_calls']==D['cost']['extra_decoder_calls']==2
    assert C['cost']['extra_encoder_calls']==C['cost']['extra_448_calls']==C['cost']['extra_expert_calls']==0
    assert len(C['attention_audit'])==4
    for a in C['attention_audit']+D['attention_audit']:
        assert a['heads']==2 and a['queries']==1 and a['frames']==3
        assert a['text_max_error']==a['padding_max_error']==0
        assert a['visual_mass_max_error']<1e-6
    for c,d in zip(C['priors'],D['priors']):
        assert torch.equal(c['reliability'],d['reliability']) and torch.equal(c['legal_visual'],d['legal_visual'])
        assert c['frame_ids']==d['frame_ids']
        for t in range(3):
            valid=c['legal_visual'][t]
            assert torch.equal(c['M'][t,valid].sort().values,d['M'][t,valid].sort().values)
    assert any(not torch.equal(z,x['native_logits'])for z,x in zip(C['logits'],cache['offsets']))
    for p,before in zip(model.ground_decoder.time_decoder.parameters(),params):
        assert torch.equal(p,before) and not p.requires_grad and p.grad is None
    assert all('forward'not in layer.cross_attn_image.__dict__ for layer in model.ground_decoder.time_decoder.layers)


def test_guided_attention_restores_original_instance_paths_after_exception():
    model=small_native();decoder=model.ground_decoder.time_decoder
    with pytest.raises(RuntimeError,match='synthetic'):
        with tube.guided_attention(decoder,torch.zeros(3,4),torch.ones(3),torch.ones(3,4,dtype=torch.bool),.5):
            raise RuntimeError('synthetic')
    assert all('forward'not in layer.cross_attn_image.__dict__ for layer in decoder.layers)


def test_cache_rejects_wrong_offset_or_reconstructed_padding_shape():
    model,cache,boxes,base,records=cache_fixture()
    wrong=copy.deepcopy(records);wrong[1]['frame_ids'][0]=999
    with pytest.raises(ValueError,match='Physical offset'):tube.cache_temporal(model,base,wrong,cache['frame_ids'])
    invalid=copy.deepcopy(base);invalid.time_inputs[0]['encoded_mask']=torch.zeros(3,5,dtype=torch.bool)
    with pytest.raises(ValueError,match='padding mask'):tube.cache_temporal(model,invalid,records,cache['frame_ids'])


def test_layout_restore_preserves_broadcast_transpose_gaps_and_storage_offset_without_arithmetic():
    broadcast=torch.arange(12,dtype=torch.float32).reshape(3,4).T[None].expand(5,4,3)
    gaps=torch.arange(8*4*3,dtype=torch.float32).reshape(8,4,3)[2:6,1:3]
    for original in (broadcast,gaps):
        layout=dict(shape=list(original.shape),stride=list(original.stride()),storage_offset=original.storage_offset())
        values=original.clone()
        reconstructed=tube.restore_layout(values,layout,'cpu')
        assert torch.equal(reconstructed,original)
        assert reconstructed.stride()==original.stride()
        assert reconstructed.storage_offset()==original.storage_offset()
    assert broadcast.clone().stride()!=broadcast.stride()
    invalid=broadcast.clone();invalid[1,0,0]+=1
    layout=dict(shape=list(broadcast.shape),stride=list(broadcast.stride()),storage_offset=0)
    with pytest.raises(ValueError,match='Broadcast layout disagrees'):tube.restore_layout(invalid,layout,'cpu')
