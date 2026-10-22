"""Original native224 second-pass hidden export, without temporal interventions."""
import ast,hashlib,inspect,textwrap
import torch
from .tensors import detached
SCHEMA='ta_native224_time_decoder_inputs_v1'
TIME_FORWARD_AST='ba35c70f4cb2b9b0727776848d09c1d5e68296bdea23fb16110c3002099e1a37'
LAYER_FORWARD_AST='f675458e2dbfd0043a222b4ffd9656915da285b7d06834456a721352f88cba87'
INPUT_KEYS={'query_tgt','query_content','query_time','query_mask','encoded_feature','encoded_pos','encoded_mask'}

def _require(condition,message):
    if not condition:raise ValueError(message)

def _ast(function):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    return hashlib.sha256(ast.dump(tree.body[0],include_attributes=False).encode()).hexdigest()

def verify_decoder(decoder):
    _require(_ast(type(decoder).forward)==TIME_FORWARD_AST,'Unsupported native TimeDecoder forward')
    _require(len(decoder.layers)>0,'Empty temporal decoder')
    for layer in decoder.layers:
        _require(_ast(type(layer).forward)==LAYER_FORWARD_AST,'Unsupported temporal layer forward')
        a=layer.cross_attn_image
        _require(type(a)is torch.nn.MultiheadAttention and not a.batch_first,
                 'Expected the actual native per-frame MultiheadAttention')
        _require(a._qkv_same_embed_dim and a.bias_k is None and a.bias_v is None and not a.add_zero_attn,
                 'Unsupported attention token layout')
    _require(not decoder.training and not any(p.requires_grad for p in decoder.parameters()),
             'Temporal replay requires frozen eval decoder weights')
    return dict(time_decoder=TIME_FORWARD_AST,time_layer=LAYER_FORWARD_AST,layers=len(decoder.layers))

def _check_inputs(kwargs,grid,frames):
    _require(set(kwargs)==INPUT_KEYS,'Cache must contain the actual seven TimeDecoder kwargs')
    h,w=map(int,grid);_require(h>0 and w>0,'Positive visual token grid required')
    q=kwargs['query_tgt'];memory=kwargs['encoded_feature'];S,N,C=memory.shape
    _require(memory.dtype==torch.float32 and N==frames and S>h*w,'Wrong native memory/frame layout')
    _require(q.shape==(N,1,C) and kwargs['query_content'].shape==q.shape and kwargs['query_time'].shape==q.shape,
             'Wrong temporal query/content/time layout')
    _require(kwargs['encoded_pos'].shape==memory.shape,'Position and memory layout differ')
    _require(kwargs['encoded_mask'].shape==(N,S) and kwargs['encoded_mask'].dtype==torch.bool,
             'Preserve the original encoded padding mask exactly')
    _require(kwargs['query_mask'].shape==(1,N) and kwargs['query_mask'].dtype==torch.bool,'Wrong query padding mask')
    for value in kwargs.values():
        _require(torch.is_tensor(value),'Decoder inputs must be tensors')
        if value.is_floating_point():_require(bool(torch.isfinite(value).all()),'Nonfinite cached input')
    return h,w

def cache_temporal(model,base,records,frame_ids,source_identity=None):
    """Export native second-pass inputs already captured by FrozenNativeReplay.

    ``base.time_inputs`` is populated by hooks at the actual TimeDecoder call;
    ``temporal_inputs`` is the corresponding full-layer hidden output. No
    queries or memory are reconstructed from a spatial cache or class scores.
    """
    decoder=model.ground_decoder.time_decoder;interface=verify_decoder(decoder)
    _require(len(records)==len(base.time_inputs)==len(base.temporal_inputs)==2,'Expected the two original offsets')
    _require(hasattr(base,'time_input_layouts') and len(base.time_input_layouts)==2,
             'Actual input layouts must be captured before cloning the native kwargs')
    ids=[int(x)for x in frame_ids]
    _require(len(ids)>=4 and all(b>a for a,b in zip(ids,ids[1:])),'Physical frame IDs must increase')
    _require(not model.temp_embed.training and not any(p.requires_grad for p in model.temp_embed.parameters()),
             'Frozen temporal head required')
    offsets=[]
    for offset,(record,kwargs,hidden)in enumerate(zip(records,base.time_inputs,base.temporal_inputs)):
        _require(record['offset']==offset and not record.get('flip',False),'Unexpected offset/flip routing')
        physical=[int(x)for x in record['frame_ids']]
        _require(physical==ids[offset::2],'Physical offset grid mismatch')
        grid=tuple(map(int,base.views[offset]['info']['fea_map_size']))
        _check_inputs(kwargs,grid,len(physical))
        layouts=base.time_input_layouts[offset]
        _require(set(layouts)==INPUT_KEYS,'Missing actual native tensor layout')
        for key,value in kwargs.items():
            _require(tuple(layouts[key]['shape'])==tuple(value.shape),'Captured tensor layout shape changed')
        _require(hidden.ndim==4 and hidden.shape[0]==len(decoder.layers) and hidden.shape[1:3]==(1,len(physical)),
                 'Full native temporal hidden batch shape changed')
        logits=base.zero['logits'][offset]
        _require(logits.shape==(1,len(physical),2),'Unexpected native temporal logits')
        offsets.append(dict(offset=offset,frame_ids=physical,frame_positions=list(range(offset,len(ids),2)),
            grid_size=list(grid),inputs=detached(kwargs,'cpu'),input_layouts=layouts,hidden=detached(hidden,'cpu'),
            native_logits=detached(logits,'cpu')))
    return dict(schema=SCHEMA,frame_ids=ids,offsets=offsets,interface=interface,
        source_identity=source_identity,GT_read=False,resolution=224,
        memory_layout='original text prefix then motion visual suffix; actual padding order retained',
        capture='actual final native224 FP32 TimeDecoder boundary; no reconstructed memory')
