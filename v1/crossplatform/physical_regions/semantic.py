"""Read native noun evidence and dimensions at fixed seed identities."""
import numpy as np
import torch
from torch.nn.functional import normalize
from .language import CLASSES


def prompts(model):
    result = []
    for rotation in range(len(CLASSES)):
        seq = CLASSES[rotation:] + CLASSES[:rotation]
        text = ' . '.join(seq) + ' . not mentioned'
        tokenizer = model.tokenizer
        encoded = tokenizer([text], return_tensors='pt')
        offsets = tokenizer([text], return_offsets_mapping=True)['offset_mapping'][0]
        positions = {}
        cursor = 0
        for noun in seq:
            start = text.index(noun, cursor)
            cursor = start + len(noun)
            positions[noun] = [i for i,(lo,hi) in enumerate(offsets) if lo < cursor and hi > start]
            assert positions[noun]
        encoded = encoded.to(next(model.parameters()).device)
        words = model.text_projector(model.text_encoder(**encoded).last_hidden_state)
        result.append((text, words, encoded.attention_mask.ne(1), [positions[n] for n in CLASSES]))
    return result

def encode(model, xyz, features, words, pad):
    return model.cross_encoder(vis_feats=features.transpose(1,2).contiguous(),
        pos_feats=model.pos_embed(xyz).transpose(1,2).contiguous(),
        padding_mask=torch.zeros(xyz.shape[:2],dtype=torch.bool,device=xyz.device),
        text_feats=words,text_padding_mask=pad,end_points={},detected_feats=None,detected_mask=None)

def decode(model, xyz, vision, lang, pad, ids, *, need_scores=True, need_boxes=True):
    """Keep all intermediate position updates; compute the requested terminal outputs."""
    batches = ids.shape[0]
    anchor = xyz[0,ids].contiguous()
    local = vision[0,ids].transpose(1,2).contiguous()
    query = model.decoder_query_proj(local).transpose(1,2).contiguous()
    center,size = model.proposal_head(local,base_xyz=anchor,end_points={},prefix='proposal_')
    v = vision.expand(batches,-1,-1).contiguous()
    w = lang.expand(batches,-1,-1).contiguous()
    mask = pad.expand(batches,-1).contiguous()
    assert model.self_position_embedding == 'loc_learned'
    for i,(layer,head) in enumerate(zip(model.decoder,model.prediction_heads)):
        query = layer(query,v,w,torch.cat((center,size),-1),None,mask,detected_feats=None,detected_mask=None)
        final = i+1 == model.num_decoder_layers
        if not final or need_boxes:
            prefix = 'last_' if final else str(i)+'head_'
            center,size = head(query.transpose(1,2).contiguous(),base_xyz=anchor,end_points={},prefix=prefix)
    scores = None
    if need_scores:
        query = normalize(model.contrastive_align_projection_image(query),dim=-1)
        token = normalize(model.contrastive_align_projection_text(lang),dim=-1)
        scores = (query @ token.transpose(1,2))/.07
    return scores, torch.cat((center,size),-1) if need_boxes else None


def prepare_nouns(model):
    result=[]
    for noun in CLASSES[:-1]:
        text='The '+noun+'.'
        ids=model.tokenizer([text],return_tensors='pt').to(next(model.parameters()).device)
        words=model.text_projector(model.text_encoder(**ids).last_hidden_state)
        result.append((text,words,ids.attention_mask.ne(1)))
    return result


def observe(model, sampled, prompt_bank, noun_bank):
    cloud = torch.from_numpy(np.ascontiguousarray(sampled, dtype=np.float32))[None].to(next(model.parameters()).device)
    backbone = model.backbone_net(cloud)
    xyz, features = backbone['fp2_xyz'], backbone['fp2_features']
    if xyz.shape != (1,1024,3):
        raise ValueError("The registered 3EED seed layout is 1024 by 3.")
    groups = torch.arange(1024, device=xyz.device).reshape(256,4).T.contiguous()
    restore = groups.flatten().argsort()
    scores, extents = [], []
    for _, words, pad, spans in prompt_bank:
        vision, lang = encode(model,xyz,features,words,pad)
        dots,_ = decode(model,xyz,vision,lang,pad,groups,need_boxes=False)
        logits = torch.stack([torch.logsumexp(dots[:,:,ix],dim=-1)-np.log(len(ix)) for ix in spans],dim=-1)
        scores.append(logits.reshape(1024,-1)[restore].cpu().numpy())
    for _, words, pad in noun_bank:
        vision, lang = encode(model,xyz,features,words,pad)
        _, boxes = decode(model,xyz,vision,lang,pad,groups,need_scores=False)
        extents.append(boxes.reshape(1024,6)[restore,3:].cpu().numpy())
    return np.stack(scores), dict(xyz=xyz[0].cpu().numpy(),extent=np.stack(extents,axis=1))
