"""Read the two existing frozen heads at the same physical seeds and noun position."""
import argparse
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import normalize

from token_paths import CLASSES, SOURCES, WORK, archive_npz, digest, read, roster, save, signatures
from collect import Grounder
from common_lex import sample
from observe_lex import encode


def queries(model):
    result = []
    for noun in CLASSES[:-1]:
        text = f'The {noun} . not mentioned'
        tokens = model.tokenizer([text], return_offsets_mapping=True, return_tensors='pt')
        offsets = tokens.pop('offset_mapping')[0].tolist()
        start = text.index(noun)
        pos = next(i for i, (left, right) in enumerate(offsets) if left <= start < right)
        tokens = tokens.to(next(model.parameters()).device)
        words = model.text_projector(model.text_encoder(**tokens).last_hidden_state)
        result.append((text, words, tokens.attention_mask.ne(1), pos))
    assert len({v[-1] for v in result}) == 1
    return result


def final_heads(model, xyz, vision, lang, pad, ids, pos):
    b = ids.shape[0]
    anchors = xyz[0, ids].contiguous()
    feat = vision[0, ids].transpose(1, 2).contiguous()
    q = model.decoder_query_proj(feat).transpose(1, 2).contiguous()
    center, size = model.proposal_head(feat, base_xyz=anchors, end_points={}, prefix='proposal_')
    visual = vision.expand(b,-1,-1).contiguous()
    words = lang.expand(b,-1,-1).contiguous()
    mask = pad.expand(b,-1).contiguous()
    assert model.self_position_embedding == 'loc_learned'
    for layer, head in zip(model.decoder, model.prediction_heads):
        q = layer(q, visual, words, torch.cat((center,size),-1), None, mask,
                  detected_feats=None, detected_mask=None)
        out = {}
        center,size = head(q.transpose(1,2).contiguous(), base_xyz=anchors, end_points=out, prefix='read_')
    logits = out['read_sem_cls_scores']
    assert logits.shape[-1] == 256
    align = normalize(model.contrastive_align_projection_image(q),dim=-1)
    tokens = normalize(model.contrastive_align_projection_text(lang),dim=-1)
    contrast = (align @ tokens[:,pos:pos+1].transpose(1,2)).squeeze(-1)/.07
    return torch.stack((logits[...,pos], logits[...,-1], contrast),dim=-1), torch.cat((center,size),-1)


@torch.inference_mode()
def run(src):
    rows, obs = roster()
    targets = sorted({r['frame'] for r in rows if r['source']==src})
    dest = WORK/src
    dest.mkdir(mode=0o700, exist_ok=True)
    files = {}
    checks = []
    tick = perf_counter()
    if targets:
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        model = Grounder(src)._frontend.base
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        device = next(model.parameters()).device
        nouns = queries(model)
        ids = torch.arange(1024,device=device).reshape(256,4).T.contiguous()
        restore = ids.flatten().argsort()
        for number, fid in enumerate(targets):
            f = obs[fid]
            pts = sample(f['f'])
            assert digest(f['lex']) == f['lh']
            with np.load(f['lex']) as z:
                np.testing.assert_array_equal(pts,z['sample'])
                native_xyz = z['xyz']
            data = torch.from_numpy(pts)[None].to(device)
            back = model.backbone_net(data)
            xyz,feature = back['fp2_xyz'],back['fp2_features']
            np.testing.assert_array_equal(xyz[0].cpu().numpy(),native_xyz)
            per_word = []
            for text,words,pad,pos in nouns:
                vision,language = encode(model,xyz,feature,words,pad)
                values,_ = final_heads(model,xyz,vision,language,pad,ids,pos)
                per_word.append(values.reshape(1024,3)[restore].cpu().numpy())
                if number == 0:
                    native = model(dict(point_clouds=data,text=[text]))
                    direct,boxes = final_heads(model,xyz,vision,language,pad,native['query_points_sample_inds'].long(),pos)
                    aligned = native['last_proj_queries'] @ native['proj_tokens'][:,pos:pos+1].transpose(1,2)/.07
                    truth = torch.stack((native['last_sem_cls_scores'][...,pos],native['last_sem_cls_scores'][...,-1],
                                         aligned.squeeze(-1)),dim=-1)
                    box = torch.cat((native['last_center'],native['last_pred_size']),-1)
                    checks.append(float(max((truth-direct).abs().max(),(box-boxes).abs().max())))
            values = np.stack(per_word,axis=1)
            assert np.isfinite(values).all() and values.shape==(1024,len(CLASSES)-1,3)
            path = dest/(fid+'.npz')
            archive_npz(path, xyz=native_xyz, values=values)
            files[fid] = digest(path)
            if (number+1)%16==0:
                print(src,number+1,len(targets),round(perf_counter()-tick,1),flush=True)
        assert len(checks)==len(CLASSES)-1 and max(checks)<1e-4,checks
    save(dest/'SEALED.json',dict(files=files, checks=checks, pins=signatures(), labels=False,
                                total_seconds=perf_counter()-tick,updates=0))
    print('captures',src,len(files),max(checks,default=0.),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',choices=SOURCES)
    run(p.parse_args().source)
