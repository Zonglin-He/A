"""Class evidence aligned by the physical source seed, before any task label is read."""
import argparse
from functools import lru_cache
from hashlib import sha256
from time import perf_counter
import numpy as np
import torch
from torch.nn.functional import normalize

from common_lex import *
from collect import Grounder
from observe_witness import interpret, PATTERN


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


def decode(model, xyz, vision, lang, pad, ids):
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
        prefix = 'last_' if i+1 == model.num_decoder_layers else str(i)+'head_'
        center,size = head(query.transpose(1,2).contiguous(),base_xyz=anchor,end_points={},prefix=prefix)
    query = normalize(model.contrastive_align_projection_image(query),dim=-1)
    token = normalize(model.contrastive_align_projection_text(lang),dim=-1)
    return (query @ token.transpose(1,2))/.07, torch.cat((center,size),-1)


def geometry(frame, model, prepared, compare):
    points = sample(frame)
    data = torch.from_numpy(points)[None].to(next(model.parameters()).device)
    native = model.backbone_net(data)
    xyz,features = native['fp2_xyz'],native['fp2_features']
    ids = torch.arange(xyz.shape[1],device=xyz.device).reshape(256,4).T.contiguous()
    assert xyz.shape == (1,1024,3)
    restore = ids.flatten().argsort()
    scores,boxes,errors = [],[],[]
    for r,(text,words,pad,spans) in enumerate(prepared):
        vision,lang = encode(model,xyz,features,words,pad)
        dots,box = decode(model,xyz,vision,lang,pad,ids)
        logit = torch.stack([torch.logsumexp(dots[:,:,idx],dim=-1)-np.log(len(idx)) for idx in spans],-1)
        scores.append(logit.reshape(1024,-1)[restore].cpu().numpy())
        boxes.append(box.reshape(1024,6)[restore].cpu().numpy())
        if compare and r==0:
            full = model(dict(point_clouds=data,text=[text]))
            physical = full['query_points_sample_inds'].long()
            native_scores,native_boxes = decode(model,xyz,vision,lang,pad,physical)
            actual_logits = full['last_proj_queries'] @ full['proj_tokens'].transpose(1,2)/.07
            actual_boxes = torch.cat((full['last_center'],full['last_pred_size']),-1)
            errors=[float((a-b).abs().max()) for a,b in ((native_scores,actual_logits),(native_boxes,actual_boxes))]
            assert max(errors)<1.e-4,errors
    return dict(logits=np.stack(scores),native_boxes=np.stack(boxes),xyz=xyz[0].cpu().numpy(),
                sample=points,exact_native_check=np.asarray(errors))


def nouns(frame, front):
    from parse97 import dependency_document, parse85, parse_subject, repair_inventory87
    from roles import nominal
    text=frame['text'].strip()
    if PATTERN.fullmatch(text):
        t,r,*_=interpret(text,{})
    else:
        doc=dependency_document(front.parser(text))
        graph=repair_inventory87(text,doc,parse85(text,doc,parse_subject(text,doc)))
        parsed=interpret(text,dict(graph=graph,document=doc))
        target=next((e for e in graph['entities'] if e['role']=='T'),None)
        t=target['head'] if target else ''
        r=''
        if parsed:
            t,r=parsed[:2]
    def category(phrase):
        cat=nominal(phrase)
        cat={'othervehicle':'other vehicle'}.get(cat,cat)
        return CLASSES.index(cat) if cat in CLASSES else -1
    return category(t),category(r)


@torch.inference_mode()
def run(source):
    rows=[r for r in panel() if r['source']==source]
    old_files=read(OLD/'observed/SEALED.json')['files']
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    engine=Grounder(source)
    front=engine._frontend
    front.parser=lru_cache(maxsize=1500)(front.parser)
    model=front.base
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    prepared=prompts(model)
    OBS.mkdir(exist_ok=True,mode=0o700)
    started=perf_counter()
    files={}; obs_hashes={}; cats={}; checked=[]
    code=digest(HERE/'observe_lex.py')
    for i,row in enumerate(rows,1):
        frame=row['frame'];filename=name(row)
        fid=sha256(repr((source,frame['target'],frame['points'],frame['seed'],frame['waymo_view'])).encode()).hexdigest()
        target=OBS/(fid+'.npz')
        if fid not in obs_hashes:
            receipt=target.with_suffix('.json')
            if receipt.exists():
                rec=read(receipt)
                assert rec['code']==code and digest(target)==rec['sha']
            else:
                result=geometry(frame,model,prepared,not checked)
                assert np.isfinite(result['logits']).all() and np.isfinite(result['native_boxes']).all()
                archive_npz(target,**result)
                rec=dict(code=code,sha=digest(target),native=result['exact_native_check'].tolist())
                save(receipt,rec)
            if rec['native']:checked.append(rec['native'])
            obs_hashes[fid]=rec['sha']
        with np.load(OLD/'observed'/filename) as past,np.load(target) as capture:
            assert digest(OLD/'observed'/filename)==old_files[filename]
            np.testing.assert_array_equal(capture['xyz'],past['seeds'])
        files[filename]=fid
        cats[filename]=nouns(frame,front)
        if i==1 or i%40==0 or i==len(rows):print(source,i,len(rows),round(perf_counter()-started,1),flush=True)
    assert checked
    save(OBS/(source+'.json'),dict(files=files,hashes=obs_hashes,nouns=cats,checked=checked,code=code,
         protocol=digest(SPEC),seconds=perf_counter()-started))


def seal():
    idx={};hashes={};cats={};audits={}
    for s in SOURCES:
        st=read(OBS/(s+'.json'))
        assert st['protocol']==digest(SPEC) and st['code']==digest(HERE/'observe_lex.py')
        assert not(idx.keys() & st['files'].keys())
        idx.update(st['files']);hashes.update(st['hashes']);cats.update(st['nouns']);audits[s]=st['checked']
    assert idx.keys()=={name(r) for r in panel()}
    assert all(digest(OBS/(k+'.npz'))==h for k,h in hashes.items())
    save(OBS/'SEALED.json',dict(files=idx,hashes=hashes,nouns=cats,native=audits,protocol=digest(SPEC),
        code=digest(HERE/'observe_lex.py'),common=digest(HERE/'common_lex.py'),labels_used=False,FINAL_access=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=(*SOURCES,'seal'));stage=p.parse_args().stage
    seal() if stage=='seal' else run(stage)
