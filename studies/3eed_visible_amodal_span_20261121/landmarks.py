"""Greedy, local-only bilateral feature readout from original RGB."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from common import BASE, HERE, OUT, PRIVATE, QWEN, SPEC, crop_photo, digest, inputs, read, write

PROMPT = """The centered object in this crop is the target. Inspect its actual visible pixels. Is it a rigid road vehicle with BOTH matching outer tips of its two headlamps on the same front face visible? If not, try matching outer tips of its two taillamps on the same rear face, then matching lateral corners of the same bumper. Both must be distinguishable. Never guess an occluded tip. Do not pair front with rear. Otherwise return {"part":"unknown"}. For a valid pair give the two corresponding precise points, ordered LEFT to RIGHT IN THE IMAGE, as x,y coordinates normalized from 0 to 1000 over this crop. Also mark one DIFFERENT detail on the same vehicle, on each corresponding image side (e.g. a windshield corner); use null when unclear. JSON only: {"part":"headlamp|taillamp|bumper","left":[x,y],"right":[x,y],"other_left":[x,y],"other_right":[x,y]}."""

def parse(reply):
    try:
        obj = json.loads(re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', reply.strip(), flags=re.I))
    except ValueError:
        return {'status':'invalid_json'}
    if not isinstance(obj,dict) or obj.get('part') not in ('headlamp','taillamp','bumper'):
        return {'status':'unknown' if isinstance(obj,dict) and obj.get('part')=='unknown' else 'invalid_part'}
    fields = {}
    for key in ('left','right','other_left','other_right'):
        item = obj.get(key)
        if not isinstance(item,list) or len(item)!=2 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not 0<=v<=1000 for v in item):
            item = None
        fields[key]=item
    if fields['left'] is None or fields['right'] is None or fields['left'][0]>=fields['right'][0]:
        return {'status':'invalid_pair'}
    return {'status':'pair','part':obj['part'],**fields}

def check_assets(model_dir):
    assets = read(QWEN/'MODEL.json')
    assert digest(QWEN/'MODEL.json') == read(BASE/'results/model_seal.json')['pins']['assets']
    for name,pin in assets['files'].items():
        path=model_dir/name
        assert path.stat().st_size==pin['size']
        if 'sha256' in pin:
            assert digest(path)==pin['sha256']
        else:
            data=path.read_bytes()
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\x00'+data).hexdigest()==pin['git_blob_sha1']
    return assets['revision']

@torch.inference_mode()
def run(model_dir):
    started=time.perf_counter()
    panel, frozen=inputs()
    rows=[r for r in panel if frozen[r['uid']]['selected']]
    pin=dict(revision=check_assets(model_dir), protocol=digest(SPEC), code=digest(__file__),
             common=digest(HERE/'common.py'), frozen=digest(BASE/'results/prediction_seal.json'))
    processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
    processor.image_processor.size={'shortest_edge':65536,'longest_edge':524288}
    processor.tokenizer.padding_side='left'
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_dir,dtype=torch.bfloat16,device_map={'':'cuda:0'},
             use_safetensors=True,local_files_only=True,attn_implementation='sdpa').eval().requires_grad_(False)
    msg=[{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPT}]}]
    prompt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    folder=PRIVATE/'replies'
    folder.mkdir(parents=True,exist_ok=True)
    fresh=0
    for start in range(0,len(rows),4):
        batch=[]
        for row in rows[start:start+4]:
            path=folder/(row['uid']+'.json')
            if path.exists():
                assert read(path)['pin']==pin
            else:
                batch.append(row)
        if not batch:
            continue
        photos, rects=zip(*(crop_photo(r,frozen[r['uid']]) for r in batch))
        inp=processor(text=[prompt]*len(batch),images=list(photos),padding=True,return_tensors='pt').to('cuda')
        out=model.generate(**inp,do_sample=False,max_new_tokens=128)
        answers=processor.batch_decode(out[:,inp.input_ids.size(1):],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        for row,crop,answer in zip(batch,rects,answers,strict=True):
            uid=row['uid']
            write(folder/(uid+'.json'),dict(uid=uid,crop=crop,answer=answer,parsed=parse(answer),pin=pin))
        fresh+=len(batch)
        if start%40==0: print('Landmark crops',start+len(batch),'/',len(rows),flush=True)
    hashes={r['uid']:digest(folder/(r['uid']+'.json')) for r in rows}
    write(PRIVATE/'reply_hashes.json',hashes)
    write(OUT/'landmark_seal.json',dict(pins=pin, sha256=digest(PRIVATE/'reply_hashes.json'), calls=len(rows),
             fresh=fresh,frames=len(panel),seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),
             peak_allocated_bytes=torch.cuda.max_memory_allocated(),target_labels_used=False,FINAL_access=False))
    print('Landmark replies sealed',len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_dir',type=Path)
    run(ap.parse_args().model_dir)
