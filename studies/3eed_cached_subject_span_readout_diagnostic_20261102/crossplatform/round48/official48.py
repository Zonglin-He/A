"""Literal official-span diagnostics on old development inputs only.

No model, frame, point-cloud, final cohort or GT box loader is imported.
Metadata is streamed through a class/caption allowlist; classes are supplied
only to the two explicitly annotation-informed diagnostic readouts.
"""
import ast
import hashlib
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT/'crossplatform/round3/p0/upstream/src/joint_det_dataset.py'
INDEX = ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json'
PREP = ROOT/'crossplatform/round5/baseline_engineering/PREPARATION_RECEIPT.json'
OLD_ALL = ROOT/'research/round24/input/ALL336_PRIVATE.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''):h.update(b)
    return h.hexdigest()

def _allowed(rows):
    registered=json.loads(OLD_ALL.read_text())
    byid={r['id']:r for r in registered}
    assert len(rows)==336 and len(byid)==336 and {r['id'] for r in rows}==set(byid)
    for r in rows:
        assert r==byid[r['id']], 'Only immutable old ALL336 rows are permitted'
    return byid

def historical_index(rows):
    import ijson
    wanted=_allowed(rows)
    prep=json.loads(PREP.read_text())
    assert sha(INDEX)==prep['scoring_index_sha256']
    result={}
    with INDEX.open('rb') as f:
        for entry in ijson.items(f,'item'):
            sid=entry['id']
            if sid not in wanted:continue
            r=wanted[sid]
            assert sid not in result
            assert entry['platform']==r['platform'] and entry['scene']==r['scene']
            assert Path(entry['meta_path']).resolve()==Path(r['metapath']).resolve()
            assert type(entry['object_index']) is int and entry['object_index']>=0
            keep=('id','platform','scene','meta_path','object_index','metadata_container_sha256')
            result[sid]={k:entry[k] for k in keep}
            if 'waymo_view_suffix' in entry:
                result[sid]['waymo_view_suffix']=entry['waymo_view_suffix']
    assert set(result)==set(wanted)
    return result

def read_class_allowlist(rows):
    """Never retain/interpret bbox or other annotations, even for target rows."""
    import ijson
    index=historical_index(rows)
    grouped={}
    for r in rows:grouped.setdefault(r['metapath'],[]).append(r)
    classes={};receipts=[]
    for path,rr in grouped.items():
        expected={index[r['id']]['metadata_container_sha256'] for r in rr}
        assert len(expected)==1 and sha(path)==next(iter(expected))
        selected={index[r['id']]['object_index']:{} for r in rr}
        current=-1
        with Path(path).open('rb') as f:
            for prefix,event,value in ijson.parse(f):
                if prefix=='ground_info.item' and event=='start_map':current+=1
                elif current in selected and prefix in ('ground_info.item.class','ground_info.item.caption') and event=='string':
                    selected[current][prefix.rsplit('.',1)[1]]=value
        for r in rr:
            record=selected[index[r['id']]['object_index']]
            assert set(record)=={'class','caption'}
            assert ' '.join(record['caption'].replace(',', ' ,').split())==r['caption']
            classes[r['id']]=record['class'].lower()
        receipts.append({'metadata_sha256':next(iter(expected)),'targets':len(rr),'retained_fields':['class','caption'],'bbox_values_retained':False})
    assert len(classes)==336
    return classes,receipts

def synonym_constants():
    tree=ast.parse(DATASET.read_text())
    result={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('WAYMO_SYNONYMS','M3ED_SYNONYMS'):
                    result[target.id]=ast.literal_eval(node.value)
    assert set(result)=={'WAYMO_SYNONYMS','M3ED_SYNONYMS'}
    return result

def boolean_positive_map(encoded,beg,end,T):
    """Same finite char_to_token fallback and >0 binary mask as official code."""
    assert 0<T<=256
    begin=encoded.char_to_token(int(beg))
    finish=encoded.char_to_token(int(end)-1)
    if begin is None:
        try:
            begin=encoded.char_to_token(int(beg)+1)
            if begin is None:begin=encoded.char_to_token(int(beg)+2)
        except Exception:begin=None
    if finish is None:
        try:
            finish=encoded.char_to_token(int(end)-2)
            if finish is None:finish=encoded.char_to_token(int(end)-3)
        except Exception:finish=None
    full=np.zeros(256,dtype=bool)
    if begin is not None and finish is not None:
        full[begin:finish+1]=True
    return full[:T].copy(),{'beg_token':begin,'end_token':finish,'positive_tokens_in_actualT':int(full[:T].sum()),'positive_tokens_outside_actualT':int(full[T:].sum()),'finite_official_fallback':True}

def official_masks(caption,category,tokenizer,T,platform):
    """Return LITERAL indices and fixed-character-span REMAP, never choose by GT IoU."""
    assert caption==' '.join(caption.replace(',', ' ,').split())
    formatted=' '+caption+' '
    actual=caption+' . not mentioned'
    constants=synonym_constants()
    names=constants['WAYMO_SYNONYMS' if platform=='waymo' else 'M3ED_SYNONYMS'].get(category,[category])
    positions=[]
    for word in names:
        positions.extend((m.start(),m.end(),word) for m in re.finditer(r'\b'+re.escape(word)+r'\b',formatted,re.I))
    empty=np.zeros(T,dtype=bool)
    if not positions:
        return empty,empty.copy(),{'status':'UNKNOWN_no_official_synonym_match','span_formatted':None,'span_actual':None,'literal_remap_mask_equal':True}
    beg,end,word=positions[0]
    original=tokenizer(formatted,return_offsets_mapping=True)
    current=tokenizer(actual,return_offsets_mapping=True)
    assert len(current['input_ids'])==T
    literal,lr=boolean_positive_map(original,beg,end,T)
    remap,rr=boolean_positive_map(current,beg-1,end-1,T)
    assert formatted[beg:end]==actual[beg-1:end-1]
    receipt={'status':'matched' if literal.any() and remap.any() else 'UNKNOWN_empty_token_map','span_formatted':[beg,end],'span_actual':[beg-1,end-1],'matched_word':word,'synonym_list_priority_preserved':True,'literal':lr,'remap':rr,'literal_remap_mask_equal':bool(np.array_equal(literal,remap)),'official_prefix_token_count':len(original['input_ids']),'actual_token_count':T,'prefix_vs_actual_caption_token_ids_equal':original['input_ids'][1:-1]==current['input_ids'][1:len(original['input_ids'])-1],'not_oracle_or_full_official_pipeline':True}
    return literal,remap,receipt
