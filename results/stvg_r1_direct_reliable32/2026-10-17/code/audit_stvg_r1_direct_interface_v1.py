"""GT-free audit of all128 sealed Direct responses; never admits repaired targets."""
from pathlib import Path
import collections,hashlib,json,math,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write,guard
OLD=ROOT/'artifacts/stvg_r1_geometry32_v1';BASE=ROOT/'artifacts/stvg_r1_direct_reliable32_v1'

def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate key')
        out[k]=v
    return out
def strict_json(raw):
    s=raw.strip()
    if s.startswith('```json\n') and s.endswith('\n```'):s=s[8:-4].strip()
    elif s.startswith('```\n') and s.endswith('\n```'):s=s[4:-4].strip()
    try:return json.loads(s,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite')))
    except (ValueError,TypeError):return None
def literal_box(raw,W,H,truncated=False):
    """Diagnostic bounds, not a decoder that converts invalid answers to supervision."""
    key_mentions=len(re.findall(r'["\']bbox_2d',raw));complete=[]
    for m in re.finditer(r'"bbox_2d"\s*:\s*(null|\[[^\[\]]*\])',raw):
        try:v=json.loads(m[1])
        except (ValueError,TypeError):continue
        complete.append(v)
    boxes=[v for v in complete if isinstance(v,list) and len(v)==4 and all(type(x) in [int,float] and math.isfinite(x) for x in v)]
    within=[v for v in boxes if 0<=v[0]<v[2]<=W and 0<=v[1]<v[3]<=H]
    d=strict_json(raw);wire=d[0] if isinstance(d,list) and len(d)==1 else d
    strict_single=isinstance(wire,dict) and 'bbox_2d' in wire and (wire['bbox_2d'] is None or any(wire['bbox_2d']==v for v in boxes))
    if truncated:category='truncated_not_recoverable'
    elif strict_single and key_mentions==1:
        category='strict_complete_single_box' if within else ('strict_abstention' if wire['bbox_2d'] is None else 'strict_invalid_geometry')
    elif key_mentions>1:category='multiple_or_dangling_duplicate_key_ambiguous'
    elif len(within)==1 and key_mentions==1:category='unique_complete_box_literal_in_malformed_wrapper_diagnostic_only'
    elif boxes:category='complete_box_literal_invalid_geometry'
    elif key_mentions:category='bbox_key_without_complete_four_coordinate_literal'
    else:category='no_explicit_bbox_payload'
    return dict(category=category,bbox_key_mentions=key_mentions,complete_numeric_four_coordinate_literals=len(boxes),
                in_range_four_coordinate_literals=len(within),complete_null_literals=sum(v is None for v in complete),
                strict_document_valid=d is not None,strict_single_record=strict_single,
                coordinate_unit_from_output_not_inferred=True,diagnostic_never_admitted_as_reference=True)

def main():
    sys.addaudithook(guard)
    from scripts.stvg_r1_geometry32_v2 import parse_coords
    assert not (BASE/'RAW_AUDIT_COMPLETION.json').exists();BASE.mkdir(exist_ok=True)
    bar=read(OLD/'R1_GLOBAL_BARRIER.json');assert bar['responses']==128
    jobs={j['job_id']:j for j in read(OLD/'INPUT_JOB_LOCK.json')['records']};rr=read(OLD/'ROSTER_LOCK.json')['records'];alias={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)}
    rows=[];pins={};private=[]
    for f in bar['files']:
        p=ROOT/f['path'];assert sha(p)==f['sha256'];d=read(p);j=jobs[d['job_id']];meta=d['input_meta'];W,H=meta['processed_width'],meta['processed_height']
        assert d['direct']==parse_coords(d['raw_output'],W,H) and d['physical_frame_id']==j['physical_frame_id']
        assert W==meta['grid_width']*28 and H==meta['grid_height']*28 and meta['original_width']==j['original_image_shape'][1] and meta['original_height']==j['original_image_shape'][0]
        a=literal_box(d['raw_output'],W,H,d['truncated']);cid=alias[j['case_id']]
        rows.append(dict(case_id=cid,dataset=j['dataset'],slot=j['slot'],old_parser_status=d['direct']['status'],output_sha256=hashlib.sha256(d['raw_output'].encode()).hexdigest(),
            truncated=d['truncated'],processed_width=W,processed_height=H,actual_anchor_grid_verified=True,**a))
        private.append(dict(job_id=j['job_id'],response_path=f['path'],response_sha256=f['sha256'],audit_row=rows[-1]));pins[f['path']]=f['sha256']
    assert len(rows)==128 and collections.Counter(r['dataset'] for r in rows)==dict(hc2=64,vidstg=64)
    for p,h in pins.items():assert sha(ROOT/p)==h
    domains=[]
    for ds in ['hc2','vidstg']:
        xs=[r for r in rows if r['dataset']==ds];domains.append(dict(dataset=ds,planned=64,
            old_parser=dict(collections.Counter(r['old_parser_status'] for r in xs)),categories=dict(collections.Counter(r['category'] for r in xs)),
            any_complete_in_range_literal=sum(r['in_range_four_coordinate_literals']>0 for r in xs),
            ambiguous_multiple_or_dangling_key=sum(r['bbox_key_mentions']>1 for r in xs),
            new_unique_literal_candidates_not_previously_valid=sum(r['old_parser_status']!='valid' and r['category']=='unique_complete_box_literal_in_malformed_wrapper_diagnostic_only' for r in xs)))
    result=dict(status='complete_GT_free_audit',count=32,responses=128,domains=domains,rows=rows,
        new_reference_admissions=0,new_model_calls=0,new_Student_fits=0,GT_read=False,
        semantics_wrong_object_or_wrong_frame='unknown_without_actual_pixel_review; syntax cannot establish identity or anchor correctness',
        coordinate_policy='original explicit processor-resized pixels; no original-size/normalized1000 guessing or clamp',
        old_predictions_and_parser_unmodified=True)
    write(BASE/'RAW_RESPONSE_AUDIT.json',result);write(BASE/'PRIVATE_AUDIT_BINDING.json',dict(status='pass',count=128,records=private,GT_read=False))
    write(BASE/'RAW_AUDIT_COMPLETION.json',dict(status='pass',count=128,new_model_calls=0,new_GT_reads=0,all_response_file_SHA_readback=True,result_sha256=sha(BASE/'RAW_RESPONSE_AUDIT.json'),GT_read=False))
    print(json.dumps(domains,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
