"""Extend only the unpublished response-group manifest with full sensitivity."""
from pathlib import Path
import copy,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_referent_evidence_p0_v1 import BASE,read,put,sha
def main():
    from scripts.export_stvg_lite_batch001 import public_source_text
    out=ROOT/'results/stvg_referent_evidence_p0/2026-10-16/response001';p=BASE/'publication/response001/EXPORT_MANIFEST.json';assert not(p.parent/'PUBLIC_STAGE.json').exists()
    d=read(BASE/'RESPONSE_POSTSEAL_DIAGNOSIS.json');alias={cid:f'P{i:03}'for i,cid in enumerate(sorted({r['case_id']for r in d['source_rows']}),1)}
    assoc=read(BASE/'HISTORICAL_CORRECTION_ASSOCIATION.json');observations=[];fidmap={}
    for cid in alias:
        for i,fid in enumerate(sorted({x['physical_frame_id']for x in d['frame_rows']if x['case_id']==cid}),1):fidmap[cid,fid]=f'F{i:03}'
    for r in assoc['observations']:
        row={k:v for k,v in r.items()if k!='physical_frame_id'};row.update(case_id=alias[r['case_id']],frame_alias=fidmap[r['case_id'],r['physical_frame_id']]);observations.append(row)
    put(out/'HISTORICAL_CORRECTION_ASSOCIATION.json',dict(status='complete',count=16,observations=observations,summary=assoc['summary'],GT_read=True,scope=assoc['scope']))
    pad=read(BASE/'PADDING_POSTSEAL_DIAGNOSIS.json');records=[dict(r,case_id=alias[r['case_id']])for r in pad['records']];put(out/'PADDING_DIAGNOSIS.json',dict(status='complete',count=16,records=records,GT_read=False,posthoc_trigger='fixed four-case pixel review of selected evidence'))
    v=read(BASE/'valid_content/RESPONSE_POSTSEAL_DIAGNOSIS.json');v.pop('GT_bindings');v.pop('barrier_sha256')
    for rows in ('source_rows','pair_rows','clips'):
        v[rows]=[dict(r,case_id=alias[r['case_id']])for r in v[rows]]
    ff=[]
    for r in v['frame_rows']:
        row={k:x for k,x in r.items()if k!='physical_frame_id'};row.update(case_id=alias[r['case_id']],frame_alias=fidmap[r['case_id'],r['physical_frame_id']]);ff.append(row)
    v['frame_rows']=ff;v['posthoc']=True;v['does_not_change_primary_gate']=True
    put(out/'VALID_CONTENT_SENSITIVITY.json',v);put(out/'VALID_CONTENT_DECISION.json',read(BASE/'valid_content/PHASE2_DECISION.json'))
    review=read(BASE/'PRIVATE_VISUAL_REVIEW.json');put(out/'FIXED_VISUAL_REVIEW.json',dict(status='complete',count=4,actual_pixel_view=True,case_aliases=[alias[c]for c in review['case_ids']],observations=review['anonymous_observations'],semantic_identity_GT=False,media_or_actual_text_exported=False))
    doc=ROOT/'docs/STVG_REFERENT_RESPONSE_P0_001.md';s=doc.read_text();s+='''
## Valid-content implementation sensitivity and fixed case review

The four fixed panels reveal an implementation confound: stable projected
features in letterboxed padding attract similarity and reciprocal matching.
The original connection selector spends49.71%/52.88% of selected cell centers
in padding (HC/Vid); query-only spends31.76%/26.83%. This prevents interpreting
the unmasked failure as a route-level failure of referent connectivity.

One posthoc, non-GT valid-content sensitivity excludes cell centers outside the
known physical-image rectangle, with the same vectors and per-cell token count.
It uses zero new vision/generation/Student calls. All selection indices are
independently checked by a separate Python sort before offline geometry scoring.
The original gate, predictions and source code remain immutable.

After this exclusion, consensus minus the better single-text target occupancy
is HC−2.486487pp and Vid+2.238643pp. Connection minus query target-supported
sampled-cell pair coverage is HC−2.678571pp and Vid+16.666667pp. Known source
counts are8/6 for occupancy and8/5 for pair coverage. Vid therefore supplies a
limited positive connectivity clue after correcting the padding-support
condition; HC does not. This is posthoc development sensitivity, not an
independent confirmation, an A/B/C result or a reason to silently replace the
primary resource gate. Cross-query-swap occupancy still exceeds own-query on
Vid (31.439543% versus27.311862%), so the static embedding proxy is not yet a
reliable referent/role verifier.

The fixed panels show general action descriptions that may omit the queried
gesture/interaction, and a relation involving a screen that can highlight the
screen rather than the target participant. Those are qualitative evidence
limitations, not independently labelled identity errors. No real query/caption
text or media is published.

The supplemental association with historical E1 corrections is small: only one
HC position from one source and five Vid positions from two sources have an
improved same-frame box on the exact Student grid. On these positions,
disagreement-minus-consensus occupancy is−17.345703pp and−5.932726pp respectively.
This does not support a strong claim that useful corrections are concentrated
in high-disagreement regions. Most known Teacher-sampled positions are off the
Student grid (126HC/62Vid unique positions) and remain unknown for this strict
same-frame association; no boxes are interpolated to manufacture support.

The sensitivity's geometry output completed before a missing copy of the
parent design-lock descriptor interrupted only its metadata citation. An
isolated continuation copies the exact locked descriptor and finishes receipts
from saved scalars, with zero model, selection or GT-score repeats. Original
code/failure/recovery are preserved and exported. The primary32 control and16
response jobs all exited successfully; this was not a model or scientific
failure. A/B/C remains unexecuted under the original prelocked decision.
''';doc.write_text(s)
    bindings=read(out/'SOURCE_BINDINGS.json')
    for name in ('supplement_stvg_referent_evidence_p0_v1.py','diagnose_stvg_referent_padding_v1.py','finish_stvg_referent_padding002.py','export_stvg_referent_supplement_v1.py','publish_stvg_referent_evidence_p0_v1.py'):
        src=ROOT/'scripts'/name;dest=out/'code'/name;dest.write_text(public_source_text(src.read_bytes()));bindings['records'].append(dict(name=name,original_sha256=sha(src),public_sha256=sha(dest)))
    (out/'SOURCE_BINDINGS.json').write_text(json.dumps(bindings,indent=2)+'\n')
    # The established conservative privacy scan rejects the exact JSON token
    # caption even when it is only a categorical strategy name. Rename that
    # public category instead of weakening the gate or exporting actual text.
    def safe(x):
        if isinstance(x,dict):return {('caption_proxy'if k=='caption'else k):safe(z)for k,z in x.items()}
        if isinstance(x,list):return [safe(z)for z in x]
        return 'caption_proxy'if x=='caption'else x
    for q in out.glob('*.json'):q.write_text(json.dumps(safe(read(q)),indent=2)+'\n')
    paths={str(q.relative_to(ROOT))for q in out.rglob('*')if q.is_file()}|{str(doc.relative_to(ROOT))}
    p.write_text(json.dumps(dict(status='complete',paths=sorted(paths),files=[dict(path=q,sha256=sha(ROOT/q),bytes=(ROOT/q).stat().st_size)for q in sorted(paths)]),indent=2)+'\n');print('supplement exported',len(paths))
if __name__=='__main__':main()
