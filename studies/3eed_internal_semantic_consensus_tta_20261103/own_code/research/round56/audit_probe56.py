"""Audit-only probe of association dtype difference; no GT or model calls."""
from common56 import *
from audit56 import raw_affinity
import hashlib
rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');selected=[]
for pl in ['drone','quad']:selected+=sorted([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256(('R56_RAW_MAPPING_AUDIT:'+r['id']).encode()).digest())[:2]
records=[]
for row in selected:
 sid=row['id'];l=load(ROOT/'crossplatform/round53/latent336_attempt001/samples_PRIVATE'/(sid+'.npz'),['boxes','point_indices','input_sample_xyz']);c=load(ROOT/'crossplatform/round56/consensus336_attempt001/samples_PRIVATE'/(sid+'.npz'));s=load(ROOT/'crossplatform/round56/semantic_capture336_attempt001/samples_PRIVATE'/(sid+'.npz'),['subject_phrase_boxes']);rr=read(ROOT/'crossplatform/round51/reference336_headspan_attempt001/cases_PRIVATE'/(sid+'.json'));boxes=[s['subject_phrase_boxes']]+[load(ROOT/'crossplatform/round51/reference336_headspan_attempt001'/rr['references'][str(e)]['file'],['boxes'])['boxes'] for e in c['association_entities'][1:]];_,ix=np.unique(l['point_indices'],return_index=True);xyz=l['input_sample_xyz'][np.sort(ix)].astype(float)
 for i,b in enumerate(boxes):
  v=raw_affinity(l['boxes'].astype(float),b.astype(float),xyz);expected=c['association_affinity'][i];records.append({'platform':row['platform'],'matrix_number':i,'max_abs_error':float(abs(v-expected).max()),'nonzero_support_agrees':bool(np.array_equal(v>0,expected>0)),'different_entries':int((v!=expected).sum())})
dump(ROOT/'research/round56/ASSOCIATION_DTYPE_AUDIT_PROBE.json',{'GT_reads':0,'records':records});print(records)
