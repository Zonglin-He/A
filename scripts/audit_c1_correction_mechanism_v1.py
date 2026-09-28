"""Independent metric recomputation and diagnostic intervention invariants."""
import sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.c1_correction_mechanism_v1 import *
from scripts.audit_c1_concept_state_diagnostic_v1 import overlap_np

def metric(boxes,x,g):
 io=overlap_np(boxes,g['boxes']);valid=np.asarray(g['valid'],bool);u=valid.copy();u[x['observed']]=False;ids=np.asarray(x['frame_ids']);a=x['indices']
 if a is None:v=0.
 else:
  lo,hi=ids[a[0]],ids[a[1]]+1;gs,ge=g['interval'];v=float(io[valid&(ids>=lo)&(ids<hi)].sum()/max(1,((ids>=min(lo,gs))&(ids<max(hi,ge))).sum()))
 return dict(vIoU_corrected=v,sIoU=float(io[valid].mean()),unobserved_sIoU=float(io[u].mean()) if u.any() else None)
def run():
 import torch
 from scripts.c1_concept_state_diagnostic_v1 import check_parent
 p=read(OUT/'LOCK.json');labels=read(p['labels']);results={r['key']:r for r in read(OUT/'PANEL_RESULTS.json')};process={r['key']:r for r in read(OUT/'PROCESS_RESULTS.json')};inter=read(OUT/'INTERVENTIONS.json');ev=read(OUT/'OFFICIAL_TRACK_EVIDENCE.json');counts=collections.Counter();maxerr=0.
 def close(a,b):
  nonlocal maxerr
  if a is None or b is None:assert a is None and b is None;return
  err=abs(a-b);maxerr=max(maxerr,err);assert err<1e-9,(a,b,err)
 for r in p['rows']:
  assert sha(r['parent_file'])==r['parent_sha'];x=original(r);g=labels[r['key']];res=process[r['key']]
  for t,rr in zip(x['fits']['F']['path'],res['path']):
   for m,v in metric(t['boxes'],x,g).items():close(v,rr['metrics'][m]);counts['original_path_metric_values']+=1
  close(res['local_delta'],res['F']['vIoU_corrected']-res['Before']['vIoU_corrected']);counts['process_sources']+=1
 for item in p['panel']:
  k=item['key'];r=next(r for r in p['rows'] if r['key']==k);x=load(OUT/'fits'/(stem(r)+'.pt'));old=original(r);res=results[k];d=inter[k];g=labels[k]
  assert x['indices']==old['indices'] and torch.equal(x['before'],old['before']);assert x['oracle_supervision'] and not x['future_writeback'] and not x['GT_labels_read_in_worker'];counts['arrival_and_time']+=1
  assert x['arms']['B']['anchors']==x['arms']['D']['anchors']==d['repaired_anchors'];assert x['arms']['A']['anchors']==x['arms']['C']['anchors']==old['anchors']
  for rep in d['repairs']:
   po=next(z for z in d['positions'] if z['position']==rep['position']);src=next(c for c in po['candidates'] if c['index']==rep['from_index']);dst=next(c for c in po['candidates'] if c['index']==rep['to_index']);assert src['identity'].startswith(('C:','O:')) and dst['identity']=='T';targets=sorted([c for c in po['candidates'] if c['identity']=='T'],key=lambda c:(-c['score'],c['candidate_id']));assert dst['index']==targets[0]['index'];counts['actual_pool_identity_repairs']+=1
  for arm,rr in x['arms'].items():
   assert len(rr['anchors'])==len(old['anchors'])
   for a,b in zip(rr['anchors'],old['anchors']):assert a['position']==b['position'] and a['weight']==b['weight']==1;counts['weight_position_invariants']+=1
   z=rr['fit'];assert z['failure'] is None;assert z['selected_step']==min(range(len(z['losses'])),key=lambda i:z['losses'][i]);assert torch.equal(z['final']['boxes'],z['path'][z['selected_step']]['boxes']);assert all(torch.equal(v,old['fits']['F']['initial_state'][n]) for n,v in z['initial_state'].items());counts['fresh_arrival_fits']+=1
   if arm=='A':
    for a,b in zip(z['path'],old['fits']['F']['path']):assert torch.equal(a['boxes'],b['boxes']) and a['loss']==b['loss'];counts['reused_A_path_states_exact']+=1
   for t,tr in zip(z['path'],res['arms'][arm]['path']):
    for m,v in metric(t['boxes'],x,g).items():close(v,tr['metrics'][m]);counts['new_path_metric_values']+=1
    if rr['freeze_LN']:
     for n,v in t['state'].items():
      if n!=QUERY:assert torch.equal(v,z['initial_state'][n]);counts['LN_tensor_fixed']+=1
   counts['full_reinsertions']+=len(rr['full_reinserted_steps'])
  counts['panel_sources']+=1
 parent=check_parent();assert counts['process_sources']==64 and counts['panel_sources']==16
 write(OUT/'AUDIT.json',dict(status='passed',counts=dict(counts),maximum_metric_error=maxerr,parent_artifacts_unchanged=parent,identity_limit='Visual single-reviewer annotations are inspectable evidence, not independently validated identity ground truth.',production_registry_unchanged=True))
 print('AUDIT',dict(counts),'maxerr',maxerr)
if __name__=='__main__':run()
