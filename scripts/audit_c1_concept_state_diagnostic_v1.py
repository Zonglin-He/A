"""Independent NumPy scoring and cross-state/source integrity audit."""
import sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.c1_concept_state_diagnostic_v1 import *
import numpy as np

def overlap_np(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 al=a[:,:2]-a[:,2:]/2;ah=a[:,:2]+a[:,2:]/2;bl=b[:,:2]-b[:,2:]/2;bh=b[:,:2]+b[:,2:]/2
 inter=np.maximum(0,np.minimum(ah,bh)-np.maximum(al,bl)).prod(1)
 return inter/np.maximum(1e-12,a[:,2:].prod(1)+b[:,2:].prod(1)-inter)
def independently_score(b,x,g):
 io=overlap_np(b,g['boxes']);valid=np.asarray(g['valid'],bool);unobs=valid.copy();unobs[x['observed']]=False
 ids=np.asarray(x['frame_ids']);ps=ids[x['indices'][0]];pe=ids[x['indices'][1]]+1;gs,ge=g['interval']
 inside=(ids>=ps)&(ids<pe);union=(ids>=min(gs,ps))&(ids<max(ge,pe))
 return dict(vIoU_corrected=float(io[valid&inside].sum()/max(1,union.sum())),sIoU=float(io[valid].mean()),unobserved_sIoU=float(io[unobs].mean()) if unobs.any() else None)
def run():
 import torch
 p=verify();parent_count=check_parent();labels=read(p['labels']);rows={r['key']:r for r in read(OUT/'SOURCE_RESULTS.json')};geometry={r['key']:r for r in read(OUT/'GEOMETRY.json')};matrices=load(OUT/'DIFFERENCE_MATRICES.pt');counts=collections.Counter();errors=collections.defaultdict(float)
 def close(a,b,key,tol=1e-9):
  if a is None or b is None:assert a is None and b is None;return
  err=float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) if np.size(a) else 0.;errors[key]=max(errors[key],err);assert err<=tol,(key,err)
 for r in p['rows']:
  for path,h in [(r['parent_file'],r['parent_sha']),(r['features_file'],r['features_sha']),(r['original'],r['original_sha']),(r['history'],r['history_sha'])]:assert sha(path)==h;counts['upstream_hashes']+=1
  x=load(OUT/'cross'/(stem(r)+'.pt'));px=load(r['parent_file']);feat=load(r['features_file']);res=rows[r['key']];g=labels[r['key']]
  assert x['indices']==px['indices'] and x['frame_ids']==px['frame_ids'] and x['anchors']==px['anchors'];counts['frame_time_reference_invariants']+=1
  R=px['fits'][name('R',r['config'])]['path'][-1];S=px['fits'][name('S',r['config'])]['path'][-1]
  assert x['actual_step']==R['step']==S['step']==(10 if px['anchors'] else 0)
  for tag,z in x['arms'].items():
   ln,qr=tag;states={'R':R['state'],'S':S['state']}
   for k,v in z['state'].items():assert torch.equal(v,states[qr if k==QUERY else ln][k]);counts['exact_component_tensors']+=1
   assert z['full_reinsertion'];counts['full_model_reinsertions']+=1
   if ln==qr:
    parent=(R if ln=='R' else S);assert torch.equal(z['boxes'],parent['boxes']);close(z['reference_loss'],parent['loss'],'diagonal_loss',1e-6);counts['diagonal_boxes_exact']+=1
   if not x['anchors']:assert torch.equal(z['boxes'],x['before']);counts['empty_anchor_unchanged']+=1
  boxes={a:z['boxes'] for a,z in x['arms'].items()};boxes.update(Before=x['before'],F10=x['F10'])
  for a,b in boxes.items():
   mm=independently_score(b,x,g)
   for metric,val in mm.items():close(val,res['metrics'][a][metric],'metric',1e-10);counts['independent_metric_values']+=1
  for metric in ['vIoU_corrected','sIoU','unobserved_sIoU']:
   rr,rs,sr,ss=[res['metrics'][a][metric] for a in ['RR','RS','SR','SS']]
   if rr is None:continue
   eff=dict(Q_at_R=rs-rr,Q_at_S=ss-sr,L_at_R=sr-rr,L_at_S=ss-rs,interaction=ss-sr-rs+rr,total=ss-rr,Q_symmetric=((rs-rr)+(ss-sr))/2,L_symmetric=((sr-rr)+(ss-rs))/2)
   for n,v in eff.items():close(v,res['effects'][n][metric],'effect');counts['scalar_effects']+=1
  for n,a,b in [('Q_at_R','RS','RR'),('Q_at_S','SS','SR'),('L_at_R','SR','RR'),('L_at_S','SS','RS'),('total','SS','RR')]:
   close(1-overlap_np(boxes[a],boxes[b]).mean(),res['shifts'][n]['disagreement'],'box_disagreement');counts['box_shift_checks']+=1
  for tag,group in [('S','basis'),('N','far_basis')]:
   D=np.stack([np.asarray(v['z'])-np.asarray(feat['original']['z']) for v in feat['groups'][group]],axis=1).astype(float) if feat['groups'][group] else np.empty((256,0))
   saved=matrices[r['key']][tag];assert np.array_equal(D,saved['D'].numpy());u,s,vh=np.linalg.svd(D,full_matrices=False)
   close(s,geometry[r['key']]['groups'][tag]['singular_values'],'singular_value',1e-8)
   close(np.linalg.norm(D,axis=0)/float(feat['scale']),[v['delta_relative'] for v in geometry[r['key']]['groups'][tag]['expressions']],'relative_raw',1e-8)
   close(np.asarray(saved['U'])@np.diag(np.asarray(saved['S']))@np.asarray(saved['Vh']),D,'SVD_reconstruction',1e-10)
   for expr,er in zip(feat['groups'][group],geometry[r['key']]['groups'][tag]['expressions']):close(1-overlap_np(expr['boxes'],x['before']).mean(),er['box_disagreement_all'],'expression_disagreement');counts['expression_views']+=1
   counts['difference_matrices']+=1
  counts['sources']+=1
 summary=read(OUT/'CROSS_SUMMARY.json')
 for key,s in summary.items():
  split,cohort=key.split('|');rr=[v for v in rows.values() if v['cohort']==cohort and (split=='all_historical' or v['split']==split)];assert len(rr)==s['n']
  for group,source_key in [('arms','metrics'),('effects','effects'),('shifts','shifts')]:
   for arm,metrics in s[group].items():
    for metric,rec in metrics.items():
     vals=np.asarray([v[source_key][arm][metric] for v in rr if v[source_key][arm][metric] is not None]);close(vals.mean(),rec['mean'],'aggregate');rng=np.random.default_rng(20260923);b=vals[rng.integers(len(vals),size=(10000,len(vals)))].mean(1);close(np.quantile(b,[.025,.975]),rec['ci95'],'bootstrap');counts['aggregate_mean_CI']+=1
 assert counts['sources']==64 and counts['full_model_reinsertions']==256
 write(OUT/'AUDIT.json',dict(status='passed',counts=dict(counts),maximum_errors=dict(errors),parent_artifact_hashes=parent_count,production_changed=False,new_fits=0,new_backwards=0,GT_readout_construction=False,scope='Independent geometry, scalar metrics, parameter swaps and means/CIs. Functional cross states are not trained causal trajectories.'))
 print('AUDIT',dict(counts),dict(errors),flush=True)
if __name__=='__main__':run()
