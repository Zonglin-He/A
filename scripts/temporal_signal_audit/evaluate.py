"""Independent CPU label evaluation, only after complete label-free signal seal.

Association audit only: no new interval, predictor, fitted readout or parameter
selection. GT never enters reconstruction, ROI placement or signal generation.
"""
from pathlib import Path
import os,sys,json,math,csv
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.temporal_signal_audit.extract import OUT,PUBLIC,read,write,sha

EVENT=['confidence','eref_support','aux_support','attention_placement','roi_enrichment','roi_mass','permuted_roi_mass','visual_mass','visual_entropy','temporal_key_salience','temporal_self_diagonal','temporal_anchor_mass','area','anchor_proximity','time_position','native_event_mass']
EVENT += [x+'_residual'for x in ['confidence','eref_support','aux_support','attention_placement','roi_enrichment','visual_mass','temporal_key_salience']]
EVENT += [x+f'_L{i}'for x in ['attention_placement','roi_enrichment','temporal_key_salience']for i in range(1,7)]
BOUNDARY=['link_break','box_speed','appearance_change','hidden_change','attention_placement_change','confidence','aux_support','attention_placement','roi_enrichment','roi_mass','permuted_roi_mass','visual_mass','visual_entropy','temporal_key_salience','temporal_self_diagonal','temporal_anchor_mass','native_boundary_mass','native_start_mass','native_end_mass']
BOUNDARY += [x+f'_L{i}'for x in ['attention_placement','roi_enrichment','temporal_key_salience']for i in range(1,7)]

def auc(y,x):
 y=np.asarray(y,bool);x=np.asarray(x,float);p=x[y];n=x[~y]
 if len(p)<2 or len(n)<2:return None
 return float(((p[:,None]>n).sum()+.5*(p[:,None]==n).sum())/(len(p)*len(n)))
def ci(x,seed=20261022):
 x=np.asarray(x,float)
 if not len(x):return dict(queries=0,mean=None,CI95=None)
 rng=np.random.default_rng(seed);b=x[rng.integers(0,len(x),(10000,len(x)))].mean(-1)
 return dict(queries=len(x),mean=float(x.mean()),CI95=np.quantile(b,[.025,.975]).tolist(),positive_queries=int((x>0).sum()),negative_queries=int((x<0).sum()),zero_queries=int((x==0).sum()))
def aligned(y,x):
 a=auc(y,x)
 if a is None:return None
 shifts=np.unique(np.linspace(1,len(x)-1,min(31,len(x)-1)).astype(int))
 null=[auc(y,np.roll(x,s))for s in shifts]
 return dict(auc=a,chance_delta=a-.5,circular_mean=float(np.mean(null)),alignment_delta=a-float(np.mean(null)))
def mean_summary(rows,field):return ci([r[field]for r in rows if r.get(field)is not None])
def p_signflip(v):
 v=np.asarray(v,float)
 if not len(v):return None
 rng=np.random.default_rng(20261022);b=(v*rng.choice([-1,1],(10000,len(v)))).mean(-1)
 return float((1+(abs(b)>=abs(v.mean())-1e-15).sum())/10001)
def bh(ps):
 out={};keys=sorted(ps,key=lambda k:ps[k]);last=1.
 for j in range(len(keys)-1,-1,-1):
  k=keys[j];last=min(last,ps[k]*len(keys)/(j+1));out[k]=last
 return out

def evaluate():
 p=read(OUT/'plan.json');seal=read(OUT/'sealed.json');assert seal['GT_read']is False and seal['plan_sha256']==sha(OUT/'plan.json');assert len(seal['files'])==128
 for r in seal['files']:assert sha(OUT/r['path'])==r['sha256']
 for k,v in p['sources'].items():assert sha(ROOT/k)==v
 labels={}
 for cohort,v in p['labels'].items():
  assert sha(ROOT/v['path'])==v['sha256'];labels[cohort]={r['id']:r for r in [json.loads(l)for l in (ROOT/v['path']).read_text().splitlines()if l.strip()]}
 rows=[];counts={};profiles={}
 for item,saved in zip(p['items'],seal['files']):
  assert item['anonymous']==saved['anonymous'];label=labels[item['cohort']][item['id']];s,e=map(float,label['interval']);fps=item['fps'];z=np.load(OUT/saved['path'],allow_pickle=False);t=z['frame_ids']/fps;s/=fps;e/=fps;inside=(t>=s)&(t<e);unobs=~z['observed'];cohort=item['cohort']
  counts.setdefault(cohort,[]).append(dict(nodes=len(t),unobserved=int(unobs.sum()),inside=int(inside.sum()),inside_unobserved=int((inside&unobs).sum()),start_inside=bool(t[0]<=s<=t[-1]),end_inside=bool(t[0]<=e<=t[-1]+1/fps)))
  for signal in EVENT:
   for support,mask in [('unobserved',unobs),('all',np.ones(len(t),bool))]:
    # Null shifts are on the full curve, then fixed evaluation mask is applied.
    value=auc(inside[mask],z[signal][mask])
    if value is None:continue
    shifts=np.unique(np.linspace(1,len(t)-1,min(31,len(t)-1)).astype(int));null=np.mean([auc(inside[mask],np.roll(z[signal],d)[mask])for d in shifts])
    rows.append(dict(cohort=cohort,query=item['anonymous'],analysis='event_'+support,signal=signal,auc=value,chance_delta=value-.5,circular_mean=float(null),alignment_delta=value-float(null)))
  for where,b in [('start',s),('end',e),('either',None)]:
   covered=(t[0]<=s<=t[-1])if where=='start'else(t[0]<=e<=t[-1]+1/fps)if where=='end'else(t[0]<=s<=t[-1]and t[0]<=e<=t[-1]+1/fps)
   if not covered:continue
   d=np.minimum(abs(t-s),abs(t-e))if b is None else abs(t-b);eligible=(d<=1)|(d>2);near=d<=1
   for signal in BOUNDARY:
    for support,mask0 in [('unobserved',unobs),('all',np.ones(len(t),bool))]:
     mask=mask0&eligible;value=auc(near[mask],z[signal][mask])
     if value is None:continue
     shifts=np.unique(np.linspace(1,len(t)-1,min(31,len(t)-1)).astype(int));null=np.mean([auc(near[mask],np.roll(z[signal],dlt)[mask])for dlt in shifts])
     row=dict(cohort=cohort,query=item['anonymous'],analysis=f'boundary_{where}_{support}',signal=signal,auc=value,chance_delta=value-.5,circular_mean=float(null),alignment_delta=value-float(null));rows.append(row)
   # Is the attention level higher specifically near GT endpoints than event interior?
   interior=inside&(abs(t-s)>2)&(abs(t-e)>2);edge=near&inside
   for signal in ['attention_placement','roi_enrichment','roi_mass','permuted_roi_mass','visual_mass','temporal_key_salience','confidence']:
    if edge.sum()>=2 and interior.sum()>=2:rows.append(dict(cohort=cohort,query=item['anonymous'],analysis='boundary_vs_event_interior_'+where,signal=signal,level_delta=float(z[signal][edge].mean()-z[signal][interior].mean())))
  # Boundary-aligned coarse profiles; query means, no GT coordinate export.
  for endpoint,bound in [('start',s),('end',e)]:
   if not(t[0]<=bound<=t[-1]+1/fps):continue
   for signal in ['roi_enrichment','attention_placement','temporal_key_salience','confidence','link_break','native_boundary_mass']:
    for center in [-3,-2,-1,0,1,2,3]:
     mask=(t-bound>=center-.5)&(t-bound<center+.5)
     if mask.any():profiles.setdefault((cohort,endpoint,signal,center),[]).append(float(z[signal][mask].mean()))
 summary={}
 for cohort in counts:
  group={}
  for analysis in sorted({r['analysis']for r in rows if r['cohort']==cohort}):
   group[analysis]={}
   for signal in sorted({r['signal']for r in rows if r['cohort']==cohort and r['analysis']==analysis}):
    rr=[r for r in rows if r['cohort']==cohort and r['analysis']==analysis and r['signal']==signal]
    group[analysis][signal]={field:mean_summary(rr,field)for field in (['level_delta']if analysis.startswith('boundary_vs')else['auc','chance_delta','circular_mean','alignment_delta'])}
  summary[cohort]=group
 # Predeclared eight primary questions. BH correct only these, not secondary exploration.
 tests={}
 for cohort in counts:
  ps={};stat={}
  for a,ss in [('event_unobserved',p['primary_event']),('boundary_either_unobserved',p['primary_boundary'])]:
   for signal in ss:
    v=[r['alignment_delta']for r in rows if r['cohort']==cohort and r['analysis']==a and r['signal']==signal]
    pp=p_signflip(v)
    if pp is not None:ps[a+':'+signal]=pp;stat[a+':'+signal]=dict(signflip_p_two_sided=pp,alignment_delta=ci(v))
  adjusted=bh(ps)
  for k in stat:stat[k]['BH_q_eight_primary']=adjusted[k]
  tests[cohort]=stat
 public=dict(status='completed_offline_association_audit',cohorts={k:dict(queries=len(v),nodes=sum(x['nodes']for x in v),unobserved_nodes=sum(x['unobserved']for x in v),captured_start_boundaries=sum(x['start_inside']for x in v),captured_end_boundaries=sum(x['end_inside']for x in v))for k,v in counts.items()},summary=summary,primary_tests=tests,
 profiles=[dict(cohort=k[0],endpoint=k[1],signal=k[2],relative_seconds=k[3],**ci(v))for k,v in sorted(profiles.items())],
 proof=dict(plan_sha256=sha(OUT/'plan.json'),signal_seal_sha256=sha(OUT/'sealed.json'),all_128_label_free_signals_before_labels=True,source_sha256=p['sources'],labels_sha256={k:v['sha256']for k,v in p['labels'].items()},
 maximum_CPU_hidden_absolute_error=max(r['max_hidden_error']for r in seal['files']),hidden_layer_checks=sum(r['hidden_checks']for r in seal['files']),CUDA_initialized=False,CPU_time_decoder_calls=256,new_RGB=0,new_encoder=0,new_expert=0,parameter_updates=0,formal_plan_unchanged=sha(ROOT/'outputs/boxfixed_full_v1/plan.json')==p['formal_plan_sha256'],seconds_extraction=seal['seconds']),
 limitations=['All cohorts are historically exposed development. HC32/Vid32 are parent-isolated from HC64, not pristine confirmatory test.', 'Query-level bootstrap; within-query frames are not independent observations. Minimum two near and two far nodes; eligible count changes by metric.', 'Spatial evidence and observation timing inherit model decisions. Residualization is a nuisance control, not a causal adjustment or predictor.', 'Native temporal attention uses target-conditioned fused visual/text memory; correlations do not establish the incremental effect of the added spatial branch.', 'Soft ROI uses full-timeline auxiliary BoxFixed trajectory with no GT event support. Fixed cyclic spatial displacement preserves area; it is not a verified wrong identity.', 'Higher target attention or consistency does not establish event semantics or target identity. Event annotation has no complete outside-event identity ground truth.', 'No method/readout fitted, no threshold selected, no new temporal prediction or vIoU claim. Secondary layers and signs are exploratory; no best layer selected.'])
 assert public['proof']['formal_plan_unchanged'];PUBLIC.mkdir(parents=True,exist_ok=True);assert not(PUBLIC/'report.json').exists();write(PUBLIC/'report.json',public)
 fields=['cohort','query','analysis','signal','auc','chance_delta','circular_mean','alignment_delta','level_delta']
 with open(PUBLIC/'signed_queries.tsv','w')as f:
  writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
 write(OUT/'evaluation_receipt.json',dict(status='scored',report_sha256=sha(PUBLIC/'report.json'),signed_sha256=sha(PUBLIC/'signed_queries.tsv'),plan_sha256=sha(OUT/'plan.json'),seal_sha256=sha(OUT/'sealed.json')))
 print(json.dumps(dict(status=public['status'],cohorts=public['cohorts'],proof=public['proof']),indent=2))
if __name__=='__main__':evaluate()
