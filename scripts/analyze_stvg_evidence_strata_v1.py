"""CPU cache-only full online decomposition, fixed temporal and scale gate."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,gzip,hashlib,collections,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
D=ROOT/'artifacts/stvg_evidence_strata_v1';P=ROOT/'results/stvg_evidence_strata/2026-10-18/cached_full';OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2';PAPER=ROOT/'artifacts/decota_paper_experiments_v1'
def read(p):return json.loads(Path(p).read_text())
def put(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def mean(v):
 a=[float(x) for x in v if x is not None and np.isfinite(x)];return float(np.mean(a)) if a else None
def bucket(x,e):return 'unknown' if x is None else 'Q'+str(int(np.searchsorted(e,x,side='right'))+1)
FEATURES=['event_fraction','event_start','annotation_coverage','input_annotation_fraction','target_area','target_x','target_y','speed_diagonals_sec','speed_sizes_sec','exploration_proxy']
CATS=['query_form','target_category','relation_proxy','parent_query_count','arrival_exposure']
METRICS=['Frozen_v','Before_v','After_v','Frozen_t','After_t','Frozen_s','After_s','delta_total_v','delta_current_v','delta_inherited_v','unobserved_iou_delta','observed_iou_delta','reward_variance','ESS','central_expert_reward_round_mean_delta']
def summarize(rr,bootstrap=False):
 groups=collections.defaultdict(list)
 for r in rr:groups[r['parent_alias']].append(r)
 pm={k:[mean([r.get(k) for r in g]) for g in groups.values()] for k in METRICS}
 z=dict(queries=len({r['query_alias'] for r in rr}),arrivals=len(rr),parents=len(groups),query_macro={k:mean([r.get(k) for r in rr]) for k in METRICS},parent_macro={k:mean(v) for k,v in pm.items()})
 for mode in ['query_macro','parent_macro']:
  a=z[mode];assert abs(a['delta_total_v']-a['delta_current_v']-a['delta_inherited_v'])<1e-10
 z['current_harm_gt5pp_arrivals']=sum(r['delta_current_v']<-.05 for r in rr);z['total_harm_gt5pp_arrivals']=sum(r['delta_total_v']<-.05 for r in rr)
 z['current_gain_arrivals']=sum(r['delta_current_v']>1e-10 for r in rr);z['current_loss_arrivals']=sum(r['delta_current_v']< -1e-10 for r in rr)
 if bootstrap and groups:
  rng=np.random.default_rng(20261018);items=list(groups.values());idx=rng.integers(len(items),size=(1000,len(items)))
  # Cluster resampling preserves query multiplicity for official query macro.
  z['cluster_CI95']={}
  for k in ['delta_current_v','delta_inherited_v','delta_total_v']:
   sums=np.array([sum(r[k] for r in g)for g in items]);cnt=np.array([len(g)for g in items]);a=np.array(pm[k]);z['cluster_CI95'][k]=dict(query=np.quantile(sums[idx].sum(1)/cnt[idx].sum(1),[.025,.975]).tolist(),parent=np.quantile(a[idx].mean(1),[.025,.975]).tolist())
 return z

def features():
 from scripts.score_stvg_opd_p1_v1 import truths
 lock=read(OLD/'DESIGN_LOCK.json');features={};provenance={};bindings={};types={};t0=time.time()
 for ds in ['hc2','vidstg']:
  plan=read(PAPER/ds/'PLAN.json');rows=plan['rows'];N=len(rows);dense,spans,prov=truths(ds,dict(parents=list(range(N))));provenance.update(prov)
  parents=sorted({r['source']for r in rows});aliases={s:('H'if ds=='hc2'else'V')+f'P{i+1:04}'for i,s in enumerate(parents)};counts=collections.Counter(r['source']for r in rows)
  dev={rows[q]['source']for q in lock['datasets'][ds]['development_query_ordinals']};assert len(dev)==32
  barrier=read(OLD/'stages'/('P1_hc2'if ds=='hc2'else'P1_vidstg')/'PREDICTION_BARRIER.json');im=barrier['inputs'];feat=[];qt=collections.Counter()
  for q,r in enumerate(rows):
   m=r['input'];fps=float(m['fps']);assert fps>0;fids=np.array(sorted(dense[q]));b=np.array([dense[q][int(f)]for f in fids],float);wh=b[:,2:]-b[:,:2];c=(b[:,:2]+b[:,2:])/2
   span=spans[q];start=span[0];end=span[1]+(1 if ds=='hc2'else 0);length=end-start;assert length>0
   # Model input may be a clip of a longer parent video. Use actual input grid extent.
   lo,hi=int(r['frame_ids'][0]),int(r['frame_ids'][-1])+1;duration=hi-lo;assert duration>0
   inside=(fids>=start)&(fids<end);covered=int(inside.sum());dt=np.diff(fids)/fps;dist=np.linalg.norm(np.diff(c,axis=0),axis=1);size=np.sqrt(np.maximum(wh[:,0]*wh[:,1],1e-12));pairsize=(size[:-1]+size[1:])/2
   path=OLD/'inputs'/('vidstg_to_hc2'if ds=='hc2'else'hcstvg2_to_vidstg')/'clean'/f'{q:05}.pt';rel=str(path.relative_to(OLD));assert sha(path)==im[rel];inp=load(path);assert inp['parent']==q
   box=inp['native_boxes'].double().numpy();positions=inp['expert']['positions4'];positions=[int(p)for p in positions if 0<=int(p)<len(box)];central=box[positions] if positions else box
   sigma=lock['datasets'][ds]['config']['sigma'];proxy=np.sqrt(np.mean((sigma*central[:,:2]*(1-central[:,:2])/np.maximum(central[:,2:],1e-12))**2))
   subj=read(PAPER/ds/'subjects'/f'{q:05}.json');assert subj['ordinal']==q and subj['caption_sha256']==hashlib.sha256(m['caption'].encode()).hexdigest()
   entity=subj['parses']['context'].get('entity','')or subj['parses']['old'].get('entity','');entity=entity.lower().strip();tokens=re.findall(r"[a-z]+",m['caption'].lower())
   category='person'if any(t in entity.split()for t in ['person','man','woman','boy','girl','people','child','guy','lady','male','female'])else('animal'if any(t in entity.split()for t in ['dog','cat','horse','bird','animal'])else('other_object'if entity else'unknown'))
   sequence=any(t in tokens for t in ['then','before','after','while','until']);relational=any(t in tokens for t in ['behind','beside','between','towards','toward','holding','holds','pulling','pushing','giving','chasing','following','hugging'])
   form='interrogative'if '?'in m['caption']or r['query_type'] not in ['caption','declarative']else'declarative';qt[r['query_type']]+=1
   f=dict(dataset=ds,query_ordinal=q,query_alias=('H'if ds=='hc2'else'V')+f'Q{q+1:05}',parent_alias=aliases[r['source']],primary=r['source']not in dev,
    event_fraction=float(length/duration),event_start=float((start-lo)/duration),annotation_coverage=covered/length,input_annotation_fraction=len(fids)/duration,
    target_area=float(np.mean(wh[:,0]*wh[:,1])/(m['width']*m['height'])),target_x=float(np.mean(c[:,0])/m['width']),target_y=float(np.mean(c[:,1])/m['height']),
    speed_diagonals_sec=mean(dist/dt/np.hypot(m['width'],m['height'])) if len(fids)>1 else None,speed_sizes_sec=mean(dist/dt/pairsize)if len(fids)>1 else None,
    exploration_proxy=float(proxy),query_form=form,target_category=category,relation_proxy='sequence_and_relation'if sequence and relational else'sequence'if sequence else'relation'if relational else'no_lexical_marker',parent_query_count='1'if counts[r['source']]==1 else'2-4'if counts[r['source']]<=4 else'5-16'if counts[r['source']]<=16 else'17+',GT_box_count=len(fids),event_physical_frames=length)
   feat.append(f)
   if q%1000==0:print('FEATURES',ds,q,N,flush=True)
  features[ds]=feat;bindings[ds]=dict(plan_sha256=sha(PAPER/ds/'PLAN.json'),input_barrier_sha256=sha(OLD/'stages'/('P1_hc2'if ds=='hc2'else'P1_vidstg')/'PREDICTION_BARRIER.json'),all_Source_inputs_hash_verified=N,full_raw_fits_available=0);types[ds]=dict(qt);del dense
 pooled=[f for rr in features.values()for f in rr];bins={k:np.quantile([f[k]for f in pooled if f[k]is not None],[.25,.5,.75]).tolist()for k in FEATURES}
 for f in pooled:f['bins']={k:bucket(f[k],bins[k])for k in FEATURES}
 put(D/'FEATURES_PRIVATE.json',dict(rows=features,bins=bins,provenance=provenance,bindings=bindings,query_type_counts=types,GT_read=True,seconds=time.time()-t0));print('FEATURES complete',flush=True)

def full():
 private=read(D/'FEATURES_PRIVATE.json');features=private['rows'];allrows={};outputs={};public=[]
 for ds in ['hc2','vidstg']:
  fby={f['query_ordinal']:f for f in features[ds]};seen={};rr=[];src=ROOT/'results/stvg_opd_p1_complete/2026-10-09/anonymous_rows'/ds/'spatial_opd/ROWS.jsonl.gz'
  with gzip.open(src,'rt')as stream:
   for line in stream:
    r=json.loads(line);f=fby[r['query_ordinal']];order=r['order'];parent=f['parent_alias'];key=(order,parent);first=key not in seen;seen[key]=1
    x={k:r.get(k)for k in METRICS};x.update(query_alias=f['query_alias'],parent_alias=parent,query_ordinal=r['query_ordinal'],order=order,primary=f['primary'],bins=f['bins'],arrival_exposure='first'if first else'repeated',**{k:f[k]for k in CATS if k!='arrival_exposure'})
    assert abs(x['delta_current_v']-(r['After_v']-r['Before_v']))<1e-10;assert abs(x['delta_inherited_v']-(r['Before_v']-r['Frozen_v']))<1e-10;assert r['Before_t']==r['After_t']==r['Frozen_t'];rr.append(x)
  allrows[ds]=rr;out={}
  for scope,rs in [('all',rr),('excluding_tuning_parents',[r for r in rr if r['primary']])]:
   strata={}
   for k in FEATURES+CATS:
    vs=sorted({r['bins'][k]if k in FEATURES else r[k]for r in rs});strata[k]={v:summarize([r for r in rs if(r['bins'][k]if k in FEATURES else r[k])==v])for v in vs}
   out[scope]=dict(overall=summarize(rs,True),orders={o:summarize([r for r in rs if r['order']==o])for o in sorted({r['order']for r in rs})},strata=strata)
  outputs[ds]=out
  # Public per-query means preserve anonymity and coarse bins only. Orders retained separately.
  for q,fs in fby.items():
   cells=[r for r in rr if r['query_ordinal']==q];public.append(dict(dataset=ds,query_alias=fs['query_alias'],parent_alias=fs['parent_alias'],primary=fs['primary'],bins=fs['bins'],**{k:fs[k]for k in CATS if k!='arrival_exposure'},metrics={k:mean([r[k]for r in cells])for k in METRICS},per_order=[dict(order=r['order'],arrival_exposure=r['arrival_exposure'],delta_total_v=r['delta_total_v'],delta_current_v=r['delta_current_v'],delta_inherited_v=r['delta_inherited_v'])for r in cells]))
  print('FULL',ds,len(rr),flush=True)
 dist={ds:{k:dict(count=len([f for f in fs if f[k]is not None]),mean=mean([f[k]for f in fs]),quantiles=np.quantile([f[k]for f in fs if f[k]is not None],[0,.25,.5,.75,1]).tolist())for k in FEATURES}for ds,fs in features.items()}
 put(P/'FULL_STRATA.json',dict(status='complete',groups=outputs,common_bins=private['bins'],dataset_distributions=dist,query_type_counts=private['query_type_counts'],three_orders_not_independent_queries=True,old_metrics_not_rescored=True))
 put(P/'QUERY_BINS_AND_METRICS.json',dict(rows=public,raw_GT_features_excluded=True,three_order_query_means=True))
 put(D/'FULL_PRIVATE_ROWS.json',allrows)

if __name__=='__main__':globals()[sys.argv[1]]()
