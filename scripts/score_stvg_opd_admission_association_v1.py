"""Post-seal CPU task/evidence readback; Oracle is a separate GT-assisted diagnosis.

Imports do not read annotations or result files. prepare_oracle() requires the
entire label-free seal. assemble()/score() additionally require the Oracle seal.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,copy,gzip,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_admission_association_common_v1 import *

ALL_ARMS=ARMS+['Oracle']
CONTRASTS={
 'system':{'R_Path':1,'H_Top1':-1},
 'assoc_R':{'R_Path':1,'R_Top1':-1},
 'interaction':{'R_Path':1,'R_Top1':-1,'H_Path':-1,'H_Top1':1},
 'pair':{'R_Path':1,'R_Add':-1},
 'alignment':{'R_Path':1,'R_Shuffle':-1},
}
FIELDS=['I1_dense_v_after','I1_dense_v_delta','I0_dense_v_after','I0_dense_v_delta',
 'dense_s_delta','own_U_delta','observed_delta','training_GT_delta','correct_U_delta',
 'wrong_U_delta','correct_U_retained','wrong_U_corrected','sampled_event_delta',
 'interpolated_event_delta','expert_GT_IoU','restored_expert_GT_IoU','expert_delta']


def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
def mean(x):return math.fsum(float(v) for v in x)/len(x) if len(x) else None
def rows_write(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists(),path
 with path.open('xb') as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
   for r in rows:f.write((json.dumps(r,separators=(',',':'),allow_nan=False)+'\n').encode())
def rows_read(path):
 with gzip.open(path,'rt') as f:return [json.loads(s) for s in f]


def require_seal(phase='formal'):
 assert phase in ['formal','oracle'];verify()
 name='GLOBAL_PREDICTION_BARRIER.json' if phase=='formal' else 'ORACLE_PREDICTION_BARRIER.json'
 b=read(BASE/name);assert b['status']=='sealed' and b['GT_read']==(phase=='oracle')
 assert b['logical_complete_fits']==(384 if phase=='formal' else 64)
 assert b['logical_complete_rounds']==(9600 if phase=='formal' else 1600)
 assert b['runtime_sha256']==sha(BASE/'RUNTIME.json') and b['actual_new_DINO_calls']==0
 assert b['files']
 for rel,h in b['files'].items():
  p=ROOT/rel;assert p.is_relative_to(BASE) and p.suffix in ['.pt','.json'] and sha(p)==h,rel
 for p in [ROOT/rel for rel in b['files'] if rel.endswith('.json')]:
  rc=read(p);assert rc['GT_assisted']==(phase=='oracle') and rc['runtime_sha256']==b['runtime_sha256']
  assert rc['time']<=b['time'] and b['files'][rc['payload_path']]==rc['payload_sha256']
 if phase=='oracle':
  formal=require_seal('formal');od=read(BASE/'ORACLE_DESIGN.json')
  assert formal['time']<od['time']<=b['time'] and od['GT_assisted'] is True
 return b


def estimate(values):
 v=np.array([v for v in values if v is not None],float);n=len(v)
 if not n:return dict(defined_parents=0,mean=None,ci95=None)
 assert np.isfinite(v).all()
 idx=np.random.default_rng(SEED).integers(n,size=(BOOT,n));ci=np.quantile(v[idx].mean(1),[.025,.975])
 return dict(defined_parents=n,mean=mean(v),ci95=ci.tolist())


def outcome(values,est=estimate):
 v=[x for x in values if x is not None]
 return dict(estimate=est(values),unknown=len(values)-len(v),positive=sum(x>EPS for x in v),
  negative=sum(x< -EPS for x in v),neutral=sum(abs(x)<=EPS for x in v),
  harm5pp=sum(x<-.05 for x in v),harm20pp=sum(x<-.2 for x in v),
  gross_gain=mean([max(x,0) for x in v]),gross_loss=mean([max(-x,0) for x in v]))


def parent_values(rows,fields):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['parent_id']].append(r)
 out=[]
 for p in sorted(groups):
  q=groups[p];out.append(dict(parent_id=p,**{f:mean([r[f] for r in q if r[f] is not None]) for f in fields}))
 return out


def summary(rr,est=estimate):
 s=dict(parents=64,formal_logical_fits=384,formal_logical_rounds=9600,Oracle_logical_fits=64,
  Oracle_logical_rounds=1600,bootstrap=BOOT,seed=SEED,historically_exposed=True,
  primary='I1_dense_v_after',secondary='I0_dense_v_after',formal_arms=ARMS,
  formal_contrasts=CONTRASTS,unvisited_excludes_all_actual_DINO_visits=True,datasets={})
 for ds in ['hc2','vidstg']:
  raw=[r for r in rr if r['dataset']==ds];assert len(raw)==32
  fields=[a+'_'+f for a in ALL_ARMS for f in FIELDS];rows=parent_values(raw,fields);assert len(rows)==32
  stats={}
  for field in FIELDS:
   arms={a:outcome([r[a+'_'+field] for r in rows],est) for a in ARMS};contrasts={}
   for name,coef in CONTRASTS.items():
    vals=[math.fsum(c*r[a+'_'+field] for a,c in coef.items()) if all(r[a+'_'+field] is not None for a in coef) else None for r in rows]
    contrasts[name]=outcome(vals,est)
   stats[field]=dict(arms=arms,contrasts=contrasts)
  stats['Oracle_diagnostic']={field:dict(arm=outcome([r['Oracle_'+field] for r in rows],est),
    contrasts={a:outcome([r['Oracle_'+field]-r[a+'_'+field] if r['Oracle_'+field] is not None and r[a+'_'+field] is not None else None for r in rows],est) for a in ['R_Top1','R_Path']}) for field in FIELDS}
  stats['compute']={a:dict(logical_fits=32,logical_fit_seconds_mean=mean([r[a+'_fit_seconds'] for r in raw]),
    logical_backwards=sum(r[a+'_backwards'] for r in raw),logical_empty_fits=sum(r[a+'_empty'] for r in raw),
    unique_canonical_payloads=len({r[a+'_canonical_payload_sha256'] for r in raw}),canonical_reused=sum(r[a+'_canonical_reused'] for r in raw)) for a in ALL_ARMS}
  # Physical cost is counted once per canonical payload, including cross-arm reuse.
  unique={}
  for r in raw:
   for a in ARMS:unique.setdefault(r[a+'_canonical_payload_sha256'],(r[a+'_fit_seconds'],r[a+'_backwards']))
  formal_keys=set(unique);oracle={}
  for r in raw:
   h=r['Oracle_canonical_payload_sha256']
   if h not in formal_keys:oracle.setdefault(h,(r['Oracle_fit_seconds'],r['Oracle_backwards']))
  stats['unique_physical_cost']=dict(formal_payloads=len(unique),formal_fit_seconds=math.fsum(v[0] for v in unique.values()),
    formal_backwards=sum(v[1] for v in unique.values()),additional_Oracle_payloads=len(oracle),
    additional_Oracle_fit_seconds=math.fsum(v[0] for v in oracle.values()),additional_Oracle_backwards=sum(v[1] for v in oracle.values()))
  stats['selection']={a:{k:sum(r[a+'_'+k] for r in raw) for k in ['selected_frames','restored_frames','changed_Top1_frames',
   'known_expert_frames','unknown_expert_frames','outside_event_expert_frames','missing_annotation_expert_frames',
   'correct_Top1_replaced_worse','wrong_Top1_replaced_better']} for a in ALL_ARMS}
  stats['pool_coverage']={k:sum(r[k] for r in raw) for k in ['actual_visited_positions','empty_candidate_positions',
   'original_ambiguous_positions','original_other_rejected_positions','H_degenerate_edges','R_degenerate_edges','H_singleton_edges','R_singleton_edges']}
  s['datasets'][ds]=stats
 return s


def corners(boxes,width=1,height=1,clip=False):
 b=np.asarray(array(boxes),float);xy=np.concatenate([b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2],-1)*np.array([width,height,width,height])
 return np.maximum(xy,0) if clip else xy
def overlap(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float);inter=np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0).prod(-1)
 union=np.maximum(a[...,2:]-a[...,:2],0).prod(-1)+np.maximum(b[...,2:]-b[...,:2],0).prod(-1)-inter
 return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
def score_boxes(boxes,raw,truth,span,interval,ds,independent=False):
 from vg_tta.tastvg_oracle_event5_v1 import official
 from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
 return dense_metric(array(boxes),raw,truth,span,interval,ds)[0] if independent else official(array(boxes),raw,truth,span,interval,ds)


def frame_labels(z,r,raw,truth,span,independent=False,before=None,after=None):
 f=z['fit'];before=f['before'] if before is None else before;after=f['final'] if after is None else after
 ids=z['frame_ids'];b=corners(before,raw['input']['width'],raw['input']['height'],r['dataset']=='hc2');a=corners(after,raw['input']['width'],raw['input']['height'],r['dataset']=='hc2')
 if independent:
  from vg_tta.tastvg_oracle_event5_v1 import box_iou
  iou=box_iou
 else:iou=overlap
 visited=set(r['visited']);trained=set(f['positions']);frames=[]
 for j,fid in enumerate(ids):
  if not (span[0]<=fid<span[1] and fid in truth):continue
  bv,av=float(iou(b[j],truth[fid])),float(iou(a[j],truth[fid]))
  frames.append(dict(parent_id=r['index'],dataset=r['dataset'],arm=z['arm'],GT_assisted=z['logical_GT_assisted'],
   sample_ordinal=j,before=bv,after=av,delta=av-bv,visited=j in visited,trained=j in trained,
   initially_correct=bv>=.5,retained=av>=.5,unvisited=j not in visited))
 out={}
 masks={'observed':lambda x:x['visited'],'training_GT':lambda x:x['trained'],'own_U':lambda x:x['unvisited'],
  'correct_U':lambda x:x['unvisited'] and x['initially_correct'],'wrong_U':lambda x:x['unvisited'] and not x['initially_correct'],
  'sampled_event':lambda x:True}
 for name,pred in masks.items():
  ff=[x for x in frames if pred(x)];out[name+'_frames']=len(ff)
  for fld in ['before','after','delta']:out[name+'_'+fld]=mean([x[fld] for x in ff])
  if name=='correct_U':out[name+'_retained']=mean([x['retained'] for x in ff])
  if name=='wrong_U':out[name+'_corrected']=mean([x['retained'] for x in ff])
 # Decoder grid and non-grid interpolation are evaluated separately; neither is imputed into U.
 non_grid=[fid for fid in sorted(truth) if span[0]<=fid<span[1] and fid not in set(ids)]
 def interpolated(x):
  if not non_grid:return []
  p=np.stack([np.interp(non_grid,ids,x[:,k]) for k in range(4)],-1);v=np.asarray(iou(p,np.array([truth[f] for f in non_grid])),float)
  v[(np.asarray(non_grid)<ids[0])|(np.asarray(non_grid)>ids[-1])]=0;return v.tolist()
 ib,ia=interpolated(b),interpolated(a);out.update(interpolated_event_frames=len(non_grid),
  interpolated_event_before=mean(ib),interpolated_event_after=mean(ia),interpolated_event_delta=mean([y-x for x,y in zip(ib,ia)]))
 return out,frames


def evidence_rows(z,r,selection,raw,truth,span):
 pools={p['position']:p for p in selection['pools']};fixed=z['selection'];ee=[]
 for row in fixed['rows']:
  p=pools[row['position']];fid=row['frame_id'];k=row['candidate_index'];top=int(np.argmax(p['scores']))
  assert fid==z['frame_ids'][row['position']] and row['box']==p['boxes'][k] and row['score']==p['scores'][k]
  known=span[0]<=fid<span[1] and fid in truth;outside=not span[0]<=fid<span[1]
  qualities=overlap(corners(p['boxes'],raw['input']['width'],raw['input']['height'],r['dataset']=='hc2'),truth[fid]) if known else None
  qi=float(qualities[k]) if known else None;qt=float(qualities[top]) if known else None
  ee.append(dict(parent_id=r['index'],dataset=r['dataset'],arm=z['arm'],GT_assisted=z['logical_GT_assisted'],
   sample_ordinal=row['position'],candidate_index=k,Top1_index=top,candidate_count=len(p['boxes']),
   score=row['score'],original_margin=float(max(p['scores'])-sorted(p['scores'],reverse=True)[1]) if len(p['scores'])>1 else None,
   old_admitted=row['old_admitted'],restored_margin=not row['old_admitted'],reason=p['reason'],context_active=row['context_active'],
   changed_Top1=k!=top,GT_known=known,outside_event=outside,missing_annotation=not outside and fid not in truth,
   expert_GT_IoU=qi,Top1_GT_IoU=qt,quality_delta_vs_Top1=qi-qt if known else None,
   correct_Top1_replaced_worse=bool(known and k!=top and qt>=.5 and qi<qt-EPS),
   wrong_Top1_replaced_better=bool(known and k!=top and qt<.5 and qi>qt+EPS),
   correct_Top1_destroyed_at_05=bool(known and qt>=.5 and qi<.5),
   wrong_Top1_corrected_at_05=bool(known and qt<.5 and qi>=.5)))
 return ee


def load_targets(records):
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old()
 from scripts.score_stvg_opd_p1_v1 import truths
 return {ds:truths(ds,dict(parents=[r['query_ordinal'] for r in records if r['dataset']==ds])) for ds in ['hc2','vidstg']}


def prepare_oracle():
 """GT selects only existing R-pool candidates after every formal fit is sealed."""
 import torch
 torch.set_num_threads(2);bar=require_seal();assert not torch.cuda.is_initialized()
 records=read(BASE/'DESIGN_LOCK.json')['records'];assert len(records)==64
 tick=time.time();write(BASE/'ORACLE_GT_EXPOSURE.json',dict(status='formal_global_sealed_before_Oracle_GT_selection',
  formal_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),formal_barrier_time=bar['time'],time=tick,
  GT_for_formal_selection=False,Oracle_GT_assisted=True,existing_pool_only=True,CPU_only=True))
 targets=load_targets(records);from scripts.run_decota_paper_main_v1 import read_row
 out=[];provenance={}
 for r in records:
  tt,ss,binding=targets[r['dataset']];provenance.update(binding);truth=tt[r['query_ordinal']];span=ss[r['query_ordinal']]
  raw=read_row(r['dataset'],r['query_ordinal']);selected=read(ROOT/r['selection_path']);fixed=copy.deepcopy(selected['arms']['R_Top1']);fixed['GT_assisted']=True
  pools={p['position']:p for p in selected['pools']};known=unknown=0;choices=[]
  for row in fixed['rows']:
   p=pools[row['position']];fid=row['frame_id'];use=span[0]<=fid<span[1] and fid in truth
   if use:
    q=overlap(corners(p['boxes'],raw['input']['width'],raw['input']['height'],r['dataset']=='hc2'),truth[fid]);k=int(np.argmax(q));known+=1
    row.update(candidate_index=k,candidate_id=p['candidate_ids'][k],box=p['boxes'][k],score=p['scores'][k])
   else:unknown+=1
   choices.append(dict(position=row['position'],GT_selection_used=use,
    retained_R_Top1_without_label=not use,outside_event=not span[0]<=fid<span[1]))
  out.append(dict(index=r['index'],dataset=r['dataset'],selection=fixed,known_selection_frames=known,
   retained_unscorable_frames=unknown,GT_selection_provenance=choices,GT_provenance=binding,
   original_selection_sha256=r['selection_sha256']))
 for n,h in provenance.items():assert sha(ROOT/n)==h
 require_seal();write(BASE/'ORACLE_DESIGN.json',dict(status='GT_assisted_existing_candidate_pool_only',records=out,
  GT_assisted=True,GT_provenance=provenance,formal_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
  formal_predictions_immutable=True,no_GT_boxes_added=True,unknown_keeps_R_Top1=True,logical_fits=64,
  logical_rounds=1600,not_student_performance_upper_bound=True,time=time.time()))
 assert not torch.cuda.is_initialized()


def assemble(independent=False):
 import torch
 torch.set_num_threads(2);formal=require_seal();oracle=require_seal('oracle');assert not torch.cuda.is_initialized()
 records=read(BASE/'DESIGN_LOCK.json')['records'];targets=load_targets(records)
 from scripts.run_decota_paper_main_v1 import read_row
 rr=[];ff=[];rd=[];ee=[];provenance={}
 for r in records:
  tt,sp,binding=targets[r['dataset']];provenance.update(binding);truth=tt[r['query_ordinal']];span=sp[r['query_ordinal']]
  raw=read_row(r['dataset'],r['query_ordinal']);selection=read(ROOT/r['selection_path']);row=dict(parent_id=r['index'],dataset=r['dataset'],historically_exposed=True)
  assert sorted(r['visited'])==sorted(p['position'] for p in selection['pools'])
  row.update(actual_visited_positions=len(r['visited']),empty_candidate_positions=sum(not p['boxes'] for p in selection['pools']),
   original_ambiguous_positions=sum(p['reason']=='ambiguous_distinct_instances' for p in selection['pools']),
   original_other_rejected_positions=sum(not p['accepted'] and p['reason']!='ambiguous_distinct_instances' for p in selection['pools']))
  for support in ['H','R']:
   d=selection['decomposition'][support];row[support+'_degenerate_edges']=d['degenerate_edges'];row[support+'_singleton_edges']=d['singleton_edges']
   row[support+'_interaction_Frobenius_norm']=float(np.sqrt(sum(np.square(np.asarray(c,float)).sum() for c in d['centered'])))
  fit_before=None
  for arm in ALL_ARMS:
   phase='oracle' if arm=='Oracle' else 'formal';z=resolve(r['index'],arm,phase);f=z['fit'];assert z['logical_GT_assisted']==(arm=='Oracle')
   assert z['Native_interval']==r['intervals']['I0'] and f['positions']==[p['position'] for p in z['selection']['rows']]
   if fit_before is None:fit_before=array(f['before']).copy()
   else:assert np.array_equal(fit_before,array(f['before']))
   v,frames=frame_labels(z,r,raw,truth,span,independent);ff+=frames
   for k,x in v.items():row[arm+'_'+k]=x
   for interval in ['I0','I1']:
    for name,b in [('before',f['before']),('after',f['final'])]:
     m=score_boxes(b,raw,truth,span,r['intervals'][interval],r['dataset'],independent)
     for k,x in m.items():row[arm+'_'+interval+'_dense_'+k+'_'+name]=x
     for th in [.3,.5]:row[arm+'_'+interval+'_R'+str(int(th*100))+'_'+name]=float(m['v']>th)
    for k in ['s','v','t']:row[arm+'_'+interval+'_dense_'+k+'_delta']=row[arm+'_'+interval+'_dense_'+k+'_after']-row[arm+'_'+interval+'_dense_'+k+'_before']
    assert row[arm+'_'+interval+'_dense_t_delta']==0
   row[arm+'_dense_s_delta']=row[arm+'_I1_dense_s_delta'];assert row[arm+'_I0_dense_s_delta']==row[arm+'_I1_dense_s_delta']
   evidence=evidence_rows(z,r,selection,raw,truth,span);ee+=evidence
   for name,pred in [('expert',lambda e:True),('restored_expert',lambda e:e['restored_margin'])]:
    row[arm+'_'+name+'_GT_IoU']=mean([e['expert_GT_IoU'] for e in evidence if pred(e) and e['GT_known']])
   row[arm+'_selected_frames']=len(evidence);row[arm+'_restored_frames']=sum(e['restored_margin'] for e in evidence)
   row[arm+'_changed_Top1_frames']=sum(e['changed_Top1'] for e in evidence)
   for k in ['known_expert_frames','unknown_expert_frames','outside_event_expert_frames','missing_annotation_expert_frames']:
    pred={'known_expert_frames':lambda e:e['GT_known'],'unknown_expert_frames':lambda e:not e['GT_known'],
     'outside_event_expert_frames':lambda e:e['outside_event'],'missing_annotation_expert_frames':lambda e:e['missing_annotation']}[k]
    row[arm+'_'+k]=sum(pred(e) for e in evidence)
   for k in ['correct_Top1_replaced_worse','wrong_Top1_replaced_better']:row[arm+'_'+k]=sum(e[k] for e in evidence)
   pos=f['positions'];target=np.asarray([p['box'] for p in z['selection']['rows']],float).reshape(-1,4)
   row[arm+'_expert_delta']=mean((overlap(corners(array(f['final'])[pos]),corners(target))-overlap(corners(array(f['before'])[pos]),corners(target))).tolist()) if pos else None
   row[arm+'_fit_seconds']=z['compute']['fit_seconds'];row[arm+'_backwards']=f['gradient_calls'];row[arm+'_empty']=f['empty']
   row[arm+'_canonical_reused']=z['canonical_reused'];row[arm+'_canonical_payload_sha256']=z['canonical_payload_sha256']
   row[arm+'_canonical_GT_used']=z['canonical_GT_used'];row[arm+'_logical_GT_assisted']=z['logical_GT_assisted']
   geometry={g['round']:g for g in f['geometry']['rounds']};visited=set(r['visited']);U=[j for j in range(len(z['frame_ids'])) if j not in visited]
   emap={e['sample_ordinal']:e for e in evidence}
   for k,h in enumerate(f['path'][:-1]):
    nxt=f['path'][k+1];bv=frame_labels(z,r,raw,truth,span,independent,before=h['boxes'],after=nxt['boxes'])[0]
    move=array(nxt['boxes']).astype(float)-array(h['boxes']).astype(float)
    q=dict(parent_id=r['index'],dataset=r['dataset'],arm=arm,GT_assisted=z['logical_GT_assisted'],round=k,
     updated=f['rounds'][k]['updated'],empty_evidence=f['empty'],training_frames=len(pos),
     observed_GT_delta=bv['training_GT_delta'],common_U_GT_delta=bv['own_U_delta'],
     actual_U_output_movement=float(np.linalg.norm(move[U])),actual_training_output_movement=float(np.linalg.norm(move[pos])),
     actual_parameter_displacement=float(np.sqrt(sum(np.square(array(nxt['state'][n]).astype(float)-array(v).astype(float)).sum() for n,v in h['state'].items()))),
     selected_observation_ordinals=list(pos),changed_Top1_observations=[emap[p]['changed_Top1'] for p in pos])
    if pos:
     fb=geometry[k]['feedback']
     for name in ['max_reward','reward_variance','Top1_same_samples_weight_L1','mean_gradient_norm','per_frame_informative']:
      q[name]=array(fb[name]).tolist()
    else:
     for name in ['max_reward','reward_variance','Top1_same_samples_weight_L1','mean_gradient_norm','per_frame_informative']:q[name]=[]
    rd.append(q)
  for m in ['own_U_frames','correct_U_frames','wrong_U_frames','sampled_event_frames','interpolated_event_frames']:
   assert len({row[a+'_'+m] for a in ALL_ARMS})==1
  rr.append(row);status(BASE/('ROOT_SCORING_STATUS.json' if independent else 'CPU_STATUS.json'),dict(status='independent_postseal_readback' if independent else 'postseal_task_scoring',done=len(rr),total=64,CPU_only=True,GT_read=True,time=time.time()))
 assert len(rr)==64 and len(rd)==11200 and not torch.cuda.is_initialized();verify()
 return rr,ff,rd,ee,provenance


def score():
 formal=require_seal();oracle=require_seal('oracle');write(BASE/'GT_EXPOSURE.json',dict(status='formal_and_Oracle_globally_sealed_before_task_scoring',
  formal_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),Oracle_barrier_sha256=sha(BASE/'ORACLE_PREDICTION_BARRIER.json'),
  formal_GT_assisted=False,Oracle_GT_assisted=True,GT_for_formal_selection=False,CPU_only=True,time=time.time()))
 rr,ff,rd,ee,pr=assemble();OUT.mkdir(parents=True,exist_ok=True)
 for name,rows in [('ROWS',rr),('FRAME_ROWS',ff),('ROUND_ROWS',rd),('EVIDENCE_ROWS',ee)]:rows_write(OUT/(name+'.jsonl.gz'),rows)
 write(OUT/'SUMMARY.json',summary(rr))
 write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_independent_root_PDF_view_publication_archive',parents=64,
  formal_logical_fits=384,Oracle_logical_fits=64,formal_logical_rounds=9600,Oracle_logical_rounds=1600,
  actual_unique_formal_fits=formal['actual_complete_GPU_fits'],actual_new_Oracle_fits=oracle['actual_complete_GPU_fits'],
  round_rows=len(rd),frame_rows=len(ff),evidence_rows=len(ee),GT_provenance=pr,CPU_only=True,GT_read=True,time=time.time()))


if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='prepare_oracle':prepare_oracle()
 else:score()
