"""Exact anonymous publication readback plus independent parent-summary arithmetic.

This is not a new scientific audit. It uses only the four already root-verified
anonymous tables, summary/report files and two completed audit receipts. It never
opens annotations, media, complete fits, model parameters or sampling witnesses.
No source/output file is written by public(); the publisher owns its receipt.
"""
import csv,gzip,hashlib,itertools,json,math,sys,time
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_opd_admission_association_v1'
OUT=ROOT/'results/stvg_opd_admission_association/2026-10-13'
ARMS=['H_Top1','H_Path','R_Top1','R_Path','R_Add','R_Shuffle']
ALL_ARMS=ARMS+['Oracle'];BOOT=10000;SEED=20261013;EPS=1e-12
CONTRASTS={'system':{'R_Path':1,'H_Top1':-1},'assoc_R':{'R_Path':1,'R_Top1':-1},
 'interaction':{'R_Path':1,'R_Top1':-1,'H_Path':-1,'H_Top1':1},
 'pair':{'R_Path':1,'R_Add':-1},'alignment':{'R_Path':1,'R_Shuffle':-1}}
FIELDS=['I1_dense_v_after','I1_dense_v_delta','I0_dense_v_after','I0_dense_v_delta',
 'dense_s_delta','own_U_delta','observed_delta','training_GT_delta','correct_U_delta',
 'wrong_U_delta','correct_U_retained','wrong_U_corrected','sampled_event_delta',
 'interpolated_event_delta','expert_GT_IoU','restored_expert_GT_IoU','expert_delta']
TABLES=['ROWS.jsonl.gz','FRAME_ROWS.jsonl.gz','ROUND_ROWS.jsonl.gz','EVIDENCE_ROWS.jsonl.gz']
REPORT_ATTACHMENTS=['FEEDBACK_SUMMARY.json','SOURCE_CONCENTRATION.json','CASES.json',
 'READOUT_INTERPRETATION.json','ROWS.csv']
FRAME_KEYS={'parent_id','dataset','arm','GT_assisted','sample_ordinal','before','after','delta',
 'visited','trained','initially_correct','retained','unvisited'}
ROUND_KEYS={'parent_id','dataset','arm','GT_assisted','round','updated','empty_evidence','training_frames',
 'observed_GT_delta','common_U_GT_delta','actual_U_output_movement','actual_training_output_movement',
 'actual_parameter_displacement','selected_observation_ordinals','changed_Top1_observations',
 'max_reward','reward_variance','Top1_same_samples_weight_L1','mean_gradient_norm','per_frame_informative'}
EVIDENCE_KEYS={'parent_id','dataset','arm','GT_assisted','sample_ordinal','candidate_index','Top1_index',
 'candidate_count','score','original_margin','old_admitted','restored_margin','reason','context_active',
 'changed_Top1','GT_known','outside_event','missing_annotation','expert_GT_IoU','Top1_GT_IoU',
 'quality_delta_vs_Top1','correct_Top1_replaced_worse','wrong_Top1_replaced_better',
 'correct_Top1_destroyed_at_05','wrong_Top1_corrected_at_05'}
FORBIDDEN_KEYS={'box','boxes','bbox','bboxes','gt_box','gt_boxes','expert_box','expert_boxes',
 'student_boxes','prediction_boxes','coordinates','xyxy','cxcywh','query','query_text','caption','text',
 'prompt','sentence','frame_id','frame_ids','physical_frame_id','physical_frame_ids','physical_interval',
 'gt_span','samples','actions','rewards','weights','used_rewards','chosen_rewards','chosen_weights',
 'same_samples_top1_rewards','same_samples_top1_weights','gradient','gradients','state','initial',
 'final','adam_before','adam_after','exp_avg','exp_avg_sq','features','hidden','fit','path',
 'gt','truth','ground_truth','annotations','detections','tracks'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def average(values):return math.fsum(float(x) for x in values)/len(values) if values else None


def require_root_readback():
 """Fail before any actual OUT file is opened; there is no wait or retry loop."""
 p=BASE/'ROOT_READBACK.json';q=BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json'
 assert p.is_file() and q.is_file(),'Both completed actual root readbacks are required'
 a,b=read(p),read(q)
 assert a['status']==b['status']=='pass' and a['CPU_only'] is True and b['CPU_only'] is True
 assert a['GT_read'] is True and b['GT_read'] is True,'Qualification or no-GT metadata is not a completed task audit'
 assert b['parents']==64 and b['logical_fits']==448 and b['positive_negative_neutral_unknown_complete'] is True
 assert a['counts']['logical_fits']==384 and a['Oracle_counts']['logical_fits']==64
 assert a['independent_scalar_checks']>0 and b['scalar_checks']>0
 assert isinstance(a['time'],(int,float)) and isinstance(b['time'],(int,float))
 return a,b,{p.name:sha(p),q.name:sha(q)}


def privacy(value,where='root'):
 """Reject coordinates/text/physical IDs/full action vectors independent of hashes."""
 if isinstance(value,dict):
  for k,v in value.items():
   low=str(k).lower()
   assert low not in FORBIDDEN_KEYS,(where,'private field',k)
   assert 'frame_id' not in low and 'frameid' not in low,(where,'physical frame identifier',k)
   assert not any(low.endswith('_'+x) for x in ['gt_boxes','expert_boxes','query_text','physical_frame_ids','sample_vectors','gradient_vectors']), (where,k)
   privacy(v,where+'.'+str(k))
 elif isinstance(value,list):
  # Per-observation feedback has at most four scalars. Parent lists are mappings;
  # complete 32-action vectors and flattened state/gradient vectors are forbidden.
  if value and all(isinstance(v,(int,float,bool)) for v in value):assert len(value)<=4,(where,'full numeric vector')
  for i,v in enumerate(value):privacy(v,where+f'[{i}]')
 elif isinstance(value,float):assert math.isfinite(value),(where,'nonfinite scalar')
 elif isinstance(value,str):
  assert not any(s in value.lower() for s in ['data:video/','data:image/','/annotations/','/annos/','file://']), (where,'private data reference')
 elif value is not None:assert isinstance(value,(int,bool)),(where,type(value))


def identity(row):
 assert isinstance(row['parent_id'],int) and not isinstance(row['parent_id'],bool) and 0<=row['parent_id']<64
 assert row['dataset'] in ['hc2','vidstg']
 if 'arm' in row:
  assert row['arm'] in ALL_ARMS and row['GT_assisted']==(row['arm']=='Oracle')


def inspect_row(row,name):
 identity(row);privacy(row,name)
 if name=='ROWS.jsonl.gz':
  assert all(not isinstance(v,(dict,list)) for v in row.values()),'Parent outcomes must be anonymous scalar fields'
  assert row['historically_exposed'] is True
  for k,v in row.items():
   if isinstance(v,str):assert k=='dataset' or k.endswith('_canonical_payload_sha256'),(k,'unexpected free text')
  for a in ALL_ARMS:
   for f in FIELDS:assert a+'_'+f in row
   h=row[a+'_canonical_payload_sha256'];assert len(h)==64 and all(c in '0123456789abcdef' for c in h)
   assert row[a+'_logical_GT_assisted']==(a=='Oracle')
 elif name=='FRAME_ROWS.jsonl.gz':
  assert set(row)==FRAME_KEYS
  assert isinstance(row['sample_ordinal'],int) and row['sample_ordinal']>=0
  for k in ['before','after']:assert isinstance(row[k],(int,float)) and 0<=row[k]<=1
  assert isinstance(row['delta'],(int,float)) and -1<=row['delta']<=1
 elif name=='ROUND_ROWS.jsonl.gz':
  assert set(row)==ROUND_KEYS and 0<=row['round']<(40 if row['dataset']=='hc2' else 10)
  n=row['training_frames'];assert isinstance(n,int) and 0<=n<=4
  for k in ['selected_observation_ordinals','changed_Top1_observations','max_reward','reward_variance',
            'Top1_same_samples_weight_L1','mean_gradient_norm','per_frame_informative']:
   assert isinstance(row[k],list) and len(row[k])==n,(k,n)
  assert all(isinstance(x,int) and not isinstance(x,bool) and x>=0 for x in row['selected_observation_ordinals'])
  for k in ['changed_Top1_observations','per_frame_informative']:assert all(isinstance(x,bool) for x in row[k])
 elif name=='EVIDENCE_ROWS.jsonl.gz':
  assert set(row)==EVIDENCE_KEYS and 1<=row['candidate_count']<=3
  assert 0<=row['candidate_index']<row['candidate_count'] and 0<=row['Top1_index']<row['candidate_count']
  assert row['GT_known']==(not row['outside_event'] and not row['missing_annotation'])
  assert (row['expert_GT_IoU'] is not None)==row['GT_known']


@lru_cache(maxsize=33)
def bootstrap_counts(n):
 draws=np.random.default_rng(SEED).integers(n,size=(BOOT,n))
 # Count-based execution is independent of the scorer's advanced-index mean.
 return np.array([(draws==j).sum(1) for j in range(n)]).T


def independent_estimate(values):
 v=np.asarray([x for x in values if x is not None],float);n=len(v)
 if not n:return dict(defined_parents=0,mean=None,ci95=None)
 assert np.isfinite(v).all()
 boots=(bootstrap_counts(n)*v[None,:]).sum(1)/n
 return dict(defined_parents=n,mean=math.fsum(v)/n,ci95=np.quantile(boots,[.025,.975]).tolist())


def independent_outcome(values):
 defined=[v for v in values if v is not None]
 return dict(estimate=independent_estimate(values),unknown=len(values)-len(defined),
  positive=sum(v>EPS for v in defined),negative=sum(v< -EPS for v in defined),neutral=sum(abs(v)<=EPS for v in defined),
  harm5pp=sum(v<-.05 for v in defined),harm20pp=sum(v<-.2 for v in defined),
  gross_gain=average([max(v,0) for v in defined]),gross_loss=average([max(-v,0) for v in defined]))


def independent_summary(rr):
 result=dict(parents=64,formal_logical_fits=384,formal_logical_rounds=9600,Oracle_logical_fits=64,
  Oracle_logical_rounds=1600,bootstrap=BOOT,seed=SEED,historically_exposed=True,
  primary='I1_dense_v_after',secondary='I0_dense_v_after',formal_arms=ARMS,formal_contrasts=CONTRASTS,
  unvisited_excludes_all_actual_DINO_visits=True,datasets={})
 for ds in ['hc2','vidstg']:
  raw=[r for r in rr if r['dataset']==ds];assert len(raw)==32
  groups=defaultdict(list)
  for r in raw:groups[r['parent_id']].append(r)
  assert len(groups)==32
  fields=[a+'_'+f for a in ALL_ARMS for f in FIELDS]
  parents=[{k:average([r[k] for r in groups[p] if r[k] is not None]) for k in fields} for p in sorted(groups)]
  stats={}
  for f in FIELDS:
   arms={a:independent_outcome([p[a+'_'+f] for p in parents]) for a in ARMS};contrasts={}
   for name,coeff in CONTRASTS.items():
    delta=[math.fsum(c*p[a+'_'+f] for a,c in coeff.items()) if all(p[a+'_'+f] is not None for a in coeff) else None for p in parents]
    contrasts[name]=independent_outcome(delta)
   stats[f]=dict(arms=arms,contrasts=contrasts)
  stats['Oracle_diagnostic']={f:dict(arm=independent_outcome([p['Oracle_'+f] for p in parents]),
   contrasts={a:independent_outcome([p['Oracle_'+f]-p[a+'_'+f] if p['Oracle_'+f] is not None and p[a+'_'+f] is not None else None for p in parents]) for a in ['R_Top1','R_Path']}) for f in FIELDS}
  stats['compute']={a:dict(logical_fits=32,logical_fit_seconds_mean=average([r[a+'_fit_seconds'] for r in raw]),
   logical_backwards=sum(r[a+'_backwards'] for r in raw),logical_empty_fits=sum(r[a+'_empty'] for r in raw),
   unique_canonical_payloads=len({r[a+'_canonical_payload_sha256'] for r in raw}),canonical_reused=sum(r[a+'_canonical_reused'] for r in raw)) for a in ALL_ARMS}
  unique={}
  for r in raw:
   for a in ARMS:unique.setdefault(r[a+'_canonical_payload_sha256'],(r[a+'_fit_seconds'],r[a+'_backwards']))
  oracle={}
  for r in raw:
   h=r['Oracle_canonical_payload_sha256']
   if h not in unique:oracle.setdefault(h,(r['Oracle_fit_seconds'],r['Oracle_backwards']))
  stats['unique_physical_cost']=dict(formal_payloads=len(unique),formal_fit_seconds=math.fsum(v[0] for v in unique.values()),
   formal_backwards=sum(v[1] for v in unique.values()),additional_Oracle_payloads=len(oracle),
   additional_Oracle_fit_seconds=math.fsum(v[0] for v in oracle.values()),additional_Oracle_backwards=sum(v[1] for v in oracle.values()))
  stats['selection']={a:{k:sum(r[a+'_'+k] for r in raw) for k in ['selected_frames','restored_frames','changed_Top1_frames',
   'known_expert_frames','unknown_expert_frames','outside_event_expert_frames','missing_annotation_expert_frames',
   'correct_Top1_replaced_worse','wrong_Top1_replaced_better']} for a in ALL_ARMS}
  stats['pool_coverage']={k:sum(r[k] for r in raw) for k in ['actual_visited_positions','empty_candidate_positions',
   'original_ambiguous_positions','original_other_rejected_positions','H_degenerate_edges','R_degenerate_edges','H_singleton_edges','R_singleton_edges']}
  result['datasets'][ds]=stats
 return result


def near(a,b):
 if isinstance(a,dict):
  assert isinstance(b,dict) and set(a)==set(b),'Summary schema mismatch'
  return sum(near(a[k],b[k]) for k in a)
 if isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b)
  return sum(near(x,y) for x,y in zip(a,b))
 if isinstance(a,(int,float)) and not isinstance(a,bool):
  assert isinstance(b,(int,float)) and not isinstance(b,bool) and math.isfinite(a) and math.isfinite(b)
  assert abs(a-b)<=1e-10,(a,b);return 1
 assert a==b,(a,b);return 1


def public(folder):
 tick=time.monotonic();folder=Path(folder).resolve()
 root,anonymous,audit_hashes=require_root_readback()  # No OUT reads precede this gate.
 counters=Counter();files={};tables={};readback_time=min(root['time'],anonymous['time'])
 for name in TABLES:
  local=OUT/name;remote=folder/name
  assert local.is_file() and remote.is_file(),name
  assert local.stat().st_mtime<=readback_time,'Anonymous table changed after completed root readback'
  assert sha(local)==sha(remote),(name,'publication bytes differ')
  files[name]=dict(sha256=sha(remote),bytes=remote.stat().st_size);rows=[]
  with gzip.open(local,'rt') as a,gzip.open(remote,'rt') as b:
   for ordinal,(x,y) in enumerate(itertools.zip_longest(a,b),1):
    assert x is not None and y is not None and x==y,(name,ordinal,'line mismatch')
    lr,rr=json.loads(x),json.loads(y);assert lr==rr
    inspect_row(rr,name);rows.append(rr);counters[name+'_rows']+=1
    for k,v in rr.items():
     if v is None:counters[name+'_unknown_scalars']+=1
     elif isinstance(v,(int,float)) and not isinstance(v,bool):counters[name+'_numeric_scalars']+=1
     if (k.endswith('_delta') or k in ['delta','quality_delta_vs_Top1','observed_GT_delta','common_U_GT_delta']) and isinstance(v,(int,float)):
      counters[name+('_positive_metric_scalars' if v>EPS else '_negative_metric_scalars' if v< -EPS else '_neutral_metric_scalars')]+=1
  tables[name]=rows
 assert len(tables['ROWS.jsonl.gz'])==64
 for name,key in [('FRAME_ROWS.jsonl.gz','frame_rows'),('ROUND_ROWS.jsonl.gz','round_rows'),('EVIDENCE_ROWS.jsonl.gz','evidence_rows')]:
  assert len(tables[name])==anonymous[key],(name,'root-verified row count')
 assert len(tables['ROUND_ROWS.jsonl.gz'])==11200
 for parent in tables['ROWS.jsonl.gz']:
  q=[r for r in tables['ROUND_ROWS.jsonl.gz'] if r['parent_id']==parent['parent_id']]
  for arm in ALL_ARMS:
   ids=sorted(r['round'] for r in q if r['arm']==arm)
   assert ids==list(range(40 if parent['dataset']=='hc2' else 10)),(parent['parent_id'],arm)
 # Root-verified summary is byte-bound; its estimates are independently reconstructed.
 for name in ['SUMMARY.json']+REPORT_ATTACHMENTS:
  local,remote=OUT/name,folder/name
  assert local.exists()==remote.exists(),(name,'incomplete or unbound report attachment')
  if not local.exists():continue
  assert sha(local)==sha(remote),(name,'report publication bytes differ')
  files[name]=dict(sha256=sha(remote),bytes=remote.stat().st_size);counters['exact_report_attachments']+=1
  if name.endswith('.json'):privacy(read(remote),name)
  else:
   with remote.open(newline='') as f:
    csvrows=list(csv.DictReader(f));assert len(csvrows)==64
    for row,original in zip(csvrows,tables['ROWS.jsonl.gz']):
     assert set(row)==set(original)
     for k,v in original.items():
      if v is None:assert row[k]==''
      elif isinstance(v,bool):assert row[k]==str(v)
      elif isinstance(v,(int,float)):assert float(row[k])==float(v)
      else:assert row[k]==v
      counters['CSV_fields']+=1
 assert (folder/'SUMMARY.json').is_file()
 counters['independent_summary_scalars']=near(independent_summary(tables['ROWS.jsonl.gz']),read(folder/'SUMMARY.json'))
 # Reject unbound files that expose private assets or vector witnesses under new names.
 for p in folder.rglob('*'):
  if not p.is_file():continue
  assert p.suffix.lower() not in ['.pt','.pth','.npy','.npz','.mp4','.avi','.mov','.pkl','.pickle'],p.name
  rel=p.relative_to(folder)
  if str(rel) not in files and (p.suffix.lower() in ['.json','.gz','.csv']):
   local=OUT/rel
   assert local.is_file() and sha(local)==sha(p),(str(rel),'unbound additional public table')
   if p.suffix.lower()=='.json':privacy(read(p),str(rel))
   elif p.suffix.lower()=='.gz':raise AssertionError('Unexpected compressed table outside the four declared anonymous tables')
   else:raise AssertionError('Unexpected CSV outside the scalar parent table')
   counters['exact_additional_JSON_attachments']+=1
 for name,info in files.items():assert sha(OUT/name)==sha(folder/name)==info['sha256'],'Readback input changed during audit'
 for name,digest in audit_hashes.items():assert sha(BASE/name)==digest,'Completed root receipt changed during publication readback'
 return dict(status='pass',scope='Exact publication readback of all root-verified anonymous parent/frame/round/evidence rows and report attachments; independent parent-summary bootstrap arithmetic. Not a new scientific audit.',
  publication_readback_only=True,new_scientific_audit=False,root_readback_receipts=audit_hashes,
  checks=dict(counters),files=files,bootstrap=BOOT,seed=SEED,all_positive_negative_neutral_unknown_rows_preserved=True,
  no_annotations_media_complete_fits_or_model_read=True,new_GT_read=False,new_model_calls=0,new_GPU_calls=0,
  CPU_only=True,CPU_wall_seconds=time.monotonic()-tick,time=time.time())


if __name__=='__main__':
 assert len(sys.argv)==2,'usage: audit_stvg_opd_admission_association_public_v1.py PUBLIC_FOLDER'
 print(json.dumps(public(Path(sys.argv[1])),indent=2,allow_nan=False))
