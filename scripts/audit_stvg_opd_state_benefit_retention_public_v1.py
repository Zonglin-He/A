"""Exhaustive anonymous frame -> parent/round -> bootstrap audit, no raw labels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,csv,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_benefit_retention_common_v1 import *
from scripts.audit_stvg_opd_state_benefit_retention_audit_v1 import near
from scripts.score_stvg_opd_state_benefit_retention_audit_v1 import summarise

def run(root):
 p=Path(root)/'results/stvg_opd_state_benefit_retention_audit/2026-10-13'
 pp=rows(p/'PARENT_ROWS.jsonl.gz');cc=rows(p/'ROUND_ROWS.jsonl.gz');ss=read(p/'SUMMARY.json')
 assert len(pp)==64 and len(cc)==1664 and len({r['parent_id'] for r in pp})==64
 lookup={(r['parent_id'],r['step']):r for r in cc};parents={r['parent_id']:r for r in pp}
 checks=near(summarise(pp,cc,True),ss);frame_count=0;current=None;buffer=[]
 def process(parent,items):
  count=0;r=parents[parent];K=r['total_steps'];F=r['genuine_output_frames'];assert len(items)==(K+1)*F
  matrix={(x['step'],x['frame_ordinal']):x for x in items};assert len(matrix)==len(items)
  masks={m:[j for j in range(F) if matrix[0,j][m]] for m in MASKS};masks['all_frames']=list(range(F))
  for j in range(F):
   for k in range(K+1):
    x=matrix[k,j];assert x['GT_event_available']==matrix[0,j]['GT_event_available'] and all(x[m]==matrix[0,j][m] for m in MASKS)
    assert x['own_U'] or not x['common_U'];assert not(x['own_U'] and x['observed'])
    assert x['own_U'] or x['observed'] or not x['GT_event_available']
    for arm in ARMS:
     assert (x[arm+'_Q'] is not None)==x['GT_event_available']
     if x[arm+'_Q'] is not None:assert 0<=x[arm+'_Q']<=1+1e-12
    count+=8
  for k in range(K+1):
   curve=lookup[parent,k]
   for ai,arm in enumerate(ARMS):
    for m in MASKS:
     val=mean([matrix[k,j][arm+'_Q'] for j in masks[m]]);count+=near(val,curve['arms'][arm][m+'_Q'])
     count+=near(difference(val,mean([matrix[0,j][arm+'_Q'] for j in masks[m]])),curve['arms'][arm][m+'_gain'])
     if k==0:count+=near(val,r['arms'][arm][m+'_before'])
     if k==K:count+=near(val,r['arms'][arm][m+'_after'])
    if r['arms'][arm]['empty_evidence']:
     assert curve['arms'][arm]['teacher_agreement'] is None and not curve['arms'][arm]['actual_update_outgoing']
   for a,b in PAIRS:
    key=a+'_to_'+b
    for m,mask in masks.items():
     rms=mean([matrix[k,j][key+'_RMS'] for j in mask]);count+=near(rms,curve['contrasts'][key][m+'_RMS'])
     agreement=mean([matrix[k,j][key+'_box_agreement'] for j in mask]);count+=near(agreement,curve['contrasts'][key][m+'_box_agreement'])
     if m!='all_frames':
      advantage=difference(mean([matrix[k,j][b+'_Q'] for j in mask]),mean([matrix[k,j][a+'_Q'] for j in mask]))
      count+=near(advantage,curve['contrasts'][key][m+'_advantage'])
      if k==0:count+=near(advantage,r['contrasts'][key][m+'_initial_advantage'])
      if k==K:count+=near(advantage,r['contrasts'][key][m+'_final_advantage'])
  for a,b in PAIRS:
   key=a+'_to_'+b;v=r['contrasts'][key]
   for m,mask in masks.items():
    count+=near(mean([matrix[K,j][key+'_RMS']-matrix[0,j][key+'_RMS'] for j in mask]),v[m+'_RMS_delta'])
    if m=='all_frames':continue
    before=np.array([matrix[0,j][b+'_Q'] for j in mask]);first=np.array([matrix[0,j][a+'_Q'] for j in mask]);after=np.array([matrix[K,j][b+'_Q'] for j in mask])
    advantages=before-first;drops=after-before
    late=30 if r['dataset']=='hc2' else 8
    persistent=np.array([all(matrix[k,j][b+'_Q']<matrix[0,j][b+'_Q']-EPS for k in range(late,K+1)) for j in mask],bool)
    strata=dict(initial_positive=advantages>EPS,initial_1pp=advantages>=.01,initial_5pp=advantages>=.05,initial_correct=before>=.5,correct_advantaged=(before>=.5)&(advantages>=.01))
    for label,keep in strata.items():
     prefix=m+'_'+label;count+=near(int(keep.sum()),v[prefix+'_frames'])
     count+=near(int((keep & persistent).sum()),v[prefix+'_persistent_loss_frames'])
     count+=near(int((keep & (drops< -EPS)).sum()),v[prefix+'_absolute_loss_frames'])
     count+=near(mean(drops[keep]),v[prefix+'_mean_second_gain'])
     count+=near(mean(persistent[keep].astype(float)),v[prefix+'_persistent_loss_fraction'])
    count+=near(difference(v[m+'_second_gain'],v[m+'_first_gain']),v[m+'_erosion'])
  return count
 with gzip.open(p/'FRAME_ROUND_ROWS.jsonl.gz','rt') as f:
  for line in f:
   x=json.loads(line);assert not ({'frame_ids','query_ordinal','source_id','caption','GT_boxes','native_boxes','gradient','weights'} & set(x))
   if current is not None and x['parent_id']!=current:checks+=process(current,buffer);buffer=[]
   current=x['parent_id'];buffer.append(x);frame_count+=1
 if buffer:checks+=process(current,buffer)
 with (p/'ALL_PARENT_CONTRASTS.csv').open() as f:table=list(csv.DictReader(f))
 assert len(table)==192
 expected=[dict(parent_id=r['parent_id'],dataset=r['dataset'],contrast=key,**value) for r in pp for key,value in r['contrasts'].items()]
 for csvrow,r in zip(table,expected):
  assert set(csvrow)==set(r)
  for k,v in r.items():assert csvrow[k]==('' if v is None else str(v));checks+=1
 return dict(status='pass',scope='exhaustive anonymous every real-frame boundary/all192 parent contrasts/round aggregation/initial-correct strata/persistent loss/CSV/all paired parent bootstrap; no raw GT/media/vector payload read',
  parents=64,fits=192,arm_boundaries=4992,frame_round_rows=frame_count,parent_round_rows=1664,scalar_checks=checks,
  CPU_only=True,GT_read=False)
if __name__=='__main__':print(json.dumps(run(sys.argv[1]),allow_nan=False))
