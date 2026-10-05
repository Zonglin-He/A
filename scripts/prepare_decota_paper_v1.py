"""Lock all official inputs, one-query/video roster and finite paper stages."""
import os,sys,time,collections,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *

def prepare():
 if (BASE/'DESIGN_LOCK.json').exists():return
 parent=ROOT/'artifacts/decota_fixed_full_corruption_v1'
 assert read(parent/'STATUS.json')['status']=='paused_by_user_revised_paper_scope'
 assert not (parent/'GLOBAL_PREDICTION_BARRIER.json').exists()
 config=read(ROOT/'methods/DECOTA_FIXED_FULL_RESEARCH_CONFIG.json')
 assert config['LN_writeback']==1/16 and config['spatial_lr']==.03 and config['native_when']
 inputs={};datasets={};rosters={};orders={}
 for ds in DATASETS:
  old=read(parent/ds/'PLAN.json');rows=copy.deepcopy(old['rows']);n=len(rows)
  clusters=collections.defaultdict(list);videos=collections.defaultdict(list)
  for r in rows:
   clusters[r['source']].append(r['ordinal']);videos[r['input']['original_video_id']].append(r)
  third=[]
  sources=sorted(clusters,key=lambda s:digest(['PaperOrder3-20261005',ds,s]))
  for source in sources:
   third.extend(sorted(clusters[source],key=lambda i:digest(['PaperWithin3-20261005',ds,rows[i]['key']])))
  full={**old['orders'],'order3':third}
  for o,seq in full.items():assert len(seq)==n and sorted(seq)==list(range(n))
  chosen=[min(rr,key=lambda r:(digest(['PaperOneQueryVideo-20261005',ds,r['key'],r['input']['caption']]),r['ordinal']))['ordinal'] for _,rr in sorted(videos.items())]
  assert len(chosen)==len(videos)==(732 if ds=='vidstg' else 3482)
  selected=set(chosen);one=[i for i in full['order1'] if i in selected]
  assert len(one)==len(chosen)
  rosters[ds]=dict(parent_ordinals=chosen,order1=one,videos=len(videos),source_clusters=len(clusters),
   hash_salt='PaperOneQueryVideo-20261005',selection_before_predictions=True,GT_used=False,
   parent_queries=n,official_split='test' if ds=='vidstg' else 'validation',
   parent_plan_sha256=sha(parent/ds/'PLAN.json'))
  orders[ds]={o:digest(seq) for o,seq in full.items()}
  plan={**old,'orders':full,'conditions':['clean'],'arrivals_per_checkpoint':n*3,
   'official_videos':len(videos),'parent_source_clusters':len(clusters)}
  write(BASE/ds/'PLAN.json',plan);write(BASE/ds/'ONE_QUERY_VIDEO_ROSTER.json',rosters[ds])
  dest=BASE/ds/'subjects';dest.mkdir(parents=True,exist_ok=True)
  oldbar=read(parent/ds/'SUBJECT_BARRIER.json')
  for name,h in oldbar['files'].items():
   src=parent/ds/'subjects'/name;assert sha(src)==h
   os.link(src,dest/name)
  write(BASE/ds/'SUBJECT_BARRIER.json',oldbar)
  for f in [BASE/ds/'PLAN.json',BASE/ds/'ONE_QUERY_VIDEO_ROSTER.json',BASE/ds/'SUBJECT_BARRIER.json']:
   inputs[str(f.relative_to(ROOT))]=sha(f)
  datasets[ds]={k:plan[k] for k in ['queries','sources','outside_development_sources','historically_exposed','official_videos','parent_source_clusters','arrivals_per_checkpoint']}
 stages=[dict(id='table1',setting='clean_cross_domain',all_queries=True,orders=3,ours_arrivals=41355,
   methods=['Source Only','TENT-STVG','EATA-STVG','SAR-STVG','DINO-Refine','Ours','Target-trained reference'],
   status='ours_runtime_implementation',baseline_qualification='pending_port_source_Fisher_and_smoke'),
  dict(id='table2',setting='same_domain',one_query_per_video=True,orders=1,conditions=16,arrivals_per_online_method=67424,
   methods=['Source Only','TENT-STVG','EATA-STVG','SAR-STVG','DINO-Refine','Ours'],status='locked_cohort_pending_runtime'),
  dict(id='table3',variants=['Frozen','DINO-only','w/o query residual','w/o LN adaptation','w/o consolidation','Full'],status='locked_scope_pending_runtime'),
  dict(id='table4',variants=['Direct L1+GIoU','Multi-proposal energy','Admitted Top1 energy'],status='locked_scope_pending_runtime'),
  dict(id='table5',variants=['Native','Actionness adaptation','Real transform consensus','GT-head oracle'],UVTG='old_audited_appendix_only',status='locked_scope_pending_matched_runtime'),
  dict(id='budget_alpha_efficiency',observation_counts=[1,2,4,8],LN_writeback=[0,1/32,1/16,1/8],status='locked_scope_pending_runtime'),
  dict(id='sealed_analyses',groups=['expert_quality','duration','motion','query_type'],pipeline_GT_required=True,status='waiting_all_seals')]
 master=dict(version='decota_paper_experiments_v1',stages=stages,datasets=datasets,order_hashes=orders,
  canonical_method=config,severity_physical_burst_percent=[2.5,5,10],one_query_video_counts={'vidstg':732,'hc2':3482},
  source_cluster_counts={'vidstg':732,'hc2':237},no_total_deadline=True,no_target_metric_tuning=True,
  old_partial_scored=False,GT_inference=False,production_promoted=False,time=time.time())
 write(BASE/'MASTER_PLAN.json',master);inputs[str((BASE/'MASTER_PLAN.json').relative_to(ROOT))]=sha(BASE/'MASTER_PLAN.json')
 write(BASE/'DESIGN_LOCK.json',dict(version=master['version'],datasets=datasets,config=config,inputs=inputs,
  protected={'methods/CURRENT_METHOD.json':sha(ROOT/'methods/CURRENT_METHOD.json'),'methods/DECOTA_FIXED_FULL_RESEARCH_CONFIG.json':sha(ROOT/'methods/DECOTA_FIXED_FULL_RESEARCH_CONFIG.json')},
  online_arrivals=41355,unique_frozen_inputs=13785,jobs=[dict(job=j,target=ds,source=src) for j,ds,src in JOBS],
  no_total_deadline=True,GT_inference=False,hc_annotation_metadata_sha256=read(parent/'DESIGN_LOCK.json')['hc_annotation_metadata_sha256'],time=time.time()))
 status(BASE/'STATUS.json',dict(status='prepared_pending_runtime_lock_and_smoke',phase='table1',predictions=0,GT_read=False,time=time.time()))
 archive('完整名单、第三完整序、每视频hash query及全部正式实验职责已锁；复用13785 query-only parses，零本轮新预测；HC2单位已纠正为3482片段/237父来源')
 print('PAPER_PREPARED',datasets,'OURS_TABLE1',41355,flush=True)

if __name__=='__main__':prepare()
