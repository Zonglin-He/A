"""Whitelist full official metadata, fix selections/orders and reuse only subject parses."""
import sys,hashlib,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_best_full_common_v1 import *
h=lambda s:hashlib.sha256(s.encode()).hexdigest()
def run():
 assert read(ROOT/'artifacts/tastvg_extended_sensitivity_v3/FINAL_COMPLETION.json')['status']=='completed_verified_and_published'
 if (BASE/'DESIGN_LOCK.json').exists():return
 # Annotation-bearing official HC intake, whitelist before building predictor plan.
 ann=ROOT/'data/hcstvg2_official_metadata/val_v2.json';meta={k:{f:v[f] for f in ['img_size','img_num','English']} for k,v in read(ann).items()}
 inventory=read(Path('/tmp/hc_full_inventory.json'));assert set(meta)==set(inventory)
 from scripts.tastvg_paper48_p5_common_v1 import frame_ids
 prior=read(ROOT/'artifacts/decota_final_freeze_v1/LOCK.json')['rows']['vidstg_test']
 raw={};raw['vidstg']=[dict(ordinal=i,key=r['key'],source=r['input']['source'],input=r['input'],frame_ids=r['input']['frame_ids'],query_type=r['query_type']) for i,r in enumerate(prior)]
 assert len(raw['vidstg'])==10303 and not any(r['input_unavailable'] for r in prior)
 raw['hc2']=[]
 for i,k in enumerate(sorted(meta)):
  v=meta[k];height,width=v['img_size'][:2];count=v['img_num'];path=Path(inventory[k]);assert path.is_file();source=Path(k).stem.split('_',1)[1];ids=frame_ids(count)
  q=dict(index=i,source=source,original_video_id=Path(k).stem,kind='hcstvg',caption=v['English'].lower(),width=width,height=height,frame_count=count,fps=count/20.,duration=20.,video_path=str(path),video_sha256=sha(path),frame_ids=ids)
  raw['hc2'].append(dict(ordinal=i,key='full_hc2:'+Path(k).stem,source=source,input=q,frame_ids=ids,query_type='declarative',annotation_key=k))
 metadata={}
 for dataset,rows in raw.items():
  out=BASE/dataset;sel=ROOT/'artifacts/tastvg_extended_sensitivity_v3'/dataset/'SELECTION.json';cfg=read(sel)['params'];overlap=set()
  for version in ['tastvg_optuna_v1','tastvg_coordinate_tuning_v2','tastvg_extended_sensitivity_v3']:
   overlap.update(r['source'] for r in read(ROOT/'artifacts'/version/dataset/'PLAN.json')['rows'])
  for r in rows:r['tuning_source_exposed']=r['source'] in overlap
  orders={f'order{j}':sorted(range(len(rows)),key=lambda i:(h(f'BestFull-v1-source|{j}|'+rows[i]['source']),rows[i]['source'],h('BestFull-v1-query|'+rows[i]['key']),rows[i]['key'])) for j in [1,2]}
  needed=sorted({i for seq in orders.values() for i in seq[::4]})
  plan=dict(dataset=dataset,rows=rows,orders=orders,conditions=CONDS,params=cfg,total=len(rows)*12,sources=len({r['source'] for r in rows}),queries=len(rows),availability=25,selected_config_sha256=sha(sel),selected_config_path=str(sel.relative_to(ROOT)),historically_exposed=True,full_official_cohort=True,tuning_sources=sorted(overlap),GT_used_for_selection=False)
  write(out/'PLAN.json',plan);write(out/'EXPERT_PLAN.json',dict(rows=rows,conditions=CONDS,expert_needed=needed,conditions_by_parent={str(i):CONDS for i in needed},total=len(needed)*6))
  write(out/'COHORT.json',{k:v for k,v in plan.items() if k not in ['rows','orders','tuning_sources'] }|dict(orders=2,expert_pairs=len(needed)*6,tuning_overlap_sources=len({r['source'] for r in rows if r['tuning_source_exposed']}),outside_tuning_sources=len({r['source'] for r in rows if not r['tuning_source_exposed']}),annotation_metadata_whitelist=True))
  for f in ['PLAN.json','EXPERT_PLAN.json','COHORT.json']:metadata[f'{dataset}/{f}']=sha(out/f)
 write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_new_predictions',metadata=metadata,protocol_sha256=sha(ROOT/'protocols/tastvg_best_full_v1.md'),predecessor_completion_sha256=sha(ROOT/'artifacts/tastvg_extended_sensitivity_v3/FINAL_COMPLETION.json'),hc_annotation_metadata_sha256=sha(ann),predictions=0,GT_geometry_used=False,time=time.time()))
 status(BASE/'STATUS.json',dict(status='design_locked_implementation_in_progress',time=time.time()))
 print({d:{k:read(BASE/d/'COHORT.json')[k] for k in ['queries','sources','total','expert_pairs','outside_tuning_sources']} for d in DATASETS},flush=True)
if __name__=='__main__':run()
