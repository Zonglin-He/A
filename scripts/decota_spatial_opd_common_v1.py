import sys,os,json,time,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,save,load
BASE=ROOT/'artifacts/decota_spatial_opd_v1'
PUB=ROOT/'results/decota_spatial_opd/2026-10-06'
PAPER=ROOT/'artifacts/decota_paper_experiments_v1'
from scripts.decota_paper_common_v1 import digest,budget

def verify():
 lock=read(BASE/'RUNTIME_LOCK.json')
 pins=dict(lock['pins'])
 for revision in sorted((BASE/'revisions').glob('*.json')):
  record=read(revision)
  assert record['original_runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json')
  assert record['scientific_change'] is False
  pins.update(record['pin_overrides'])
 for f,h in pins.items():assert sha(ROOT/f)==h,f
 for f,h in lock['protected'].items():assert sha(ROOT/f)==h,f
 assert sha(BASE/'DESIGN_LOCK.json')==lock['design_sha256']
 return lock

def revision_receipt():
 return {str(f.relative_to(BASE)):sha(f) for f in sorted((BASE/'revisions').glob('*.json'))}

def archive(event):
 f=ROOT/'docs/RESEARCH_HISTORY.md';s=f.read_text()
 line='**2026-10-06｜独立显式空间策略OPD：'+event+'。** 附件5ab425de授权保存旧paper队列后执行新实验再接续；Native-WHEN/原Uniform4/单DINO admissionTop1/1792参数/Adam.03/每帧32 antithetic/.25探索/.25教师温度/固定10轮末轮/LN末轮delta1/16，生产CURRENT不改，旧Ours不改名为新结果。预锁各32历史曝光开发+128来源互斥但历史曝光确认、双clean跨域两序/同域开发五5%一序，3适应臂共2880到达；全部部署arm seal后GT离线诊断。旧TENT Vid11791预测保全，完整optimizer checkpoint+短尾bitwise replay接续；旧HC媒体依赖/Table2单位hold不擅自解除。见[协议](</home/wwww/visual grounding/protocols/decota_spatial_opd_v1.md>)、[状态](</home/wwww/visual grounding/artifacts/decota_spatial_opd_v1/STATUS.json>)、[旧队列保存](</home/wwww/visual grounding/artifacts/decota_paper_experiments_v1/user_opd_pause_20261006/PAUSE_COMPLETION.json>)。'
 f.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+line+'\n',1)+'\n\n### Explicit spatial-policy OPD update\n\n'+line+'\n')
 with (BASE/'ARCHIVE.log').open('a') as log:
  for action in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def prepare():
 BASE.mkdir(parents=True,exist_ok=True)
 if (BASE/'DESIGN_LOCK.json').exists():return read(BASE/'DESIGN_LOCK.json')
 data={};stages={}
 for ds in ['vidstg','hc2']:
  p=read(PAPER/ds/'PLAN.json');groups={}
  for r in p['rows']:groups.setdefault(r['source'],[]).append(r)
  pick=lambda rows:min(rows,key=lambda r:digest(['opd-query-v1',r['key']]))
  dev=sorted([s for s,rs in groups.items() if any(r['development_source_exposed'] for r in rs)],key=lambda s:digest(['opd-dev-v1',ds,s]))[:32]
  confirm=sorted([s for s,rs in groups.items() if not any(r['development_source_exposed'] for r in rs)],key=lambda s:digest(['opd-confirm-v1',ds,s]))[:128]
  assert len(dev)==32 and len(confirm)==128 and not set(dev)&set(confirm)
  data[ds]=dict(dev=[pick(groups[s])['ordinal'] for s in dev],confirm=[pick(groups[s])['ordinal'] for s in confirm])
  for split in ['dev','confirm']:
   parents=data[ds][split];orders={f'order{j+1}':sorted(parents,key=lambda i:digest(['opd-order-v1',j,ds,p['rows'][i]['source']])) for j in range(2)}
   stages[f'{split}_{ds}']=dict(dataset=ds,source='vidstg' if ds=='hc2' else 'hcstvg2',split=split,conditions=['clean'],orders=orders,parents=parents)
  stages[f'mechanism_{ds}']=dict(dataset=ds,source='hcstvg2' if ds=='hc2' else 'vidstg',split='mechanism',
   conditions=[f'{f}_5' for f in ['frame_drop','frame_freeze','motion_blur','occlusion','exposure']],orders={'order1':data[ds]['dev']},parents=data[ds]['dev'])
 cfg=dict(version='spatial_opd_v1',config=dict(sigma=.25,tau=.25,M=32,steps=10,lr=.03,LN_writeback=1/16),
  data=data,stages=stages,adapted_arms=['on_policy','frozen_rollout','shuffled_feedback'],adapted_arrivals=2880,
  GT_inference=False,confirmation_historically_exposed=True,confirmation_selection_use=False,protocol_sha256=sha(ROOT/'protocols/decota_spatial_opd_v1.md'),time=time.time())
 write(BASE/'DESIGN_LOCK.json',cfg);status(BASE/'STATUS.json',dict(status='prepared_pending_runtime_and_qualification',GT_read=False,time=time.time()))
 return cfg

def lock():
 prepare()
 if (BASE/'RUNTIME_LOCK.json').exists():return verify()
 files=['vg_tta/decota_spatial_opd_v1.py','vg_tta/decota_spatial_opd_audit_v1.py','scripts/test_decota_spatial_opd_v1.py',
  'scripts/decota_spatial_opd_common_v1.py','scripts/run_decota_spatial_opd_v1.py','scripts/score_decota_spatial_opd_v1.py',
  'scripts/continue_decota_spatial_opd_v1.py','scripts/resume_decota_paper_baseline_saved_v1.py','protocols/decota_spatial_opd_v1.md']
 from scripts.decota_paper_common_v1 import verify as paper_verify
 old=paper_verify();pins=dict(old['pins'])
 for f in sorted((PAPER/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
 pins.update({f:sha(ROOT/f) for f in files})
 write(BASE/'RUNTIME_LOCK.json',dict(pins=pins,protected=old['protected'],design_sha256=sha(BASE/'DESIGN_LOCK.json'),paper_runtime_sha256=sha(PAPER/'RUNTIME_LOCK.json'),GT_read=False,time=time.time()))
 return verify()

if __name__=='__main__':prepare()
