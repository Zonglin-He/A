"""Pin runtime and reuse immutable matching observations, without GT/model construction."""
import sys,os,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_coordinate_common_v2 import *
def run():
 assert not (BASE/'RUNTIME_LOCK.json').exists()
 from scripts.tastvg_optuna_common_v1 import verify as old_verify
 old_verify();oldlock=read(ROOT/'artifacts/tastvg_optuna_v1/LOCK.json');design=read(BASE/'DESIGN_LOCK.json');metadata={};cache_links=0
 for dataset in DATASETS:
  out=BASE/dataset;p=read(out/'PLAN.json');assert sha(out/'PLAN.json')==design['plans'][dataset+'/PLAN.json']
  paper=ROOT/'artifacts/tastvg_paper48_v1'/p['paper48_panel'];sb=read(paper/'SUBJECT_BARRIER.json')
  experts=sorted({parent for sp in p['splits'].values() for order in sp['orders'].values() for at,parent in enumerate(order) if at%4==0})
  ep=dict(rows=p['rows'],conditions=p['conditions'],expert_needed=experts,conditions_by_parent={str(i):p['conditions'] for i in experts},total=len(experts)*len(p['conditions']))
  write(out/'EXPERT_PLAN.json',ep);files={}
  for i,row in enumerate(p['rows']):
   src=paper/'subjects'/f"{row['original_parent']:05}.json";assert sha(src)==sb['files'][src.name]
   dst=out/'subjects'/f'{i:05}.json';dst.parent.mkdir(parents=True,exist_ok=True)
   if dst.exists():assert sha(dst)==sha(src)
   else:shutil.copy2(src,dst)
   files[dst.name]=sha(dst)
  write(out/'SUBJECT_BARRIER.json',dict(count=48,files=files))
  prior=ROOT/'artifacts/tastvg_optuna_v1'/dataset;pp=read(prior/'PLAN.json');cb=read(prior/'CAPTURE_BARRIER.json');reuse=[]
  for i in range(32):
   assert p['rows'][i]==pp['rows'][i]
   for cond in p['conditions']:
    rf=prior/'capture'/cond/f'{i:05}.json';assert sha(rf)==cb['files'][str(rf.relative_to(prior))];r=read(rf);src=prior/r['cache'];assert sha(src)==r['sha256'] and r['source_native_parity']
    dst=out/r['cache'];dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():os.link(src,dst)
    else:assert sha(dst)==r['sha256']
    reuse.append(dict(parent=i,condition=cond,cache=str(dst.relative_to(out)),sha256=r['sha256'],prior_receipt_sha256=sha(rf)));cache_links+=1
  write(out/'CACHE_REUSE.json',dict(status='matching_old32_source_cache_linked',checkpoint_state_sha256=cb['checkpoint_state_sha256'],cells=len(reuse),receipts=reuse,GT_read=False))
  for f in [out/'EXPERT_PLAN.json',out/'SUBJECT_BARRIER.json',out/'CACHE_REUSE.json',*(out/'subjects').glob('*.json')]:metadata[str(f.relative_to(BASE))]=sha(f)
 pins=dict(oldlock['pins'])
 for f in sorted((ROOT/'scripts').glob('*coordinate*v2.py')):pins[str(f.relative_to(ROOT))]=sha(f)
 for name in ['protocols/tastvg_coordinate_tuning_v2.md','tests/test_tastvg_coordinate_v2.py']:pins[name]=sha(ROOT/name)
 write(BASE/'RUNTIME_LOCK.json',dict(status='locked_before_new_model_execution_or_GT',pins=pins,metadata=metadata,fig1_completion_sha256=sha(ROOT/'artifacts/stvg_native_support_fig1_v1/uniform64_v2/FINAL_COMPLETION.json'),old_optuna_lock_sha256=sha(ROOT/'artifacts/tastvg_optuna_v1/LOCK.json'),dataset_checkpoints='same official per-dataset Paper48 checkpoints; source capture validates state hashes',fixed_rho=.05,selection='corrupt_all_source_macro_dense_delta_vIoU',cache_reused_cells=cache_links,time=time.time()))
 verify();status(BASE/'STATUS.json',dict(status='runtime_locked_ready_for_serial_execution',model_predictions=0,GT_read=False,cache_reused_cells=cache_links,time=time.time()));print('RUNTIME_READY',cache_links)
if __name__=='__main__':run()
