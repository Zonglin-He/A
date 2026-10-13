"""One finite CPU-only cached readout -> labels -> independent audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_benefit_retention_common_v1 import *

def cached(record,independent=False,validate=False):
 import torch
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from scripts.run_decota_paper_main_v1 import unpack_expert
 from vg_tta.decota_fixed_full_audit_v1 import top1_support
 pair=old.pair_for(record);inp=normalized(load(ROOT/record['originals'][0]['input_path']))
 assert old.digest(inp['expert'])==pair['evidence_sha256'][0]
 support=top1_support(unpack_expert(inp['expert']));positions=[int(p) for p,b in support];teacher=np.array([b for p,b in support],float).reshape(-1,4)
 fits=[];hashes={};states=[];arm_metrics={};counts=collections.Counter();gradients=[]
 for arm in ARMS:
  p=old.BASE/'formal'/f"{record['index']:05}"/(arm+'_repeat0.pt');h=sha(p)
  assert h==read(old.BASE/'GLOBAL_PREDICTION_BARRIER.json')['files'][str(p.relative_to(ROOT))];z=load(p);f=z['fit']
  assert z['index']==record['index'] and z['dataset']==record['dataset'] and z['arm']==arm and not z['GT_read']
  assert z['frame_ids']==inp['frame_ids'] and z['pixel_sha256']==inp['pixel_sha256']==pair['pixel_sha256']
  assert z['source_model_sha256']==inp['source_model_state_sha256']==pair['source_model_sha256']
  assert z['evidence_sha256']==old.digest(inp['expert']) and z['runtime_sha256']==sha(old.BASE/'RUNTIME.json')
  assert z['Query_Residual_reset'] and z['Adam_reset'] and z['no_cross_query_state_writeback'] and not f['GT_used']
  assert old.digest(f['initial'])==record['initial_hashes'][arm] and f['config']==record['config']
  assert f['positions']==positions and len(f['path'])==record['config']['steps']+1 and len(f['rounds'])==record['config']['steps']
  assert torch.equal(f['path'][0]['boxes'],f['before']) and torch.equal(f['path'][-1]['boxes'],f['final'])
  assert f['selected_step']==record['config']['steps'] and not torch.count_nonzero(f['initial'][old.Q])
  if validate:
   from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare,mathematical_readback
   from scripts.stvg_opd_design_ablation_common_v1 import audit
   from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
   m=mathematical_readback(f);assert m==z['audit']['Adam'];counts['Adam_scalar_checks']+=m['scalar_checks']
   assert audit(f,unpack_expert(inp['expert']))==z['audit']['objective']
   from scripts.run_decota_paper_main_v1 import read_row
   rawrow=read_row(record['dataset'],record['query_ordinal']);assert rawrow['frame_ids']==inp['frame_ids']
   trace,n=old.noise(f,rawrow['key']);assert trace==z['audit']['noise'];counts['sample_coordinates']+=n
   if arm!='Source':
    checks=dict(tensors=0,tensor_coordinates=0,scalar_values=0);equal(bare(f),bare(load(ROOT/record['originals'][ARMS.index(arm)-1]['path'])['fit']),checks)
    counts['original_full_fits']+=1;counts['original_tensor_coordinates']+=checks['tensor_coordinates']
  boxes=np.stack([x['boxes'].numpy().astype(float) for x in f['path']]);st=np.stack([vector(x['state']) for x in f['path']])
  assert boxes.shape==(record['config']['steps']+1,len(z['frame_ids']),4) and np.isfinite(boxes).all() and np.isfinite(st).all()
  mets=[];gg=[];cumulative=np.zeros(3);initial=st[0]
  for k,path in enumerate(f['path']):
   u=path.get('update');g=np.zeros(1792) if u is None else u['gradient'].numpy().astype(float)
   raw=np.zeros(1792) if u is None else u['raw'].numpy().astype(float)
   gg.append(g)
   if u is not None:assert u['names']==list(f['initial']) and u['optimizer']=='adam' and u['lr']==f['config']['lr']
   if k<len(f['rounds']):
    assert bool(u is not None)==f['rounds'][k]['updated']
    assert np.array_equal((st[k+1].astype(np.float32)-st[k].astype(np.float32)).astype(float),raw)
   else:assert u is None
   agreement=iou(corners(boxes[k,positions]),corners(teacher),independent,floor=1e-7) if positions else np.array([])
   if positions and k<len(f['rounds']):
    expected=f['rounds'][k]['central_reward_before'].numpy().astype(float);assert np.max(np.abs(agreement-expected))<3e-6
   if positions and k:
    expected=f['rounds'][k-1]['central_reward_after'].numpy().astype(float);assert np.max(np.abs(agreement-expected))<3e-6
   v=dict(teacher_agreement=mean(agreement),actual_update_outgoing=u is not None,empty_evidence=f['empty'])
   for j,(label,sl) in enumerate([('joint',slice(None)),('QR',slice(0,256)),('LN',slice(256,None))]):
    v[label+'_gradient_norm']=norm(g[sl]);v[label+'_step_norm']=norm(raw[sl]);v[label+'_cumulative_step_norm']=float(cumulative[j])
    v[label+'_displacement_norm']=norm((st[k]-initial)[sl]);cumulative[j]+=norm(raw[sl])
   v['output_movement_incoming_RMS']=0. if not k else float(np.sqrt(np.square(boxes[k]-boxes[k-1]).mean()))
   mets.append(v)
  fits.append(boxes);states.append(st);gradients.append(gg);arm_metrics[arm]=mets;hashes[str(p.relative_to(ROOT))]=h
  counts['fits']+=1;counts['updates']+=len(f['rounds']);counts['backwards']+=f['gradient_calls'];counts['empty_fits']+=f['empty']
  assert sha(p)==h
 state_dist={};output_dist={};pair_metrics={}
 for a,b in PAIRS:
  key=a+'_to_'+b;i,j=ARMS.index(a),ARMS.index(b);delta=states[j]-states[i]
  state_dist[key]={name:np.sqrt(np.square(delta[:,sl]).mean(1)) for name,sl in [('joint',slice(None)),('QR',slice(0,256)),('LN',slice(256,None))]}
  output_dist[key]=[distances(x,y,independent) for x,y in zip(fits[i],fits[j])]
  pair_metrics[key]=[{} for k in range(record['config']['steps']+1)]
 # Pair gradient cosines use the saved outgoing gradients; zero vectors are unknown.
 for a,b in PAIRS:
  key=a+'_to_'+b;i,j=ARMS.index(a),ARMS.index(b)
  for k,item in enumerate(pair_metrics[key]):
   for name,sl in [('joint',slice(None)),('QR',slice(0,256)),('LN',slice(256,None))]:item[name+'_gradient_cosine']=cosine(gradients[i][k][sl],gradients[j][k][sl])
 return dict(parent_id=record['index'],dataset=record['dataset'],frame_ids=inp['frame_ids'],visited=pair['visited'],positions=positions,
  boxes=np.stack(fits),state_dist=state_dist,output_dist=output_dist,arm_metrics=arm_metrics,pair_metrics=pair_metrics,
  source_bindings=hashes,config=record['config'],CPU_only=True,GT_read=False),dict(counts)

def inventory():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();verify()
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old();files={};counts=collections.Counter()
 for r in read(BASE/'DESIGN_LOCK.json')['records']:
  value,c=cached(r);p=BASE/'readouts'/f"{r['index']:05}.pt";save(p,value);files[str(p.relative_to(ROOT))]=sha(p);counts.update(c)
  counts['arm_boundaries']+=len(value['boxes'][0])*3;counts['frame_boundaries']+=len(value['frame_ids'])*len(value['boxes'][0])
  status(BASE/'STATUS.json',dict(status='reading_all_cached_boundaries_before_labels',done=counts['fits'],total=192,CPU_only=True,GT_read=False,time=time.time()))
  print('RETENTION_CACHE',r['index']+1,64,flush=True)
 assert counts['fits']==192 and counts['updates']==4800 and counts['arm_boundaries']==4992 and counts['empty_fits']==21
 verify();assert not torch.cuda.is_initialized()
 write(BASE/'GLOBAL_READOUT_BARRIER.json',dict(status='sealed',files=files,counts=dict(counts),parents=64,fits=192,
  arm_boundaries=4992,saved_updates=4800,runtime_sha256=sha(BASE/'RUNTIME.json'),CPU_only=True,GT_read=False,time=time.time()))
def run():
 import torch
 try:
  if not BASE.exists():prepare()
  inventory()
  from scripts.score_stvg_opd_state_benefit_retention_audit_v1 import score
  score()
  from scripts.audit_stvg_opd_state_benefit_retention_audit_v1 import root
  root()
  verify();assert not torch.cuda.is_initialized()
  status(BASE/'STATUS.json',dict(status='pending_actual_PDF_pixel_view_publication_archive',done=192,total=192,CPU_only=True,GT_read=True,time=time.time()))
 except BaseException:
  if BASE.exists():
   p=BASE/'failures';p.mkdir(exist_ok=True);(p/('CPU_'+str(os.getpid())+'.log')).write_text(traceback.format_exc())
   status(BASE/'STATUS.json',dict(status='failed_CPU_readout_or_audit_preserved',pid=os.getpid(),CPU_only=True,time=time.time()))
  raise
if __name__=='__main__':run()
