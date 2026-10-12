"""One fixed receiving-LN intervention; old science and baselines are immutable."""
import collections,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
BASE=ROOT/'artifacts/stvg_opd_LN_state_intervention_v1'
OUT=ROOT/'results/stvg_opd_LN_state_intervention/2026-10-13'
P0=ROOT/'artifacts/stvg_opd_state_conditional_actionability_v1'
S=ROOT/'artifacts/stvg_opd_supplement_v1'
ARMS=['Source','Inherited_order1','Inherited_order2']
Q='spatial.query_residual';BOOT=10000;SEED=20261013;EPS=1e-12

def canonical(x):
 import torch
 if torch.is_tensor(x):return dict(dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
 if isinstance(x,np.ndarray):return dict(dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(x.tobytes()).hexdigest())
 if isinstance(x,dict):return {str(k):canonical(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [canonical(v) for v in x]
 if isinstance(x,np.generic):return x.item()
 return x
def digest(x):return hashlib.sha256(json.dumps(canonical(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def transplant(source,ln):
 import torch
 assert set(source)==set(ln) and len(source)==7 and Q in source
 assert all(source[n].shape==ln[n].shape==(256,) and source[n].dtype==ln[n].dtype==torch.float32 and torch.isfinite(ln[n]).all() for n in source)
 out={n:(torch.zeros_like(source[n]) if n==Q else ln[n].detach().cpu().clone()) for n in source}
 assert sum(v.numel() for n,v in out.items() if n!=Q)==1536 and not torch.count_nonzero(out[Q]).item()
 return out
def states(pair):return dict(zip(ARMS,[transplant(pair['source_initial_state'],s) for s in [pair['source_initial_state'],*pair['initial_states']]]))
def noise(f,key):
 import torch
 out=[];coords=0
 for k,r in enumerate(f['rounds']):
  assert r['round']==k
  if 'rollout' not in r:
   assert f['empty'] and r['no_information'] and not r['updated'];continue
  a=r['rollout'];seed=int(hashlib.sha256(f'spatial-opd-v1|{key}|{k}'.encode()).hexdigest()[:15],16);assert a['seed']==seed
  gen=torch.Generator(device='cpu').manual_seed(seed);n=torch.randn(len(a['mean']),16,4,generator=gen,dtype=torch.float32);n=torch.cat((n,-n),1)
  assert a['mean'].dtype==a['samples'].dtype==torch.float32 and torch.equal(a['mean'][:,None,:]+f['config']['sigma']*n,a['samples'])
  out.append(dict(round=k,seed=seed,noise_sha256=digest(n),noise_shape=list(n.shape)));coords+=a['samples'].numel()
 assert len(out)==(0 if f['empty'] else f['config']['steps'])
 return out,coords
def verify():
 r=read(BASE/'RUNTIME.json')
 assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256'] and sha(BASE/'CPU_CONTRACTS.json')==r['contracts_sha256']
 for n,h in {**r['pins'],**r['bindings'],**r['protected']}.items():assert sha(ROOT/n)==h,n
 assert read(S/'baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json')['status']=='remaining_formal_cancelled_by_explicit_human_instruction'
 return r
def pair_for(record):
 p=ROOT/record['pair_path'];assert sha(p)==record['pair_sha256'];z=load(p)
 assert z['dataset']==record['dataset'] and z['parent_id']==record['parent_id'] and z['previous_P6']
 for rel,h in z['source_bindings'].items():assert sha(ROOT/rel)==h,rel
 return z
def cpu_contracts():
 import torch
 assert not torch.cuda.is_initialized();source={Q:torch.zeros(256),**{f'LN{i}':torch.ones(256)*(i+1) for i in range(6)}}
 altered={n:v.clone()+.25 for n,v in source.items()};z=transplant(source,altered)
 assert not torch.count_nonzero(z[Q]) and all(torch.equal(z[n],altered[n]) for n in z if n!=Q)
 assert all(torch.equal(source[n],torch.zeros(256) if n==Q else torch.ones(256)*(int(n[2:])+1)) for n in source)
 bad={**altered,'LN0':torch.zeros(255)}
 try:transplant(source,bad)
 except AssertionError:pass
 else:raise AssertionError('Invalid state was accepted')
 p=torch.nn.Parameter(torch.zeros(2));a=torch.optim.Adam([p]);assert a.state_dict()['state']=={}
 p.grad=torch.ones(2);a.step();fresh=torch.optim.Adam([p]);assert fresh.state_dict()['state']=={} and a.state_dict()['state']
 n=torch.randn(2,16,4,generator=torch.Generator().manual_seed(7));n=torch.cat((n,-n),1);assert torch.equal(n[:,:16],-n[:,16:])
 assert not torch.cuda.is_initialized()
 return dict(status='pass',contracts=7,scope='synthetic exact LN transplant/zero Query Residual/source preservation/invalid schema rejection/fresh Adam/common antithetic noise/CPU-only; not GPU qualification',CPU_only=True,GT_read=False,time=time.time())
def prepare():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 assert not BASE.exists() and not OUT.exists();BASE.mkdir();OUT.mkdir(parents=True)
 assert read(P0/'FINAL_COMPLETION.json')['status']=='complete' and read(S/'FINAL_COMPLETION.json')['status']=='complete'
 assert read(ROOT/'artifacts/stvg_opd_claim_strengthening_v1/FINAL_COMPLETION.json')['status']=='complete'
 rs=read(P0/'DESIGN_LOCK.json')['records'];bar=read(P0/'GLOBAL_PAIRED_READOUT_BARRIER.json');groups=collections.defaultdict(list)
 for r in rs:
  if r['previous_P6']:groups[r['parent_id']].append(r)
 selected=[];bindings={};empty={};coordinates=0
 for index,(parent,group) in enumerate(sorted(groups.items())):
  group=sorted(group,key=lambda r:r['order']);assert [r['order'] for r in group]==['order1','order2']
  p=P0/'pairs'/f'{parent:05}.pt';rel=str(p.relative_to(ROOT));assert sha(p)==bar['files'][rel];z=load(p);ss=states(z)
  assert all(digest(ss[ARMS[j+1]])==z['initial_sha256'][j] for j in [0,1])
  for n,h in z['source_bindings'].items():assert sha(ROOT/n)==h;bindings[n]=h
  originals=[]
  for r in group:
   f=load(ROOT/r['original_Full']['path'])['fit'];assert digest(f['initial'])==digest(ss['Inherited_'+r['order']]);assert f['config']==r['config']
   originals.append(dict(**r['original_Full'],public_cell=r['public_cell'],order=r['order']));coordinates+=1792
  assert group[0]['config']==group[1]['config'];empty[index]=f['empty']
  selected.append(dict(index=index,dataset=z['dataset'],source=group[0]['source'],parent_id=parent,query_ordinal=group[0]['query_ordinal'],
   capture_index=group[0]['capture_index'],previous_P6=True,config=group[0]['config'],pair_path=rel,pair_sha256=sha(p),
   originals=originals,initial_hashes={n:digest(v) for n,v in ss.items()},source_model_sha256=z['source_model_sha256'],pixel_sha256=z['pixel_sha256']))
  bindings[rel]=sha(p)
 assert len(selected)==64 and collections.Counter(r['dataset'] for r in selected)=={'hc2':32,'vidstg':32}
 qual=[r['index'] for r in selected if r['dataset']=='hc2'][:2]
 vid=[r for r in selected if r['dataset']=='vidstg'];qual += [next(r['index'] for r in vid if not empty[r['index']]),next(r['index'] for r in vid if empty[r['index']])]
 baseline=read(S/'baselines/DESIGN_LOCK.json');overlap={ds:len({r['query_ordinal'] for r in selected if r['dataset']==ds}&set(baseline['datasets'][ds]['development_parents'])) for ds in ['hc2','vidstg']}
 write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_actual_GPU_GT',records=selected,parents=64,arms=ARMS,formal_fits=192,
  qualification_indices=qual,qualification_fits=24,qualification_captures=8,formal_captures=64,total_formal_rounds=4800,
  baseline_dev_query_overlap=overlap,historical_GT_exposure=True,primary='own_U_delta_Omega0_real_unvisited_frames',secondary='common_U_delta_Omega0_Omega1_union',
  bootstrap=BOOT,seed=SEED,tiers_pp=[1,5,20],human_selected_P0_previous_P6_32=True,single_factor_LN_initialization=True,
  same_common_noise=True,zero_Query_Residual=True,Adam_reset=True,new_DINO_calls=0,new_online_state_writeback=False,
  gate_or_parameter_search=False,production_promoted=False,remaining_formal_baseline_authority=False,time=time.time()))
 write(BASE/'CPU_CONTRACTS.json',cpu_contracts())
 own=['protocols/stvg_opd_LN_state_intervention_v1.md','scripts/stvg_opd_LN_state_intervention_common_v1.py',
  'scripts/run_stvg_opd_LN_state_intervention_v1.py','scripts/score_stvg_opd_LN_state_intervention_v1.py','scripts/audit_stvg_opd_LN_state_intervention_v1.py']
 shared=['scripts/run_decota_paper_main_v1.py','scripts/stvg_opd_design_ablation_common_v1.py','scripts/stvg_opd_legacy_input_bridge003.py',
  'scripts/stvg_opd_paper_inputs_v1.py','scripts/run_stvg_opd_p1_softmax_recovery_002.py',
  'scripts/run_stvg_opd_expanded_after_p8_priority001.py','vg_tta/stvg_opd_p4_joint_precision001.py','vg_tta/stvg_opd_common_state_geometry_arithmetic004.py',
  'vg_tta/stvg_opd_paper_ablations_v1.py','vg_tta/decota_spatial_opd_tunable_v1.py']
 shared += [str(p.relative_to(ROOT)) for p in (ROOT/'vg_tta').glob('stvg_opd_frozen_*0*.py')]
 for p in [P0/'RUNTIME.json',P0/'GLOBAL_PAIRED_READOUT_BARRIER.json',P0/'ROOT_READBACK.json',P0/'FINAL_COMPLETION.json',
  S/'FINAL_COMPLETION.json',S/'baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json']:
  bindings[str(p.relative_to(ROOT))]=sha(p)
 protected={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'methods/CURRENT_METHOD.json',ROOT/'methods/decota_spatial_opd_v1/configs.json',
  ROOT/'artifacts/stvg_opd_temporal_guided_spatial_candidate_v1/P7_A/FINAL_COMPLETION.json']}
 from scripts.pause_stvg_opd_for_geometry_v1 import process
 identity=[process(pid) for pid in [509176,3091846]];assert all(p['state']=='T' for p in identity)
 write(BASE/'BASELINE_STANDBY_IDENTITY.json',dict(status='pass',processes=identity,scope='same explicitly cancelled baseline processes remain T; no signals authorized',GT_read=False,time=time.time()))
 bindings[str((BASE/'BASELINE_STANDBY_IDENTITY.json').relative_to(ROOT))]=sha(BASE/'BASELINE_STANDBY_IDENTITY.json')
 write(BASE/'RUNTIME.json',dict(status='locked_before_actual_GPU_GT',pins={n:sha(ROOT/n) for n in own+shared},bindings=bindings,protected=protected,
  design_sha256=sha(BASE/'DESIGN_LOCK.json'),contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),new_DINO_calls=0,GT_read=False,time=time.time()))
 write(BASE/'SOURCE_STATE_CUSTODY_READBACK.json',dict(status='pass',parents=64,original_inherited_fits=128,state_coordinates=coordinates,
  source_hash_bindings=len(bindings),baseline_dev_query_overlap=overlap,CPU_only=True,GT_read=False,time=time.time()))
 status(BASE/'STATUS.json',dict(status='prepared_pending_actual24_qualification_then192_formal',done=0,total=192,CPU_only=True,GT_read=False,time=time.time()))
 verify()
