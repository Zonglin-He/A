"""T0/S0 finite CPU readouts; source logits, not hard intervals, supply q."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
D=ROOT/'artifacts/stvg_posterior_decision_tree_v1';OLD=ROOT/'artifacts/stvg_interval_decision_v1'
P=ROOT/'results/stvg_posterior_decision/2026-10-18/cached64'
ARMS=['Native','projection','marginal_MAP','per_offset_MBR','final_MBR']
LAYOUTS=['N','U','P','projection_uniform4']

def prepare():
 import torch
 from vg_tta.stvg_posterior_decision_tree_v1 import contract
 assert not(D/'PLAN_LOCK.json').exists();old=read(OLD/'PLAN_LOCK.json');bar=read(OLD/'GLOBAL_READOUT_BARRIER.json');refs={Path(r['path']).stem:r for r in bar['files']};files=[];parents={ds:set()for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in parents}
 for r in old['files']:
  assert sha(ROOT/r['path'])==r['sha256'];ref=refs[r['alias']];assert sha(ROOT/ref['path'])==ref['sha256'];z=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False);q=z['query_ordinal'];parent=plans[r['dataset']][q]['source'];assert parent not in parents[r['dataset']];parents[r['dataset']].add(parent);files.append(dict(**r,reference=ref,query_ordinal=q,parent=parent))
 extension=[];seen={ds:set(v)for ds,v in parents.items()};counts={ds:0 for ds in parents}
 for f in sorted((ROOT/'artifacts/stvg_spatial_controls256_v1/source_cases').glob('*.pt')):
  rc=read(f.with_suffix('.receipt.json'));assert sha(f)==rc['sha256'];z=torch.load(f,map_location='cpu',weights_only=False);ds=z['dataset'];ds='hc2'if ds in ['hc2','hcstvg2']else'vidstg';q=z['query_ordinal'];parent=plans[ds][q]['source']
  if parent in seen[ds]or counts[ds]>=32:continue
  seen[ds].add(parent);counts[ds]+=1;extension.append(dict(alias=('EH'if ds=='hc2'else'EV')+f'{counts[ds]:03}',dataset=ds,path=str(f.relative_to(ROOT)),sha256=rc['sha256'],query_ordinal=q,parent=parent))
 assert len(files)==64 and len(extension)==64
 own=['vg_tta/stvg_posterior_decision_tree_v1.py','scripts/run_stvg_posterior_decision_tree_v1.py','protocols/stvg_posterior_decision_tree_v1.md','scripts/stvg_temporal_marginal_decode_util256_v1.py','methods/decota_final_simplified_v1/observations.py']
 attachment=Path('/home/wwww/.codex/attachments/a4b6305d-bd9b-4dba-a6c8-73c97e06f721/已粘贴的文本.txt')
 put(D/'PLAN_LOCK.json',dict(status='locked_before_readouts_GT',count=64,files=files,extension=extension,attachment_sha256=sha(attachment),arms=ARMS,layouts=LAYOUTS,primary_geometry='After',critical_min_parents=8,critical_bins=dict(short=['event_length_bin','Q1'],fast=['motion_bin','Q4'],small=['target_area_bin','Q1']),worthwhile_pp=.5,projection_noninferiority_pp=-.5,subgroup_risk_pp=-1.,coverage_gain=.05,zero_hit_risk=.05,coverage_subgroup_risk=-.05,max_extensions=1,CPU_T0_seconds_cap=3600,CPU_S0_seconds_cap=3600,created_epoch=time.time(),deadline_epoch=time.time()+7200,GT_before_all_readout_seal=False,new_DINO=0,new_models=0,extension_primary_After_available=False,historical_full_exposure=True))
 put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in own},registries={f:sha(ROOT/f)for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']},plan_sha256=sha(D/'PLAN_LOCK.json'),strata_sha256=sha(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json')))
 put(D/'CPU_CONTRACT_COMPLETION.json',contract());print('LOCKED64 + one fixed64 extension',flush=True)

def verify():
 p=read(D/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'];l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 for f,h in {**l['pins'],**l['registries']}.items():assert sha(ROOT/f)==h,f
 return p

def compute(records,extension=False):
 import torch,numpy as np
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf,mode
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,greedy4,coverage_value,brute_utility
 from methods.decota_final_simplified_v1.observations import uniform_positions
 torch.set_num_threads(4);files=[];start=time.perf_counter();T0=S0=0.;folder='extension_readouts'if extension else'readouts'
 for k,r in enumerate(records):
  verify();path=D/folder/(r['alias']+'.pt');assert sha(ROOT/r['path'])==r['sha256'];z=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False);c=z if extension else z['capture'];zero=c['zero'];records0=c['records'];ids=z['frame_ids'];tick=time.perf_counter()
  grids=probabilities(records0,zero['logits']);U,q=mass_table(grids);assert U==ids;cdf=independent_cdf(grids,U);err=float(abs(cdf-q).max());assert err<1e-12;assert np.count_nonzero(np.tril(q))==0
  iv,mass=mode(U,q);mb=final_mbr(U,q);s,e=np.triu_indices(len(U),1);checks=np.unique(np.linspace(0,len(s)-1,min(17,len(s))).astype(int));direct=brute_utility(U,q,checks);probe_error=float(abs(direct-mb['utility_grid'][s[checks],e[checks]]).max());assert probe_error<1e-11
  sourceboxes=zero['boxes'];boxes={'Source':sourceboxes};ivs=dict(marginal_MAP=iv,final_MBR=mb['interval'])
  if extension:
   native=z['Native_interval'];proj=z['Projection_interval'];ivs.update(Native=native,projection=proj)
  else:
   ref=load(ROOT/r['reference']['path']);assert ref['input_source_sha256']==r['sha256'];ivs.update(Native=ref['fused_intervals']['Native'],projection=ref['fused_intervals']['projection'],per_offset_MBR=ref['fused_intervals']['native_MBR']);boxes.update(Before=z['original_spatial_before']['boxes'],After=z['predictions']['SpatialOPD']['boxes'])
  t0=time.perf_counter()-tick;T0+=t0;tick=time.perf_counter();a,b=[ids.index(v)for v in [ivs['Native'][0],ivs['Native'][1]-1]];N=uniform_positions(ids,[a,b],4);U4=uniform_positions(ids,[0,len(ids)-1],4);pp,trace=greedy4(ids,q);a,b=[ids.index(v)for v in [ivs['projection'][0],ivs['projection'][1]-1]];proj4=uniform_positions(ids,[a,b],4);layouts=dict(N=N,U=U4,P=pp,projection_uniform4=proj4);assert all(len(v)==len(set(v))==4 for v in layouts.values());s0=time.perf_counter()-tick;S0+=s0
  out=dict(alias=r['alias'],dataset=r['dataset'],query_ordinal=r['query_ordinal'],frame_ids=ids,q=q,utility_grid=mb['utility_grid'],intervals=ivs,boxes=boxes,layouts=layouts,greedy_trace=trace,model_coverage={a:coverage_value(q,v)for a,v in layouts.items()},q_mass=float(q.sum()),marginal_mode_mass=mass,MBR_expected_tIoU=mb['utility'],mode_expected_tIoU=float(mb['utility_grid'].ravel()[int(q.argmax())]),q_CDF_max_error=err,utility_probe_max_error=probe_error,source_sha256=r['sha256'],GT_read=False,compute=dict(T0_seconds=t0,S0_seconds=s0))
  files.append(save(path,out));put(D/'STATUS.json',dict(status='running',stage=folder,done=k+1,total=len(records),GT_read=False),True);print('SEALED',r['alias'],len(ids),'T0',round(t0,3),'S0',round(s0,3),flush=True)
  assert T0<3600 and S0<3600;del z,c,zero,q,cdf,mb,out
 put(D/('EXTENSION_READOUT_BARRIER.json'if extension else'GLOBAL_READOUT_BARRIER.json'),dict(status='complete',count=len(files),files=files,GT_read=False,T0_seconds=T0,S0_seconds=S0,CPU_wall_seconds=time.perf_counter()-start,new_models=0,new_DINO=0));return files

def readouts():compute(verify()['files'])
def extend():
 assert read(D/'INITIAL_DECISION.json')['S0_extension_required'];assert not(D/'EXTENSION_READOUT_BARRIER.json').exists();compute(verify()['extension'],True)

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','readouts','extend']);stage=a.parse_args().stage
 try:globals()[stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',type=type(e).__name__,error=str(e),partial_preserved=True));raise
