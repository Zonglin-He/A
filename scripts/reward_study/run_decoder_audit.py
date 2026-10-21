"""Finite frozen-cache derivative audit. Default only describes; --run is explicit.

No RGB, source encoding, teacher forward, labels, metrics, tuning or formal outputs.
Six fixed canonical states (3 queries x steps0/20) and all three initial full Jacobians.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,copy,gc,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()
TRIAL=ROOT/'outputs/layout_reward_dev'
WORK=TRIAL/'workspace'
OUT=TRIAL/'research/decoder_audit'
CASE_ORDINALS=[36,84,125];STEPS=[0,20]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(z,indent=2)+'\n');os.replace(tmp,p)
def describe():
 return dict(status='prepared_not_run',HC_development_ordinals=CASE_ORDINALS,states=STEPS,canonical_trajectory='saved LayoutOPD for step20; same eta and same reconstructed OPD Adam buffers for both gradients',Jacobian='HC development rows 36/84/125 initial, all 65x4 corrected chart outputs vs all 8x256 content coordinates',no_new_source_encoder=True,no_new_RGB=True,no_new_teacher=True,GT_read=False,strict_state_replay='every saved Adam state_step and replayed path boxes must match bitwise; stop on incompatibility',gradient_tolerance=dict(atol=1e-8,rtol=1e-5),plan_budget='6 matched-state comparisons + 3 initial Jacobians and cached seed-gradient mappings; shadow outputs are diagnostic only')


def main(run):
 if not run:print(json.dumps(describe(),indent=2));return
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'sealed.json').exists():raise RuntimeError('Existing sealed audit is immutable; do not overwrite')
 sys.path.insert(0,str(WORK));os.chdir(WORK)
 import torch
 from stvg.runtime import setup,lease,load_source
 from stvg.state import Replay
 from stvg.policy import chart,diagonal,noise_for,unchart,mixture_loss
 from stvg.layout_reward import LayoutReward
 setup();torch.set_num_threads(1)
 assert torch.cuda.is_available();lock=lease()
 start=time.perf_counter();model=None;rp=None
 cfg=read(TRIAL/'config.json')['datasets']['hc2'];assert cfg['lr']==.014 and cfg['steps']==40 and cfg['layout_tau']==.1
 hc_development_rows=[x for x in [json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l.strip()]if x['dataset']=='hc2']
 rows={x['id']:x for x in hc_development_rows};case_ids=[hc_development_rows[i-1]['id']for i in CASE_ORDINALS];case_public_ids=dict(zip(case_ids,('H%03d'%i for i in CASE_ORDINALS)))
 original_run=read(TRIAL/'run.json');fingerprint=original_run['fingerprint']
 computed=hashlib.sha256((TRIAL/'config.json').read_bytes()+b''.join(p.read_bytes()for p in sorted((WORK/'stvg').glob('*.py')))+(WORK/'run_trial.py').read_bytes()).hexdigest()
 assert computed==fingerprint, 'Original executed workspace fingerprint changed'
 script_sha=sha(__file__);atomic(OUT/'run.json',dict(status='running',PID=os.getpid(),start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),script_sha256=script_sha,executed_inference_fingerprint=fingerprint,plan=describe(),GT_read=False))
 outputs=[];checks=[]
 def savept(name,z):
  p=OUT/name;tmp=p.with_suffix('.pt.tmp');torch.save(z,tmp);os.replace(tmp,p);outputs.append(dict(path=name,sha256=sha(p)))
 def tensorco(a,b):
  a=a.detach().double().flatten();b=b.detach().double().flatten();d=a.norm()*b.norm();return float((a@b/d).cpu())if float(d)>0 else None
 def chartdisp(b1,b0):return torch.cat(((b1[:,:2]-b0[:,:2])/b0[:,2:],torch.log(b1[:,2:]/b0[:,2:])),dim=-1)
 def optimizer(param):return torch.optim.Adam([param],lr=cfg['lr'],betas=(.9,.999),eps=1e-8)
 def checkfrozen(r):assert all(not p.requires_grad and p.grad is None for p in r.decoder.parameters())
 def reconstruct(r,trace,step):
  with torch.no_grad():r.state.zero_()
  opt=optimizer(r.state);maxerr=0.
  for kk in range(step):
   before=r.state.detach().clone();opt.zero_grad(set_to_none=True);r.state.grad=trace['rounds'][kk]['gradient'].to('cuda');opt.step();actual=(r.state.detach()-before).cpu();expected=trace['rounds'][kk]['state_step'];err=float((actual-expected).abs().max());maxerr=max(maxerr,err)
   assert torch.equal(actual,expected),f'Adam numerical replay mismatch at saved step{kk}, error={err}'
  v=r.values();expected=trace['path'][step]['boxes'];actual=v['boxes'].detach().cpu();err=float((actual-expected).abs().max())
  assert torch.equal(actual,expected),f'Decoder path replay mismatch at state{step}, error={err}'
  return opt,v,dict(reconstructed_steps=step,state_steps_bit_exact=True,state_step_max_error=maxerr,path_boxes_bit_exact=True,path_boxes_max_error=err)
 def release():
  gc.collect();torch.cuda.empty_cache()
 def loadcase(key):
  cp=TRIAL/'student_cache'/f'{key}.pt';tp=TRIAL/'traces'/f'{key}.pt';rec=read(TRIAL/'predictions'/f'{key}.json');cc=read(cp.with_suffix('.json'))
  assert sha(cp)==cc['sha256'] and cc['fingerprint']==fingerprint;assert sha(tp)==rec['trace_sha256'];assert rec['fingerprint']==fingerprint
  c=torch.load(cp,map_location='cpu',weights_only=False);z=torch.load(tp,map_location='cpu',weights_only=False);return c,z
 def modulehash(module):
  h=hashlib.sha256()
  for name,t in module.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 try:
  torch.cuda.reset_peak_memory_stats();t=time.perf_counter();model=load_source(cfg);model.cpu();release();model.ground_decoder.decoder.cuda().eval().requires_grad_(False);decoder_hash=modulehash(model.ground_decoder.decoder);load_seconds=time.perf_counter()-t
  for key in case_ids:
   c,z=loadcase(key);assert len(c['planned_positions'])==8 and len(c['Source'])==65
   rp=Replay(model,c,c['Source'],c['E'],c['planned_positions'],lambda:None);reward=LayoutReward(c['teacher']['layout_features'],c['teacher']['layout_mask'],c['teacher']['boxes']);active=reward.active
   assert active.all(), 'These prespecified three cases require all eight active teachers'
   for step in STEPS:
    qtime=time.perf_counter();opt,v,check=reconstruct(rp,z['LayoutOPD'],step);base_eta=rp.state.detach().clone();base_box=v['boxes'].detach().clone();bs=v['boxes'][rp.positions];mu=chart(bs);D=diagonal(bs)
    with torch.no_grad():
     raw,rng=noise_for(rows[key]['key'],41,step,8,torch.float32);eps=torch.cat((raw[:,:8],raw[:,16:24],raw[:,8:16],raw[:,24:]),1).to(mu);sig=mu.new_tensor([cfg['scales'][0]]*16+[cfg['scales'][1]]*16);actions=(mu.detach()[:,None]+D[:,None]*sig[None,:,None]*eps).detach();cand=unchart(actions);rew=reward(cand);weights=(rew/cfg['layout_tau']).softmax(-1).detach()
    saved=z['LayoutOPD']['rounds'][step]['rollout'];assert torch.equal(actions.cpu(),saved['actions'])
    weight_error=float((weights.cpu()-saved['weights']).abs().max());assert weight_error<1e-6
    lopd=mixture_loss(mu[active],actions[active],D[active],weights[active],cfg['scales']);ldirect=(1-reward(bs[:,None]).squeeze(1)[active]).mean()
    gopd=torch.autograd.grad(lopd,rp.state,retain_graph=True)[0];gdirect=torch.autograd.grad(ldirect,rp.state)[0]
    savedg=z['LayoutOPD']['rounds'][step]['gradient'].to(gopd);gerr=float((gopd-savedg).abs().max());assert torch.allclose(gopd,savedg,atol=1e-8,rtol=1e-5),f'Original OPD gradient replay incompatible: {gerr}'
    buffers=copy.deepcopy(opt.state_dict());shadow={};normbox=chartdisp(base_box,base_box)
    for name,g in [('LayoutOPD',gopd),('LayoutDirect',gdirect)]:
     ss=base_eta.detach().clone().requires_grad_(True);so=optimizer(ss);so.load_state_dict(copy.deepcopy(buffers));so.zero_grad(set_to_none=True);ss.grad=g.detach().clone();so.step();delta=ss.detach()-base_eta
     with torch.no_grad():rp.state.copy_(ss.detach())
     new=rp.values()['boxes'].detach();dc=chartdisp(new,base_box)
     if name=='LayoutOPD':
      assert torch.equal(delta.cpu(),z['LayoutOPD']['rounds'][step]['state_step']), 'Original next-state delta mismatch'
      assert torch.equal(new.cpu(),z['LayoutOPD']['path'][step+1]['boxes']), 'Original next-output mismatch'
     shadow[name]=dict(gradient=g.detach().cpu(),state_delta=delta.cpu(),state_after=ss.detach().cpu(),boxes_after=new.cpu(),normalized_box_displacement=dc.cpu())
     del so,ss,new
    with torch.no_grad():rp.state.copy_(base_eta)
    check.update(query=key,public_id=case_public_ids[key],step=step,canonical_path='LayoutOPD',same_Adam_buffers_for_both=True,original_OPD_weight_max_error=weight_error,original_OPD_gradient_max_error=gerr,content_gradient_cosine=tensorco(gopd,gdirect),Adam_delta_cosine=tensorco(shadow['LayoutOPD']['state_delta'],shadow['LayoutDirect']['state_delta']),output_displacement_cosine=tensorco(shadow['LayoutOPD']['normalized_box_displacement'],shadow['LayoutDirect']['normalized_box_displacement']),OPD_gradient_norm=float(gopd.norm()),Direct_gradient_norm=float(gdirect.norm()),OPD_Adam_delta_norm=float(shadow['LayoutOPD']['state_delta'].norm()),Direct_Adam_delta_norm=float(shadow['LayoutDirect']['state_delta'].norm()),seconds=time.perf_counter()-qtime)
    checks.append(check);savept(f'{key}_state{step:02d}.pt',dict(GT_read=False,before_state=base_eta.cpu(),before_boxes=base_box.cpu(),Adam_before=buffers,positions=rp.positions,shadow=shadow,check=check));atomic(OUT/'progress.json',dict(status='running',states_complete=len(checks),states_total=6,latest=check,GT_read=False));print(json.dumps(check),flush=True)
    checkfrozen(rp);del v,opt,mu,bs,gopd,gdirect,lopd,ldirect,shadow,buffers,actions,cand,rew,weights,base_eta,base_box;release()
   del rp,reward,c,z;rp=None;release()
  # Three complete local derivative matrices; retain the single already-loaded source model.
  jacobians={};sampling=read(TRIAL/'research/matched_state_sampling.json');sample_rows={r['id']:r for r in sampling['queries']};sampling_sha=sha(TRIAL/'research/matched_state_sampling.json')
  for key in case_ids:
   c,z=loadcase(key);rp=Replay(model,c,c['Source'],c['E'],c['planned_positions'],lambda:None);opt,v,check=reconstruct(rp,z['LayoutOPD'],0);base=v['boxes'].detach();yy=chart(v['boxes']).flatten();assert yy.numel()==260 and rp.state.numel()==2048
   t=time.perf_counter();J=[]
   for ix in range(yy.numel()):
    gg=torch.autograd.grad(yy[ix],rp.state,retain_graph=ix+1<len(yy))[0];J.append(gg.detach().cpu().flatten())
    if (ix+1)%40==0:print(json.dumps(dict(Jacobian_query=key,Jacobian_rows=ix+1,total=260,seconds=time.perf_counter()-t)),flush=True)
   J=torch.stack(J);assert J.shape==(260,2048)and torch.isfinite(J).all();checkfrozen(rp)
   pos=rp.positions;Dout=torch.cat((base[:,2:],torch.ones_like(base[:,2:])),dim=-1).cpu().flatten();Jn=J.double()/Dout[:,None];jj=Jn.reshape(65,4,8,256);energy=jj.square().sum((1,3));same=torch.tensor([[t%2==p%2 for p in pos]for t in range(65)]);own=torch.tensor([[t==p for p in pos]for t in range(65)]);obs=torch.tensor([t in pos for t in range(65)]);total=float(energy.sum());crossmax=float(jj.permute(0,2,1,3)[~same].abs().max())
   assert crossmax==0.,'Unexpected cross-offset dependency in actual frozen replay'
   def spectrum(a):
    sv=torch.linalg.svdvals(a.double());en=sv.square();pp=en/en.sum();cumulative=pp.cumsum(0);return dict(shape=list(a.shape),singular_values=sv.tolist(),energy_rank_90=int((cumulative<.9).sum()+1),energy_rank_95=int((cumulative<.95).sum()+1),energy_rank_99=int((cumulative<.99).sum()+1),effective_rank_entropy_energy=float(torch.exp(-(pp[pp>0]*pp[pp>0].log()).sum())),effective_rank_participation=float(1/pp.square().sum()),ranks_at_relative_singular_threshold={str(tol):int((sv>sv[0]*tol).sum())for tol in [1e-2,1e-3,1e-4,1e-5,1e-6]},Frobenius_norm=float(a.double().norm()))
   # CPU fixed-initial-box score gradients from the already completed five-seed audit.
   sr=sample_rows[key];assert sr['positions']==pos;Ja=J.reshape(65,4,2048)[pos].reshape(32,2048).double();seedmaps=[]
   for ss in sr['sampling']['seeds']:
    boxg=torch.tensor(ss['box_gradient'],dtype=torch.float64).flatten();cg=Ja.T@boxg;adam=-cfg['lr']*cg/(cg.abs()+1e-8);linear=Jn@adam
    item=dict(seed=ss['seed'],box_gradient=boxg,content_gradient=cg,initial_Adam_delta=adam,linearized_dimensionless_full_output_delta=linear)
    if ss['seed']==41:
     refg=z['LayoutOPD']['rounds'][0]['gradient'].double().flatten();rel=float((cg-refg).norm()/refg.norm());item['seed41_saved_content_gradient_relative_L2_error']=rel;item['seed41_saved_content_gradient_cosine']=tensorco(cg,refg);assert rel<1e-4,'Jacobian seed41 content-gradient mapping incompatible'
    seedmaps.append(item)
   seedpairs=[]
   for ii in range(5):
    for jj0 in range(ii+1,5):
     x,y=seedmaps[ii],seedmaps[jj0];seedpairs.append(dict(seed0=x['seed'],seed1=y['seed'],box_gradient_cosine=tensorco(x['box_gradient'],y['box_gradient']),content_gradient_cosine=tensorco(x['content_gradient'],y['content_gradient']),initial_Adam_delta_cosine=tensorco(x['initial_Adam_delta'],y['initial_Adam_delta']),linearized_output_delta_cosine=tensorco(x['linearized_dimensionless_full_output_delta'],y['linearized_dimensionless_full_output_delta'])))
   actual=chart(z['LayoutOPD']['path'][1]['boxes'].double()).flatten()-chart(z['LayoutOPD']['path'][0]['boxes'].double()).flatten();actual=actual/Dout;linear_actual=Jn@z['LayoutOPD']['rounds'][0]['state_step'].double().flatten()
   linearcheck=dict(actual_first_step_saved_state_used=True,dimensionless_output_cosine=tensorco(linear_actual,actual),relative_L2_prediction_error=float((linear_actual-actual).norm()/actual.norm()),linearized_delta_norm=float(linear_actual.norm()),actual_delta_norm=float(actual.norm()),note='First-order prediction along the saved actual finite Adam step; not an extra rollout or proof of global controllability.')
   jac=dict(query=key,public_id=case_public_ids[key],state=0,output='all corrected cx,cy,logw,logh',state_dimensions=[8,256],normalization='centres divided by fixed initial width/height; log sizes unchanged',same_offset_energy_fraction=float(energy[same].sum())/total,cross_offset_energy_fraction=float(energy[~same].sum())/total,cross_offset_max_abs=crossmax,own_anchor_frame_energy_fraction=float(energy[own].sum())/total,other_frame_same_offset_energy_fraction=float(energy[same&~own].sum())/total,observed_output_energy_fraction=float(energy[obs].sum())/total,unobserved_output_energy_fraction=float(energy[~obs].sum())/total,chart=spectrum(J),normalized_chart=spectrum(Jn),normalized_observed=spectrum(Jn.reshape(65,4,2048)[obs].reshape(-1,2048)),normalized_unobserved=spectrum(Jn.reshape(65,4,2048)[~obs].reshape(-1,2048)),per_frame_per_anchor_energy=energy.tolist(),seed_gradient_mapping=dict(source_CPU_audit_sha256=sampling_sha,seed_pairs=seedpairs,seed41_saved_content_gradient_relative_L2_error=seedmaps[0]['seed41_saved_content_gradient_relative_L2_error'],seed41_saved_content_gradient_cosine=seedmaps[0]['seed41_saved_content_gradient_cosine'],interpretation='Three mechanism cases only; Jacobian maps frozen initial box-score gradients to content, not all-HC sampling stability.'),saved_first_step_linearization=linearcheck,seconds=time.perf_counter()-t,interpretation='One exact local Jacobian under fixed cached encoder memory. Not global reachability, nonlinear optimization or GT-correctness evidence.')
   savept(key+'_state00_full_Jacobian.pt',dict(J=J,J_dimensionless=Jn,positions=pos,seed_gradient_maps=seedmaps,source_cache_sha256=sha(TRIAL/'student_cache'/f'{key}.pt'),GT_read=False));jacobians[key]=jac
   assert modulehash(rp.decoder)==decoder_hash;del rp,c,z,opt,v,J,Jn,Ja,jj,seedmaps;rp=None;release()
  atomic(OUT/'jacobian_summary.json',jacobians)
  assert modulehash(model.ground_decoder.decoder)==decoder_hash
  summary=dict(status='completed_sealed_before_any_GT_scoring',plan=describe(),script_sha256=script_sha,original_inference_fingerprint=fingerprint,matched_states=checks,Jacobian=jacobians,decoder_weights_before_after_sha256=decoder_hash,model_weights_frozen=True,GT_read=False,source_encoder_forwards=0,expert_forwards=0,load_seconds=load_seconds,total_seconds=time.perf_counter()-start,peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved())
  atomic(OUT/'summary.json',summary);outputs.append(dict(path='summary.json',sha256=sha(OUT/'summary.json')));outputs.append(dict(path='jacobian_summary.json',sha256=sha(OUT/'jacobian_summary.json')));atomic(OUT/'sealed.json',dict(status='sealed_before_offline_explanation',GT_read=False,script_sha256=script_sha,original_inference_fingerprint=fingerprint,files=outputs));atomic(OUT/'run.json',dict(status='completed_sealed',GT_read=False,total_seconds=summary['total_seconds'],script_sha256=script_sha));print(json.dumps(summary,indent=2))
 except Exception:
  atomic(OUT/'failure.json',dict(status='stopped_on_incompatibility_or_error',GT_read=False,error=traceback.format_exc(),partial_outputs_preserved=outputs,seconds=time.perf_counter()-start));raise
 finally:
  del rp,model;release();lock.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();main(a.run)
