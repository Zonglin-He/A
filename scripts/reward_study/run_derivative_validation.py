"""Fixed cached derivative audit of native inter-layer reference detach.
No permanent module/source/method change; no labels/encoder/teacher or updates.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,json,hashlib,gc,inspect,textwrap,types,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();T=ROOT/'outputs/layout_reward_dev';W=T/'workspace';O=T/'research/decoder_audit/derivative_validation';sys.path.insert(0,str(W));os.chdir(W)
from stvg.runtime import setup,lease,load_source
from stvg.state import Replay
from stvg.policy import chart

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,z):p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(z,indent=2)+'\n');os.replace(tmp,p)
def co(a,b):
 a=a.double().flatten();b=b.double().flatten();den=a.norm()*b.norm();return float(a@b/den)if den>0 else None

def comparison(pred,fd,scale=None):
 if scale is not None:pred=pred/scale;fd=fd/scale
 return dict(relative_L2_error=float((pred-fd).norm()/fd.norm()),cosine=co(pred,fd),predicted_norm=float(pred.norm()),FD_norm=float(fd.norm()),norm_ratio=float(pred.norm()/fd.norm()),max_abs_error=float((pred-fd).abs().max()))

def singular(a):
 ss=torch.linalg.svdvals(a.double());ee=ss.square();p=ee/ee.sum();return dict(Frobenius_norm=float(a.double().norm()),singular_values=ss.tolist(),energy_rank99=int((p.cumsum(0)<.99).sum()+1),energy_effective_rank=float(torch.exp(-(p[p>0]*p[p>0].log()).sum())),rank_rel1e6=int((ss>ss[0]*1e-6).sum()))

class KeptReplay(Replay):
 def values(self):
  z=super().values();self.last=z;return z

setup();torch.set_num_threads(1);assert read(ROOT/'outputs/formal/state.json')['status']=='paused_by_user';state_sha=sha(ROOT/'outputs/formal/state.json');lock=lease();start=time.perf_counter();model=rp=None;files=[];summary=[]
try:
 assert not (O/'sealed.json').exists(), 'Existing derivative audit immutable'
 fingerprint=read(T/'run.json')['fingerprint'];computed=hashlib.sha256((T/'config.json').read_bytes()+b''.join(p.read_bytes()for p in sorted((W/'stvg').glob('*.py')))+(W/'run_trial.py').read_bytes()).hexdigest();assert computed==fingerprint
 cfg=read(T/'config.json')['datasets']['hc2'];model=load_source(cfg);model.cpu();gc.collect();torch.cuda.empty_cache();model.ground_decoder.decoder.cuda().eval().requires_grad_(False)
 load_seconds=time.perf_counter()-start;source_file=Path(inspect.getsourcefile(model.ground_decoder.decoder.forward));source_file_sha=sha(source_file);body=textwrap.dedent(inspect.getsource(model.ground_decoder.decoder.forward));needle='pred_boxes = new_pred_boxes.detach()';assert body.count(needle)==1;changed=body.replace(needle,'pred_boxes = new_pred_boxes');assert changed.replace('pred_boxes = new_pred_boxes\n',needle+'\n')==body
 ns=dict(vars(sys.modules[model.ground_decoder.decoder.forward.__module__]));exec(compile(changed,'<diagnostic-in-memory-reference-full-gradient>','exec'),ns);fullforward=ns['forward'];Hs=[.01,.001,.0001,.00001]
 hc_development_rows=[json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l.strip()and json.loads(l)['dataset']=='hc2'];case_ordinals=[36,84,125];case_ids=[hc_development_rows[i-1]['id']for i in case_ordinals];case_public_ids=dict(zip(case_ids,('H%03d'%i for i in case_ordinals)))
 write(O/'run.json',dict(status='running',PID=os.getpid(),GT_read=False,queries=case_ids,directions=['saved_first_state_step_L2_normalized','fixed_random_seed20261021_L2_normalized'],h_values=Hs,source_file_sha256=source_file_sha,original_forward_sha256=hashlib.sha256(body.encode()).hexdigest(),diagnostic_forward_sha256=hashlib.sha256(changed.encode()).hexdigest(),permanent_source_changed=False))
 for key in case_ids:
  begin=time.perf_counter();cp=T/'student_cache'/f'{key}.pt';tp=T/'traces'/f'{key}.pt';assert sha(cp)==read(cp.with_suffix('.json'))['sha256'];assert sha(tp)==read(T/'predictions'/f'{key}.json')['trace_sha256'];c=torch.load(cp,map_location='cpu',weights_only=False);z=torch.load(tp,map_location='cpu',weights_only=False);oldJ=torch.load(T/'research/decoder_audit'/f'{key}_state00_full_Jacobian.pt',map_location='cpu',weights_only=False)['J'].double();rp=KeptReplay(model,c,c['Source'],c['E'],c['planned_positions'],lambda:None);base=rp.last['boxes'].detach().cpu();assert torch.equal(base,z['LayoutOPD']['path'][0]['boxes']);Dout=torch.cat((base[:,2:],torch.ones_like(base[:,2:])),dim=-1).double().flatten()
  dirs={};dd=z['LayoutOPD']['rounds'][0]['state_step'].double();dirs['saved_first_state_step']= (dd/dd.norm()).float();generator=torch.Generator(device='cpu').manual_seed(20261021);dd=torch.randn(8,256,generator=generator,dtype=torch.float32).double();dirs['fixed_random']=(dd/dd.norm()).float();fddata={};originalforward=rp.decoder.forward
  # All central differences use original unchanged forward and gradient-mode dispatch.
  for name,d in dirs.items():
   hdata=[]
   for h in Hs:
    vv=[];spacing=[]
    for sign in [1,-1]:
     with torch.no_grad():rp.state.copy_(d.to('cuda')*(sign*h))
     v=rp.values();yy=chart(v['boxes']).detach().cpu().flatten().double();native=v['native'].detach().cpu().float();ulp=(torch.nextafter(native,torch.full_like(native,float('inf')))-native).double();q=torch.cat((ulp[:,:2],ulp[:,2:]/native[:,2:].double()),dim=-1).flatten();vv.append(yy);spacing.append(q)
     del v
    fd=(vv[0]-vv[1])/(2*h);outquant=(spacing[0]+spacing[1])/(2*h)
    hdata.append(dict(h=h,FD=fd,output_one_ULP_quantization_floor_raw_L2=float(outquant.norm()),output_one_ULP_quantization_floor_dimensionless_L2=float((outquant/Dout).norm()),quantization_note='Conservative one-output-ULP-per-forward scale only, not a bound on all accumulated FP32 arithmetic error.'))
   fddata[name]=hdata
  with torch.no_grad():rp.state.zero_()
  # Change only this isolated decoder INSTANCE's backward graph; native source file intact.
  rp.decoder.forward=types.MethodType(fullforward,rp.decoder);v=rp.values();fullbase=v['boxes'].detach().cpu();assert torch.equal(fullbase,base), 'Removing reference detach changed forward values';yy=chart(v['boxes']).flatten();J=[]
  for ix in range(260):
   grad=torch.autograd.grad(yy[ix],rp.state,retain_graph=ix<259)[0];J.append(grad.detach().cpu().flatten())
  fullJ=torch.stack(J).double();assert torch.isfinite(fullJ).all();rp.decoder.forward=originalforward
  comparisons=[]
  for name,d in dirs.items():
   d=d.double().flatten();orig=oldJ@d;full=fullJ@d
   for hh in fddata[name]:
    fd=hh['FD'];comparisons.append(dict(direction=name,h=hh['h'],original_autograd_vs_FD=comparison(orig,fd),full_reference_gradient_vs_FD=comparison(full,fd),original_autograd_vs_FD_dimensionless=comparison(orig,fd,Dout),full_reference_gradient_vs_FD_dimensionless=comparison(full,fd,Dout),output_one_ULP_quantization_floor_raw_L2=hh['output_one_ULP_quantization_floor_raw_L2'],output_one_ULP_quantization_floor_dimensionless_L2=hh['output_one_ULP_quantization_floor_dimensionless_L2']))
  row=dict(query=key,public_id=case_public_ids[key],initial_original_boxes_bit_exact=True,full_reference_gradient_forward_bit_exact=True,physical_Replay_calls=rp.calls,native_offset_decoder_calls=2*rp.calls,seconds=time.perf_counter()-begin,original_Jacobian=singular(oldJ),full_reference_gradient_Jacobian=singular(fullJ),relative_Jacobian_change=float((fullJ-oldJ).norm()/fullJ.norm()),comparisons=comparisons)
  target=O/f'{key}.pt';torch.save(dict(originalJ=oldJ,full_reference_gradient_J=fullJ,Dout=Dout,directions=dirs,finite_differences=fddata,GT_read=False),target);files.append(dict(path=target.name,sha256=sha(target)));summary.append(row);write(O/'progress.json',dict(completed=len(summary),total=3,latest=row,GT_read=False));print(json.dumps(row),flush=True)
  del rp,c,z,v,J,fullJ,oldJ;rp=None;gc.collect();torch.cuda.empty_cache()
 assert sha(source_file)==source_file_sha;assert sha(ROOT/'outputs/formal/state.json')==state_sha;assert read(ROOT/'outputs/formal/state.json')['status']=='paused_by_user';assert all(not p.requires_grad and p.grad is None for p in model.parameters())
 result=dict(status='completed_and_sealed_before_interpretation',GT_read=False,source_encoder_forwards=0,expert_forwards=0,permanent_source_changed=False,registered_method_changed=False,formal_still_paused=True,formal_state_sha256=state_sha,source_file_sha256=source_file_sha,script_sha256=sha(__file__),load_seconds=load_seconds,seconds=time.perf_counter()-start,physical_replay_calls=sum(r['physical_Replay_calls']for r in summary),native_offset_decoder_calls=sum(r['native_offset_decoder_calls']for r in summary),central_difference_calls=48,original_baseline_calls=3,full_reference_gradient_baseline_calls=3,queries=summary,note='A diagnostic full backward graph for the same forward function, not an optimizer/method result. Output quantization scales do not bound all FP32 rounding.')
 write(O/'summary.json',result);files.append(dict(path='summary.json',sha256=sha(O/'summary.json')));write(O/'sealed.json',dict(status='sealed_before_explanation',GT_read=False,script_sha256=sha(__file__),source_file_sha256=source_file_sha,files=files));write(O/'run.json',dict(status='completed_sealed',GT_read=False,seconds=result['seconds']));print(json.dumps(result,indent=2))
except Exception:
 write(O/'failure.json',dict(status='failed',GT_read=False,error=traceback.format_exc(),partial_files_preserved=files,seconds=time.perf_counter()-start));raise
finally:
 del rp,model;gc.collect();torch.cuda.empty_cache();lock.close()
