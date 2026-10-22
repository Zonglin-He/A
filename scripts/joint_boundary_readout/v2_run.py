"""Immutable CPU cached actual-boundary trial. Label-free predictions seal first."""
import os,sys,json,time,argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.temporal_signal_audit.extract import sha,read,write,arr
from scripts.tube_fusion.core import restore_geometry
from stvg.spatial import evidence_dp
from scripts.joint_boundary_readout.v2_core import candidate_pool,boundary_features,select
OUT=ROOT/'outputs/joint_boundary_readout_v2';PUBLIC=ROOT/'results/joint_boundary_readout_v2'
ARMS={'Hidden10':(.10,'hidden',True,False),'Joint10':(.10,'joint',True,False),'Permuted10':(.10,'joint',True,True),'Hidden25':(.25,'hidden',True,False),'Joint25':(.25,'joint',True,False),'Permuted25':(.25,'joint',True,True),'JointAll10':(.10,'joint',False,False),'JointAll25':(.25,'joint',False,False),'EvidenceOnly':(0,'evidence_joint',True,False),'HiddenOnly':(0,'evidence_hidden',True,False),'PermutedOnly':(0,'evidence_joint',True,True)}
NAMES=['Source','Baseline',*ARMS]
def sources():return {str(p.relative_to(ROOT)):sha(p)for p in sorted(Path(__file__).parent.glob('*.py'))}
def validate_plan(p):
 for k,v in p['sources'].items():assert sha(ROOT/k)==v,k
 assert sha(ROOT/'outputs/boxfixed_full_v1/plan.json')==p['formal_plan_sha256']
 formal=read(ROOT/'outputs/boxfixed_full_v1/plan.json');assert all(sha(ROOT/k)==v for k,v in formal['sources'].items())

def prepare():
 assert not(OUT/'plan.json').exists();prior=read(ROOT/'outputs/joint_boundary_readout_v1/sealed.json');dev=[1-.5*(r['base_expected_iou']+r['view_agreement'])for r in prior['files']if r['cohort']=='HC64'];threshold=float(np.median(dev));audit=read(ROOT/'outputs/temporal_signal_audit_v1/plan.json');seal=read(ROOT/'outputs/temporal_signal_audit_v1/sealed.json');items=[]
 for item,s in zip(audit['items'],seal['files']):
  c=item['cohort'];key=item['id'];base=ROOT/('outputs/interval_pool/development'if c=='HC64'else'outputs/interval_pool/recheck_'+item['dataset']);pack=base/'posteriors'/f'{key}.pt';pred=base/'predictions'/f'{key}.npz'
  signal=ROOT/'outputs/temporal_signal_audit_v1'/s['path'];assert sha(signal)==s['sha256']
  items.append(dict(item,signal=str(signal.relative_to(ROOT)),signal_sha256=sha(signal),pack=str(pack.relative_to(ROOT)),pack_sha256=sha(pack),old_prediction=str(pred.relative_to(ROOT)),old_prediction_sha256=sha(pred)))
 p=dict(schema='joint_boundary_readout_v2',prior_attempt='v1 retained; gate covered only3/128, not sufficient execution of correction test',ambiguity_threshold_from_unlabeled_HC64=threshold,prior_attempt_seal_sha256=sha(ROOT/'outputs/joint_boundary_readout_v1/sealed.json'),status='locked_before_new_GT_scores',items=items,labels=audit['labels'],sources=sources(),formal_plan_sha256=sha(ROOT/'outputs/boxfixed_full_v1/plan.json'),prior_signal_seal_sha256=sha(ROOT/'outputs/temporal_signal_audit_v1/sealed.json'),GT_read=False,
 protocol=dict(arms=ARMS,candidates='Exactly twelve per-layer/offset MBRs plus existing Global6 MBR, duplicate removal; no GT oracle online.',base='Current Global6 paired hull and original physical tIoU MBR; source native224 overall reference.',
 ambiguity='Uncertainty=1-.5*(best expected tIoU+mean12 view agreement). Threshold is HC64 unlabeled median, frozen identically for HC32/Vid32. Coverage is not correctness.',
 evidence='Hidden change empirical rank at two ends times positive target-ROI-enrichment entry/exit contrasts. Continuous physical-time weighted window means, each window min(interval_length/4,1 second). Scores in[0,1].',
 prior='Score=original expected tIoU(candidate)+lambda*evidence. lambda=.10/.25 only, one coverage-motivated revision, no third attempt or further sweep; same weight and candidate budget for hidden-only and permuted.',
 gate='Gated main arms retain exactly incumbent on nonambiguous inputs. Ungated .10/.25 arms detect gate blocking. EvidenceOnly/HiddenOnly/PermutedOnly diagnose signal ranking without the prior, not eligible for adoption.',
 exposure='All cohorts historically exposed. HC64 development selects at most one of two fixed strengths; HC32/Vid32 frozen parent-isolated development rechecks, not pristine tests.',
 selection='Choose best HC64 joint tIoU among .10/.25 passing: delta_t>=.0025,delta_full_v>0,delta_fixed_space_v>0,delta_t versus both matched hidden/permuted>0; no more than2 additional Source-relative severe v drops. If no passing strength, no new method.',
 recheck='If HC64 selects: both HC32/Vid32 delta_t>0 anddelta_full_v>=0, at most2 additional Source-relative severe v drops/domain; equal-domain stratified-bootstrap pooled delta_t lower95>0 and full_v lower95>-.005. No tuning on recheck.',
 evaluation='Every query retained; GT tIoU<.5/.7 wrong subsets only offline, crossed with label-free ambiguity. Report rescued/harmed correct cases, start/end errors, full and same-spatial vIoU, signed severe losses, candidate oracle as diagnostic only.',
 cost='one CPU thread nice15; existing signals and actual geometry/posteriors only; zero GPU/RGB/encoder/decoder/head/expert/update. Candidate construction, rerank, DP and cache IO separately timed. Not standalone end-to-end.'))
 validate_plan(p);write(OUT/'plan.json',p);print('locked128',flush=True)

@torch.no_grad()
def execute():
 torch.set_num_threads(1);os.nice(15);p=read(OUT/'plan.json');validate_plan(p);assert not(OUT/'sealed.json').exists();rows=[];begin=time.perf_counter()
 for j,item in enumerate(p['items']):
  key=item['anonymous'];start=time.perf_counter()
  for name in ['signal','pack','old_prediction','capture']:assert sha(ROOT/item[name])==item[name+'_sha256']
  cap=torch.load(ROOT/item['capture'],map_location='cpu',weights_only=False);pack=torch.load(ROOT/item['pack'],map_location='cpu',weights_only=False);z=np.load(ROOT/item['signal'],allow_pickle=False);old=np.load(ROOT/item['old_prediction'],allow_pickle=False);io=time.perf_counter()-start;ids=cap['frame_ids'];assert np.array_equal(ids,z['frame_ids']);assert pack['GT_read']is False
  tick=time.perf_counter();pool=candidate_pool(pack,ids);pool['ambiguous']=bool(1-.5*(pool['base_expected_iou']+pool['view_agreement'])>=p['ambiguity_threshold_from_unlabeled_HC64']);pooltime=time.perf_counter()-tick;t=np.asarray(ids)/item['fps'];ints=pool['intervals']/item['fps'];area=z['roi_mass']-z['roi_enrichment'];real=z['roi_enrichment'];perm=z['permuted_roi_mass']-area
  tick=time.perf_counter();f=boundary_features(ints,t,z['hidden_change'],real);fp=boundary_features(ints,t,z['hidden_change'],perm);featuretime=time.perf_counter()-tick
  reference='Global6'if item['cohort']=='HC64'else'Hull_MBR';source='Source'if item['cohort']=='HC64'else'Source_Native224';assert np.array_equal(pool['base_interval'],old[reference+'_interval'])
  geom=restore_geometry(cap['geometry']);values=dict(Source=arr(cap['Source']),Source_interval=old[source+'_interval'],Baseline=old[reference],Baseline_interval=np.asarray(pool['base_interval']),FixedSpatial=old[reference],frame_ids=np.asarray(ids),positions8=np.asarray(cap['planned_positions']),candidate_intervals=pool['intervals'],candidate_utilities=pool['utilities'],candidate_features=f,permuted_features=fp)
  assert np.array_equal(values['Source'],old[source]);costs={};selected={};saved_dp={tuple(pool['base_interval']):values['Baseline']}
  for name,(weight,kind,gated,permuted)in ARMS.items():
   tick=time.perf_counter();k=select(pool,fp if permuted else f,weight,kind,gated);reranktime=time.perf_counter()-tick;span=pool['intervals'][k];selected[name]=k;signature=tuple(span)
   tick=time.perf_counter()
   if signature not in saved_dp:saved_dp[signature]=arr(evidence_dp(geom.for_interval(span.tolist()))['final'])
   boxes=saved_dp[signature];dptime=time.perf_counter()-tick;values[name]=boxes;values[name+'_interval']=span
   costs[name]=dict(rerank_seconds=reranktime,actual_DP_seconds=dptime,DP_reused=signature in {tuple(values[a+'_interval'])for a in ['Baseline']+list(selected)[:-1]})
  # Restoring the original spatial readout is part of interface validation.
  restored=evidence_dp(geom.for_interval(pool['base_interval']))
  assert np.array_equal(arr(restored['final']),values['Baseline'])
  dest=OUT/'predictions'/f'{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();np.savez_compressed(dest,**values)
  receipt=dict(anonymous=key,cohort=item['cohort'],GT_read=False,path=str(dest.relative_to(OUT)),sha256=sha(dest),ambiguous=pool['ambiguous'],view_agreement=pool['view_agreement'],base_expected_iou=pool['base_expected_iou'],candidates=len(pool['intervals']),selected=selected,input_IO_seconds=io,pool_seconds=pooltime,feature_seconds=featuretime,costs=costs,seconds=time.perf_counter()-start,baseline_space_exact=True,zero_weight_baseline_exact=select(pool,f,0,gated=False)==0)
  rows.append(receipt);write(OUT/'progress.json',dict(completed=j+1,total=128,GT_read=False))
  if j%16==0:print(j+1,'/128',round(time.perf_counter()-begin,2),flush=True)
 validate_plan(p);assert not torch.cuda.is_initialized();write(OUT/'sealed.json',dict(status='all128_predictions_sealed_before_GT',plan_sha256=sha(OUT/'plan.json'),files=rows,GT_read=False,CUDA_initialized=False,seconds=time.perf_counter()-begin,network_calls=0,parameter_updates=0));print('sealed128',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','execute']);x=a.parse_args();globals()[x.action]()
