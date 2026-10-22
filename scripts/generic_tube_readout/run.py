"""Cached adapter to the generic box/posterior readout; no internal-head input."""
import os,sys,json,time,argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.temporal_signal_audit.extract import read,write,sha,arr
from scripts.tube_fusion.core import restore_geometry
from stvg.spatial import evidence_dp
from scripts.generic_tube_readout.core import consensus,expected_overlap,select
OUT=ROOT/'outputs/generic_tube_readout_v1';PUBLIC=ROOT/'results/generic_tube_readout_v1'
ARMS=['ExpectedTube','FixedTrackTube','MeanQuality','ShiftedQuality','PriorMatched','RandomMatched','OldJoint10'];NAMES=['Source','Baseline',*ARMS]
def validate(p):
 for k,v in p['sources'].items():assert sha(ROOT/k)==v
 formal=read(ROOT/'outputs/boxfixed_full_v1/plan.json');assert all(sha(ROOT/k)==v for k,v in formal['sources'].items());assert sha(ROOT/'outputs/boxfixed_full_v1/plan.json')==p['formal_plan_sha256']
def prepare():
 assert not(OUT/'plan.json').exists();old=read(ROOT/'outputs/joint_boundary_readout_v2/plan.json');items=[]
 for i in old['items']:
  path=ROOT/'outputs/joint_boundary_readout_v2/predictions'/f"{i['anonymous']}.npz";items.append(dict(i,old_trial_prediction=str(path.relative_to(ROOT)),old_trial_prediction_sha256=sha(path)))
 p=dict(schema='generic_expected_tube_readout_v1',items=items,labels=old['labels'],sources={str(p.relative_to(ROOT)):sha(p)for p in Path(__file__).parent.glob('*.py')},formal_plan_sha256=sha(ROOT/'outputs/boxfixed_full_v1/plan.json'),GT_read=False,
 protocol=dict(user_constraint='No actionness or architecture-specific auxiliary signal. Main algorithm inputs only interval posterior, box tracks and spatial box hypotheses/validity; generic core uses only NumPy.',
 question='Can local spatial uncertainty improve expected tube-overlap decisions without pretending to predict event semantics?',candidate='Same13 existing interval decisions, de-duplicated. For each candidate execute the same frozen BoxFixed spatial DP; algorithm can accept any interval/trajectory proposal bank.',
 quality='Per node uniform mean box IoU of the candidate track with available ERef/forward/backward complete-box hypotheses. Consensus is a spatial-quality proxy, not calibrated true IoU, identity or event probability; duplicate/correlated hypotheses are not assumed independent.',
 score='Sum over native temporal hypotheses q(s,e) of physical integral of candidate spatial quality on candidate intersection hypothesis, divided by temporal union. No reward weights or temperature.',
 controls='Constant quality1 exactly recovers old temporal utility/incumbent; fixed-baseline track isolates temporal changes; per-candidate mean quality removes alignment; fixed half-video quality shift preserves histogram; same-query prior/random controls match intervention queries.',
 exposure='All128 have repeated development exposure. Earlier failure cases inform design, so no untouched confirmation claim.',budget='Single fixed generic structural trial; no parameter search, new backbone/head/decoder/expert/GPU/updates. CPU one thread/nice15.',
 adoption='HC64 t>=0/fullv>0/fixedv>0 and ExpectedTube better t than ShiftedQuality; severe Source-relative v losses<=baseline+2. Both32 groups t>=0 and fullv>0; equal-domain pooled v lower95>0. No production switch during active full.'))
 validate(p);write(OUT/'plan.json',p)
@torch.no_grad()
def execute():
 torch.set_num_threads(1);os.nice(15);p=read(OUT/'plan.json');validate(p);assert not(OUT/'sealed.json').exists();rows=[];begin=time.perf_counter();maxerr=0.
 for j,item in enumerate(p['items']):
  start=time.perf_counter()
  for k in ['capture','pack','old_trial_prediction']:assert sha(ROOT/item[k])==item[k+'_sha256']
  cap=torch.load(ROOT/item['capture'],map_location='cpu',weights_only=False);pack=torch.load(ROOT/item['pack'],map_location='cpu',weights_only=False);old=np.load(ROOT/item['old_trial_prediction']);io=time.perf_counter()-start;geom=restore_geometry(cap['geometry']);ids=old['frame_ids'];intervals=old['candidate_intervals'];u=old['candidate_utilities'];hyp=arr(geom.boxes);valid=arr(geom.valid);tracks=[];qualities=[];dpsecs=0.;scoresecs=0.;scores={k:[]for k in ['ExpectedTube','FixedTrackTube','MeanQuality','ShiftedQuality']};fixedquality=consensus(old['Baseline'],hyp,valid)
  for n,span in enumerate(intervals):
   tick=time.perf_counter();track=arr(evidence_dp(geom.for_interval(span.tolist()))['final']);dpsecs+=time.perf_counter()-tick
   if n==0:assert np.array_equal(track,old['Baseline'])
   tick=time.perf_counter();quality=consensus(track,hyp,valid);tracks.append(track);qualities.append(quality)
   controls=dict(ExpectedTube=quality,FixedTrackTube=fixedquality,MeanQuality=np.full(len(ids),quality.mean()),ShiftedQuality=np.roll(quality,len(ids)//2))
   for name,q in controls.items():scores[name].append(expected_overlap(ids,pack['hull'],span,q))
   restored=expected_overlap(ids,pack['hull'],span,np.ones(len(ids)));maxerr=max(maxerr,abs(restored-u[n]));assert abs(restored-u[n])<2e-12;scoresecs+=time.perf_counter()-tick
  selected={a:select(v,u)for a,v in scores.items()};active=selected['ExpectedTube']!=0;other=np.arange(1,len(u));selected['PriorMatched']=int(other[np.argmax(u[1:])])if active else 0;selected['RandomMatched']=int(np.random.default_rng(20261022+j).choice(other))if active else 0
  values={k:old[k]for k in ['Source','Source_interval','Baseline','Baseline_interval','FixedSpatial','frame_ids','positions8','candidate_intervals','candidate_utilities']};values.update(OldJoint10=old['Joint10'],OldJoint10_interval=old['Joint10_interval'],qualities=np.stack(qualities),**{'score_'+k:np.asarray(v)for k,v in scores.items()})
  for arm,k in selected.items():values[arm]=tracks[k];values[arm+'_interval']=intervals[k]
  dest=OUT/'predictions'/f"{item['anonymous']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();np.savez_compressed(dest,**values);rows.append(dict(query=item['anonymous'],cohort=item['cohort'],path=str(dest.relative_to(OUT)),sha256=sha(dest),selected=selected,GT_read=False,input_IO_seconds=io,candidate_spatial_DP_seconds=dpsecs,expected_overlap_seconds=scoresecs))
  write(OUT/'progress.json',dict(completed=j+1,total=128,GT_read=False))
 validate(p);assert not torch.cuda.is_initialized();write(OUT/'sealed.json',dict(files=rows,GT_read=False,plan_sha256=sha(OUT/'plan.json'),constant_quality_temporal_utility_max_error=maxerr,network_calls=0,GPU_calls=0,parameter_updates=0,seconds=time.perf_counter()-begin));print('sealed128',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','execute']);a=p.parse_args();globals()[a.action]()
