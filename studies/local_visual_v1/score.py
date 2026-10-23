"""Read official labels only after all64 fixed paired outputs are sealed."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(Path(__file__).parent))
from run import D,P,B,ARMS,verify
from full import PROVIDER,sha,read,atomic,identity
from full_score import metric,cluster
import numpy as np,torch

def main():
 p=verify();s=read(D/'sealed.json');assert s['count']==64 and s['GT_read']is False and s['plan_sha256']==sha(D/'plan.json');assert not(D/'score_receipt.json').exists();torch.set_num_threads(1)
 for e in s['files'].values():
  assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256']
 atomic(D/'pre_GT_check.json',dict(all64_predictions_sealed_and_hashed=True,seal_sha256=sha(D/'sealed.json'),GPU_initialized=torch.cuda.is_initialized()))
 begin=time.perf_counter();domains={};checks=0;maxerr=0;old=read(B.parents[1]/'results/general_region_recheck_v3/report.json')
 for ds in ['hc2','vidstg']:
  label=PROVIDER/p['labels'][ds]['path'];assert sha(label)==p['labels'][ds]['sha256'];truth={x['id']:x for x in map(json.loads,label.read_text().splitlines())if x['dataset']==ds};rows=[];baseline={x['query']:x for x in old['domains'][ds]['rows']}
  for j,item in enumerate(x for x in p['items']if x['dataset']==ds):
   key=item['alias'];r=item['row'];gt=truth[r['id']];span=gt['interval'];boxes={int(k):v for k,v in gt['boxes'].items()};values={};fixed={}
   with np.load(D/s['files'][key]['path'],allow_pickle=False)as z:
    for a in ARMS:
     iv=z[a+'_interval'].tolist();values[a],err=metric(z[a],r,boxes,span,iv,z['positions8']);checks+=1;maxerr=max(maxerr,err)
     fixed[a],err=metric(z['FixedSpace'],r,boxes,span,iv,z['positions8']);checks+=1;maxerr=max(maxerr,err)
    for a,b in [('CurrentRegion','RegionKernel'),('NativeG6','NativeG6'),('Source','Source')]:
     for f in ['v','t','s']:assert abs(values[a][f]-baseline[key]['values'][b][f])<2e-14,(key,a,f)
   rows.append(dict(query=key,parent=f'p{j+1:03d}',duration_group='short_le5s'if(span[1]-span[0])/r['metadata']['fps']<=5 else'long_gt5s',values=values,fixed_space=fixed))
  assert len(rows)==32 and len(set(x['parent']for x in rows))==32
  means={a:{f:float(np.mean([x['values'][a][f]for x in rows if x['values'][a][f]is not None]))if any(x['values'][a][f]is not None for x in rows)else None for f in rows[0]['values'][a]}for a in ARMS}
  pairs=[('Native2D',b)for b in ['CurrentRegion','Source','NativeG6','Native3D','Fused2D','Reflected2D','WrongTime2D']]+[('Fused2D','CurrentRegion'),('Native3D','CurrentRegion')]
  paired={a+'-'+b:cluster(rows,a,b)for a,b in pairs};groups={g:dict(queries=sum(x['duration_group']==g for x in rows),delta_vs_CurrentRegion={a:{f:float(np.mean([100*(x['values'][a][f]-x['values']['CurrentRegion'][f])for x in rows if x['duration_group']==g]))for f in ['t','v']}for a in ARMS},delta_vs_Source={a:{f:float(np.mean([100*(x['values'][a][f]-x['values']['Source'][f])for x in rows if x['duration_group']==g]))for f in ['t','v']}for a in ARMS})for g in ['short_le5s','long_gt5s']}
  fixedrows=[{**x,'values':x['fixed_space']}for x in rows];fixedpaired=cluster(fixedrows,'Native2D','CurrentRegion')
  severe={a:{ref:{f:sum(x['values'][a][f]-x['values'][ref][f]<-.05 for x in rows)for f in ['t','v']}for ref in ['Source','NativeG6','CurrentRegion']}for a in ARMS}
  domains[ds]=dict(queries=32,means=means,paired=paired,groups=groups,severe=severe,fixed_space_paired=fixedpaired,rows=rows)
 out=dict(status='completed_scored',primary='Native2D',domains=domains,checks=checks,max_metric_error=maxerr,prediction_seconds=s['seconds'],scoring_seconds=time.perf_counter()-begin,counts=s['counts'],exposure=p['exposure'],automatic_method_promotion=False,current_full_frozen_method_unchanged=True,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'))
 atomic(P/'report.json',out);lines=['dataset\tquery\tmethod\tvIoU\ttIoU\tdelta_v_CurrentRegion_pp\tdelta_t_CurrentRegion_pp\tdelta_v_Source_pp\tdelta_t_Source_pp'];bad=[lines[0]]
 for ds,d in domains.items():
  for r in d['rows']:
   for a,v in r['values'].items():
    ref=r['values']['CurrentRegion'];src=r['values']['Source'];line=f"{ds}\t{r['query']}\t{a}\t{v['v']:.12g}\t{v['t']:.12g}\t{100*(v['v']-ref['v']):.12g}\t{100*(v['t']-ref['t']):.12g}\t{100*(v['v']-src['v']):.12g}\t{100*(v['t']-src['t']):.12g}";lines.append(line)
    if min(v['v']-ref['v'],v['t']-ref['t'])<-.05:bad.append(line)
 (P/'signed.tsv').write_text('\n'.join(lines)+'\n');(P/'severe.tsv').write_text('\n'.join(bad)+'\n')
 md=['# TA-STVG: replace fused motion feedback with native local appearance','','Fixed previously exposedHC32/Vid32, one candidate, no parameter selection. All64 outputs sealed before independentCPUofficiallabels. Original Source/E/Global6/eightobservations/space stayfixed.','', '| Direction | Representation | tIoU (%) | Complete vIoU (%) | Δt vs current(pp) | Δv vs current(pp) |','|---|---|---:|---:|---:|---:|']
 for ds,d in domains.items():
  for a in ['Source','NativeG6','CurrentRegion','Native2D','Native3D','Fused2D']:
   v=d['means'][a];ref=d['means']['CurrentRegion'];md.append(f"|{ds}|{a}|{100*v['t']:.4f}|{100*v['v']:.4f}|{100*(v['t']-ref['t']):+.4f}|{100*(v['v']-ref['v']):+.4f}|")
 md+=['','Native2D is original input_proj(ResNet appearance) grid BEFORE text/vision fusion. Native3D/Fused2D are explanatorycontrols, not candidates automatically promoted. CurrentRegion is original fused motion memory, restored bitwise. No new externalencoder/teacher/observation/training. All4views collected in the same one nativeSource forward/query; newSource64 recorded as research cost, not deployment latency.','', 'The generic algorithm does not need motion or a class vocabulary. This experiment evaluates one concrete TA pre-fusion2D choice; it does not prove every spatially-addressable representation will transfer or improve event recognition. OnlyTA tested; noMLLM/crossbackbone empiricalclaim.','', 'See report.json for all pairedparent-bootstrapCIs, group/boundary/length/severe/fixedspace values and signed.tsv/severe.tsv for every sign. No statisticalequivalence/adoption gate is invented after outcomes.']
 (P/'ANALYSIS.md').write_text('\n'.join(md)+'\n');assert not torch.cuda.is_initialized();atomic(D/'score_receipt.json',dict(report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),checks=checks,max_metric_error=maxerr,CUDA_initialized=False,**identity()));atomic(D/'state.json',dict(status='completed_scored',fingerprint=p['fingerprint']));print(json.dumps({ds:dict(paired=d['paired']['Native2D-CurrentRegion'],means={a:d['means'][a]for a in ['Native2D','CurrentRegion']},groups=d['groups'],severe=d['severe']['Native2D'])for ds,d in domains.items()},indent=2))
if __name__=='__main__':main()
