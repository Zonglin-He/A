"""Both fixed panels must seal before an independent CPU labels process."""
import os,sys,json,time
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.native_event_semantic_recheck_v1.run import *
import numpy as np,torch
from stvg.metrics import DenseTube,official,box_iou
from stvg.metrics_base import xyxy

def evaluate(boxes,row,gt,interval,positions,ds):
 truth={int(k):v for k,v in gt['boxes'].items()};rr=dict(input=row['metadata'],frame_ids=row['metadata']['frame_ids']);a=official(boxes,rr,truth,gt['interval'],interval,ds);b=DenseTube(boxes,rr,truth,gt['interval'],clip=(ds=='hc2')).score(interval);error=max(abs(a[k]-b[k])for k in a);assert error<2e-10
 ids=rr['frame_ids'];obs={ids[int(i)]for i in positions};pix=xyxy(boxes,row['metadata']['width'],row['metadata']['height']);pix=np.maximum(pix,0)if ds=='hc2'else pix
 for name,want in [('observed_s',True),('unobserved_s',False)]:
  values=[box_iou(pix[i],truth[f])for i,f in enumerate(ids)if f in truth and(f in obs)==want];a[name]=float(np.mean(values))if values else None
 fps=row['metadata']['fps'];span=gt['interval'];a.update(boundary_MAE_seconds=float((abs(interval[0]-span[0])+abs(interval[1]-span[1]))/(2*fps)),length_seconds=float((interval[1]-interval[0])/fps),zero_time_overlap=float(a['t']==0));return {k:float(v)if v is not None else None for k,v in a.items()},error

def main():
 torch.set_num_threads(1);p=verify();s=read(D/'sealed.json');assert s['count']==64 and not s['GT_read']and s['plan_sha256']==sha(D/'plan.json');assert not(D/'score_receipt.json').exists()
 for e in s['files'].values():assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256']
 put(D/'pre_GT_validation.json',dict(all64_both_domains_sealed=True,seal_sha256=sha(D/'sealed.json'),GT_read=False));reports={};checks=0;error=0
 for ds in ['hc2','vidstg']:
  lf=p['labels'][ds];assert sha(PROVIDER/lf['path'])==lf['sha256'];gt={x['id']:x for x in map(json.loads,(PROVIDER/lf['path']).read_text().splitlines())};rows=[];parents=[]
  for item in [x for x in p['items']if x['dataset']==ds]:
   key=item['alias'];truth=gt[item['key']];e=s['files'][key];values={};fixed={};parents.append(item['parent'])
   with np.load(D/e['path'],allow_pickle=False)as z:
    for a in ARMS+['Source']:
     values[a],err=evaluate(z[a],item['row'],truth,z[a+'_interval'].tolist(),z['positions8'],ds);error=max(error,err);checks+=1;fixed[a],err=evaluate(z['NativeG6'],item['row'],truth,z[a+'_interval'].tolist(),z['positions8'],ds);error=max(error,err);checks+=1
    movement={a:(z[a+'_interval']-z['NativeG6_interval']).tolist()for a in ARMS}
   rows.append(dict(query=key,parent=f'p{len(rows)+1:03d}',duration_group='short_le5s'if(truth['interval'][1]-truth['interval'][0])/item['row']['metadata']['fps']<=5 else'long_gt5s',values=values,fixed_space=fixed,movement=movement));put(D/'score_progress.json',dict(dataset=ds,completed=len(rows),total=32))
  assert len(set(parents))==32;put(D/(ds+'_parents.json'),dict(parents=parents));ix=np.random.default_rng(20261022).integers(0,32,(10000,32));means={a:{m:float(np.mean([r['values'][a][m]for r in rows if r['values'][a][m]is not None]))for m in rows[0]['values'][a]}for a in ARMS+['Source']};paired={}
  for a,b in [(a,'NativeG6')for a in ARMS[1:]]+[('SemanticPrototypeLoop2',a)for a in ARMS[2:]+['Source']]:
   mm={}
   for m in ['v','t','unobserved_s','boundary_MAE_seconds']:
    delta=np.array([r['values'][a][m]-r['values'][b][m]for r in rows])*(1 if m.endswith('seconds')else 100);benefit=delta*(-1 if m.endswith('seconds')else 1);win=int((benefit>1e-10).sum());harm=int((benefit< -1e-10).sum());mm[m]=dict(mean=float(delta.mean()),CI95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),improved=win,harmed=harm,unchanged=32-win-harm,severe_below_minus5pp=int((delta< -5).sum())if not m.endswith('seconds')else None)
   paired[a+'-'+b]=mm
  groups={g:dict(count=sum(x['duration_group']==g for x in rows),time_gain_vs_Source={a:float(np.mean([100*(r['values'][a]['t']-r['values']['Source']['t'])for r in rows if r['duration_group']==g]))for a in ARMS})for g in ['short_le5s','long_gt5s']};a='SemanticPrototypeLoop2';passed=all(paired[a+'-'+b][m]['mean']>0 for b in ['NativeG6','Source']for m in ['v','t'])and all(groups[g]['time_gain_vs_Source'][a]>0 for g in groups)and paired[a+'-NativeG6']['t']['improved']>paired[a+'-NativeG6']['t']['harmed']and all(paired[a+'-'+b]['t']['mean']>0 for b in ['RandomProjectionLoop2','ReversedSemanticLoop2','SharpenOriginal']);reports[ds]=dict(queries=32,parents=32,means=means,paired=paired,groups=groups,rows=rows,basic_development_screen=passed)
 report=dict(status='completed_scored_exposed_fixed_recheck',domains=reports,all64_both_domains_seal_before_GT=True,checks=checks,max_metric_error=error,model_calls=s['counts'],seconds=s['seconds'],passed_both_domains=all(v['basic_development_screen']for v in reports.values()),promoted=False,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'));put(P/'report.json',report)
 header='dataset\tquery\tmethod\tvIoU\ttIoU\tdelta_v_G6_pp\tdelta_t_G6_pp\tboundary_MAE_seconds';lines=[header];bad=[header]
 for ds,v in reports.items():
  for row in v['rows']:
   for a,x in row['values'].items():
    b=row['values']['NativeG6'];line=f"{ds}\t{row['query']}\t{a}\t{x['v']:.12g}\t{x['t']:.12g}\t{100*(x['v']-b['v']):.12g}\t{100*(x['t']-b['t']):.12g}\t{x['boundary_MAE_seconds']:.12g}";lines.append(line)
    if min(x['v']-b['v'],x['t']-b['t'])<-.05:bad.append(line)
 (P/'signed.tsv').write_text('\n'.join(lines)+'\n');(P/'severe_declines.tsv').write_text('\n'.join(bad)+'\n');put(D/'score_receipt.json',dict(report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),CUDA_initialized=torch.cuda.is_initialized(),checks=checks,max_error=error));put(D/'state.json',dict(status='completed_scored',fingerprint=p['fingerprint']));print(json.dumps({ds:dict(pass_=v['basic_development_screen'],primary=v['paired']['SemanticPrototypeLoop2-NativeG6'],groups=v['groups'])for ds,v in reports.items()},indent=2))
if __name__=='__main__':main()
