"""Independent locked-label CPU screen after all HC63 decisions are sealed."""
import os,sys,csv,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.singleview_budget.run import OUT,PUBLIC,validate,read,write,sha
from scripts.interval_pool.score import score_boxes,summarize,paired,signed

def score():
 torch.set_num_threads(1);p=validate();seal=read(OUT/'sealed.json');assert seal['queries']==63 and not seal['GT_read'];assert not(PUBLIC/'report.json').exists()
 for r in seal['files']:assert sha(OUT/r['path'])==r['sha256']and sha(OUT/r['posterior_path'])==r['posterior_sha256']
 assert sha(ROOT/p['labels']['path'])==p['labels']['sha256'];labels={g['id']:g for g in map(json.loads,(ROOT/p['labels']['path']).read_text().splitlines())}
 base=read(ROOT/'results/highres_selfhull_v1/report.json');old={r['query']:r for r in base['queries_anonymous']};rows=[];checks=0;error=0.
 for item,r in zip(p['items'],seal['files']):
  key=r['query'];z=np.load(OUT/r['path']);b=old[key];values={k:b['values'][v]for k,v in dict(Source='Source',Global6='Only224six',Strong='Fusion',Single448_0='SelfFusion0',Single448_1='SelfFusion1').items()};fixed={k:b['fixed_spatial'][v]for k,v in dict(Source='Source',Global6='Only224six',Strong='Fusion',Single448_0='SelfFusion0',Single448_1='SelfFusion1').items()}
  gt=labels[item['id']];truth={int(k):v for k,v in gt['boxes'].items()};span=z['Single320_interval'].tolist();values['Single320'],e=score_boxes(z['Single320'],item['record'],truth,span,gt['interval'],z['positions8'].tolist());error=max(error,e);checks+=1
  from scripts.highres_priority.run import SUP
  prior=read(SUP/'receipts'/f'{key}.json');zz=np.load(SUP/prior['prediction_path']);fixed['Single320'],e=score_boxes(zz['FixedSpatial'],item['record'],truth,span,gt['interval'],z['positions8'].tolist());error=max(error,e);checks+=1
  rows.append(dict(query=key,parent=gt['parent'],values=values,fixed_spatial=fixed))
 names=list(rows[0]['values']);pairs=[('Single320',a)for a in ['Strong','Global6','Single448_0','Single448_1','Source']]+[('Single448_0','Strong'),('Single448_1','Strong')];means=summarize(rows,names,'values');bs=read(OUT/'benchmark.json')['files'];cost={arm:dict(mean=float(np.mean([r['seconds'][arm]for r in bs])),median=float(np.median([r['seconds'][arm]for r in bs])),per_query=[dict(query=r['query'],seconds=r['seconds'][arm])for r in bs])for arm in p['benchmark_arms']};reduction=1-cost['320_0']['mean']/cost['448_01']['mean'];loss=100*(means['Strong']['m_vIoU']-means['Single320']['m_vIoU']);dt=100*(means['Single320']['m_tIoU']-means['Global6']['m_tIoU']);dv=100*(means['Single320']['m_vIoU']-means['Global6']['m_vIoU'])
 gate=dict(v_loss_vs_Strong_pp=loss,t_gain_vs_Global6_pp=dt,v_gain_vs_Global6_pp=dv,extra_branch_cost_reduction=reduction,checks=dict(v_loss_at_most1=loss<=1,t_positive=dt>0,v_positive=dv>0,cost_reduction_at_least40percent=reduction>=.4));gate['pass']=all(gate['checks'].values())
 result=dict(status='completed_screen',queries=63,exposure='Previously exposed HC63 development; fixed320 before its first predictions/labels, no resolution/weight sweep.',means=means,paired=paired(rows,pairs,'values'),fixed_spatial_means=summarize(rows,names,'fixed_spatial'),fixed_spatial_paired=paired(rows,pairs,'fixed_spatial'),cost=cost,cost_scope=p['timing'],gate=gate,proof=dict(official_dense_checks=checks,official_dense_max_error=error,seal_sha256=sha(OUT/'sealed.json'),plan_fingerprint=p['fingerprint'],labels_sha256=p['labels']['sha256'],GT_only_after_all63_sealed=True),queries_anonymous=[{k:v for k,v in r.items()if k!='parent'}for r in rows],production_changed=False)
 PUBLIC.mkdir(parents=True,exist_ok=True);write(PUBLIC/'report.json',result);ss=signed(rows,pairs,'H')
 for s in ss:s['query']=rows[int(s['query'][1:])-1]['query']
 with(PUBLIC/'signed_queries.tsv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(ss[0]),delimiter='\t');w.writeheader();w.writerows(ss)
 write(PUBLIC/'protocol.json',{k:v for k,v in p.items()if k not in ['items','assets','sources','labels','boundaries','config']});write(OUT/'decision.json',dict(gate=gate,next='fixed320 HC32/Vid32 plus448 rechecks'if gate['pass']else'End320; continue previously authorized448 HC32/Vid32 recheck. No288/352/384.'));state=read(OUT/'state.json');state['status']='scored_awaiting_publication';write(OUT/'state.json',state);print(json.dumps(dict(means=means,gate=gate,cost=cost),indent=2))
if __name__=='__main__':score()
