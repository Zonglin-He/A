"""New sealed decisions are scored independently; teacher audit is separate."""
import os,sys,time
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.native_event_joint_caption_v1.run import R,D,P,ARMS,sha,read,put,verify,records
from research.boundary_feedback_v1.score import evaluate
import torch,numpy as np

def main():
 torch.set_num_threads(1);p=verify();s=read(D/'sealed.json');assert s['count']==64 and not s['GT_read']and s['plan_sha256']==sha(D/'plan.json');assert not(D/'score_receipt.json').exists()
 for e in s['files'].values():assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256']
 put(D/'pre_GT_validation.json',dict(count=64,all_hashes_valid=True,seal_sha256=sha(D/'sealed.json'),GT_read=False))
 lab=R/'data/evaluation/development_labels.jsonl';assert sha(lab)==p['labels_sha256'];gt={x['id']:x for x in map(__import__('json').loads,lab.read_text().splitlines())if x['dataset']=='hc2'};prior={r['query']:r for r in read(R/'results/joint_event_hc64_v3/report.json')['rows']};rows=[];checks=0;error=0.;parents=[]
 for row in records():
  k=row['id'];truth=gt[k];e=s['files'][k];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');values={};fixed={};parents.append(str(truth.get('parent',row['metadata']['source'])))
  with np.load(D/e['path'],allow_pickle=False)as z:
   for a in ARMS+['Source']:
    values[a],err=evaluate(z[a],row,truth,z[a+'_interval'].tolist(),z['positions8']);error=max(error,err);checks+=1
    fixed[a],err=evaluate(z['NativeG6'],row,truth,z[a+'_interval'].tolist(),z['positions8']);error=max(error,err);checks+=1
   assert values['NativeG6']==prior[k]['values']['NativeG6']and values['Source']==prior[k]['values']['Source']
   quality={};movement={a:(z[a+'_interval']-z['NativeG6_interval']).tolist()for a in ARMS}
  rows.append(dict(query=k,parent=f'p{len(rows)+1:03d}',supported=bool(e['supported']),duration_group='short_le5s'if(truth['interval'][1]-truth['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values,fixed_space=fixed,decision={a:dict(accept=movement[a]!=[0,0])for a in ARMS[1:]},anchor_auc=quality,movement=movement))
  put(D/'score_progress.json',dict(completed=len(rows),total=64))
 assert len(set(parents))==64;put(D/'parent_membership.json',dict(parents=parents));ix=np.random.default_rng(20261022).integers(0,64,(10000,64));arms=ARMS+['Source'];metrics=list(rows[0]['values']['NativeG6']);means={a:{m:float(np.mean([r['values'][a][m]for r in rows if r['values'][a][m]is not None]))for m in metrics}for a in arms};paired={}
 pairs=[(a,'NativeG6')for a in ARMS[1:]]+[('JointCaptionFusion',a)for a in ['DirectJoint','ImplicitSubjectFusion','PredicateBindingFusion','ReverseJointFeedback','SharpenOriginal','OldClauseFusion','OldSelfHull','NoComposition','Source']]
 for a,b in pairs:
  values={}
  for m in ['v','t','unobserved_s','boundary_MAE_seconds']:
   scale=1 if m.endswith('seconds')else 100;delta=np.array([r['values'][a][m]-r['values'][b][m]for r in rows])*scale;direction=-1 if scale==1 else 1;benefit=direction*delta;win=int((benefit>1e-10).sum());harm=int((benefit< -1e-10).sum())
   values[m]=dict(mean=float(delta.mean()),CI95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),improved=win,harmed=harm,unchanged=64-win-harm,severe_below_minus5pp=int((delta< -5).sum())if scale==100 else None,harmed_fraction_all=harm/64,harmed_fraction_changed=harm/(win+harm)if win+harm else None)
  paired[a+'-'+b]=values
 groups={g:{a:{m:float(np.mean([100*(r['values'][a][m]-r['values']['NativeG6'][m])for r in rows if r['duration_group']==g]))for m in['v','t']}for a in ARMS[1:]}for g in ['short_le5s','long_gt5s']}
 report=dict(status='completed_scored_development',queries=64,parents=64,means=means,paired=paired,groups_vs_G6=groups,rows=rows,anchor_auc={a:float(np.mean([r['anchor_auc'][a]for r in rows if r['anchor_auc'][a]is not None]))for a in []},changed={a:sum(r['decision'][a]['accept']for r in rows)for a in ARMS[1:]},fixed_space_gain={a:float(np.mean([100*(r['fixed_space'][a]['v']-r['fixed_space']['NativeG6']['v'])for r in rows]))for a in ARMS[1:]},checks=checks,max_metric_error=error,mean_feedback_seconds=float(np.mean([e['seconds']for e in s['files'].values()])),all64_seal_before_GT=True,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),GT_in_inference=False,model_calls=s['counts'],full_status='paused_by_user')
 put(P/'report.json',report);header='query\tmethod\tvIoU\ttIoU\tunobserved_sIoU\tdelta_v_G6_pp\tdelta_t_G6_pp\tboundary_MAE_seconds';lines=[header];severe=[header]
 for r in rows:
  for a in arms:
   v=r['values'][a];base=r['values']['NativeG6'];line=f"{r['query']}\t{a}\t{v['v']:.12g}\t{v['t']:.12g}\t{v['unobserved_s']:.12g}\t{100*(v['v']-base['v']):.12g}\t{100*(v['t']-base['t']):.12g}\t{v['boundary_MAE_seconds']:.12g}";lines.append(line)
   if min(v['v']-base['v'],v['t']-base['t'])<-.05:severe.append(line)
 (P/'signed.tsv').write_text('\n'.join(lines)+'\n');(P/'severe_declines.tsv').write_text('\n'.join(severe)+'\n');put(D/'score_receipt.json',dict(checks=checks,max_error=error,report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),CUDA_initialized=torch.cuda.is_initialized()));put(D/'state.json',dict(status='completed_scored',fingerprint=p['fingerprint']));print(__import__('json').dumps({k:v for k,v in report.items()if k not in ['rows']},indent=2))
if __name__=='__main__':main()
