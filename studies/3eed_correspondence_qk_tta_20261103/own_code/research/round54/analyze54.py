"""Descriptive objective/decision-gap analysis after the frozen task score."""
import json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round54'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def stats(x):
 a=np.asarray(x,float);return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),'mean':float(a.mean()),'p90':float(np.quantile(a,.9)),'max':float(a.max())}
def main():
 start=time.perf_counter();assert read(H/'saved_scalar_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');rows=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');summary=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];arms=list(summary['balanced_mIoU_percent']);out=H/'analysis_attempt001';assert not out.exists();out.mkdir();metrics=[]
 for arm in arms:
  x={'arm':arm,'drone_mIoU_percent':summary['drone'][arm]['mIoU_percent'],'quad_mIoU_percent':summary['quad'][arm]['mIoU_percent'],'balanced_mIoU_percent':summary['balanced_mIoU_percent'][arm],'gain_vs_source_pp':summary['balanced_mIoU_percent'][arm]-summary['balanced_mIoU_percent']['source']}
  for ref in ['source','free_20','qk_3']:
   d=np.array([r['IoU'][arm]-r['IoU'][ref] for r in rows]);x['improved_vs_'+ref]=int((d>1e-10).sum());x['worsened_vs_'+ref]=int((d< -1e-10).sum());x['query_changes_vs_'+ref]=sum(r['queries'][arm]!=r['queries'][ref] for r in rows)
  for t,n in [(.25,'25'),(.5,'50')]:
   x['balanced_Acc'+n+'_percent']=sum(summary[p][arm]['Acc'+n+'_percent'] for p in ['drone','quad'])/2;x['rescue'+n]=sum(r['IoU']['source']<=t and r['IoU'][arm]>t for r in rows);x['break'+n]=sum(r['IoU']['source']>t and r['IoU'][arm]<=t for r in rows)
  metrics.append(x)
 dump(out/'ARM_METRICS.json',metrics);collect={a:{k:[] for k in ['loss_decrease','assignment_KL','parameter_delta_L2','initial_unknown_mass','final_unknown_mass','foreground_mass_retention','conditional_foreground_peak','last_gradient_norm','weighted_output_keep','weighted_RGB','weighted_geometry','weighted_graph','competitive_gap_fraction']} for a in arms if a not in ['source','zero_step']};byname={};rgbcontrast=[];zeroRGB=0
 for row in rows:
  sid=row['id_PRIVATE'];main=read(ROOT/'crossplatform/round54/adapted336_attempt001/cases_PRIVATE'/(sid+'.json'))['statistics']['trajectories'];strong=read(ROOT/'crossplatform/round54/free20_control336_attempt001/cases_PRIVATE'/(sid+'.json'))['details']
  with np.load(ROOT/'crossplatform/round54/predicted336_attempt001/samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as z:
   s0=z['source_scores'].astype(float);q0=int(s0.argmax());other=np.arange(256)!=q0;gap=s0[q0]-s0[other];assert (gap>0).all()
   for i,arm in enumerate(arms):
    if arm in ['source','zero_step']:continue
    if arm.endswith('_20'):r=strong[arm];diag=r['final_diagnostics'];terms=diag['loss_terms'];decrease=r['initial_loss']-r['final_loss']
    else:
     name,step=arm.rsplit('_',1);r=main[name];diag=r['steps'][int(step)-1];terms=r['final_terms'] if int(step)==3 else diag['loss_terms'];decrease=r['initial_loss']-r['final_loss'] if int(step)==3 else np.nan
    c=collect[arm]
    if np.isfinite(decrease):c['loss_decrease'].append(float(decrease))
    for key in ['assignment_KL','parameter_delta_L2']:c[key].append(diag[key])
    u=diag['unknown_mass'];u0=diag['source_unknown_mass'];c['initial_unknown_mass'].append(u0);c['final_unknown_mass'].append(u);c['foreground_mass_retention'].append((1-u)/max(1-u0,1e-12));c['conditional_foreground_peak'].append(diag['max_query_mass']/max(1-u,1e-12));c['last_gradient_norm'].append(diag['gradient_norm_before_clipping']);c['weighted_output_keep'].append(terms['keep']);c['weighted_RGB'].append(terms['RGB']);c['weighted_geometry'].append(.25*terms['geometry']);c['weighted_graph'].append(.1*terms['graph'])
    delta=z['scores'][i].astype(float)-s0;ratio=(delta[other]-delta[q0])/np.maximum(gap,1e-30);c['competitive_gap_fraction'].append(float(ratio.max()))
  with np.load(ROOT/'crossplatform/round54/observations336_attempt001/samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as z:
   cost=z['RGB_cost'];valid=z['allowed']&z['point_RGB_known'][None];n=valid.sum(0);good=n>=2;zeroRGB+=not np.count_nonzero(cost)
   if good.any():
    mean=(cost*valid).sum(0)/np.maximum(n,1);variance=((cost-mean)**2*valid).sum(0)/np.maximum(n,1);magnitude=(np.abs(cost)*valid).sum(0)/np.maximum(n,1);rgbcontrast.append(float(np.sqrt(variance[good]).mean()/max(float(magnitude[good].mean()),1e-12)))
 objective={a:{k:stats(v) for k,v in fields.items() if v} for a,fields in collect.items()}
 for a in objective:
  raw=collect[a];objective[a]['cases_with_lower_final_loss']=sum(x>0 for x in raw['loss_decrease']);objective[a]['candidate_gap_reached_cases']=sum(x>=1 for x in raw['competitive_gap_fraction']);objective[a]['foreground_mass_drops_below_tenth_source_cases']=sum(x<.1 for x in raw['foreground_mass_retention'])
 dump(out/'OBJECTIVE_AND_DECISION_GAP_DIAGNOSTICS.json',objective);dump(out/'OBSERVATION_CONTRAST.json',{'RGB_unary_zero_cases':zeroRGB,'local_query_RGB_cost_std_over_mean_abs':stats(rgbcontrast),'these_are_descriptive_not_identity_ground_truth':True})
 scenes=read(H/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json');aliases={pl:sorted({r['scene_PRIVATE'] for r in rows if r['platform']==pl}) for pl in ['drone','quad']};sc=[]
 for r in scenes:
  if r['scope']=='ALL336':sc.append({'platform':r['platform'],'scene_alias':('D' if r['platform']=='drone' else 'Q')+'_S'+str(aliases[r['platform']].index(r['scene_PRIVATE'])+1).zfill(2),'n':r['n'],'mIoU_percent':r['mIoU_percent']})
 dump(out/'ALL_SCENE_METRICS.json',sc);dump(out/'ANALYSIS_RECEIPT.json',{'state':'R54_POSTSEAL_DESCRIPTIVE_NO_NEW_METHOD_OR_FIT','n':336,'arms':13,'new_model_or_GT_or_optimizer_steps':0,'seconds':time.perf_counter()-start});print(json.dumps({'state':'ANALYZED','arms':13,'RGB_unary_zero_cases':zeroRGB}))
if __name__=='__main__':main()
