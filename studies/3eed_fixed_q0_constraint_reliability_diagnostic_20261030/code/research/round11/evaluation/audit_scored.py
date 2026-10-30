"""Independent scalar reduction audit of sealed R11 scores. No new IoU/GT IO.
Does not import scorer/metrics/discrimination; uses sealed scalar outcome rows.
"""
from pathlib import Path
import argparse,json,hashlib,math,collections
import numpy as np
from scipy.stats import spearmanr

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
class Compare:
 def __init__(self):self.n=0;self.max=0.;self.fail=[]
 def __call__(self,a,b,label):
  self.n+=1
  if a is None or b is None:
   if a is not b:self.fail.append({'field':label,'expected':a,'actual':b})
  elif isinstance(a,(int,float,np.number)) and isinstance(b,(int,float,np.number)):
   e=abs(float(a)-float(b));self.max=max(self.max,e)
   if not math.isfinite(e) or e>1e-11:self.fail.append({'field':label,'absolute_error':e})
  elif a!=b:self.fail.append({'field':label,'expected':a,'actual':b})
 def tree(self,a,b,label):
  if isinstance(a,dict):
   for k,v in a.items():
    if k not in b:self.fail.append({'field':label+'.'+k,'reason':'missing'})
    else:self.tree(v,b[k],label+'.'+k)
  elif isinstance(a,list):
   self(len(a),len(b),label+'.length')
   for j,(x,y) in enumerate(zip(a,b)):self.tree(x,y,label+'.'+str(j))
  else:self(a,b,label)

def avg(x):return float(np.mean(x)) if len(x) else None

def signstat(x,tol):
 x=np.array(x,float);p=x[x>tol];n=x[x< -tol]
 return {'n':len(x),'mean':avg(x),'sum':float(x.sum()),'positive_count':len(p),'negative_count':len(n),'tie_count':int((abs(x)<=tol).sum()),'positive_sum':float(p.sum()),'negative_sum':float(n.sum()),'gain_sum_exceeds_absolute_harm_sum':float(p.sum())>abs(float(n.sum())),'min':float(x.min()) if len(x) else None,'max':float(x.max()) if len(x) else None,'tolerance':tol}

def scenestat(rr,delta):
 scene={}
 for row,d in zip(rr,delta):scene.setdefault(row['scene'],[]).append(d)
 a=np.array([avg(scene[k]) for k in sorted(scene)])
 ci=np.quantile(np.random.default_rng(1101).choice(a,size=(2000,len(a))).mean(1),[.025,.975]).tolist() if len(a)>1 else None
 return {'n':len(delta),'defined_scene_clusters':len(a),'micro_mean':avg(delta),'equal_scene_mean':avg(a),'descriptive_scene_CI95':ci,'bootstrap_seed':1101,'bootstrap_draws':2000}

def discriminant(units,key='effect',tol=1e-8):
 candidate=[u for u in units if u['candidate']];nper=collections.Counter(u['case_id'] for u in candidate)
 category=lambda u:'beneficial' if u[key]>tol else 'harmful' if u[key]< -tol else 'zero_effect'
 defined=[u for u in candidate if u['score_defined']];undefined=[u for u in candidate if not u['score_defined']]
 binary=[u for u in defined if category(u)!='zero_effect'];positive=[u for u in binary if category(u)=='beneficial'];negative=[u for u in binary if category(u)=='harmful'];both=bool(positive and negative)
 w=lambda u:1/nper[u['case_id']]
 if binary:
  pos=math.fsum(w(u) for u in positive);total=math.fsum(w(u) for u in binary);prev=pos/total
 else:prev=None
 def weighted_rank_auc():
  groups=collections.defaultdict(lambda:[0.,0.])
  for u in binary:groups[u['score']][0 if category(u)=='beneficial' else 1]+=w(u)
  negbelow=0.;wins=0.
  for score,(p,n) in sorted(groups.items()):wins+=p*(negbelow+.5*n);negbelow+=n
  return wins/(math.fsum(w(u) for u in positive)*math.fsum(w(u) for u in negative))
 def ap(harmful=False):
  groups=collections.defaultdict(lambda:[0.,0.]);tot=0.
  for u in binary:
   ispos=(category(u)=='harmful') if harmful else (category(u)=='beneficial');score=-u['score'] if harmful else u['score'];groups[score][0 if ispos else 1]+=w(u)
   if ispos:tot+=w(u)
  tp=0.;seen=0.;out=0.
  for score,(p,n) in sorted(groups.items(),reverse=True):tp+=p;seen+=p+n;out+=(p/tot)*(tp/seen)
  return out
 ranking={'both_classes':both,'helpful_positive_AUROC_casebalanced':weighted_rank_auc() if both else None,'helpful_positive_AP_casebalanced':ap() if both else None,'harmful_positive_AP_negscore_casebalanced':ap(True) if both else None,'helpful_prevalence_casebalanced':prev,'harmful_prevalence_casebalanced':1-prev if prev is not None else None}
 if both:
  wins=sum((u['score']>v['score'])+.5*(u['score']==v['score']) for u in positive for v in negative);ranking['helpful_positive_AUROC_unweighted']=wins/(len(positive)*len(negative))
 else:ranking['helpful_positive_AUROC_unweighted']=None
 a=np.array([u['score'] for u in defined],float);b=np.array([u[key] for u in defined],float)
 spearman=None if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0 else float(spearmanr(a,b).statistic)
 return {'n_units_requested':len(units),'n_non_candidate':len(units)-len(candidate),'n_candidate':len(candidate),'candidate_cases':len(nper),'n_score_defined':len(defined),'n_score_undefined':len(undefined),'effect_counts_all_candidate':dict(collections.Counter(category(u) for u in candidate)),'effect_counts_defined':dict(collections.Counter(category(u) for u in defined)),'effect_counts_undefined':dict(collections.Counter(category(u) for u in undefined)),'ranking':ranking,'Spearman_defined_candidates_including_zeroeffect':{'value':spearman,'n':len(defined)}}

def risk(delta,n):
 x=np.asarray(delta,float);p=x[x>1e-8];bad=x[x< -1e-8]
 return {'K':len(x),'allcase_n':n,'beneficial_n':len(p),'harmful_n':len(bad),'zeroeffect_n':int((abs(x)<=1e-8).sum()),'harmful_rate_at_retained':len(bad)/len(x) if len(x) else None,'harmful_mass':math.fsum(max(-float(v),0.) for v in x),'beneficial_mass':math.fsum(max(float(v),0.) for v in x),'net_effect':math.fsum(map(float,x)),'retained_mean_effect':math.fsum(map(float,x))/len(x) if len(x) else None,'allcase_deployment_mean_gain_with_sourcefallback':math.fsum(map(float,x))/n if n else None,'allreject_reference_not_valid_success':not len(x)}

def ranked_selection(rr,control,coverage):
 def key(row):
  x=row['case_controls'][control];defined=True if control=='hash_random' else x['defined'];score=0. if control=='hash_random' else x['value']
  return (not defined,-score if defined else 0.,hashlib.sha256(('R11_samecoverage_hash_tie_v1|'+row['id']).encode()).hexdigest())
 return sorted(rr,key=key)[:math.ceil(len(rr)*coverage)]

def main(a):
 seal=read(a.scored/'SEALED.json')
 for name,h in seal['files'].items():
  p=(a.scored/name).resolve();assert p.is_relative_to(a.scored.resolve()) and sha(p)==h
 r=read(a.scored/'RESULTS_PRIVATE.json');scalar=read(a.scored/'SCALAR_SUMMARY.json')
 assert r['state']=='R11_reliability_scored_after_all_arm_seal' and r['n']==320
 assert scalar=={k:v for k,v in r.items() if k!='rows_PRIVATE'}
 rows=r['rows_PRIVATE'];assert len(rows)==320 and len({x['id'] for x in rows})==320
 C=Compare();mask={'all':lambda x:True,'original_G_changed':lambda x:x['case_unit']['candidate'],'weighted_union_changed':lambda x:any(x['arms'][k]['actual_changed'] for k in ('unconditional_full','weighted_constraints','uniform_weighted_constraints')),'RGB_term_changed':lambda x:x['unit_RGB_term']['candidate']}
 for tab in r['summary']:
  rr=[x for x in rows if x['platform']==tab['platform'] and mask[tab['population']](x)];arms=[x['arms'][tab['arm']] for x in rr];n=len(rr)
  expected={'n':n,'scene_clusters':len({x['scene'] for x in rr}),'mean_iou':avg([x['iou'] for x in arms]),'Acc25':sum(x['iou']>.25 for x in arms)/n if n else None,'Acc50':sum(x['iou']>.5 for x in arms)/n if n else None,'center_L2_m_mean':avg([x['coordinate_errors']['center_L2_m'] for x in arms]),'size_L2_m_mean':avg([x['coordinate_errors']['size_L2_m'] for x in arms]),'actual_changed_n':sum(x['actual_changed'] for x in arms),'actual_changed_axis_counts':[sum(x['actual_changed_axes'][j] for x in arms) for j in range(6)],'status_histogram':dict(collections.Counter(x['status'] for x in arms))};C.tree(expected,tab,'summary.'+tab['platform']+'.'+tab['arm']+'.'+tab['population'])
 for pair in r['paired_contrasts']:
  rr=[x for x in rows if x['platform']==pair['platform'] and mask[pair['population']](x)];arm,ref=pair['arm'],pair['reference'];d=[x['arms'][arm]['iou']-x['arms'][ref]['iou'] for x in rr];ce=[x['arms'][ref]['coordinate_errors']['center_L2_m']-x['arms'][arm]['coordinate_errors']['center_L2_m'] for x in rr];se=[x['arms'][ref]['coordinate_errors']['size_L2_m']-x['arms'][arm]['coordinate_errors']['size_L2_m'] for x in rr]
  C.tree({'n':len(rr),'IoU_gain':signstat(d,1e-8),'center_error_reduction_m':signstat(ce,1e-6),'size_error_reduction_m':signstat(se,1e-6),'IoU_scene':scenestat(rr,d),'center_scene':scenestat(rr,ce),'size_scene':scenestat(rr,se)},pair,'paired.'+arm+'.'+ref+'.'+pair['population'])
 for platform in ('drone','quad'):
  pp=[x for x in rows if x['platform']==platform];coord=[u for x in pp for u in x['units_coordinate_application']];rgb=[x['unit_RGB_term'] for x in pp];cases=[x['case_unit'] for x in pp];dr=r['discrimination'][platform]
  items={'coordinate_primary_IoU':(coord,'effect',1e-8),'coordinate_secondary_coordinate_error':(coord,'coordinate_error_reduction_m',1e-6),'RGB_primary_matched_IoU':(rgb,'effect',1e-8),'RGB_secondary_center_error':(rgb,'center_error_reduction_m',1e-6),'case_originalG_IoU':(cases,'effect',1e-8)}
  for label,(uu,key,tol) in items.items():C.tree(discriminant(uu,key,tol),dr[label],platform+'.'+label)
  for j in range(6):C.tree(discriminant([u for u in coord if u['coordinate']==j]),dr['coordinate_primary_by_axis'][str(j)],platform+'.axis'+str(j))
  for name,indices in [('center',range(3)),('size',range(3,6))]:C.tree(discriminant([u for u in coord if u['coordinate'] in indices]),dr['coordinate_primary_by_family'][name],platform+'.'+name)
  for control in dr['case_control_discrimination']:
   uu=[{**x['case_unit'],'score':x['case_controls'][control]['value'],'score_defined':x['case_controls'][control]['defined']} for x in pp];C.tree(discriminant(uu),dr['case_control_discrimination'][control],platform+'.control.'+control)
 for tab in r['coverage_risk_primary50']:
  pp=[x for x in rows if x['platform']==tab['platform']];selected=[x for x in pp if x['gate_selected'][tab['policy']]];d=[x['case_unit']['effect'] for x in selected]
  expected={'candidate_n':sum(x['case_unit']['candidate'] for x in pp),'candidate_defined_n':sum(x['case_unit']['candidate'] and x['case_unit']['score_defined'] for x in pp),'selected_n':len(selected),'selected_undefined_primary_reliability_n':sum(not x['case_unit']['score_defined'] for x in selected),'risk':risk(d,len(pp)),'scene':scenestat(pp,[x['case_unit']['effect'] if x['gate_selected'][tab['policy']] else 0. for x in pp])};C.tree(expected,tab,'coverage50.'+tab['platform']+'.'+tab['policy'])
 for platform in ('drone','quad'):
  pp=[x for x in rows if x['platform']==platform];cand=[x for x in pp if x['case_unit']['candidate']];dr=r['discrimination'][platform]
  for tab in dr['secondary_coverage_curves']:
   selected=ranked_selection(cand,tab['control'],tab['coverage']);effects=[x['case_unit']['effect'] for x in selected]
   undefined=sum(not x['case_controls'][tab['control']]['defined'] for x in selected) if tab['control']!='hash_random' else 0
   C.tree({'n_candidate':len(cand),'K':len(selected),'n_selected_undefined':undefined,'risk':risk(effects,len(pp))},tab,'curves.'+platform+'.'+tab['control']+'.'+str(tab['coverage']))
  ref=dr['random_reference_primary50']
  if not cand:C(ref['state'],'undefined_no_originalG_changed_candidates',platform+'.randomempty');continue
  observed=risk([x['case_unit']['effect'] for x in pp if x['gate_selected']['reliable_50']],len(pp));series=collections.defaultdict(list);k=math.ceil(len(cand)/2)
  for draw in range(1000):
   selected=sorted(cand,key=lambda x:hashlib.sha256(('R11_random_reference_v1|0|'+str(draw)+'|'+x['id']).encode()).hexdigest())[:k];z=risk([x['case_unit']['effect'] for x in selected],len(pp))
   for name in ('harmful_mass','beneficial_mass','net_effect','harmful_rate_at_retained','allcase_deployment_mean_gain_with_sourcefallback'):series[name].append(z[name])
  expected={}
  for name,values in series.items():
   v=np.array(values);o=observed[name];expected[name]={'random_mean':float(v.mean()),'random_min':float(v.min()),'random_max':float(v.max()),'random_quantiles_025_25_50_75_975':np.quantile(v,[.025,.25,.5,.75,.975]).tolist(),'reliable_observed':o,'fraction_random_less_or_equal_reliable':float((v<=o).mean()),'fraction_random_greater_or_equal_reliable':float((v>=o).mean())}
  C.tree(expected,ref['distribution'],platform+'.random_reference')
 out={'state':'PASS_R11_INDEPENDENT_SCALAR_REDUCTION_AUDIT' if not C.fail else 'FAIL_R11_INDEPENDENT_SCALAR_REDUCTION_AUDIT','n':320,'results_sha256':sha(a.scored/'RESULTS_PRIVATE.json'),'scored_seal_sha256':sha(a.scored/'SEALED.json'),'prediction_seal_sha256':seal['prediction_seal_sha256'],'release_sha256':seal['release_sha256'],'auditor_sha256':sha(Path(__file__)),'numeric_or_exact_checks':C.n,'max_absolute_numeric_difference':C.max,'failures':C.fail,'all_sealed_score_files_rehashed':True,'private_scalar_mirror_exact':True,'covered':'All11arms/matchedpopulations/strictthresholds/statuscounts/pairedsignedeffects/sceneCI/allunitandcasecontrol discrimination/sameK coverage-risk/curves/1000 randomreference; no new IoU or GTmetadata.','no_new_GT_metadata_model_or_GPU_reads':True,'few_scene_limit':'Only2Drone/3Quad scenes; descriptive, not independent coordinate/sample significance.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'state':out['state'],'output_sha256':sha(a.output),'checks':C.n,'max_difference':C.max}));assert not C.fail
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scored',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
