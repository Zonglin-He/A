"""Pure R11 reliability discrimination/coverage primitives; no file/data/GT IO.
A score is heuristic, not a calibrated probability. Highscore direction locked.
"""
import collections,hashlib,math
import numpy as np

def average_ranks(values):
 x=np.asarray(values,float);order=np.argsort(x,kind='stable');r=np.empty(len(x),float);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  r[order[i:j]]=(i+j-1)/2+1;i=j
 return r

def spearman(scores,effects):
 x=np.asarray(scores,float);y=np.asarray(effects,float)
 if len(x)<3:return {'value':None,'reason':'fewer_than3_defined_units','n':len(x)}
 if np.ptp(x)==0:return {'value':None,'reason':'constant_score','n':len(x)}
 if np.ptp(y)==0:return {'value':None,'reason':'constant_effect','n':len(x)}
 a=average_ranks(x);b=average_ranks(y);a-=a.mean();b-=b.mean();return {'value':float((a@b)/np.sqrt((a@a)*(b@b))),'reason':None,'n':len(x),'case_scene_dependence_not_an_independent_unit_test':True}

def weighted_auc(scores,labels,weights):
 s=np.asarray(scores,float);y=np.asarray(labels,bool);w=np.asarray(weights,float);positive=np.flatnonzero(y);negative=np.flatnonzero(~y)
 if not len(positive) or not len(negative):return None
 win=(s[positive,None]>s[None,negative]).astype(float)+.5*(s[positive,None]==s[None,negative]);ww=w[positive,None]*w[None,negative]
 return float((win*ww).sum()/(w[positive].sum()*w[negative].sum()))

def weighted_ap(scores,labels,weights):
 s=np.asarray(scores,float);y=np.asarray(labels,bool);w=np.asarray(weights,float)
 if not y.any() or y.all():return None
 order=np.argsort(-s,kind='stable');total_pos=w[y].sum();tp=0.;mass=0.;ap=0.;i=0
 while i<len(order):
  j=i+1
  while j<len(order) and s[order[j]]==s[order[i]]:j+=1
  block=order[i:j];newtp=w[block][y[block]].sum();tp+=newtp;mass+=w[block].sum();ap+=newtp/total_pos*(tp/mass);i=j
 return float(ap)

def assay(units,effect_key='effect',effect_tolerance=1e-8):
 """Each unit: case_id, candidate,score_defined,score, effect postseal.
 Weights1/n_candidateunits in that case, BEFORE score/effect availability filtering.
 """
 candidate=[r for r in units if r['candidate']];counts=collections.Counter(r['case_id'] for r in candidate);weights={id(r):1/counts[r['case_id']] for r in candidate}
 def category(r):
  d=float(r[effect_key]);assert math.isfinite(d)
  return 'beneficial' if d>effect_tolerance else 'harmful' if d< -effect_tolerance else 'zero_effect'
 for r in candidate:
  if r['score_defined']:assert math.isfinite(float(r['score']))
 allcounts=collections.Counter(category(r) for r in candidate);defined=[r for r in candidate if r['score_defined']];undefined=[r for r in candidate if not r['score_defined']];dc=collections.Counter(category(r) for r in defined);uc=collections.Counter(category(r) for r in undefined)
 binary=[r for r in defined if category(r)!='zero_effect'];scores=np.array([r['score'] for r in binary],float);labels=np.array([category(r)=='beneficial' for r in binary],bool);w=np.array([weights[id(r)] for r in binary]);both=bool(len(binary) and labels.any() and (~labels).any());prevalence=float(w[labels].sum()/w.sum()) if len(w) else None
 ranking={'both_classes':both,'undefined_reason':None if both else 'missing_beneficial_or_harmful_class_after_definedscore_filter','helpful_positive_AUROC_casebalanced':weighted_auc(scores,labels,w) if both else None,'helpful_positive_AP_casebalanced':weighted_ap(scores,labels,w) if both else None,'harmful_positive_AP_negscore_casebalanced':weighted_ap(-scores,~labels,w) if both else None,'helpful_prevalence_casebalanced':prevalence,'harmful_prevalence_casebalanced':1-prevalence if prevalence is not None else None,'helpful_positive_AUROC_unweighted':weighted_auc(scores,labels,np.ones(len(w))) if both else None}
 return {'n_units_requested':len(units),'n_non_candidate':len(units)-len(candidate),'n_candidate':len(candidate),'candidate_cases':len(counts),'n_score_defined':len(defined),'n_score_undefined':len(undefined),'effect_counts_all_candidate':dict(allcounts),'effect_counts_defined':dict(dc),'effect_counts_undefined':dict(uc),'effect_tolerance':effect_tolerance,'weights':'1/n_candidateunits_per_case before score/effectfilter','ranking':ranking,'Spearman_defined_candidates_including_zeroeffect':spearman([r['score'] for r in defined],[r[effect_key] for r in defined]),'not_a_probability_calibration_or_independent_unit_significance_claim':True}

def deterministic_select(candidates,coverage,policy_score,defined,*,salt,platform):
 """OriginalGchanged caseIDs only; NA last without deleting candidates.
 Returns exactceil(N*coverage); coverage0 source-reference only, not a success.
 Hash tie matches the literal producer salt|id; platform partitions eligibility only.
 """
 ids=list(candidates);assert len(ids)==len(set(ids)) and 0<=coverage<=1
 for ident in ids:
  if defined[ident]:assert math.isfinite(float(policy_score[ident]))
 key=lambda ident:(0 if defined[ident] else 1,-float(policy_score[ident]) if defined[ident] else 0,hashlib.sha256((salt+'|'+ident).encode()).hexdigest())
 order=sorted(ids,key=key);k=math.ceil(len(ids)*coverage);return {'selected':order[:k],'ordered':order,'n_candidate':len(ids),'K':k,'requested_coverage':coverage,'actual_coverage':k/len(ids) if ids else None,'n_selected_undefined':sum(not defined[x] for x in order[:k]),'NA_last_no_candidate_removal':True}

def risk_summary(retained_effects,allcase_n,tolerance=1e-8):
 d=list(map(float,retained_effects));assert allcase_n>=len(d) and all(math.isfinite(x) for x in d);positive=[x for x in d if x>tolerance];negative=[x for x in d if x< -tolerance]
 return {'K':len(d),'allcase_n':allcase_n,'beneficial_n':len(positive),'harmful_n':len(negative),'zeroeffect_n':sum(abs(x)<=tolerance for x in d),'harmful_rate_at_retained':len(negative)/len(d) if d else None,'harmful_mass':math.fsum(max(-x,0) for x in d),'beneficial_mass':math.fsum(max(x,0) for x in d),'net_effect':math.fsum(d),'retained_mean_effect':math.fsum(d)/len(d) if d else None,'allcase_deployment_mean_gain_with_sourcefallback':math.fsum(d)/allcase_n if allcase_n else None,'allreject_reference_not_valid_success':len(d)==0}

RANDOM_REFERENCE_SALT='R11_random_reference_v1'
RANDOM_REFERENCE_SEED=0
RANDOM_REFERENCE_DRAWS=1000

def random_reference(candidates,effect_by_id,allcase_n,*,reliable_selected,coverage=.5):
 """Conditional finite-cohort random subset reference, not a significance test.
 Every draw selects exactK using SHA(salt|0|draw|id), without labels in ordering.
 Effects are postseal offline outcomes only, never selection inputs.
 """
 ids=list(candidates);assert len(ids)==len(set(ids)) and set(reliable_selected)<=set(ids)
 k=math.ceil(len(ids)*coverage);assert len(reliable_selected)==k
 if not ids:return {'state':'undefined_no_originalG_changed_candidates','n_candidate':0,'K':0,'allreject_not_success':True}
 observed=risk_summary([effect_by_id[x] for x in reliable_selected],allcase_n)
 measures=collections.defaultdict(list)
 for draw in range(RANDOM_REFERENCE_DRAWS):
  chosen=sorted(ids,key=lambda ident:hashlib.sha256((RANDOM_REFERENCE_SALT+'|'+str(RANDOM_REFERENCE_SEED)+'|'+str(draw)+'|'+ident).encode()).hexdigest())[:k]
  r=risk_summary([effect_by_id[x] for x in chosen],allcase_n)
  for name in ('harmful_mass','beneficial_mass','net_effect','harmful_rate_at_retained','allcase_deployment_mean_gain_with_sourcefallback'):measures[name].append(r[name])
 result={}
 for name,values in measures.items():
  a=np.asarray(values,float);z=observed[name]
  result[name]={'random_mean':float(a.mean()),'random_min':float(a.min()),'random_max':float(a.max()),'random_quantiles_025_25_50_75_975':np.quantile(a,[.025,.25,.5,.75,.975]).tolist(),'reliable_observed':z,'fraction_random_less_or_equal_reliable':float((a<=z).mean()),'fraction_random_greater_or_equal_reliable':float((a>=z).mean())}
 return {'state':'computed_conditional_random_reference_not_population_pvalue','n_candidate':len(ids),'K':k,'allcase_n':allcase_n,'draws':RANDOM_REFERENCE_DRAWS,'seed':RANDOM_REFERENCE_SEED,'salt':RANDOM_REFERENCE_SALT,'key_literal':'salt|seed|draw|id','effect_not_used_in_selection':True,'distribution':result,'not_bestseed_search_or_method_arm':True}
