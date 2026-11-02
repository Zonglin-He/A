"""Independent saved-IoU aggregation only; no GT/metadata/geometric evaluator."""
from pathlib import Path
import collections,hashlib,itertools,json,math,statistics,time
ROOT=Path(__file__).resolve().parents[3];S=ROOT/'research/round50/independent_score/dev336_attempt001';OUT=ROOT/'research/round50/independent_scalar_recheck/dev336_attempt001'
ARMS=['source','old_G','wideentry_oldW','oldentry_newW_k3','wideentry_newW_k3','oldentry_newW_k10','wideentry_newW_k10']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
ERROR=0.;COUNT=0
def same(a,b,path='root'):
 global ERROR,COUNT
 if isinstance(a,dict):assert set(a)==set(b),(path,'keys');[same(a[k],b[k],path+'.'+k) for k in a]
 elif isinstance(a,list):assert len(a)==len(b),(path,'length');[same(x,y,path+f'[{i}]') for i,(x,y) in enumerate(zip(a,b))]
 elif type(a) in (float,int):
  assert type(b) in (int,float) and math.isfinite(a) and math.isfinite(b),path
  d=abs(a-b);ERROR=max(ERROR,d);COUNT+=1;assert d<=1e-10,(path,a,b)
 else:assert a==b,(path,a,b)
def avg(v):return statistics.fmean(v)
def main():
 start=time.perf_counter();assert not OUT.exists();OUT.mkdir(parents=True);seal=read(S/'SEALED.json')
 for f,h in seal['files'].items():assert sha(S/f)==h
 tasks=read(S/'ALL336_TASK_ROWS_PRIVATE.json');assert len(tasks)==336 and len({r['row'] for r in tasks})==336 and collections.Counter(r['platform'] for r in tasks)=={'drone':168,'quad':168}
 for r in tasks:assert set(r['IoU'])==set(r['queries'])==set(ARMS) and all(math.isfinite(v) and 0<=v<=1 for v in r['IoU'].values()) and all(type(q) is int and 0<=q<256 for q in r['queries'].values())
 summary={};scenes=[];negative=[];effects={}
 for part in ('ALL336','D208','V64','M64'):
  rr=tasks if part=='ALL336' else [r for r in tasks if r['split']==part];ss={}
  for pl in ('drone','quad'):
   pr=[r for r in rr if r['platform']==pl];ss[pl]={}
   for arm in ARMS:
    v=[r['IoU'][arm] for r in pr];ds=[r['IoU'][arm]-r['IoU']['source'] for r in pr];do=[r['IoU'][arm]-r['IoU']['old_G'] for r in pr]
    ss[pl][arm]={'n':len(pr),'mIoU_percent':avg(v)*100,'Acc25_percent':sum(x>.25 for x in v)/len(pr)*100,'Acc50_percent':sum(x>.5 for x in v)/len(pr)*100,'gain_vs_source_pp':avg(ds)*100,'gain_vs_old_G_pp':avg(do)*100,'improved_vs_source':sum(x>1e-10 for x in ds),'worsened_vs_source':sum(x< -1e-10 for x in ds),'improved_vs_old_G':sum(x>1e-10 for x in do),'worsened_vs_old_G':sum(x< -1e-10 for x in do),'query_changes_vs_source':sum(r['queries'][arm]!=r['queries']['source'] for r in pr),'query_changes_vs_old_G':sum(r['queries'][arm]!=r['queries']['old_G'] for r in pr),'scene_clusters':len({r['scene_PRIVATE'] for r in pr})}
   for scene in sorted({r['scene_PRIVATE'] for r in pr}):
    sr=[r for r in pr if r['scene_PRIVATE']==scene];scenes.append({'part':part,'platform':pl,'scene_PRIVATE':scene,'n':len(sr),'mIoU_percent':{a:avg(r['IoU'][a] for r in sr)*100 for a in ARMS}})
  ss['balanced_mIoU_percent']={a:avg(ss[pl][a]['mIoU_percent'] for pl in ('drone','quad')) for a in ARMS};summary[part]=ss;effects[part]={}
  for pl in ('drone','quad','balanced'):
   m=ss['balanced_mIoU_percent'] if pl=='balanced' else {a:ss[pl][a]['mIoU_percent'] for a in ARMS};e={'coverage_oldW_minus_old_G_pp':m['wideentry_oldW']-m['old_G']}
   for k in (3,10):
    old=m[f'oldentry_newW_k{k}'];wide=m[f'wideentry_newW_k{k}'];e.update({f'newW_oldentry_k{k}_minus_old_G_pp':old-m['old_G'],f'newW_wideentry_k{k}_minus_wide_oldW_pp':wide-m['wideentry_oldW'],f'coverage_newW_k{k}_pp':wide-old,f'entry_W_interaction_k{k}_pp':wide-old-m['wideentry_oldW']+m['old_G']})
   effects[part][pl]=e
 for r in tasks:
  for a,b in itertools.product(ARMS,repeat=2):
   d=r['IoU'][a]-r['IoU'][b]
   if a!=b and d< -1e-10:negative.append({'row':r['row'],'platform':r['platform'],'split':r['split'],'arm':a,'reference':b,'delta_pp':d*100})
 same(summary,read(S/'TASK_SUMMARY.json'));same(scenes,read(S/'ALL_SCENE_SUMMARIES_PRIVATE.json'));same(negative,read(S/'ALL_ORDERED_NEGATIVE_COMPARISONS.json'));same(effects,read(S/'FACTORIAL_CONTRASTS.json'));assert len(negative)==1758
 m=summary['ALL336']['balanced_mIoU_percent'];k=3 if m['wideentry_newW_k3']-m['wideentry_newW_k10']>1e-10 else 10;priority=['wideentry_oldW','oldentry_newW','wideentry_newW'];selected={x:x if x=='wideentry_oldW' else f'{x}_k{k}' for x in priority};maximum=max(m[selected[x]] for x in priority);family=next(x for x in priority if maximum-m[selected[x]]<=1e-10);winner=selected[family];gain=m[winner]-m['old_G']
 choice=read(S/'DEV_SELECTION_AND_FULL_GATE.json');assert choice['selected_shared_k']==k==10 and choice['selected_candidate_family']==family=='wideentry_oldW' and choice['selected_candidate_arm']==winner and choice['full_gate_positive_balanced_net_vs_old_G']==(gain>1e-10)
 same(choice['selected_candidate_balanced_mIoU_percent'],m[winner]);same(choice['selected_gain_vs_old_G_pp'],gain);same(choice['selected_gain_vs_source_pp'],m[winner]-m['source'])
 receipt=read(S/'SCORING_RECEIPT.json');assert receipt['task_values']==2352 and receipt['ordered_negative_comparisons']==1758 and receipt['full_gate_PASS'] is True
 dump(OUT/'RECOMPUTED_TASK_SUMMARY.json',summary);dump(OUT/'RECOMPUTED_FACTOR_CONTRASTS.json',effects);dump(OUT/'FIXED_SELECTION_RECHECK.json',{'selected_shared_k':k,'selected_candidate':winner,'gain_vs_oldG_pp':gain,'full_gate_PASS':gain>1e-10,'selected_full_candidate_has_no_k_parameter':True,'new_conditional_W_k10_minus_wide_oldW_pp':m['wideentry_newW_k10']-m['wideentry_oldW'],'no_posthoc_case_or_platform_gate':True})
 result={'state':'PASS_R50_INDEPENDENT_SAVED_DEV_SCALARS_AND_FIXED_SELECTION_REAGGREGATION','n':336,'arms':7,'IoU_values_reaggregated':2352,'ordered_negative_comparisons_reproduced':1758,'scene_summary_rows_reproduced':len(scenes),'numeric_comparisons':COUNT,'max_abs_scalar_error':ERROR,'selected_shared_k':k,'selected_candidate':winner,'full_gate_PASS':True,'full_candidate_independent_of_k':True,'new_conditional_W_not_claimed_to_beat_coverage_only':True,'no_GT_metadata_IoU_recomputation_models_or_FINAL_read':True,'score_seal_sha256':sha(S/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(OUT/'AUDIT_RESULT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
