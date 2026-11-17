"""Source-only same native R145 reader, q90 cumulative-mass calibration; no new model."""
from common148 import *
from mass148 import *
import collections

def main():
 check();assert rd(R/'research/round145/SOURCE_AUDIT.json')['status']=='PASS';seal=rd(R/'crossplatform/round145/source/SEALED.json');rows=[];scores=collections.defaultdict(list);counts=collections.Counter();oldk=rd(R/'research/round145/SOURCE_KAPPAS_LOCK.json')['kappa'];oldrows={(e['source'],e['id_PRIVATE']):e for e in rd(R/'research/round145/SOURCE_CALIBRATION_ROWS_PRIVATE.json')}
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  src,sid=e['entry']['source'],e['entry']['row']['id'];root=R/'crossplatform/round145/source';p=root/src/(sid+'.npz');j=root/src/(sid+'.json');assert sha(p)==seal['files'][str(p.relative_to(root))] and sha(j)==seal['files'][str(j.relative_to(root))];a=np.load(p);has=rd(j)['has_B'];ass,S=a['assignment'],a['S'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];row={'source':src,'id_PRIVATE':sid,'eligible':False,'views':[]}
  if has and S.sum()>=2:
   l=e['entry']['files']['label'];assert sha(R/l['path'])==l['sha256'];y=np.load(R/l['path'])['ious'];good=np.array([y[g].max()>.25 for g in gs])[S]
   if good.any():
    row['eligible']=True;counts[src]+=1
    for tag in ['A','B']:
     m=conditional_mass(a['p'+tag],gs,S);v=score_first_good(m,good);scores[src].append(v);lp=np.log(m);old=lp.max()-lp<=oldk[src];row['views'].append({'view':tag,'score':v,'mass':m.tolist(),'good_PRIVATE':good.tolist(),'old_retained_fraction':float(old.mean()),'S_size':len(m),'top_mass':float(m.max()),'entropy':float(-(m*np.log(m)).sum())})
  assert row['eligible']==oldrows[src,sid]['eligible'];rows.append(row)
 q={s:float(np.quantile(v,.9)) for s,v in scores.items()};stats=[]
 for src in ['waymo','drone','quad']:
  vv=[v for r in rows if r['source']==src and r['eligible'] for v in r['views']];inside=[];sizes=[]
  for v in vv:
   m=np.array(v['mass']);good=np.array(v['good_PRIVATE']);C=mass_set(m,q[src]);assert not(v['score']<=q[src] and not(C&good).any());inside.append(bool((C&good).any()));sizes.append(float(C.mean()))
  stats.append({'source':src,'source_inputs_eligible':counts[src],'correlated_view_cases':len(vv),'q90_cumulative':q[src],'empirical_view_good25_covered_fraction':float(np.mean(inside)),'mean_retained_group_fraction':float(np.mean(sizes)),'mean_old_loggap_retained_group_fraction':float(np.mean([v['old_retained_fraction'] for v in vv])),'mean_top_mass':float(np.mean([v['top_mass'] for v in vv]))})
 assert dict(counts)=={'waymo':22,'drone':18,'quad':37};dump(H/'SOURCE_CALIBRATION_ROWS_PRIVATE.json',rows);dump(H/'SOURCE_MASS_LOCK.json',{'status':'SOURCE_ONLY_CUMULATIVE_MASS_Q90_LOCKED','q90':q,'source_counts':dict(counts),'summary':stats,'source_labels_used':True,'target_GT_choice':False,'source_weight_training':False,'new_models':0,'no_formal_target_conformal_guarantee':True,'same_R145_reader':True,'source_prediction_seal_sha256':sha(root/'SEALED.json'),'utc':utc()});print(stats,flush=True)
if __name__=='__main__':main()
