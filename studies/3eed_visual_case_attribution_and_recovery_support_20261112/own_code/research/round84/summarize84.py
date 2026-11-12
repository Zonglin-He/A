"""Posthoc selected-case summaries only; no models, gradients or new GT scoring."""
from pathlib import Path
import json,collections
R=Path(__file__).resolve().parents[2];H=R/'research/round84';C=R/'crossplatform/round84/capture33_attempt001'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
cases=read(H/'experiment/CASES33_PRIVATE.json');paths=read(H/'CASE_PATH_ROWS.json');reviews={x['case']:x for x in read(H/'MANUAL_VISUAL_REVIEW_PRIVATE.json')};cov=read(H/'ORACLE_PROXY_SUPERVISION_ROWS.json');hybrid=read(H/'HEAD_FUSION_2X2_ROWS.json');loss=read(H/'LOSS_REGION_PROXY_ROWS.json')
aliases={x['case']:'CASE'+str(i+1).zfill(3) for i,x in enumerate(cases)}
def anon(rows):return [dict(x,case=aliases[x['case']]) for x in rows]
def group(s):
 if 'UNKNOWN' in s:return 'UNKNOWN'
 if s.startswith(('different_','nonperson_')):return 'different_region_to_target'
 return 'same_selected_object_or_region'
rp=[]
for r in paths:
 v=reviews[r['case']];d=r['BT_IoU']-r['frozen_IoU'];rp.append(dict(r,case=aliases[r['case']],change='improved' if d>1e-12 else 'harmed' if d< -1e-12 else 'unchanged_supplement',visual_reviewed=v['visual_reviewed'],visual_class=v['frozen_to_BT_identity'],coarse_visual_class=group(v['frozen_to_BT_identity'])))
dump(H/'PUBLIC_CASE_ROWS.json',rp);dump(H/'PUBLIC_HYBRID_ROWS.json',anon(hybrid));dump(H/'PUBLIC_SUPERVISION_ROWS.json',anon(cov));dump(H/'PUBLIC_LOSS_REGION_ROWS.json',anon(loss))
groups=[]
for change in ['improved','harmed','unchanged_supplement']:
 for label,n in sorted(collections.Counter(x['coarse_visual_class'] for x in rp if x['change']==change).items()):groups.append({'change':change,'visual_class':label,'n':n})
dump(H/'VISUAL_GROUP_COUNTS.json',groups)
private=[read(C/(x['case']+'_PRIVATE.json')) for x in cases];keys=['original_features_original_head','original_features_prepared_head','prepared_features_original_head','prepared_features_prepared_head']
def switches(k,l):return sum(x['hybrid_PRIVATE'][k]['query_PRIVATE']!=x['hybrid_PRIVATE'][l]['query_PRIVATE'] for x in private)
summary={'n':len(cases),'head_swap_on_original_query_changes':switches(keys[0],keys[1]),'head_swap_on_prepared_query_changes':switches(keys[2],keys[3]),'feature_group_swap_with_original_head_query_changes':switches(keys[0],keys[2]),'source_to_prepared_IoU_decreased':sum(x['frozen_IoU']<x['source_IoU']-1e-12 for x in paths),'source_to_prepared_IoU_increased':sum(x['frozen_IoU']>x['source_IoU']+1e-12 for x in paths),'oracle_max_below_25':sum(x['oracle_candidate_IoU']<.25 for x in paths),'oracle_max_below_50':sum(x['oracle_candidate_IoU']<.5 for x in paths),'selected_case_diagnostic_not_population_test':True,'hybrid_may_be_off_manifold':True,'feature_group_not_RGB_alone':True}
dump(H/'CASE_DIAGNOSIS_SUMMARY.json',summary)
rankrows=[]
for x,c in zip(private,cases):
 if not c['priority']:continue
 # Compare the identical candidate alternatives across saved states. No new labels.
 for arm,rows in x['ranks_PRIVATE'].items():
  for item in rows:
   q=item['query_PRIVATE'];roles=[k for k,v in x['chosen_PRIVATE'].items() if v==q]
   rankrows.append({'case':aliases[c['case']],'state':arm,'candidate_local_alias':'ALT'+str(sorted({v for v in x['chosen_PRIVATE'].values()}).index(q)+1),'selected_by':'+'.join(roles),'rank':item['rank'],'within_state_probability':item['probability'],'IoU':item['IoU']})
dump(H/'PUBLIC_PRIORITY_RANK_ROWS.json',rankrows)
local=['# R84 本地原图核验\n','本页含本地案例标识；不进入公开导出。原句/原图/坐标/索引仅保留本机。\n','所有33条已完成完整RGB和裁剪初步目视核验；两条重点案例进一步核验支持与误差图。其余误差图均已生成并数值审计，不冒充全部完成语义归因。\n','[完整132页图册](WORKSPACE_ROOT/crossplatform/round84/atlas_PRIVATE/index.html)\n','|案例|Source|Frozen|Raw|BT|Oracle|视觉分类|\n|---|---:|---:|---:|---:|---:|---|']
for r in paths:
 a=r['case'];local.append('|['+a+'](WORKSPACE_ROOT/crossplatform/round84/atlas_PRIVATE/pages/'+a+'_overview.jpg)|'+'|'.join(f'{r[k]:.4f}' for k in ['source_IoU','frozen_IoU','raw_IoU','BT_IoU','oracle_candidate_IoU'])+'|'+reviews[a]['frozen_to_BT_identity']+'|')
(H/'LOCAL_GALLERY_PRIVATE.md').write_text('\n'.join(local)+'\n')
print(json.dumps({'summary':summary,'visual_groups':groups},ensure_ascii=False))
