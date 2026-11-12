"""Descriptive soft-neighborhood density and decisions, not a density ablation."""
from core87 import *
tasks=read(H87/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');out=[]
for t in tasks:
 a=load(B87/'evidence336_attempt002/samples_PRIVATE'/(t['id_PRIVATE']+'.npz'));degree=a['alias'].sum(1);qs=t['queries'];d=t['IoU']['full_direct']-t['IoU']['frozen_bi'];out.append({'row':t['row'],'platform':t['platform'],'soft_degree_frozen_winner':float(degree[qs['frozen_bi']]),'soft_degree_direct_winner':float(degree[qs['full_direct']]),'soft_degree_TTA_winner':float(degree[qs['full_tta1']]),'direct_changes_query':qs['full_direct']!=qs['frozen_bi'],'direct_selects_lower_density':bool(degree[qs['full_direct']]<degree[qs['frozen_bi']]-1e-10),'direct_task_delta_pp':float(d*100),'TTA_task_delta_pp':float((t['IoU']['full_tta1']-t['IoU']['frozen_bi'])*100),'descriptive_not_isolated_causality':True})
dump(H87/'ALIAS_DENSITY_DIAGNOSTIC_ROWS.json',out)
for name,rr in [('all',out),('direct_harms',[r for r in out if r['direct_task_delta_pp']< -1e-8]),('direct_improves',[r for r in out if r['direct_task_delta_pp']>1e-8])]:
 print(name,len(rr),{'lower_density_n':sum(r['direct_selects_lower_density'] for r in rr),'median_frozen_degree':float(np.median([r['soft_degree_frozen_winner'] for r in rr])),'median_direct_degree':float(np.median([r['soft_degree_direct_winner'] for r in rr]))})
