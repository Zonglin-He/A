"""Independent saved-case association, visual page and scalar audit. No scoring."""
from pathlib import Path
import json,hashlib,collections
from PIL import Image
R=Path(__file__).resolve().parents[2];H=R/'research/round84';A=R/'crossplatform/round84/atlas_PRIVATE'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cases=read(H/'experiment/CASES33_PRIVATE.json');rows=read(H/'CASE_PATH_ROWS.json');public=read(H/'PUBLIC_CASE_ROWS.json');review=read(H/'MANUAL_VISUAL_REVIEW_PRIVATE.json');cov=read(H/'ORACLE_PROXY_SUPERVISION_ROWS.json');covsum=read(H/'ORACLE_PROXY_SUPERVISION_SUMMARY.json');assert len(cases)==len(rows)==len(public)==len(review)==len(cov)==33
assert len({x['case'] for x in cases})==33 and {x['case'] for x in cases}=={x['case'] for x in rows}=={x['case'] for x in review}=={x['case'] for x in cov}
assert sum(x['R82_changed'] for x in cases)==31 and sum(x['priority'] for x in cases)==2
pagecount=0
for c,r,p in zip(cases,rows,public):
 assert c['case']==r['case'];assert p['case']=='CASE'+str(pagecount//4+1).zfill(3)
 for k,v in [('source_IoU','source'),('frozen_IoU','frozen_bi'),('raw_IoU','R80_raw_correct'),('BT_IoU','bt_correct')]:assert r[k]==c['task']['IoU'][v] and p[k]==r[k]
 assert c['R82_changed']==(abs(r['BT_IoU']-r['frozen_IoU'])>1e-12)
 assert r['seed_raw_coordinate_max_error']<1e-4
 for kind in ['overview','support_point','support_RGB','loss']:
  f=A/'pages'/(c['case']+'_'+kind+'.jpg')
  with Image.open(f) as im:assert im.format=='JPEG' and min(im.size)>400;im.verify()
  pagecount+=1
assert pagecount==132 and len(list((A/'pages').glob('*.jpg')))==132
for name in ['SEALED.json']:
 seal=read(A/name)
 for f,v in seal['files'].items():assert sha(A/f)==v
assert sum(r['BT_IoU']>r['frozen_IoU']+1e-12 for r in rows)==23
assert sum(r['BT_IoU']<r['frozen_IoU']-1e-12 for r in rows)==8
for r in review:assert r['visual_reviewed'] is True
assert sum(x['fit_supervised_point_seeds_inside_oracle']==0 for x in cov)==covsum['zero_fit_point_proxy_cases']==27
assert sum(x['visible_seeds_inside_oracle']>0 and x['visible_seeds_inside_oracle']==x['protected_anchor_seeds_inside_oracle'] for x in cov)==covsum['visible_proxy_seeds_but_all_protected_cases']==18
native=read(H/'independent_native_audit_attempt001/AUDIT_RESULT.json');mask=read(H/'independent_supervision_support_audit_attempt001/AUDIT_RESULT.json');assert native['state'].startswith('PASS_') and mask['state'].startswith('PASS_') and native['max_native_probability_error']==native['max_map_error']==native['max_original_hidden_bridge_error']==0
summary=read(H/'CASE_DIAGNOSIS_SUMMARY.json');assert summary['head_swap_on_original_query_changes']==summary['head_swap_on_prepared_query_changes']==0 and summary['feature_group_swap_with_original_head_query_changes']==25
o=H/'independent_scalar_visual_audit_attempt001';assert not o.exists();o.mkdir();result={'state':'PASS_R84_33_CASE_ASSOCIATIONS_132_JPEGS_SAVED_VALUES_VISUAL_CLASS_COUNTS_AND_NATIVE_AUDITS','n':33,'changed_cases':31,'improved_cases':23,'harmed_cases':8,'supplemental_cases':2,'readable_visual_pages':pagecount,'RGB_initial_review_cases':33,'priority_detailed_support_and_loss_review_cases':2,'all_loss_maps_not_claimed_semantically_reviewed':True,'coordinate_max_error':max(x['seed_raw_coordinate_max_error'] for x in rows),'new_task_scores':0,'new_parameter_updates':0,'FINAL_access':False,'native_probability_and_map_errors':0,'unknowns_preserved':True}
(o/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
