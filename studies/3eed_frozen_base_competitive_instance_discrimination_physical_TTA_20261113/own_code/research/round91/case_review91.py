"""Postseal case descriptions; box overlap is a diagnostic, not object identity truth."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round91';B=R/'crossplatform/round91';read=lambda p:json.loads(p.read_text())
cases=read(H/'experiment/GOOD_BAD_CASES_PRIVATE.json')+read(H/'experiment/FIXED_R90_CASES_PRIVATE.json');notes=read(H/'experiment/CASE_OBSERVATIONS_PRIVATE.json');assert {c['case'] for c in cases}==set(notes)
out=[];private=[]
def boxiou(a,b):
 lo=np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2);hi=np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2);inter=np.maximum(hi-lo,0).prod();return float(inter/(a[3:].prod()+b[3:].prod()-inter+1e-30))
for c in cases:
 n=notes[c['case']];assert n['full_scene_reviewed'] and n['recovery_page_reviewed'];assert n['identity_conclusion'] in ['different_visible_objects','same_visible_object_duplicate_boxes','same_candidate_no_identity_change','UNKNOWN'];t=c['task'];z=np.load(B/'prediction336_attempt001/samples_PRIVATE'/(c['row']['id']+'.npz'));q=t['queries'];a='R76_full_disc_frozen';b='R76_full_disc_physical_bt';rec=read(B/'prediction336_attempt001/records_PRIVATE'/(c['row']['id']+'.json'))[b]
 x={'case':c['case'],'platform':t['platform'],'full_scene_reviewed':True,'recovery_page_reviewed':True,'identity_conclusion':n['identity_conclusion'],'description':n['description'],'static_identity_conclusion':n['static_identity_conclusion'],'changed_candidate':q[a]!=q[b],'selected_box_mutual_IoU':boxiou(z['source_boxes'][q[a]],z['source_boxes'][q[b]]),'new_frozen_IoU':t['IoU'][a],'physical_bt_IoU':t['IoU'][b],'physical_bt_change_pp':100*(t['IoU'][b]-t['IoU'][a]),'source_interface_change_pp':100*(t['IoU'][a]-t['IoU']['R76_frozen']),'physical_auxiliary_change':rec['after']-rec['before'],'postseal_descriptive_not_deployment_gate':True}
 out.append(x);private.append({**x,'caption':c['row']['caption'],'reasons':c['reasons']})
(H/'CASE_REVIEW_ANONYMOUS_ROWS.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');(H/'CASE_REVIEW_PRIVATE.json').write_text(json.dumps(private,indent=2,ensure_ascii=False)+'\n');print({'reviewed_pages':len(out),'unknown_identity_n':sum(x['identity_conclusion']=='UNKNOWN' for x in out),'new_task_scoring':False})
