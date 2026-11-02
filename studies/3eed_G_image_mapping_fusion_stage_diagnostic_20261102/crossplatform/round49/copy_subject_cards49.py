"""Copy-only Phase2 cards; no model, numerical method, GT, or new text parsing."""
import json,hashlib,pathlib,datetime
R=pathlib.Path('WORKSPACE_ROOT')
def read(p):return json.loads((R/p).read_text())
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def write(p,d): (R/p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
a=read('research/round49/blind_rgb48_draft/ROOT_PHASE2_AUTHORIZATION.json')
src='research/round49/subject_inputs_attempt001/TEXT_INPUTS_PRIVATE.json'; ph='crossplatform/round49/blind_rgb48_attempt001/phase1_visual_review/ANNOTATIONS_PRIVATE.json'
assert sha(src)==a['actual_subject_table_sha256'];assert sha(ph)==a['phase1_annotation_sha256']
b='crossplatform/round49/blind_rgb48_attempt001';out=R/b/'phase2_subject_cards';out.mkdir(exist_ok=False)
cohort=read(b+'/COHORT_PRIVATE.json');bindings={x['review_alias']:x for x in read(b+'/RENDER_BINDINGS_PRIVATE.json')};texts={x['id_PRIVATE']:x for x in read(src)}
files={}
for c in cohort:
 t=texts[c['id_PRIVATE']];assert t['full_caption_sha256']==hashlib.sha256(c['caption_PRIVATE'].encode()).hexdigest()
 card={'review_alias':c['review_alias'],'subject_query':t['subject_query_PRIVATE'],'subject_query_cache_sha256':t['cached_text_file_sha256'],'subject_query_string_sha256':t['subject_query_sha256'],'phase1_annotation_sha256':a['phase1_annotation_sha256'],'display_ids':[x['display_id'] for x in bindings[c['review_alias']]['numbered_candidates_PRIVATE']],'phase':'PHASE2_ACTUAL_LEGACY_SUBJECT_QUERY_COMPATIBILITY','no_method_scores_GT_or_3D_data':True}
 p=str((out/(c['review_alias']+'.json')).relative_to(R));write(p,card);files[p]=sha(p)
write(str((out/'SEALED.json').relative_to(R)),{'state':'COPY_ONLY_PHASE2_CARDS_SEALED','n':48,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_table_sha256':sha(src),'phase1_annotation_sha256':sha(ph),'authorization_sha256':sha('research/round49/blind_rgb48_draft/ROOT_PHASE2_AUTHORIZATION.json'),'files':files,'no_NLP_model_method_score_GT_or_FINAL':True})
print('cards48 sealed',out)
