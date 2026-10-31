"""No model/GT/scoring. Diagnose and repair explicit field ownership in texts."""
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round19'))
from common import *
from text_binding_patch import bind_explicit_subject_field
def main():
 records=[]
 for r in read(EXP.parent/'input/ALL112_v2_PRIVATE.json'):
  case=read(ROOT/'crossplatform/round19/G112_attempt004/cases_PRIVATE'/(r['id']+'.json'));g=case.get('text_graph_PRIVATE',{});fixed=bind_explicit_subject_field(g,r['caption']);old=[a for a in g.get('attributes',[]) if a['entity']=='T'];new=[a for a in fixed.get('attributes',[]) if a['id']=='POSTSCORE_SUBJECT_FIELD']
  if new:
   a=new[0];lo,hi=a['quote']['span'];assert r['caption'][lo:hi]==a['quote']['text'] and a['entity']=='T';records.append({'alias':r['alias'],'stage':r['stage'],'platform':r['platform'],'old_subject_attribute_count':len(old),'explicit_subject_field_preserved':True,'visual_truth_not_certified':True})
 dump(EXP/'TEXT_BINDING_PATCH_AUDIT.json',{'state':'Postscore_text_only_binding_patch_checked_on_real_captions','no_new_model_or_GT_scoring':True,'not_integrated_into_scored_G':True,'score_improvement_not_measured':True,'scored_G_attributes_not_overwritten':True,'code_sha256':sha(ROOT/'crossplatform/round19/text_binding_patch.py'),'field_cases':len(records),'previously_zero_subject_attributes':sum(r['old_subject_attribute_count']==0 for r in records),'rows':records})
 print('TEXT_BINDING_PATCH_AUDITED',len(records),sum(r['old_subject_attribute_count']==0 for r in records))
if __name__=='__main__':main()
