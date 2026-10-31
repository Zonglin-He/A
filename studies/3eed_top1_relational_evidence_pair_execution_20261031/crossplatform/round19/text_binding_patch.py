"""Post-score CPU repair, NOT wired into any scored G or reported as accuracy.
3EED's explicit subject appearance field is outside the short noun phrase.
Binding that literal field to T preserves text; it does not certify visual truth.
"""
import re,copy
def bind_explicit_subject_field(graph,caption):
 g=copy.deepcopy(graph)
 if g.get('status')!='PARSED_text_graph':return g
 pattern=r'(?i)\bmain\s+appearance\s+features\s*:\s*(?P<appearance>[^;]+)(?=;)'
 matches=list(re.finditer(pattern,caption))
 if len(matches)!=1:return g
 m=matches[0];lo,hi=m.span('appearance');text=caption[lo:hi].strip();lo+=len(caption[lo:hi])-len(caption[lo:hi].lstrip());hi=lo+len(text)
 if not text:return g
 a={'id':'POSTSCORE_SUBJECT_FIELD','entity':'T','kind':'appearance','quote':{'text':text,'span':[lo,hi]},'status':'PARSED','owner_binding':'explicit_main_appearance_features_subject_field','visual_truth_certified':False}
 g['attributes']=list(g.get('attributes',[]))+[a];g['postscore_text_binding_repair_only']=True;return g
