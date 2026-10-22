from pathlib import Path
import sys,json,hashlib,time
R=Path('/home/wwww/visual grounding/stvg-mainline');sys.path.insert(0,str(R))
from stvg.query import QuerySubjectParser
from research.joint_event_v1.run import read,put,sha
L=R/'outputs/overnight_event_loop_20261023';p=read(R/'outputs/native_event_three_state_v1/plan.json');parser=QuerySubjectParser(R/'checkpoints/stanza');start=time.perf_counter();rows=[]
for item in p['items']:
 caption=item['row']['metadata']['caption'];doc=parser.nlp(caption);words=[]
 for si,sent in enumerate(doc.sentences):
  for token in sent.tokens:
   for w in token.words:words.append(dict(key=(si,w.id),parent=(si,w.head),root=w.head==0,pos=w.upos,rel=w.deprel.split(':')[0],lemma=w.lemma,start=token.start_char,end=token.end_char))
 roots={w['key']for w in words if w['root']and w['pos']=='VERB'};chosen=set(roots)
 for _ in words:
  more={w['key']for w in words if w['pos']=='VERB'and w['parent']in chosen and w['rel']in{'conj','xcomp','ccomp','advcl'}}
  if more<=chosen:break
  chosen|=more
 args=sorted([w for w in words if w['parent']in chosen and w['pos']in{'NOUN','PROPN'}and w['rel']in{'obj','iobj','obl'}],key=lambda w:w['start']);arg=None
 if args:
  head=args[0];take={head['key']}
  for _ in words:
   more={w['key']for w in words if w['parent']in take and w['rel']in{'det','amod','compound','nummod','flat','nmod','case','poss'}}
   if more<=take:break
   take|=more
  spans=[w for w in words if w['key']in take];lo=min(w['start']for w in spans);hi=max(w['end']for w in spans);arg=dict(subject=head['lemma'].lower(),nounphrase=caption[lo:hi].strip(' .'),start=lo,end=hi,relation=head['rel'],all_argument_count=len(args))
 rows.append(dict(alias=item['alias'],group=item['group'],dataset=item['dataset'],supported=arg is not None,argument=arg,caption_sha256=hashlib.sha256(caption.encode()).hexdigest(),native_capture_sha256=item['capture']['sha256']))
summary={g:dict(total=sum(x['group']==g for x in rows),supported=sum(x['group']==g and x['supported']for x in rows))for g in ['HC64','hc2_recheck','vidstg_recheck']};put(L/'role_context_input_preflight.json',dict(schema='SOURCEONLYsyntaxfeasibility_NOTmethodexperiment',source_script_sha256=sha(Path(__file__)),selection_plan_sha256=sha(R/'outputs/native_event_three_state_v1/plan.json'),parser_version=parser.version,main_root_VERB_only=True,no_GT_read=True,no_visual_or_expert_forward=True,model_experts_added=0,rows=rows,summary=summary,seconds=time.perf_counter()-start));print(json.dumps(summary,indent=2))
