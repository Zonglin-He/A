"""Independent query-only POS/predicate sensitivity labels; never reads GT or scores."""
from pathlib import Path
import sys,os,json,time,hashlib,re,collections
R=Path(__file__).resolve().parents[2];D=R/'outputs/temporal_feedback_risk_v1';A=D/'semantic_annotation_assets_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(temp,p)
def identity():
 s=Path('/proc/self/stat').read_text().split(') ',1)[1].split();return dict(pid=os.getpid(),startticks=int(s[19]))
COPULA=set('am is are was were be been being do does did have has had can could would should will may might must'.split())
STATE=set('hold holds held holding sit sits sitting sat stand stands standing stood lean leans leaned leaning lie lies lying lay rest rests resting hang hangs hung hanging remain remains remaining'.split())
APP_VERBS=set('wear wears wearing wore worn dress dresses dressed dressing'.split())
VISUAL=r'\b(?:red|blue|green|yellow|black|white|pink|purple|brown|gray|grey|cyan|orange|silver|mint|blond|blonde|bald|striped|plaid|floral|young|younger|old|older|elderly|tall|short|fat|thin|slim|naked|bearded|hooded|white-haired|black-haired|long-haired|short-haired|shirt|shirts|t-shirt|suit|suits|coat|jacket|dress|skirt|pants|trousers|jeans|sleeves|hat|cap|helmet|glasses|uniform|clothes|clothing|gown|hair|beard|fur|turban|overalls|vest|scarf|apron|sweater|bikini)\b'
RELATION=r'\b(?:above|beneath|below|behind|beside|under|over|inside|outside|between|near|away|towards|next to|in front of|in the|on the|at home)\b'
def label(caption,tagger):
 text=' '.join(caption.lower().split());tokens=re.findall(r"[a-z]+(?:[-'][a-z]+)*|[.,?!]",text);tags=tagger.tag(tokens);verbs=[];states=[];appearance=re.findall(VISUAL,text)
 for i,(token,pos)in enumerate(tags):
  if not pos.startswith('VB')or token in COPULA:continue
  particle=tokens[i+1]if i+1<len(tokens)else''
  if token in APP_VERBS:appearance.append(token);continue
  if token in STATE and particle not in ['up','down']:states.append(token);continue
  verbs.append(token)
 # A noun compound swimming pool is not a motion predicate. Keep POS evidence,
 # and conservatively remove that one contextual token if a tagger marks VBG.
 if re.search(r'\bswimming\s+pool\b',text):verbs=[v for v in verbs if v!='swimming']
 relation=bool(re.search(RELATION,text));fine='mixed_action_appearance'if verbs and appearance else'action_centric'if verbs else'state_or_posture'if states else'static_relation'if relation else'appearance_only'if appearance else'uncertain'
 broad='event_centric'if fine in ['mixed_action_appearance','action_centric']else'static_visual_grounding_proxy'if fine in ['static_relation','appearance_only']else'uncertain_or_state'
 return dict(fine_group=fine,broad_group=broad,POS_tags=tags,predicate_verbs=verbs,state_verbs=states,appearance_cues=appearance,caption_sha256=hashlib.sha256(text.encode()).hexdigest(),not_semantic_gold=True)
def run():
 plan=read(D/'semantic_annotation_plan_v1.json');assert plan['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in plan.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
 for field in ['code_sha256','input_sha256']:
  for f,h in plan[field].items():assert sha(R/f)==h,f
 assets=read(A/'receipt.json')
 for f,h in assets['files_sha256'].items():assert sha(A/f)==h,f
 sys.path.insert(0,str(A/'python'));import nltk
 assert nltk.__version__=='3.9.2';nltk.data.path[:]=[str(A/'nltk_data')]
 from nltk.tag.perceptron import PerceptronTagger
 start=time.perf_counter();tagger=PerceptronTagger(lang='eng');source=read(D/'text_classification_v4.json');out={}
 for ds,domain in source['datasets'].items():
  qq={key:label(v['private_caption'],tagger)for key,v in domain['queries'].items()};out[ds]=dict(queries=qq,fine_counts=dict(collections.Counter(x['fine_group']for x in qq.values())),broad_counts=dict(collections.Counter(x['broad_group']for x in qq.values())))
 sample=read(D/'root_semantic_blind_sample_v1.json');review=read(D/'root_text_semantic_review_v1.json');assert set(review['labels'])=={r['query']for r in sample['rows']};agreement=[]
 for row in sample['rows']:
  k=row['query'];pred=out[row['dataset']]['queries'][k]['fine_group'];gold=review['labels'][k];agreement.append(dict(dataset=row['dataset'],query=k,POS_group=pred,root_text_review_group=gold,agrees=pred==gold))
 result=dict(status='all12205_query_text_POS_predicate_sensitivity_labels_complete_not_gold',datasets=out,source_text_classification_sha256=sha(D/'text_classification_v4.json'),root_blind_review_sha256=sha(D/'root_text_semantic_review_v1.json'),review_agreement=agreement,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,NLP_POS_model_constructors=1,STVG_model_constructors=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT=0,no_private_text_sent_to_network=True)
 atomic(D/'semantic_annotation_v1.json',result);public=dict(status=result['status'],counts={ds:{k:v for k,v in e.items()if k!='queries'}for ds,e in out.items()},root_AI_text_review_not_independent_human_gold=True,sampled_queries=len(agreement),sample_agreement=sum(x['agrees']for x in agreement),sample_disagreements=len(agreement)-sum(x['agrees']for x in agreement),pure_appearance_vs_static_relation_separate=True,scope='Secondary event vs static visual-grounding proxy is not a pure appearance-only contrast. All initial keyword results retained. Neither POS predictions nor root single AI text review are semantic gold. No grouping selected from accuracy.',annotation_sha256=sha(D/'semantic_annotation_v1.json'),elapsed_wall_seconds=result['elapsed_wall_seconds'],NLP_POS_model_constructors=1,new_Source=0,new_expert=0,new_GT=0);atomic(R/'results/temporal_feedback_risk_v1/semantic_validation_v1.json',public)
 print(json.dumps(public),flush=True)
if __name__=='__main__':run()
