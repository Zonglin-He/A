"""Mechanism tests on synthetic data, not dataset cases or GT."""
from evidence86 import *
from pathlib import Path
import json
text='The car left of the truck, which is right of the bus in the image.'
pieces=[('The',2,'det','DET'),('car',0,'root','NOUN'),('left',2,'acl','ADJ'),('of',6,'case','ADP'),('the',6,'det','DET'),('truck',3,'obl','NOUN'),(',',6,'punct','PUNCT'),('which',9,'nsubj','PRON'),('is',6,'acl:relcl','VERB'),('right',9,'xcomp','ADJ'),('of',13,'case','ADP'),('the',13,'det','DET'),('bus',10,'obl','NOUN'),('in',16,'case','ADP'),('the',16,'det','DET'),('image',10,'obl','NOUN'),('.',2,'punct','PUNCT')];words=[];cursor=0
for i,(word,head,dep,pos) in enumerate(pieces,1):
 lo=text.index(word,cursor);hi=lo+len(word);cursor=hi;words.append({'id':i,'text':word,'lemma':word.lower(),'head':head,'deprel':dep,'upos':pos,'start_char':lo,'end_char':hi})
ent=[]
for role,head in [('T','car'),('R0','truck'),('R1','bus')]:
 w=next(x for x in words if x['text']==head);ent.append({'role':role,'head':head,'span':[w['start_char'],w['end_char']],'phrase':head,'attributes':[]})
g={'usable':True,'entities':ent};edges=relations86(text,{'sentences':[{'words':words}]},g);actual={(x['subject'],x['reference'],x['predicate']) for x in edges};assert actual=={('T','R0','LEFT'),('R0','R1','RIGHT')},actual
# UNKNOWN mass never changes, and cannot become an authorized winner.
p=np.array([.4,.3,.3]);s=np.array([0.,1.,0.]);c=np.array([0.,0.,2.]);q,active=transfer86(p,s,c);assert active and q[0]==p[0] and q[1]>p[1] and q[2]<p[2]
pi,eligible,logs,union=target86(p,[{'kind':'test','support':s,'counter':c}],np.arange(3),'synthetic');assert pi.argmax()==1 and guarded86(pi,eligible).argmax()==0 and logs[0]['unknown_probability_error']==0
q,a=transfer86(p,s,np.zeros(3));assert not a and np.array_equal(p,q)
result={'state':'PASS_R86_NO_SPURIOUS_TRANSITIVE_EDGE_UNKNOWN_MASS_AND_READOUT_QUALIFICATION','synthetic_edges':len(edges),'spurious_target_bus_edge':False,'UNKNOWN_mass_preserved':True,'unsupported_unknown_winner_not_certified':True,'single_sided_observation_abstains':True,'dataset_GT_read':False};out=Path(__file__).resolve().parent/'SYNTHETIC_AUDIT.json';out.write_text(json.dumps(result,indent=2)+'\n');print(result)
