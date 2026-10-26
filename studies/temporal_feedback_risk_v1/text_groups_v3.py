"""Final outcome-blind operational rubric; ambiguity is retained, never force binary."""
import re,hashlib
from text_groups_v1 import ACTION,APPEARANCE,SEQUENCE
from text_groups_v2 import EXTRA_ACTION,STATE_AMBIGUITY
EXTRA_DYNAMIC=r'\b(?:tie|ties|tied|tying|tell(?:s|ing)?|told|clos(?:es|ed|ing)|step(?:s|ped|ping)?|stuff(?:s|ed|ing)?|help(?:s|ed|ing)?|lay(?:s|ing)?|laid|lead(?:s|ing)?|led|salut(?:e|es|ed|ing)|adjust(?:s|ed|ing)?|arrang(?:e|es|ed|ing)|count(?:s|ed|ing)?|unbutton(?:s|ed|ing)?|button(?:s|ed|ing)?|remov(?:e|es|ed|ing))\b'
EXTRA_VISUAL=r'\b(?:older|younger|turban|plaid|overalls|necktie|sweater|sweatshirt|vest|scarf|apron|barefoot|bareheaded)\b'
def classify(caption):
 s=' '.join(caption.lower().split());a=re.findall(ACTION,s)+re.findall(EXTRA_ACTION,s)+re.findall(EXTRA_DYNAMIC,s);v=re.findall(APPEARANCE,s)+re.findall(EXTRA_VISUAL,s);seq=re.findall(SEQUENCE,s);state=re.findall(STATE_AMBIGUITY,s)
 group='mixed'if a and v else'action_centric'if a else'appearance_centric'if v and not state and not seq else'uncertain'
 return dict(group=group,action_cues=a,appearance_cues=v,sequence_cues=seq,state_or_action_ambiguity_cues=state,explicit_sequence_and_multiple_action_cues=bool(seq and len(a)>=2),caption_sha256=hashlib.sha256(s.encode()).hexdigest(),rule_based_not_gold_semantic_annotation=True)
