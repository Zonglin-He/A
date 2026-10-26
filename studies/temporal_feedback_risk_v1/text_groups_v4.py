"""Frozen text-only rubric: exclude noun hand and swimming-pool action false cues."""
import re,hashlib
from text_groups_v1 import ACTION,APPEARANCE,SEQUENCE
from text_groups_v2 import EXTRA_ACTION,STATE_AMBIGUITY
from text_groups_v3 import EXTRA_DYNAMIC,EXTRA_VISUAL
assert ACTION.count('hand(?:s|ed|ing)?')==1
DYNAMIC=ACTION.replace('hand(?:s|ed|ing)?',r'hand(?:ed|ing)|hands(?=\s+(?:the|a|an|his|her|their|it|him))')
def classify(caption):
 s=' '.join(caption.lower().split());s_action=re.sub(r'\bswimming(?=\s+pool\b)','',s);a=re.findall(DYNAMIC,s_action)+re.findall(EXTRA_ACTION,s_action)+re.findall(EXTRA_DYNAMIC,s_action)+re.findall(r'\bdriven\b',s_action);v=re.findall(APPEARANCE,s)+re.findall(EXTRA_VISUAL,s);seq=re.findall(SEQUENCE,s);state=re.findall(STATE_AMBIGUITY,s)
 group='mixed'if a and v else'action_centric'if a else'appearance_centric'if v and not state and not seq else'uncertain'
 return dict(group=group,action_cues=a,appearance_cues=v,sequence_cues=seq,state_or_action_ambiguity_cues=state,explicit_sequence_and_multiple_action_cues=bool(seq and len(a)>=2),caption_sha256=hashlib.sha256(s.encode()).hexdigest(),rule_based_not_gold_semantic_annotation=True)
