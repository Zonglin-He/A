"""Pre-score text-only refinement: keep static/action ambiguities out of pure appearance."""
import re,hashlib
from text_groups_v1 import ACTION,APPEARANCE,SEQUENCE
EXTRA_ACTION=r'\b(?:rais(?:e|es|ed|ing)|lower(?:s|ed|ing)?|drag(?:s|ged|ging)?|press(?:es|ed|ing)?|accompan(?:y|ies|ied|ying)|get(?:s|ting)?|got|count(?:s|ed|ing)?|pinch(?:es|ed|ing)?|cover(?:s|ed|ing)?|twist(?:s|ed|ing)?|draw(?:s|ing)?|drew|stir(?:s|red|ring)?|embrac(?:e|es|ed|ing)|smil(?:e|es|ed|ing)|rubb?(?:s|ed|ing)?|extend(?:s|ed|ing)?|stretch(?:es|ed|ing)?|kneel(?:s|ed|ing)?|remov(?:e|es|ed|ing)|fasten(?:s|ed|ing)?|dropp?(?:s|ed|ing)?|pick(?:s|ed|ing)?|clap(?:s|ped|ping)?|bow(?:s|ed|ing)?|reach(?:es|ed|ing)?|grasp(?:s|ed|ing)?|salut(?:e|es|ed|ing)|slap(?:s|ped|ping)?|patt?(?:s|ed|ing)?|throw(?:s|n|ing)?|carry|carries|carried|carrying)\b'
STATE_AMBIGUITY=r'\b(?:hold(?:s|ing)?|held|sit(?:s|ting)?|sat|stand(?:s|ing)?|stood|lean(?:s|ed|ing)?|lie|lies|lying|lay|hang(?:s|ing)?|hung|rest(?:s|ed|ing)?|face(?:s|d)?|facing|wait(?:s|ed|ing)?)\b'
def classify(caption):
 s=' '.join(caption.lower().split());a=re.findall(ACTION,s)+re.findall(EXTRA_ACTION,s);v=re.findall(APPEARANCE,s);seq=re.findall(SEQUENCE,s);state=re.findall(STATE_AMBIGUITY,s)
 group='mixed'if a and v else'action_centric'if a else'appearance_centric'if v and not state and not seq else'uncertain'
 return dict(group=group,action_cues=a,appearance_cues=v,sequence_cues=seq,state_or_action_ambiguity_cues=state,explicit_sequence_and_multiple_action_cues=bool(seq and len(a)>=2),caption_sha256=hashlib.sha256(s.encode()).hexdigest(),rule_based_not_gold_semantic_annotation=True)
