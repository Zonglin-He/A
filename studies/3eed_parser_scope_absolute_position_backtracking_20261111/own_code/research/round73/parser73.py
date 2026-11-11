"""Finite clause grammar with original character spans; no model/GT parsing."""
from common73 import *
import re
import text72
DIRECTION_PATTERN=text72.REL_RE.pattern
DIRECTION=re.compile(DIRECTION_PATTERN,re.I)
OTHER_PATTERN=r'\b(?:next to|beside|behind|in front of|near|closest to|nearest to|between)\b'
ALLREL=re.compile('(?:'+OTHER_PATTERN+')|(?:'+DIRECTION_PATTERN+')',re.I)
FRAME=r'(?:image|view|picture|photo|screen|frame)'
ABSOLUTE=re.compile(r'\b(?:(?:to|on|in|at)\s+(?:the\s+)?)?(?:(?:upper|lower|top|bottom)\s+)?(?P<direction>left|right)(?:\s+(?:side|part|region|portion|corner|quadrant|area|section))?\s+(?:of|in)\s+(?:the\s+)?(?:(?:center|centre)\s+of\s+(?:the\s+)?)?'+FRAME+r'\b',re.I)
NEGATION=re.compile(r'\b(?:not|no|without|except|neither|nor)\b',re.I)
def classify(caption):
 result={'kind':'UNKNOWN','reason':'no_explicit_direction_constraint','direction':None,'target_span':None,'constraint_span':None,'reference_span':None,'single_target_role':False}
 if NEGATION.search(caption):result['reason']='negation_scope_not_supported';return result
 first=text72.ENT_RE.search(caption)
 if first is None:result['reason']='target_entity_not_recognized';return result
 match=ABSOLUTE.search(caption);edge=DIRECTION.search(caption)
 if match is not None:
  before=caption[:match.start()];entities=list(text72.ENT_RE.finditer(before));prior=ALLREL.search(before)
  if first.end()>match.start() or len(entities)!=1:result['reason']='absolute_subject_role_ambiguous';return result
  if prior is not None:result['reason']='earlier_relation_changes_scope';return result
  if re.search(r'[.;!?]',before) or re.search(r'\b(?:there is|there are|another|other|while|whereas)\b',before,re.I):result['reason']='independent_clause_or_competing_subject';return result
  # A possessive reference frame is not the camera image frame.
  result.update(kind='IMAGE_ABSOLUTE',reason='explicit_camera_frame_and_first_subject',direction=match.group('direction').lower(),target_span=[0,match.start()],constraint_span=list(match.span()),reference_span=[match.end()-len(re.search(FRAME+r'\b',match.group(),re.I).group()),match.end()],single_target_role=True);return result
 if edge is not None:
  firstrel=ALLREL.search(caption)
  if edge.start()<first.end() or firstrel is None or firstrel.span()!=edge.span():result['reason']='relative_relation_scope_ambiguous';return result
  a=edge.end();tail=caption[a:];stop=re.search(r'[,.;!?]|\b(?:which|that|who|next to|behind|in front of|between)\b',tail,re.I);b=a+(stop.start() if stop else len(tail))
  if text72.ENT_RE.search(caption[a:b]):result.update(kind='ENTITY_RELATIVE',reason='simple_named_anchor',direction=edge.group(1).lower(),target_span=[0,edge.start()],constraint_span=list(edge.span()),reference_span=[a,b],single_target_role=True)
  else:result['reason']='anchor_not_an_explicit_entity'
 return result
def patched_parse(caption,ntokens):
 # The original parsing logic is retained; only the shared relation expression
 # changes. Restore the original module state after each deterministic call.
 old=text72.ALLREL;text72.ALLREL=ALLREL
 try:t=text72.parse(caption,ntokens)
 finally:text72.ALLREL=old
 edge=DIRECTION.search(caption);first=ALLREL.search(caption);simple=edge is not None and first is not None and edge.span()==first.span();t['relation']=edge.group(1).lower() if simple and t['ref_mask'].any() else None
 return t
def absolute_negatives(rects,valid,reliability,width,direction,margin=.05):
 rects=np.asarray(rects,float);known=np.asarray(valid,bool)&(np.asarray(reliability)>=.99)&np.isfinite(rects).all(1)&(width>0)&(rects[:,2]>rects[:,0])&(rects[:,3]>rects[:,1]);interval=rects[:,[0,2]]/max(float(width),1.)
 # Entire full-box interval must lie in the opposite half; boundary crossing,
 # clipping and missing projections do not certify a contradiction.
 negative=known&((interval[:,0]>.5+margin) if direction=='left' else (interval[:,1]<.5-margin))
 return negative,known,interval
