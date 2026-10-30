"""Strictfirst-target globalHORIZONTAL region parser; no learned parsing/GT.
Intervals are constraints, not preferredlocations. Not a generic relation parser.
"""
from dataclasses import dataclass,asdict
import re

@dataclass(frozen=True)
class Region:
    low:float
    high:float
    relation:str
    evidence_text:str
    evidence_start:int
    evidence_end:int
    axis:str='horizontal'

OBJECT_WORDS=r'car|sedan|suv|vehicle|truck|bus|pedestrian|person|man|woman|cyclist|tree|house|building|lamppost|streetlight|pole|road|sidewalk|hydrant|shrubs|it|him|her|them'

def parse_target_horizontal_region(text):
    """Ignore reference-object and bodyorientation clauses; failclosed onambiguity."""
    if not isinstance(text,str):raise TypeError('caption text required')
    # Appendednegative sentinel is notpartof thereferringexpression.
    clean=re.split(r'\bnot mentioned\b',text,maxsplit=1,flags=re.I)[0]
    cut=re.search(r'\b(?:there\s+(?:is|are)|relationship\s+with|relationships\s+with)\b',clean,re.I)
    head=clean[:cut.start()] if cut else clean
    m=re.search(r'\bposition\s+information\s*:\s*([^;]+)',head,re.I)
    if m:fragment=m.group(1);start=m.start(1);kind='explicit_target_position_section'
    else:
        # A view/image/frameanchor is mandatory for prose. Otherobjects do not
        # become targets merelybecause theirrelative relation containsleft/right.
        m=re.search(r'\b(?:located|positioned|situated|stationary\s+at)\b(?P<position>.*?\b(?:view|frame|image)\b)',head,re.I)
        if not m:
            m=re.search(r'\b(?:located|positioned|situated)\b(?P<position>[^.;]*?\b(?:center|centre)\b)',head,re.I)
            if m and re.match(r'\s+of\s+(?:(?:the|a)\s+)?(?:road|street|car|vehicle|body)\b',head[m.end():],re.I):m=None
        if not m:return None,{'reason':'no_explicit_firsttarget_global_position_clause'}
        fragment=m.group('position');start=m.start('position');kind='target_location_with_explicit_view_anchor'
    # Remove clauses marking orientation/occlusion, after positionevidence.
    end=re.search(r'\b(?:partially|obscured|with\s+(?:its|their|the)\s+(?:body|rear|front|side|left\s+side|right\s+side)|facing|showing)\b',fragment,re.I)
    if end:fragment=fragment[:end.start()]
    # Relative-to-object is not globalimageposition; source classmentions are
    # allowed onlyoutside this extractedlocationclause.
    if re.search(r'\b(?:of|beside|behind|near|between)\s+(?:(?:the|a|an|another|other)\s+)?(?:'+OBJECT_WORDS+r')\b',fragment,re.I):return None,{'reason':'location_clause_contains_otherobject_relative_reference','evidence_text':fragment}
    left=bool(re.search(r'\bleft\b',fragment,re.I));right=bool(re.search(r'\bright\b',fragment,re.I))
    if left and right:return None,{'reason':'contradictory_horizontal_directions','evidence_text':fragment}
    if left:low,high,relation=0.,.5,'global_left_half'
    elif right:low,high,relation=.5,1.,'global_right_half'
    elif re.search(r'\b(?:center|centre|middle)\b',fragment,re.I):
        if re.search(r'\b(?:above|below|upper|lower|top|bottom)\b',fragment,re.I):return None,{'reason':'vertical_only_center_phrase_not_horizontal_constraint','evidence_text':fragment}
        low,high,relation=1/3,2/3,'explicit_global_horizontal_center'
    else:return None,{'reason':'no_horizontal_position_in_target_clause','evidence_text':fragment}
    region=Region(low,high,relation,fragment.strip(),start,start+len(fragment))
    return region,{'reason':'parsed','rule':kind,'no_fixed_mean_target':True,'only_firsttarget_global_clause':True}
