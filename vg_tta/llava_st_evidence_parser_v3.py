"""Official generated prose compatibility; never edit raw boxes or resolve conflicting spans."""
import re
from vg_tta.external_privileged_views import parse_teacher_text as old_parse,NUMBER

def parse_teacher_text(text,frame_ids,*,clip_bounds=None):
 out=old_parse(text,frame_ids,clip_bounds=clip_bounds);out['parser_v2_errors']=list(out['errors']);normalized=out['normalized_text']
 spans=re.findall(r'\{\s*('+NUMBER+r')\s*,\s*('+NUMBER+r')\s*\}',normalized)
 numeric=[list(map(float,s)) for s in spans];out['all_interval_mentions']=numeric;out['syntax_repairs']=[]
 if len(numeric)>1 and all(x==numeric[0] for x in numeric):
  a,b=numeric[0]
  if 0<=a<=b<=1:
   lo,hi=out['clip_bounds'];out['interval_physical']=[lo+a*(hi-lo),lo+b*(hi-lo)];out['temporal_usable']=True
   out['errors']=[e for e in out['errors'] if e!='missing_or_ambiguous_interval'];out['syntax_repairs'].append('identical_repeated_interval_mentions')
 # Only the exact official-style prose label is exempt, not arbitrary colons.
 headings=len(re.findall(r'\bObject bounding box\s*:',normalized,flags=re.IGNORECASE))
 out['recognized_prose_colons']=headings
 if headings and normalized.count(':')-headings==len(out['boxes']):
  out['errors']=[e for e in out['errors'] if e!='unparsed_box_entries'];out['syntax_repairs'].append('recognized_box_heading_colon')
 out['format_valid']=not out['errors'];out['schema']='physical_clip_time_v3_official_prose';return out
