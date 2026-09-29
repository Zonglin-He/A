from vg_tta.llava_st_evidence_parser_v3 import parse_teacher_text
from vg_tta.external_privileged_views import parse_teacher_text as old

def test_repeated_identical_special_tokens_and_prose():
 s='During the span of {<TEMP-000><TEMP-099>}. Object bounding box: <TEMP-000>:[<WIDTH-010><HEIGHT-010><WIDTH-030><HEIGHT-040>] time {<TEMP-000><TEMP-099>}.'
 e=parse_teacher_text(s,[10,11,90]);o=old(s,[10,11,90])
 assert e['interval_physical']==[10,90] and e['format_valid'] and e['boxes']==o['boxes'] and e['frame_groups']==o['frame_groups']
 assert e['all_interval_mentions']==[[0,1],[0,1]] and len(e['syntax_repairs'])==2

def test_conflicting_or_bad_mentions_and_unparsed_entries_retained():
 e=parse_teacher_text('{0,1} Object bounding box: .2:[.1,.1,.3,.4] {0,.8}',[10,20])
 assert not e['temporal_usable'] and 'missing_or_ambiguous_interval' in e['errors']
 e=parse_teacher_text('{0,1} Object bounding box: .2:[0,0,0,0] bad:[]',[10,20])
 assert 'invalid_box_geometry' in e['errors'] and 'unparsed_box_entries' in e['errors']

def test_old_valid_numerics_unchanged():
 s='{.2,.9} .3:[.1,.1,.3,.4] .8:[.3,.2,.5,.6]';o=old(s,[10,20,30]);e=parse_teacher_text(s,[10,20,30])
 for k in ['interval_physical','boxes','frame_groups','errors','temporal_usable','spatial_usable','format_valid']:assert o[k]==e[k]
