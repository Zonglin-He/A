"""Synthetic-only schema check, before any real target score/annotation access."""
from common69 import *
import io,ijson,sys
sys.path.insert(0,str(ROOT/'research/round5/audit'));from independent_core_v2 import transform_gt
text=b'{"ground_info":[{"bbox_3d":[1.2,2.3,3.4,4.5,1.6,1.7,0.8,0.0,0.0]}]}'
old=next(ijson.items(io.BytesIO(text),'ground_info.item'));failure=None
try:transform_gt(old['bbox_3d'],'drone')
except TypeError:failure='TypeError_Decimal_plus_float'
assert failure is not None
patched=next(ijson.items(io.BytesIO(text),'ground_info.item',use_float=True));actual=transform_gt(np.asarray(patched['bbox_3d'],dtype=np.float64),'drone');expected=np.array([1.2,2.3,5.2,4.5,1.6,1.7,0.8],np.float32);assert np.array_equal(actual,expected)
quad=transform_gt(patched['bbox_3d'],'quad');assert np.array_equal(quad,np.array([1.2,2.3,3.4,4.5,1.6,1.7,0.8],np.float32));integers=transform_gt(np.asarray([1,2,3,4,5,6,0,0,0],dtype=np.float64),'drone');assert integers[2]==np.float32(4.8);assert not (ROOT/'research/round69/score_all2304_attempt001').exists()
out=ROOT/'research/round69/attempts/score_parser_preflight001/PREFLIGHT_RESULT.json';assert not out.exists();dump(out,{'state':'PASS_SYNTHETIC_ONLY_STREAMING_FLOAT_SCHEMA_AND_OFFICIAL_DRONE_SHIFT','old_default_parser_failure':failure,'patched_float_parser_matches_explicit_float32':True,'explicit_float64_box_input_prevents_integer_shift_truncation':True,'actual_annotation_media_feature_or_GT_file_access':False,'task_predictions_or_method_changed':False});print(read(out))
