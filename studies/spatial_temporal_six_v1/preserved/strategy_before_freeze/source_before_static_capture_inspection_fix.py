"""Five spatial estimators and a common temporal front, no timing claims.

This module has no environment/GPU/cache side effects. Models and input/GT/SHA
contracts belong to separately frozen execution plans. Pure's exact original
function is loaded by AST once at startup, never by importing its runtime.
"""
import ast,inspect,textwrap
from pathlib import Path
import numpy as np
import torch
import matching
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval
from stvg.representation import extract
ROOT=Path(__file__).resolve().parents[2]
_tree=ast.parse((ROOT/'studies/pure_expert_interpolation_v1/run.py').read_text())
_node=[n for n in _tree.body if isinstance(n,ast.FunctionDef)and n.name=='pure_interpolation'];assert len(_node)==1
_scope={'np':np};exec(compile(ast.Module(body=_node,type_ignores=[]),'<original_pure_interpolation_numeric_function>','exec'),_scope)
pure_interpolation=_scope['pure_interpolation']
ARMS=['DirectE','PureExpert','Matching','Independent','Current']
def temporal_front(cache):
 features,support=extract(cache,np.asarray(cache['E']));q,detail=one_step(features,cache['posterior']['Native'],kappa=0,beta=1);return map_interval(cache['frame_ids'],q),q,features
def spatial(arm,*,ids,positions,expert_boxes,E,evidence,interval,image_size,matching_config=None):
 assert arm in ARMS
 if arm=='DirectE':return np.asarray(E,dtype=np.float64).copy()
 if arm=='PureExpert':return pure_interpolation(ids,positions,expert_boxes)
 if arm=='Matching':
  assert matching_config is not None
  out,diag=matching.propagate(evidence.features,evidence.valid,positions,evidence.seeds,expert_boxes,(evidence.height,evidence.width),image_size,**matching_config);return out
 geometry=SpatialGeometry(evidence,E,expert_boxes);bank=geometry.for_interval(interval)
 if arm=='Independent':
  valid=bank.valid.clone();valid[~bank.active,1:]=False;actions=bank.score_unary.masked_fill(~valid,-torch.inf).argmax(1);return bank.boxes[torch.arange(bank.frames),actions].numpy()
 return evidence_dp(bank)['final'].numpy()
def capture_front_class():
 """Same live Source+teacher+E, without constructing unused spatial geometry.

Every strategy computes the actual common Current temporal posterior. Pure and
DirectE need no postfusion mask evidence/transport. Matching needs evidence;
Independent/Current build their geometry inside the spatial strategy boundary.
The optional fixed positions are reserved for the explicit density intervention.
 """
 from stvg import predictor as old
 from stvg.current_shared_roi import predictor_class
 selected=predictor_class();code=textwrap.dedent(inspect.getsource(selected.capture));line="positions=observation_positions(U,native['indices'])";assert code.count(line)==1
 code=code.replace(line,"positions=(list(self.fixed_positions)if self.fixed_positions is not None else observation_positions(U,native['indices']))")
 block="evidence=SourceEvidence.from_capture(spatial,U,positions,teacher['native_mask_logits'],teacher['input_shape'][-2:],selected_query=teacher['selected_query'])"
 assert code.count(block)==1;code=code.replace(block,"evidence=(SourceEvidence.from_capture(spatial,U,positions,teacher['native_mask_logits'],teacher['input_shape'][-2:],selected_query=teacher['selected_query'])if self.requires_feature_evidence else None)")
 geom="cache['geometry']=build_geometry(evidence,cache['E'],teacher['boxes'],collect_mask_diagnostics=collect_mask_diagnostics)";assert code.count(geom)==1;code=code.replace(geom,"cache['geometry']=None")
 scope={**vars(old),'map_interval':map_interval};exec(compile(code,'<same_Source_teacher_E_optional_evidence_no_spatial_geometry_capture>','exec'),scope)
 return type('SameCurrentTemporalFrontWithRequiredSpatialInputs',(selected,),dict(capture=scope['capture'],fixed_positions=None,requires_feature_evidence=True))
