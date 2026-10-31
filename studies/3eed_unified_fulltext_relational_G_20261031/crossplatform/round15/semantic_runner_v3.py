"""Safe v3 CPU adapter: unchanged v2 prompt functions, v3 parsing only."""
from semantic_runner_v2 import text_messages,visual_messages,linguistic_payload,json_safe
from semantic_graph_v3 import DEFAULT_CONFIG,parse_caption_graph,parse_visual_graph,combine_visual_graphs,match_scene_graph,proposal_consensus,match_relation_flip_diagnostic
def interpret_answers(*,caption,text_raw,visual_raws,candidates_by_order,scene,config=DEFAULT_CONFIG):
 if len(visual_raws)!=2 or len(candidates_by_order)!=2:raise ValueError('Exactly2 image numberings')
 expected={x['mask_id'] for x in scene['objects']}
 for candidates in candidates_by_order:
  if len(candidates)!=len(expected) or {x['mask_id'] for x in candidates}!=expected:raise ValueError('Whole independent mask inventory, no beam or q0 crop')
 graph=parse_caption_graph(text_raw,caption,config);parsed=[parse_visual_graph(raw,cand,graph) for raw,cand in zip(visual_raws,candidates_by_order)];visual=combine_visual_graphs(parsed)
 full=match_scene_graph(graph,scene,visual,relation_check=True,config=config);control=match_scene_graph(graph,scene,visual,relation_check=False,config=config);proposal=proposal_consensus(visual);flip=match_relation_flip_diagnostic(graph,scene,visual,config)
 return {'text_graph':graph,'visual_parsed':parsed,'combined_visual':visual,'graph_match':full,'proposal_consensus':proposal,'graph_relation_disabled':control,'graph_relation_flip':flip,'scope':'Fullquery VLM still contains relation language in control. Full search handles independent measured relations; no semantic identity truth, no neural adaptation or box refit.'}
