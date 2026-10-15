"""Identical candidate/render algorithm; one crop prompt serialization for cached and fresh clocks."""
from scripts import stvg_evidence_spatial_inputs005 as _base
_original_prompt=_base._prompt
def _matched_prompt(query,context_times,local_times,ids,presentation):
 if presentation=='crop':return _base._confirmation003_prompt(query,context_times,local_times,ids)
 return _original_prompt(query,context_times,local_times,ids,presentation)
_base._prompt=_matched_prompt
from scripts.stvg_evidence_spatial_inputs005 import *
