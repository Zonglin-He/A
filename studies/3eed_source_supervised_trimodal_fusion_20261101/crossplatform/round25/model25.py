"""Source-only model-family factory, shared by cached and cache-free inference."""
from common25 import *
from fusion25 import TrimodalFusion
from rank_fusion25 import RankFusion

def make_model(family,projection=None):return (RankFusion if family=='residual_rank' else TrimodalFusion)(projection=projection)
def state_path(config):return BASE/config['training_folder']/'states_PRIVATE'/config['state_name']
