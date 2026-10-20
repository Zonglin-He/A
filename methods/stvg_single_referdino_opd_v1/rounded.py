"""Same tested predictor, using the accepted shorter numerical configuration."""
import json
from pathlib import Path
from .tuned_gamma import InitializationTunedSingleReferDINOOPDPredictor
class RoundedSingleReferDINOOPDPredictor(InitializationTunedSingleReferDINOOPDPredictor):
 def __init__(self,dataset='hc2',config=None,source=None,check=lambda:None):
  if config is None:config=json.loads(Path(__file__).with_name('rounded_config_20261020.json').read_text())['datasets'][dataset]
  super().__init__(dataset,config=config,source=source,check=check)
