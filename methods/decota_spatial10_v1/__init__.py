"""F34 Spatial10 working core; separate from the historical production registry."""

from .predictor import Spatial10Predictor, load_config

__all__ = ["Spatial10Predictor", "load_config"]
