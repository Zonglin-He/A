"""User-selected DeCoTA C+D. No experiment runners are imported on import."""

from .config import MethodConfig


def __getattr__(name):
    if name == "DeCoTAPredictor":
        from .predictor import DeCoTAPredictor
        return DeCoTAPredictor
    raise AttributeError(name)


__all__ = ["DeCoTAPredictor", "MethodConfig"]
