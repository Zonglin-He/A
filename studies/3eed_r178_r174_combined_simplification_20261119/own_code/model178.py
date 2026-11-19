"""Inherited R174 weights with explicitly listed inference combinations."""
from dataclasses import replace

from scope178 import ARMS, PROTOCOL
from model177 import Ablation, OPTIONS, ReplayDecoder


def configurations():
    result = {}
    for name in ARMS:
        definition = PROTOCOL['arms'][name]
        parent = definition['parent']
        initial = Ablation() if parent is None else result[parent]
        result[name] = replace(initial,**definition['overrides'])
        assert result[name].rounds in PROTOCOL['design']['rounds']
    return result


CONFIGS = configurations()


class CombinedDecoder(ReplayDecoder):
    def run_recipe(self,evidence,name):
        option = CONFIGS[name]
        registered = f'r178_{name}'
        if registered in OPTIONS:
            assert OPTIONS[registered] == option
        else:
            OPTIONS[registered] = option
        return self.replay(evidence,registered)
