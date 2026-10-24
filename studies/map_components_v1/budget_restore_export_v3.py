"""Original post-audit scalar export, bound to the restored budget namespace."""
import sys, types
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from studies.map_components_v1 import export as original
scope = dict(vars(original))
scope.update(D=R/'outputs/map_budget_restore_v3', P=R/'results/map_budget_restore_v3')
export = types.FunctionType(original.export.__code__, scope, 'export')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['hc2', 'vidstg'])
    export('budget', parser.parse_args().dataset)
