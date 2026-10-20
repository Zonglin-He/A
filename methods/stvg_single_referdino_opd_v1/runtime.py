"""Frozen TA-STVG loading. No independent detector construction."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
from methods.stvg_referdino_opd_v1.runtime import setup,lease

def load_source(config):
    from methods.decota_final_simplified_v1._tastvg_load import official_imports,load_model_on_device
    official_imports();pipeline=sys.modules['models.pipeline'];old=pipeline.load_json;seen=[]
    def empty(path):
        assert str(path).endswith(('/annos/train.json','/annos/test.json'));seen.append(str(path));return {}
    pipeline.load_json=empty
    try:
        src='vidstg'if config['source_dataset']=='vidstg'else'hcstvg2';runtime='vidstg'if src=='vidstg'else'hc-stvg2'
        model,_,_=load_model_on_device(src,ROOT/'artifacts/tastvg_runtime'/runtime,
            checkpoint=ROOT/config['source_checkpoint'],device='cuda',source_dataset=src)
    finally:pipeline.load_json=old
    assert len(seen)==2 and model.verb_label=={}and model.verb_label2=={}
    return model.eval().requires_grad_(False)
