"""Strict legacy input metadata recovery from its original hash-bound NPZ.

No payload is edited and no hash/box/expert equality is weakened. A missing
source-state digest is read from the exact old cache referenced by the P0 input,
after every shared pixel/interval/frame/readout/expert field is checked.
"""
import torch
from scripts.stvg_opd_paper_hc2_revision_common_v2 import ROOT,sha
from scripts.decota_paper_common_v1 import load_npz
from scripts.run_decota_paper_main_v1 import pack_expert,unpack_expert
from scripts.decota_opd_tuning_common_v1 import digest

LEGACY={'dataset','source','parent','pixel_sha256','frame_ids','native_boxes','interval','indices',
        'expert','corruption_spec','new_DINO_calls','old_cache_sha256','GT_read'}


def normalized(inp):
    assert not inp['GT_read']
    if 'source_model_state_sha256' in inp:
        assert inp['dataset'] in ['hc2','vidstg']
        assert isinstance(inp['source_model_state_sha256'],str) and len(inp['source_model_state_sha256'])==64
        assert isinstance(inp['observation_budget'],int) and inp['observation_budget'] in [1,2,4,8]
        return inp
    assert set(inp)==LEGACY,'Unsupported legacy input schema must fail closed'
    ds=inp['dataset'];assert ds in ['hc2','vidstg']
    assert inp['source']==('vidstg' if ds=='hc2' else 'hcstvg2')
    assert inp['corruption_spec'] is None and inp['new_DINO_calls']==0
    assert isinstance(inp['parent'],int) and inp['parent']>=0
    assert isinstance(inp['old_cache_sha256'],str) and len(inp['old_cache_sha256'])==64
    oldjob='t1_ours_hc2' if ds=='hc2' else 't1_ours_vid'
    path=ROOT/'artifacts/decota_paper_experiments_v1'/oldjob/'inputs/clean'/f'{inp["parent"]:05}.npz'
    assert sha(path)==inp['old_cache_sha256'],'Legacy provenance cache bytes differ'
    arrays,md,receipt=load_npz(path)
    assert receipt['sha256']==inp['old_cache_sha256'] and not md['GT_read']
    for field in ['pixel_sha256','interval','frame_ids','indices']:
        assert md[field]==inp[field],field
    assert torch.equal(torch.from_numpy(arrays['native_boxes']),inp['native_boxes'])
    assert digest(pack_expert(unpack_expert(md['expert'])))==digest(inp['expert'])
    source=md['source_model_state_sha256'];assert isinstance(source,str) and len(source)==64
    return {**inp,'source_model_state_sha256':source,'query_ordinal':inp['parent'],'observation_budget':4}


def input_equivalent(left,right):
    a,b=normalized(left),normalized(right)
    for field in ['dataset','source','query_ordinal','pixel_sha256','source_model_state_sha256',
                  'interval','frame_ids','indices','observation_budget']:
        assert a[field]==b[field],field
    assert torch.equal(a['native_boxes'],b['native_boxes'])
    assert digest(a['expert'])==digest(b['expert']),'Different admitted expert evidence'


def install():
    import scripts.stvg_opd_paper_later_common_v1 as common
    import scripts.run_stvg_opd_paper_components_v2 as runner
    common.input_equivalent=input_equivalent;runner.input_equivalent=input_equivalent
