import itertools
import torch
from methods.decota_final_simplified_v1.objectives import prediction
from vg_tta.tastvg_corruption_c0c1_v1 import candidates,ranked_spans,envelope


def test_temporal_heap_matches_exhaustive_cartesian_with_ties():
    ids=list(range(8));records=[{'frame_ids':ids[::2]},{'frame_ids':ids[1::2]}]
    for seed in range(3):
        torch.manual_seed(seed);zz=[torch.zeros(1,4,2) if seed==0 else torch.randn(1,4,2) for _ in range(2)]
        pred=prediction(zz,torch.rand(8,4),records,ids);actual=candidates([pred],records,ids)['temporal'];aa,bb=map(ranked_spans,zz)
        expected=[pred['indices']]
        for _,i,j in sorted([(-(a[0]+b[0]),i,j) for i,a in enumerate(aa) for j,b in enumerate(bb)]):
            ij=envelope(aa[i][1:],bb[j][1:],records,ids)
            if ij not in expected:expected.append(ij)
            if len(expected)==8:break
        assert [x['indices'] for x in actual]==expected


def test_spatial_candidates_are_whole_tubes_and_native_first():
    ids=list(range(8));records=[{'frame_ids':ids[::2]},{'frame_ids':ids[1::2]}];z=[torch.zeros(1,4,2)]*2
    a=prediction(z,torch.zeros(8,4),records,ids);b=prediction(z,torch.ones(8,4),records,ids)
    out=candidates([a,b,b],records,ids)
    assert len(out['spatial'])==2 and out['spatial'][0]['origin']=='layer3'
    assert torch.equal(out['spatial'][0]['boxes'],b['boxes']) and torch.equal(out['spatial'][1]['boxes'],a['boxes'])
    assert out['temporal'][0]['indices']==b['indices']
