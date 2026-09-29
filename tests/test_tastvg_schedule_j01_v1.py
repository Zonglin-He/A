from scripts.run_tastvg_schedule_j01_v1 import source_order

def test_order_is_input_permutation_invariant():
    rows=[dict(source='source'+str(i),ordinal=i) for i in range(16)]
    for j in range(1,6):assert source_order(rows,j)==source_order(list(reversed(rows)),j)

def test_order_ignores_metrics_and_covers_roster_once():
    rows=[dict(source='source'+str(i),ordinal=i) for i in range(16)]
    changed=[dict(r,GT_score=i%3,expert_reward=100-i) for i,r in enumerate(rows)]
    for j in range(1,6):
        got=[r['ordinal'] for r in source_order(changed,j)]
        assert sorted(got)==list(range(16)) and got==[r['ordinal'] for r in source_order(rows,j)]
        assert len({got[i] for i in [0,4,8,12]})==4

def test_orders_use_fixed_hash_namespaces():
    rows=[dict(source='source'+str(i),ordinal=i) for i in range(16)]
    assert len({tuple(r['ordinal'] for r in source_order(rows,j)) for j in range(1,6)})==5
