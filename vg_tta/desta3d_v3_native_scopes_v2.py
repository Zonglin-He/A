"""Native positive-control scopes, v2. Historical source helper stays pinned.

L2 includes branch query pooling and the event temporal reader's output mix.
L3 means shared projection/conv scope, NOT full shared representation tuning.
No optimizer or model inference occurs here.
"""

def configure_branch(adapter,branch,level=0):
    if branch not in ('event','spatial') or level not in (0,1,2,3):raise ValueError((branch,level))
    if adapter.architecture!='dual3d':raise ValueError('requires independent dual readers')
    prefixes=[f'film_{branch}.',f'norm_{branch}.'];exact=set()
    if level>=1:prefixes.append(f'out_proj_{branch}.');exact.add(f'gate_{branch}')
    if level>=2:
        prefixes.extend([f'{branch}_reader.pointwise.',f'query_pool_{branch}.'])
        if branch=='event':prefixes.extend(f'event_temporal_reader.paths.{i}.1.' for i in range(len(adapter.event_temporal_reader.paths)))
    if level>=3:prefixes.extend(['input_proj.','shared_stem.'])
    selected=[]
    for name,p in adapter.named_parameters():
        p.grad=None;p.requires_grad_(name in exact or any(name.startswith(s) for s in prefixes))
        if p.requires_grad:selected.append((name,p))
    assert selected and all(not n.startswith('norm_stem.') for n,_ in selected)
    if level>=2:
        assert any(n.startswith(f'query_pool_{branch}.') for n,_ in selected)
        if branch=='event':assert any(n.startswith('event_temporal_reader.paths.') for n,_ in selected)
    return selected
