"""Exact archived DECOTA burst coverage and operators, current decoder only."""
from pathlib import Path
import ast, hashlib, math
import numpy as np

REF = Path(__file__).parent / 'reference'

def original(name, file):
    tree = ast.parse((REF / file).read_text())
    f = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    scope = dict(np=np, hashlib=hashlib, math=math)
    exec(compile(ast.Module(body=[f], type_ignores=[]), str(REF / file), 'exec'), scope)
    return scope[name]

burst_spec = original('burst_spec', 'deployment_original.py')
corrupt_local = original('corrupt_local', 'local_original.py')
FAMILIES = ['frame_drop', 'frame_freeze', 'motion_blur', 'occlusion', 'exposure']
COVERAGES = [2.5, 5., 10.]

def conditions():
    return ([dict(id='legacy_' + n, kind='legacy', name=n) for n in ['Blur', 'JPEG', 'Dim']] +
            [dict(id=f'{f}_p{p:g}'.replace('.', 'p'), kind='burst', family=f, coverage_percent=p)
             for f in FAMILIES for p in COVERAGES])

def apply(frames, ids, metadata, ds, condition):
    from stvg.decode import decode
    assert frames.dtype == np.uint8 and len(frames) == len(ids)
    if condition['kind'] == 'legacy':
        from studies.paper_review_suite_v1.shifts import transform
        result = transform(frames, condition['name'])
        info = dict(kind='legacy_whole_sampled_clip', selected_positions=list(range(len(ids))),
                    magnitude='Original Blur5sigma1.5/JPEG30/Dim.5, single fixed strength')
    else:
        spec = burst_spec(str(metadata['source']), metadata['frame_count'], ids, condition['coverage_percent'])
        family = condition['family']
        if family == 'frame_freeze':
            result = frames.copy()
            if spec['positions']:
                donor = max(0, spec['physical_start'] - 1)
                image = frames[ids.index(donor)] if donor in ids else decode({**metadata, 'frame_ids': [donor]}, ds)[0][0]
                result[spec['positions']] = image
                spec['donor_physical_frame'] = donor
        else:
            result = corrupt_local(frames, spec['positions'],
                                   {'frame_drop': 'black', 'exposure': 'overexposure'}.get(family, family),
                                   str(metadata['source']))
        info = dict(kind='DECOTA_physical_burst', family=family, requested_coverage_percent=condition['coverage_percent'],
                    **spec, no_sampled_hit=not bool(spec['positions']))
    assert result.shape == frames.shape and result.dtype == frames.dtype
    return result, info
