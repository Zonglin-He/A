"""Check raw scene deployment against sealed boxes and the previous frozen semantic outputs."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


def examples():
    from predict_direct import PRIVATE, catalog, digest, read, row_id
    from common import prior
    from scope183 import VIEWS, frame, ordinary
    from score_roles import sensor_frame_box
    assert digest(PRIVATE/'decisions.npz') == read(PRIVATE/'SEAL.json')['decisions']
    with np.load(PRIVATE/'decisions.npz') as z:
        boxes = dict(zip(z['keys'],z['subjects']))
    roster = ordinary()
    groups = defaultdict(list)
    controlled = {}
    for row in catalog()['rows']:
        if row['scope']=='ordinary':
            groups[row['source'],row['target']].append(row)
        elif row['scope']=='noun':
            controlled.setdefault(row['source'],row)
    def example(src,form,f,b):
        return dict(source=src,form=form,inputs=f,box=b.tolist())
    for (src,tgt),rows in sorted(groups.items()):
        chosen = min(rows,key=lambda r:(np.array_equal(boxes[row_id(r)][0],boxes[row_id(r)][-1]),r['ordinal']))
        rec = roster[tgt][chosen['ordinal']]
        yield example(src,'simple',frame(rec),sensor_frame_box(boxes[row_id(chosen)][-1],tgt,rec['suffix']))
        folder = prior.FULL/'predictions'/(src+'_to_'+tgt)
        assert digest(folder/'all.npz')==read(folder/'SEALED.json')['all']
        with np.load(folder/'all.npz') as z:
            ix = int(np.flatnonzero(~z['registered'])[0])
            yield example(src,'fallback',frame(roster[tgt][ix]),z['completed'][ix,prior.old.ARMS.index('shared_cut')])
    obs = {(v['source'],v['uid']):v['frame'] for v in prior.noun_inputs()}
    for src,row in sorted(controlled.items()):
        f = obs[src,row['uid']]
        yield example(src,'ordered',f,sensor_frame_box(boxes[row_id(row)][-1],f['target'],VIEWS.index(f['waymo_view'])))


def worker(package, asset_root, old_package, manifest, output):
    import torch
    sys.path.insert(0,str(package.resolve()))
    from crossplatform.ground_regions import Grounder, semantic
    from crossplatform import ground as frozen
    old = old_package/'crossplatform/physical_regions/semantic.py'
    assert sha256(old.read_bytes()).hexdigest()=='a6789b893c545fb7bae104350ff8cd0e9bc87697bb1a8715e191bf61037c46ef'
    spec = spec_from_file_location('crossplatform.physical_regions.old_semantic',old)
    previous = module_from_spec(spec)
    spec.loader.exec_module(previous)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    current = engine = None
    checked = []
    costs = {}
    observe = semantic.observe
    for ex in sorted(json.loads(manifest.read_text()),key=lambda e:e['source']):
        src,form,f = ex['source'],ex['form'],ex['inputs']
        if current!=src:
            engine = None
            engine = Grounder(src,asset_root=asset_root)
            current = src
        def verify_semantic(model,*args):
            def call(fn):
                counts = Counter()
                def count(name):
                    def hook(*_):
                        counts[name]+=1
                    return hook
                heads = {'box':model.prediction_heads[-1],'visual':model.contrastive_align_projection_image,
                         'text':model.contrastive_align_projection_text}
                handles = [head.register_forward_hook(count(name)) for name,head in heads.items()]
                try:
                    return fn(model,*args),dict(counts)
                finally:
                    for handle in handles:
                        handle.remove()
            after,a = call(observe)
            before,b = call(previous.observe)
            np.testing.assert_array_equal(after[0],before[0])
            for field in ('extent','xyz'):
                np.testing.assert_array_equal(after[1][field],before[1][field])
            assert a==dict(box=6,visual=7,text=7) and b==dict(box=13,visual=13,text=13)
            costs[src]=dict(before=b,after=a,all_noun_logits_and_extents_equal=True)
            return after
        comparison = verify_semantic if form=='simple' and src not in costs else observe
        with (patch.object(semantic,'observe',side_effect=comparison) as noun,
              patch.object(frozen,'decoder_evidence',wraps=frozen.decoder_evidence) as fallback):
            actual = engine.predict(**f)
        np.testing.assert_array_equal(actual,np.asarray(ex['box']))
        assert noun.call_count==int(form!='fallback') and fallback.call_count==int(form=='fallback')
        checked.append(dict(source=src,target=f['target'],form=form,seven_coordinates_exact=True))
    assert len(checked)==14 and len(costs)==3
    output.write_text(json.dumps(dict(cases=checked,prompt_bank_terminal_calls=costs)))


def check(package,old_package):
    from predict_direct import OUT, PREVIOUS, ROOT, save
    env = {k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        cases = list(examples())
        manifest = folder/'examples.json'
        manifest.write_text(json.dumps(cases,default=str))
        def run(*args):
            process = subprocess.run([sys.executable,'-I',*map(str,args)],cwd=folder,env=env,
                                     capture_output=True,text=True)
            if process.returncode:
                raise RuntimeError(process.stderr[-4000:])
            return process.stdout
        report = folder/'results.json'
        run(__file__,'worker',package,ROOT,old_package,manifest,report)
        result = json.loads(report.read_text())
        result['cli'] = []
        for form in ('simple','ordered','fallback'):
            case = next(c for c in cases if c['form']==form)
            args = ['--asset-root',str(ROOT),'--source',case['source']]
            for key,value in case['inputs'].items():
                args.extend(['--'+key.replace('_','-'),value])
            got = json.loads(run(package/'crossplatform/ground_regions.py',*args))
            assert got['frame']=='target_lidar'
            np.testing.assert_array_equal(got['box'],case['box'])
            result['cli'].append(dict(form=form,source=case['source'],target=case['inputs']['target'],exact=True))
        single = folder/'single.json'
        single.write_text(json.dumps(next(c for c in cases if c['form']=='simple'),default=str))
        versions=[]
        for p in (old_package,package):
            run(PREVIOUS/'fallback.py','worker',p,ROOT,single,report)
            versions.append(json.loads(report.read_text()))
        for name in ('empty_inventory','no_feasible_pair'):
            np.testing.assert_array_equal(versions[0][name],versions[1][name])
        result['late_fallbacks_exact'] = sorted(versions[0])
        result['FINAL_access'] = False
        OUT.mkdir(exist_ok=True)
        save(OUT/'entry.json',result)
        print('Checked 14 raw cases, 3 isolated CLI cases, 3 real prompt banks and 2 late fallbacks')


if __name__=='__main__':
    if sys.argv[1]=='worker':
        worker(*map(Path,sys.argv[2:]))
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('current_package',type=Path)
        parser.add_argument('previous_package',type=Path)
        args=parser.parse_args()
        check(args.current_package.resolve(),args.previous_package.resolve())
