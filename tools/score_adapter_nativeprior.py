"""Independent CPU scorer: repair ONLY the frozen scorer's local-name collision.

The inference plan, its bound source and every sealed prediction stay unchanged.
The only AST edit alpha-renames the severe-TSV loop's local `metric` variable.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import ast, argparse, copy, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from full import RUN, PUBLIC, read, sha, atomic, identity, check

def adapt(source):
    tree=ast.parse(source);original=ast.dump(tree,include_attributes=False)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    loops=[n for n in ast.walk(run)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='metric']
    assert len(loops)==1
    loop=loops[0];assert isinstance(loop.iter,ast.List)and [n.value for n in loop.iter.elts]==['v','t']
    changed=0
    for node in ast.walk(loop):
        if isinstance(node,ast.Name)and node.id=='metric':node.id='metric_field';changed+=1
    assert changed==4
    restored=copy.deepcopy(tree)
    for node in ast.walk(restored):
        if isinstance(node,ast.Name)and node.id=='metric_field':node.id='metric'
    assert ast.dump(restored,include_attributes=False)==original
    ast.fix_missing_locations(tree)
    return tree,dict(edit='alpha rename severe-output loop variable metric -> metric_field only',renamed_Name_nodes=changed,reverse_edit_restores_entire_original_AST=True,original_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),adapter_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())

def main(ds):
    p=check();dest=RUN/ds;seal=read(dest/'sealed.json')
    assert seal['queries']==p['jobs'][ds]['queries']and seal['fingerprint']==p['fingerprint']and seal['GT_read']is False
    original=ROOT/'tools/full_score.py';assert sha(original)==p['code']['tools/full_score.py']
    tree,proof=adapt(original.read_text());before={k:sha(dest/e['path'])for k,e in seal['files'].items()}
    for k,e in seal['files'].items():assert before[k]==e['sha256']
    proof.update(original_source_sha256=sha(original),adapter_source_sha256=sha(__file__),plan_sha256=sha(RUN/'plan.json'),seal_sha256=sha(dest/'sealed.json'),labels_sha256=p['jobs'][ds]['labels_sha256'],dataset=ds,queries=seal['queries'],CPU_only=True,new_inference=0,new_expert=0,**identity())
    atomic(dest/'score_adapter_worker.json',proof);started=time.perf_counter()
    namespace={'__name__':'_frozen_nativeprior_score_adapter','__file__':str(original)}
    exec(compile(tree,str(original)+'[local-name-adapter]','exec'),namespace)
    assert callable(namespace['metric']);namespace['run'](ds)
    assert all(sha(dest/e['path'])==before[k]for k,e in seal['files'].items())
    proof.update(status='passed_completed_CPU_score',seconds=time.perf_counter()-started,score_receipt_sha256=sha(dest/'score_receipt.json'),all_predictions_byteidentical=True,original_bound_source_still_unchanged=sha(original)==p['code']['tools/full_score.py'])
    atomic(dest/'score_adapter_receipt.json',proof);atomic(PUBLIC/ds/'CPU_SCORE_ADAPTER.json',proof)
    print(json.dumps({k:v for k,v in proof.items()if not k.endswith('_sha256')}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['hc2','vidstg']);a=ap.parse_args();main(a.dataset)
