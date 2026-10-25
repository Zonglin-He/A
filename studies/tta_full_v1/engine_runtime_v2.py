"""Independent metadata/interface fixes; frozen numerical engine remains intact."""
from full_common import *
import inspect, textwrap

old = ns.load_old('_full_TTA_engine_preserved_v1', S/'engine.py')

def replacements():
    acquire = textwrap.dedent(inspect.getsource(old.Engine.acquire))
    before = "dict(kind='sealed_existing_native_context',**e)"
    after = "dict(e,kind='sealed_existing_native_context',original_context_kind=e['kind'])"
    assert acquire.count(before) == 1
    fixed_acquire = acquire.replace(before, after)
    assert fixed_acquire.replace(after, before) == acquire
    process = textwrap.dedent(inspect.getsource(old.Engine.process))
    before_parent = 'cap,par=parent(self.ds,row)'
    after_parent = 'cap,parent_proof=parent(self.ds,row)'
    assert process.count(before_parent) == 1 and process.count('parent=par,context_origin=origin') == 1
    fixed_process = process.replace(before_parent, after_parent).replace('parent=par,context_origin=origin', 'parent=parent_proof,context_origin=origin')
    assert fixed_process.replace(after_parent,before_parent).replace('parent=parent_proof,context_origin=origin','parent=par,context_origin=origin') == process
    return acquire, fixed_acquire, process, fixed_process

original_acquire, corrected_acquire, original_process, corrected_process = replacements()
scope = dict(vars(old))
exec(compile(corrected_acquire+'\n'+corrected_process,'<full_TTA_metadata_interface_v2>','exec'),scope)

class Engine(old.Engine):
    acquire = scope['acquire']
    process = scope['process']
