"""Keep original parent local; rename only gradient-cleanup loop binding."""
from full_common import *
import inspect,textwrap
old=ns.load_old('_full_TTA_engine_original_for_v3',S/'engine.py')

def replacements():
    acquire=textwrap.dedent(inspect.getsource(old.Engine.acquire))
    before="dict(kind='sealed_existing_native_context',**e)"
    after="dict(e,kind='sealed_existing_native_context',original_context_kind=e['kind'])"
    assert acquire.count(before)==1
    fixed_acquire=acquire.replace(before,after)
    process=textwrap.dedent(inspect.getsource(old.Engine.process))
    before_loop='for par in dec.parameters():par.grad=None'
    after_loop='for reset_parameter in dec.parameters():reset_parameter.grad=None'
    assert process.count(before_loop)==1
    fixed_process=process.replace(before_loop,after_loop)
    assert fixed_acquire.replace(after,before)==acquire
    assert fixed_process.replace(after_loop,before_loop)==process
    return acquire,fixed_acquire,process,fixed_process

_,a,_,p=replacements();scope=dict(vars(old))
exec(compile(a+'\n'+p,'<full_TTA_origin_gradient_loop_interface_v3>','exec'),scope)

class Engine(old.Engine):
    acquire=scope['acquire']
    process=scope['process']
