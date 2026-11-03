"""Only correct two deprecated torchvision version guards, without vendor edits."""
from pathlib import Path
import sys,types,hashlib
def prepare(vendor):
 import torch,torchvision
 assert tuple(map(int,torchvision.__version__.split('+')[0].split('.')[:2]))>=(0,7)
 vendor=Path(vendor);sys.path.insert(0,str(vendor));import util
 assert Path(util.__file__).resolve()==(vendor/'util/__init__.py').resolve()
 path=vendor/'util/misc.py';original=path.read_text();a='float(torchvision.__version__[:3]) < 0.7';b="tuple(map(int, torchvision.__version__.split('+')[0].split('.')[:2])) < (0, 7)";assert original.count(a)==2;fixed=original.replace(a,b);assert fixed.replace(b,a)==original
 assert 'util.misc'not in sys.modules
 module=types.ModuleType('util.misc');module.__file__=str(path);module.__package__='util';sys.modules['util.misc']=module;exec(compile(fixed,str(path)+'::version_guard_only','exec'),module.__dict__);util.misc=module
 for shape in [(1,1,3,5),(0,1,3,5)]:
  x=torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float32).reshape(shape);assert torch.equal(module.interpolate(x,size=(6,10),mode='bilinear',align_corners=False),torch.nn.functional.interpolate(x,size=(6,10),mode='bilinear',align_corners=False))
 return dict(status='two_modern_torchvision_version_guards_reverse_exact_original_vendor_unchanged',torchvision_version=torchvision.__version__,original_sha256=hashlib.sha256(original.encode()).hexdigest(),runtime_source_sha256=hashlib.sha256(fixed.encode()).hexdigest(),nonempty_and_empty_interpolation_exact=True)
