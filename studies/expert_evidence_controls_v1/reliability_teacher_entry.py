"""Original teacher server, explicitly preserved original component backend."""
import torch,runpy
torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.matmul.allow_tf32=False
if __name__=='__main__':runpy.run_module('stvg.teacher_worker',run_name='__main__')
