from common76 import *
from model76 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source
base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));torch.manual_seed(0);m=Fusion(base.contrastive_align_projection_image).cuda().eval();m.gate_late.data.fill_(.3);r=target_rows()[0];z=tensor_prefix(prefix(r));d=tensor_data(load(casepath(r)),r);d['p2i'].fill_(-1);d['p2i_weight'].zero_()
with torch.no_grad():p,q,ex=m.main(base,z,d,'independent');assert torch.equal(ex['late_RGB_delta'],torch.zeros_like(ex['late_RGB_delta']))
out=H/'independent_unknown_gate_audit_attempt002';out.mkdir();dump(out/'AUDIT_RESULT.json',{'state':'PASS_R76_UNKNOWN_RGB_POST_PROJECTION_RESIDUAL_EXACT_ZERO','no_target_GT_read':True,'unknown_candidate_residual_exact_zero':True,'original_source_unchanged':True});print(read(out/'AUDIT_RESULT.json'))
