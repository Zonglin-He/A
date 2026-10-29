"""Safe CPU model/state checks required before regenerating a public local plan."""
import pathlib,sys,argparse,json,torch,hashlib
R=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'detectron2'),str(R/'upstream'),str(R/'upstream/mask2former/modeling/pixel_decoder/ops')]
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from mask2former import add_maskformer2_config
from avism import add_avism_config
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args();c=get_cfg();add_deeplab_config(c);add_maskformer2_config(c);add_avism_config(c);c.merge_from_file(str(R/'upstream/configs/avism/R50/avism_R50_IN.yaml'));c.MODEL.DEVICE='cpu';c.MODEL.WEIGHTS='';m=build_model(c);s=torch.load(a.checkpoint,weights_only=True,map_location='cpu',mmap=True);s=s['model'] if 'model' in s else s;m.load_state_dict(s,strict=True)
report={'missing':[],'unexpected':[],'shape_mismatches':{},'expected_keys':len(m.state_dict()),'metadata_keys':len(s),'audio_fusion_norm_parameters':[n for n,p in m.avism_module.named_parameters() if n.startswith(('enc_av_cross_attn.','enc_av_ffn.')) and '.norm.' in n]};(R/'logs').mkdir(exist_ok=True);(R/'logs/model_metadata_compatibility.json').write_text(json.dumps(report,indent=2));checkpoint=pathlib.Path(a.checkpoint);digest=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest();assert digest=='83543b712a3fcf6d5392e536ae512ef606adce404122105ec2b9dccf0327de5e';checkpoint.with_suffix(checkpoint.suffix+'.download.json').write_text(json.dumps({'state':'completed','bytes':checkpoint.stat().st_size,'sha256':digest,'verification':'existing file checked against known official full checkpoint SHA256'},indent=2));print('Strict CPU check passed',len(s),'tensors; official full checkpoint SHA verified')
