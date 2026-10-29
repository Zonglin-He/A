"""CPU-only full tensor checkpoint check; never runs baseline inference or touches GPU."""
import os,sys,pathlib,json,hashlib,time,argparse
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
root=pathlib.Path(__file__).resolve().parents[1];torch.set_num_threads(4)
os.environ['GREAT_TEXT_MODEL']=str(root/'assets/robbert-v2-dutch-base');sys.path.insert(0,str(root/'adapted'))
from model.GREAT import GREAT
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default=str(root/'downloads/checkpoints/best_unseen_aff.pt'));ap.add_argument('--output',default='checkpoint_full_validation.json');args=ap.parse_args();p=pathlib.Path(args.checkpoint);t=time.perf_counter();checkpoint=torch.load(p,map_location='cpu',weights_only=True);sd={k.removeprefix('module.'):v for k,v in checkpoint['model'].items()};assert all(torch.isfinite(v).all() for v in sd.values());m=GREAT(pre_train=False);loaded=m.load_state_dict(sd,strict=True)
summary={'kind':'engineering_only','checkpoint_sha256':hashlib.file_digest(open(p,'rb'),'sha256').hexdigest(),'size':p.stat().st_size,'weights_only_safe_load':True,'strict_load':not loaded.missing_keys and not loaded.unexpected_keys,'all_model_tensors_finite':True,'state_keys':len(sd),'parameter_count':sum(x.numel() for x in m.parameters()),'text_embedding_shape':list(m.text_encoder.text_encoder.embeddings.word_embeddings.weight.shape),'text_tower_embedding_equal':torch.equal(m.text_encoder.text_encoder.embeddings.word_embeddings.weight,m.text_encoder2.text_encoder.embeddings.word_embeddings.weight),'poolers_equal':torch.equal(m.text_encoder.text_encoder.pooler.dense.weight,m.text_encoder2.text_encoder.pooler.dense.weight),'seconds':time.perf_counter()-t}
(root/args.output).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
status=json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').exists() else {};status.update(checkpoint_download='complete' if p.name=='best_unseen_aff.pt' else 'original_partial_model_tensors_crc_recovered',checkpoint_strict_load='passed',validated_checkpoint=str(p),gpu_ready=True);(root/'STATUS.json').write_text(json.dumps(status,indent=2))
