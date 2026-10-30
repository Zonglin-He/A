"""Waymo-only official training components; isolated imports, no joint checkpoint path."""
import os,sys,random,json,hashlib
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1';sys.dont_write_bytecode=True
import numpy as np
import torch
from torch.utils.data import DataLoader,DistributedSampler
from types import SimpleNamespace
R=Path(__file__).resolve().parent;P=R.parent;U=P/'round3/p0/upstream';DATA=P/'dataset_full/3eed_dataset';ROBE=(P/'upstream/data/roberta_base').resolve();sys.path.insert(0,str(U))
from src.joint_det_dataset import Joint3DDataset
from models import BeaUTyDETR,HungarianMatcher,SetCriterion,compute_hungarian_loss
from utils.lr_scheduler import get_scheduler

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def configure(plan):
 runtime=R/'runtime';(runtime/'data').mkdir(parents=True,exist_ok=True);link=runtime/'data/roberta_base'
 if not link.exists():link.symlink_to(ROBE,target_is_directory=True)
 assert link.resolve()==ROBE and (ROBE/'model.safetensors').is_file();os.chdir(runtime);torch.set_num_threads(plan.get('torch_num_threads',4));seed=plan['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 # Exact released entry-point cuDNN flags; actual TF32 flags separately recorded.
 torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=True

def seed_worker(worker_id):
 seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed);np.random.seed((int(np.random.get_state()[1][0])+worker_id)%2**32)

def dataset(split):
 assert split in ['train','val'];ds=Joint3DDataset(dataset_dict={'waymo':1},test_dataset={'waymo':1},split=split,use_color=True,use_height=False,detect_intermediate=True,data_path=str(DATA),split_dir=str(DATA/'splits'),butd=False,butd_gt=False,butd_cls=False,augment_det=False,debug=False)
 assert set(a['dataset'] for a in ds.annos)=={'waymo'};assert len(ds)==(3682 if split=='train' else 3782);return ds

def order_contract(ds,split):
 join=P.parent/'research/round6/source_data_audit'/('OFFICIAL_'+split.upper()+'_ANNO_ORDER_PRIVATE.json');rows=json.loads(join.read_text());assert len(rows)==len(ds)
 for a,b in zip(ds.annos,rows):
  assert a['meta_path']==b['meta_path'];assert hashlib.sha256(a['utterance'].encode()).hexdigest()==b['caption_sha256'];assert a['target']==b['true_source_class']
 order=hashlib.sha256(json.dumps([(r['meta_path'],r['object_index'],r['caption_sha256'],r['true_source_class']) for r in rows],separators=(',',':')).encode()).hexdigest();return {'n':len(ds),'order_sha256':order,'independent_join_sha256':sha(join)}

def loader(ds,split,generator,plan):
 train=split=='train';sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=train,seed=0);kwargs=dict(batch_size=24,shuffle=False,num_workers=plan.get('num_workers',4),worker_init_fn=seed_worker,pin_memory=True,sampler=sampler,drop_last=train,generator=generator)
 if kwargs['num_workers']:kwargs['multiprocessing_context']='spawn'
 return DataLoader(ds,**kwargs)

def model():
 # Pretrained generic RoBERTa only; all visual/grounding modules use original constructor initialization.
 m=BeaUTyDETR(num_class=256,num_obj_class=485,input_feature_dim=3,num_queries=256,num_decoder_layers=6,self_position_embedding='loc_learned',contrastive_align_loss=True,butd=False,pointnet_ckpt=None,self_attend=True)
 assert all(not p.requires_grad for p in m.text_encoder.parameters());return m

def optimizer(m):
 return torch.optim.AdamW([{'params':[p for n,p in m.named_parameters() if 'backbone_net' not in n and 'text_encoder' not in n and p.requires_grad]},{'params':[p for n,p in m.named_parameters() if 'backbone_net' in n and p.requires_grad],'lr':1e-3},{'params':[p for n,p in m.named_parameters() if 'text_encoder' in n and p.requires_grad],'lr':1e-5}],lr=1e-4,weight_decay=.0005)

def scheduler(opt,nsteps):
 args=SimpleNamespace(lr_scheduler='step',lr_decay_epochs=[25,26],lr_decay_rate=.1,warmup_epoch=-1,warmup_multiplier=100,max_epoch=100);return get_scheduler(opt,nsteps,args)
def criterion():return SetCriterion(matcher=HungarianMatcher(1,0,2,True),losses=['boxes','labels','contrastive_align'],eos_coef=.1,temperature=.07)
def to_cuda(batch):return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
def compute(m,batch,crit):
 out=m({'point_clouds':batch['point_clouds'].float(),'text':batch['utterances']})
 for k,v in batch.items():assert k not in out;out[k]=v
 return compute_hungarian_loss(out,6,crit,query_points_obj_topk=4)
def state_fingerprint(m,prefix=None):
 h=hashlib.sha256()
 for name,v in m.state_dict().items():
  if prefix is not None and not name.startswith(prefix):continue
  x=v.detach().cpu().contiguous();h.update(name.encode()+b'\0');h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
 return h.hexdigest()
