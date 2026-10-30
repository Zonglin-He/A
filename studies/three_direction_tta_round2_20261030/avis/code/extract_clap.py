"""Frozen official CLAP; physically normalized PCM48k; no annotation imports."""
import pathlib,json,hashlib,sys,time,math
import numpy as np,torch,soundfile as sf
from scipy.signal import resample_poly
from transformers import ClapModel,ClapConfig,ClapProcessor
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def load(device):
 receipt=json.loads((R/'teacher/DOWNLOAD_RECEIPT.json').read_text());assert receipt['state']=='completed';assert sha(R/'teacher/pytorch_model.bin')==receipt['sha256']
 state=torch.load(R/'teacher/pytorch_model.bin',weights_only=True,map_location='cpu',mmap=True)
 model=ClapModel(ClapConfig.from_pretrained(str(R/'teacher'),local_files_only=True));loaded=model.load_state_dict(state,strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
 model.requires_grad_(False);model.eval();model.to(device);processor=ClapProcessor.from_pretrained(str(R/'teacher'),local_files_only=True)
 return model,processor

def windows(path,n):
 wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(-1);g=math.gcd(sr,48000);wave=resample_poly(wave,48000//g,sr//g).astype('float32')
 out=[]
 for t in range(n):
  lo=(t-1)*48000;hi=(t+2)*48000;segment=wave[max(0,lo):min(len(wave),hi)];segment=np.pad(segment,(max(0,-lo),max(0,hi-len(wave))))
  assert len(segment)==144000
  out.append(segment)
 return out,{'original_sample_rate':sr,'channels_averaged':True,'resample':'scipy polyphase rational','waveform_normalization':'soundfile PCM float32 physicalscale; independent of oldAVISMcachedfeatures','waveform_duration_seconds':len(wave)/48000,'window_seconds':3,'context':'noncausal centered at1second bins; zero pad outside recording then officialrepeatpad10s','peak_abs':float(np.abs(wave).max())}

def extract(manifest,output,device='cuda',planpath=None):
 if planpath is not None:
  planpath=pathlib.Path(planpath);plan=json.loads(planpath.read_text());assert sha(planpath)==planpath.with_suffix('.json.sha256').read_text().split()[0]
  for file,h in {**plan['source_hashes'],**plan['input_hashes']}.items():assert sha(file)==h,file
  assert sha(manifest)==plan['manifest_sha256']
 spec=json.loads(pathlib.Path(manifest).read_text());out=pathlib.Path(output);out.mkdir(parents=True,exist_ok=False);model,processor=load(device);prompts=json.loads((R/'PROMPTS_LOCK.json').read_text());torch.set_num_threads(4);torch.manual_seed(20261030);np.random.seed(20261030)
 flat=[p for c in prompts['classes'] for p in c['prompts']]+prompts['background_prompts'];text=processor(text=flat,return_tensors='pt',padding=True);text={k:v.to(device) for k,v in text.items()}
 with torch.no_grad():te=model.get_text_features(**text);te=torch.nn.functional.normalize(te,dim=-1);offset=0;class_emb=[]
 for c in prompts['classes']:
  e=te[offset:offset+len(c['prompts'])].mean(0);class_emb.append(torch.nn.functional.normalize(e,dim=-1));offset+=len(c['prompts'])
 class_emb=torch.stack(class_emb);background=te[offset:];all_emb=torch.cat([class_emb,background])
 np.save(out/'text_embeddings.npy',all_emb.cpu().numpy());rows=[]
 for v in spec['videos']:
  wav=pathlib.Path(v['audio_path']).parents[1]/'WAVAudios'/f"{v['video_id']}.wav";segments,frontend=windows(wav,len(v['frame_indices']));embeds=[]
  if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  tic=time.perf_counter()
  with torch.no_grad():
   for start in range(0,len(segments),8):
    data=processor(audios=segments[start:start+8],sampling_rate=48000,return_tensors='pt');data={k:v.to(device) for k,v in data.items()};e=model.get_audio_features(**data);embeds.append(torch.nn.functional.normalize(e,dim=-1))
  if device=='cuda':torch.cuda.synchronize()
  ae=torch.cat(embeds);cosine=(ae@all_emb.T).cpu().numpy();assert np.isfinite(cosine).all();np.save(out/f"{v['video_id']}.npy",cosine)
  row={'video_id':v['video_id'],'frames':len(segments),'teacher_seconds_excluding_PCM_IO':time.perf_counter()-tic,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,'wav_sha256':sha(wav),'cosine_sha256':sha(out/f"{v['video_id']}.npy"),'frontend':frontend};rows.append(row);(out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 protocol={'state':'sealed','teacher':'laion/clap-htsat-unfused','revision':json.loads((R/'teacher/MODEL_API.json').read_text())['sha'],'teacher_sha256':sha(R/'teacher/pytorch_model.bin'),'prompts_sha256':sha(R/'PROMPTS_LOCK.json'),'manifest_sha256':sha(manifest),'extractor_sha256':sha(__file__),'GT_access':False,'text_embeddings_sha256':sha(out/'text_embeddings.npy'),'rows':rows,'logit_scale_audio':float(model.logit_scale_a.exp().detach().cpu()),'cosine_not_calibrated_activity_probability':True,'feature_plan_sha256':sha(planpath) if planpath else None}
 (out/'seal.json').write_text(json.dumps(protocol,indent=2));print('SEALED',flush=True)
if __name__=='__main__':extract(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else 'cuda',sys.argv[4] if len(sys.argv)>4 else None)
