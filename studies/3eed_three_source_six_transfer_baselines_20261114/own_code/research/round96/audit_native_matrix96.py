"""Independent native source loading/readouts and complete sampled-point identity replay, no IoU scoring."""
from pathlib import Path
import sys,json,hashlib,re,time
import numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';sys.path.insert(0,str(B))
from factory import configure,model,dataset,loader,to_cuda,torch,sha,state_fingerprint
from source_validation import LEXICON
rd=lambda p:json.loads(p.read_text())
def main():
 se=rd(B/'frozen_matrix/MATRIX_SEALED.json');configure({'seed':0,'torch_num_threads':4});m=model().cuda().requires_grad_(False).eval();start=time.perf_counter();maxbox=maxscore=0.;checked=pointchecks=toperrors=0;directions=[];pl=rd(H/'experiment/MATRIX_REPAIR_LOCK_attempt002.json')
 for source in ['waymo','drone','quad']:
  ck=R/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth' if source=='waymo' else Path(rd(B/(source+'_source100')/'TRAINING_COMPLETE.json')['primary_checkpoint']);c=torch.load(ck,map_location='cpu',weights_only=True);assert c['trained_platforms']==[source] and c['epoch']==100;m.load_state_dict(c['model'],strict=True);del c;fp=state_fingerprint(m)
  for d in [v for v in se['directions'] if v['source']==source]:
   target=d['target'];p=Path(d['prediction_directory']);s=rd(p/'sealed.json');assert sha(ck)==d['checkpoint_sha256'] and s['model_state_fingerprint']==fp;configure({'seed':0,'torch_num_threads':4});ds=dataset(target,'val');g=torch.Generator().manual_seed(0);dl=loader(ds,'val',g,pl);chosen={0,len(dl)//2,len(dl)-1};ix=0;n=0
   with torch.no_grad():
    for ib,b in enumerate(dl):
     arr=[]
     for j in range(len(b['utterances'])):
      z=np.load(p/(str(ix+j).zfill(6)+'.npz'),allow_pickle=False);sample=b['point_clouds'][j].contiguous().numpy();assert hashlib.sha256(sample.tobytes()).digest()==z['point_sample_sha256'].tobytes();pointchecks+=1
      if ib in chosen:arr.append({k:z[k] for k in z.files})
     if ib in chosen:
      bb=to_cuda(b);o=m({'point_clouds':bb['point_clouds'].float(),'text':bb['utterances']});enc=m.tokenizer(bb['utterances'],padding='longest',return_offsets_mapping=True,return_tensors='pt');off=enc['offset_mapping'].numpy();mask=np.zeros(off.shape[:2],bool)
      for j,t in enumerate(bb['utterances']):
       match=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',t,re.I);mask[j]=((off[j,:,1]>match.start())&(off[j,:,0]<match.end())) if match else (off[j,:,1]>off[j,:,0])
      mk=torch.as_tensor(mask,device='cuda');T=o['proj_tokens'].shape[1];assert T==mask.shape[1]
      for ih,h in enumerate(['last_','proposal_']):
       raw=torch.cat([o[h+'center'],o[h+'pred_size']],-1);box=raw.clone();box[:,:,3:6].clamp_(min=1e-6);v=o[h+'proj_queries']@o['proj_tokens'].transpose(-1,-2)/.07;score=torch.logsumexp(v.masked_fill(~mk[:,None],-torch.inf),-1)-torch.logsumexp(v.masked_fill(mk[:,None],-torch.inf),-1)
       for j,z in enumerate(arr):
        maxbox=max(maxbox,float(np.abs(raw[j].cpu().numpy()-z['boxes_raw'][ih]).max()),float(np.abs(box[j].cpu().numpy()-z['boxes'][ih]).max()));maxscore=max(maxscore,float(np.abs(score[j].cpu().numpy()-z['text_logodds'][ih]).max()));toperrors+=int(int(score[j].argmax())!=int(z['text_logodds'][ih].argmax()))
      checked+=len(arr);n+=len(arr)
     ix+=len(b['utterances'])
   assert ix==s['n'] and state_fingerprint(m)==fp;directions.append({'source':source,'target':target,'native_readout_rows':n,'point_hash_rows':ix});print('AUDIT96',source,target,n,ix,flush=True)
 assert maxbox<1e-5 and maxscore<1e-4 and toperrors==0,(maxbox,maxscore,toperrors)
 a={'state':'PASS_R96_NATIVE_SOURCE_CHECKPOINTS_SAMPLED_READOUTS_AND_ALL_POINT_IDENTITIES','directions':directions,'native_rows':checked,'point_hash_rows':pointchecks,'max_box_error':maxbox,'max_text_logodds_error':maxscore,'Top1_mismatches':toperrors,'new_numeric_task_GT_scores':0,'model_parameters_and_buffers_unchanged':True,'matrix_seal_sha256':sha(B/'frozen_matrix/MATRIX_SEALED.json'),'FINAL_inference_or_scoring':False,'seconds':time.perf_counter()-start};out=H/'independent_native_matrix_audit';out.mkdir();(out/'AUDIT_RESULT.json').write_text(json.dumps(a,indent=2)+'\n');print(a)
if __name__=='__main__':main()
