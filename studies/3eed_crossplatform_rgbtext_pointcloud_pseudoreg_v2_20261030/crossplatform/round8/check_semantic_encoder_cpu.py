"""Synthetic CPU engineering checks only; no target data, GT, metrics or GPU."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPTokenizer
from semantic_encoder import FrozenSemanticEncoder,crop_rgb,DEFAULT_ASSETS,file_sha

def main():
 torch.set_num_threads(2);start=time.monotonic();encoder=FrozenSemanticEncoder(device='cpu',batch_size=2)
 w,h=301,173;y,x=np.mgrid[0:h,0:w];rgb=np.stack([(x*255//(w-1)),(y*255//(h-1)),((x+y)%256)],-1).astype(np.uint8);image=Image.fromarray(rgb,'RGB')
 crops,crop_receipt=crop_rgb(image,[[20.2,15.8,140.1,160.3],[-10,-10,100,100]])
 assert [c.size for c in crops]==[(121,146),(100,100)] and crop_receipt[1]['source_image_bounds_clipped']
 try:crop_rgb(image,[[500,500,600,600]])
 except ValueError:pass
 else:raise AssertionError('Empty crop admitted')
 try:encoder.preprocess_crops([np.zeros((4,4,3),np.float32)])
 except TypeError:pass
 else:raise AssertionError('Implicit floatRGB admitted')
 original=transforms.Compose([transforms.Resize(224,interpolation=InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.Lambda(lambda im:im.convert('RGB')),transforms.ToTensor(),transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
 hf=encoder.preprocess_crops([image]);reference=original(image)[None];error=float((hf-reference).abs().max());assert error<1e-6,error
 descriptions=['a red vehicle to the left of another vehicle','a person standing beside the doorway','object '+'location relation '*90]
 tokens,trunc=encoder.tokenize_texts(descriptions);assert trunc['truncated_count']==1 and trunc['used_token_lengths'][-1]==77
 for i,n in enumerate(trunc['used_token_lengths']):assert int(tokens['input_ids'][i,n-1])==encoder.tokenizer.eos_token_id
 slow=CLIPTokenizer.from_pretrained(str(DEFAULT_ASSETS),local_files_only=True)
 assert slow(descriptions,add_special_tokens=True,padding=False,truncation=False)['input_ids']==encoder.tokenizer(descriptions,add_special_tokens=True,padding=False,truncation=False)['input_ids']
 images=encoder.encode_crops(crops);texts,tr=encoder.encode_texts(descriptions[:2]);assert images.shape==(2,512) and texts.shape==(2,512)
 assert torch.allclose(images.norm(dim=-1),torch.ones(2),atol=2e-6) and torch.allclose(texts.norm(dim=-1),torch.ones(2),atol=2e-6)
 result=encoder.match(crops,descriptions[:2],crop_roles=['tight','context']);repeat=encoder.match(crops,descriptions[:2],crop_roles=['tight','context'])
 assert np.array_equal(result['cosines'],repeat['cosines']) and result['crop_receipt']['tight_context_fusion'].startswith('none')
 assert np.allclose(result['cosines'],(images@texts.T).numpy(),atol=0,rtol=0)
 encoder._assert_frozen();assert not torch.cuda.is_initialized()
 receipt={'state':'synthetic_CPU_safe_load_preprocess_text_image_inference_pass_no_target_data_or_GT_or_GPU','resource':encoder.resource,'semantic_encoder_sha256':file_sha(Path(__file__).with_name('semantic_encoder.py')),'fixture_program_sha256':file_sha(__file__),'frozen_eval_no_grad':True,'strict_safe_derived_weights_load':True,'synthetic_preprocessing_reference':'original OpenAI CLIP transform structure using torchvision Resize bicubic / CenterCrop224 / RGB / ToTensor / Normalize official mean/std','max_preprocessor_abs_difference':error,'non_square_crop_size_contract_pass':True,'empty_crop_rejected':True,'float_RGB_rejected':True,'text_context77_endtoken_and_recorded_truncation_pass':True,'synthetic_texts_fast_slow_official_HF_tokenizers_equal':True,'synthetic_truncation_receipt':trunc,'normalized_text_and_image512D_pass':True,'raw_cosine_no_softmax_no_implicit_view_fusion_pass':True,'same_input_CPU_repeat_bitexact':True,'no_CUDA_initialization':not torch.cuda.is_initialized(),'no_target_media_annotations_or_scoring_access':True,'seconds':time.monotonic()-start}
 Path(__file__).with_name('CPU_SEMANTIC_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
