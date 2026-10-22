"""Official-model meta construction and state-schema check, no real inference."""
import hashlib,json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runtime/llava_st/upstream'))
import requests,torch,transformers
from accelerate import init_empty_weights
from llava.model.language_model.llava_qwen import LlavaQwenConfig,LlavaQwenForCausalLM
from tools.llava_st.backbone import embedded_vision_initialization,REVISION
from stvg.tube_runner import write
if __name__=='__main__':
    root=ROOT/'checkpoints/llava-st/LLaVA-ST-Qwen2-7B';raw=json.loads((root/'config.json').read_text());config=LlavaQwenConfig.from_dict({**{k:v for k,v in raw.items()if k!='text_config'},'vocab_size':raw['text_config']['vocab_size'],'model_type':'llava_qwen'})
    with init_empty_weights(),embedded_vision_initialization():model=LlavaQwenForCausalLM(config)
    expected={k:list(v.shape)for k,v in model.state_dict().items()};meta=json.loads((ROOT/'runtime/llava_st/download_plan.json').read_text());index_url=f'https://huggingface.co/{meta["repo"]}/resolve/{meta["revision"]}/model.safetensors.index.json';r=requests.get(index_url,timeout=60);r.raise_for_status();index=r.json();assert set(index['weight_map'])==set(expected)
    write(ROOT/'runtime/llava_st/meta_architecture_check.json',dict(status='pass_meta_schema_only_not_actual_weights_or_GPU',upstream=REVISION,transformers=transformers.__version__,parameters=sum(p.numel()for p in model.parameters()),state_entries=len(expected),published_index_state_keys_exact=True,base_embedding_rows=model.model.embed_tokens.weight.shape[0],coordinate_output_rows=300,configuration_compatibility='flatten original nested text_config.vocab_size',embedded_vision_not_downloaded_separately=True,actual_GPU_inference=False))
    # Tiny random network exercises generation API only, not trained-model quality.
    torch.manual_seed(41)
    c=LlavaQwenConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,bos_token_id=1,eos_token_id=None,pad_token_id=0)
    from types import SimpleNamespace
    tiny=LlavaQwenForCausalLM(c).eval();tiny.model.vision_config=SimpleNamespace();out=tiny.generate(torch.tensor([[1,4,5]]),max_new_tokens=3,do_sample=False);assert out.shape[-1]==3
    print('META STATE KEYS MATCH PUBLISHED INDEX:',len(expected),'parameters',sum(p.numel()for p in model.parameters()),'API smoke passed',flush=True)
