import os
import torch
import folder_paths
import json

from transformers import AutoModelForCausalLM,AutoTokenizer, pipeline
from transformers import AutoProcessor

from PIL import Image
import numpy as np

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


class CXH_PHI_PIP:
    def __init__(self):
        self.model = None
        self.processor = None

class CXH_Phi_load:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["Phi-3.5-vision-instruct"],),
                "num_crops":([ 4, 16],{"default": 4}),
                "attention": ([ 'flash_attention_2', 'sdpa', 'eager'],{"default": 'eager'})
            },
        }

    RETURN_TYPES = ("CXH_PHI_PIP",)
    RETURN_NAMES = ("phi_mode",)
    FUNCTION = "gen"
    CATEGORY = "CXH/LLM"

    def gen(self,model,num_crops,attention):
        # 下载本地
        model_id = f"microsoft/{model}"
        model_checkpoint = os.path.join(folder_paths.models_dir, 'LLM', os.path.basename(model_id))
        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
            
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            _attn_implementation= attention
        )

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        processor = AutoProcessor.from_pretrained(model_id, 
            trust_remote_code=True, 
            num_crops=num_crops
        )
    
        p = CXH_PHI_PIP()
        p.model = model
        p.processor = processor
    
        return (p,)

class CXH_Phi_Run:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "phi_mode": ("CXH_PHI_PIP",),
                "prompt": ("STRING",{"default": '', "multiline": True}),
                "images": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 100, "max": 10000, "step": 500}),
               
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out",)
    FUNCTION = "gen"
    CATEGORY = "CXH/LLM"

    def gen(self,phi_mode,prompt,images,temperature,max_new_tokens):
        model = phi_mode.model
        processor = phi_mode.processor
        
        placeholder = ""
        imgs = []

        index = 1
        for i in images:
            i = torch.unsqueeze(i, 0)
            orig_image = tensor2pil(i).convert('RGB')
            imgs.append(orig_image)
            placeholder += f"<|image_{index}|>\n"
            index = index + 1
        
        messages = [
            {"role": "user", "content":prompt +  placeholder},
        ]
        
        prompt = processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
            )
        
        inputs = processor(prompt, imgs, return_tensors="pt").to("cuda:0") 
        
        generation_args = { 
            "max_new_tokens": max_new_tokens, 
            "temperature": temperature, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            **generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)[0]
        
        return (response,)

class CXH_Phi_chat_load:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["Phi-3.5-mini-instruct"],),
                "attention": ([ 'flash_attention_2', 'sdpa', 'eager'],{"default": 'eager'}),
            },
        }

    RETURN_TYPES = ("PHI_MIN_MODE",)
    RETURN_NAMES = ("phi_min_mode",)
    FUNCTION = "inference"
    CATEGORY = "CXH/GPT"

    def inference(self,model,attention):
        # 下载本地
        model_id = f"microsoft/{model}"
        model_checkpoint = os.path.join(folder_paths.models_dir, 'LLM', os.path.basename(model_id))
        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)

        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            _attn_implementation= attention
            ) 

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        phi_min_mode = {
            "model":model,
            "tokenizer":tokenizer
        }

        return (phi_min_mode,)

class CXH_Phi_chat_min:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("PHI_MIN_MODE",),
                "prompt": ("STRING",{"default": '', "multiline": True}),
                "temperature": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 100, "max": 20000, "step": 500}),  
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out",)
    FUNCTION = "inference"
    CATEGORY = "CXH/GPT"

    def inference(self,model,prompt, temperature,max_new_tokens):
        model_cache = model["model"]
        tokenizer = model["tokenizer"]

        # text 
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]

        pipe = pipeline(
            "text-generation",
            model=model_cache,
            tokenizer=tokenizer,
        )

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature":temperature,
            "do_sample": False,
        }
        
        output = pipe(messages, **generation_args)
        result = output[0]['generated_text']
        
        
        return (result,)