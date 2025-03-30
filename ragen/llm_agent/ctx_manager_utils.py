import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import PIL
from PIL import Image
import torch

@torch.no_grad()
def env_output_to_prompt(env_outputs: Dict, init_prompt_lookup: Dict, tokenizer, is_final_prompt: bool):
    """
    env_outputs - example be like:
    [
		{"env_id": xxx, "history": [
			{"state": "###\n#x_#", "actions": ["xx", "yy"], "reward": xxx}
			{"state", "###\n#x_#"}
        ]
    ]
    init_prompt_lookup - from env_id to initial prompt

    """
    env_prompt = []
    for env_output in env_outputs:
        chat_prompt = []
        chat_response = []
        prompt_images = []
        response_images = []
        chat_prompt.append({"role": "system", "content": init_prompt_lookup[env_output["env_id"]]})
        
        for idx, turn_history in env_output["history"]:
            if "llm_response" in turn_history:
                chat_response.append({"role": "assistant", "content": turn_history["llm_response"]})
            if idx == len(env_output["history"])-1 and is_final_prompt:
                break
            raw_prompt= f"Turn {idx}\n"
            if "reward" in turn_history:
                raw_prompt += f"Reward:\n{turn_history['reward']}\n"
            if "state" in turn_history:
                raw_prompt += f"State:\n{turn_history['state']}\n"
            if idx == 0: # initial state is for prompt
                chat_prompt.append({"role": "user", "content": raw_prompt})
                if "images" in turn_history:
                    prompt_images.extend(turn_history["images"])
            else:
                chat_response.append({"role": "user", "content": raw_prompt})
                if "images" in turn_history:
                    response_images.extend(turn_history["images"])
            
           
        chat_prompt = tokenizer.apply_chat_template(chat_prompt, add_generation_prompt=(not is_final_prompt), tokenize=False)
        chat_response = tokenizer.apply_chat_template(chat_response, add_generation_prompt=(not is_final_prompt), tokenize=False)
        env_prompt.append({
            "chat_prompt": chat_prompt,
            "chat_response": chat_response,
            "prompt_images": prompt_images,
            "response_images": response_images,
            "env_id": env_output["env_id"],
            "raw_prompt": raw_prompt,
        })
    return env_prompt


@torch.no_grad()      
def handle_multi_modal_data(
        processor,
        raw_prompt: str, 
        row_dict: Dict,
        image_data: List[Image.Image],
        do_embedding: bool = True,
    ) -> Dict:
    """
    vllm: do_embedding=False -> processd_prompt: <|vision_start|><|image_pad|><|vision_end|>
    actor: do_embedding=True -> processd_prompt: <|vision_start|><|placeholder|>...<|placeholder|><|vision_end|>
    """
    processed_prompt = raw_prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    row_dict['multi_modal_data'] = {'image': image_data}
    image_grid_thw = None
    if do_embedding:
        image_inputs = processor.image_processor(image_data, return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
    if image_grid_thw is not None:
        merge_length = processor.image_processor.merge_size**2
        index = 0
        while '<image>' in raw_prompt:
            raw_prompt = raw_prompt.replace(
                '<image>',
                '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                '<|vision_end|>',
                1,
            )
            index += 1
        processed_prompt = raw_prompt.replace('<|placeholder|>',
                                                    processor.image_token)
    return {
        'processed_prompt': processed_prompt,
        'row_dict': row_dict,
        'image_grid_thw': image_grid_thw
    }
    
    
if __name__ == "__main__":
    # Example usage
    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"llm_response": "Response 1", "reward": 0.5, "state": "State 1 <image>", "images": [Image.new('RGB', (100, 100))]},
                {"llm_response": "Response 2", "reward": 0.8, "state": "State 2 <image>", "images": [Image.new('RGB', (100, 100))]}
            ]
        }
    ]
    
    init_id_to_prompt = {1: "Initial prompt"}
    from transformers import PreTrainedTokenizer, ProcessorMixin
    