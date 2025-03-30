import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import PIL
import torch

@torch.no_grad()
def env_output_to_prompt(env_outputs: Dict,init_id_to_prompt: Dict, tokenizer,is_final_prompt: bool) -> str:
    env_prompt=[]
    for env_output in env_outputs:
        chat = []
        images = []
        chat.append({"role": "system", "content": init_id_to_prompt[env_output["env_id"]]})
        for idx, turn_history in env_output["history"]:
            if "llm_response" in turn_history:
                chat.append({"role": "assistant", "content": turn_history["llm_response"]})
            if idx==len(env_output["history"])-1 and is_final_prompt:
                continue
            cur_text= f"Turn {idx}\n"
            if "reward" in turn_history:
                cur_text+= f"Reward:\n{turn_history['reward']}\n"
            if "state" in turn_history:
                cur_text+= f"State:\n{turn_history['state']}\n"
            chat.append({"role": "user", "content": cur_text})
            if "images" in turn_history:
                images.extend(turn_history["images"])
           
        prompt_text = tokenizer.apply_chat_template(chat, add_generation_prompt=(not is_final_prompt), tokenize=False)
        env_prompt.append({
            "env_id": env_output["env_id"],
            "chat_text": prompt_text,
            "images": images
        })
    return env_prompt


@torch.no_grad()      
def handle_multi_modal_data(
        processor,
        raw_prompt: str, 
        row_dict: Dict,
        image_data: List[PIL.Image.Image],
        do_embedding: bool = True,
    ) -> str:
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