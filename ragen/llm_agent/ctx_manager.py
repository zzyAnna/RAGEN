import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
#from .ctx_manager_utils import env_output_to_prompt, handle_multi_modal_data
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn

class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        
        Args:
            config: Configuration for the ContextManager.
            tokenizer: Tokenizer to use for encoding text.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self._init_action_sep_lookup()
        self._init_prefix_lookup()
    
    def _init_prefix_lookup(self):
        self.prefix_lookup = {}
        pass
    
    def _init_action_sep_lookup(self):
        self.action_sep_lookup = {}
        pass
        
    def _parse_response(self, response: str, action_sep='|',special_token_list=None) -> List:
        pattern = r'<think>(.*?)</think><answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            return []
        think_content = match.group(1)
        action_content = match.group(2)
        if special_token_list is None:
            special_token_list=["<think>","</think>","<answer>","</answer>","|<im_start>|","|<im_end>|"]
        for special_token in special_token_list:
            action_content = action_content.replace(special_token, "").strip()
            think_content = think_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        llm_response="<think>" + think_content + "</think>" + "<answer>" + action_content + "</answer>"
        return llm_response,actions
        
    
    def get_lm_inputs(self, env_outputs: List[Dict]) -> Dict:
        # env_prompt = env_output_to_prompt(env_outputs, self.id_to_prompt, self.tokenizer, is_final_prompt=False)
        # for item in env_prompt:
        #     prompt_raw_dict = {}
        #     response_raw_dict = {}
        #     chat_prompt = item["chat_prompt"]
        #     chat_response = item["chat_response"]
        #     prompt_images = item["prompt_images"]
        #     response_images = item["response_images"]
        #     env_id = item["env_id"]
        #     if prompt_images["images"]:
        #         prompt_res = handle_multi_modal_data(processor=self.processor,raw_prompt=item["raw_prompt"], row_dict=prompt_raw_dict, image_data=item["images"], do_embedding=False)
        #         prompt_raw_dict = prompt_res['prompt_raw_dict']
        #         image_grid_thw = prompt_res['image_grid_thw']
        #         item["raw_prompt"] = image_grid_thw
        pass

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, reponse in zip(env_ids, responses):
            llm_response, actions = self._parse_response(reponse, action_sep=self.action_sep_lookup[env_id])
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": reponse,
                "llm_response": llm_response,
                "actions": actions,
            })
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> Dict:
        #env_prompt=env_output_to_prompt(env_outputs, self.id_to_prompt, self.tokenizer, is_final_prompt=True)
        pass

    


if __name__=="__main__":
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    ctx_manager = ContextManager(config=None, tokenizer=tokenizer)
    batch_list = [
        {
            "env_ids": 0,
            "chat_response": "<think><think></answer> 123. </think><answer> <answer> say | hi </answer></answer>",
        },
        {
            "env_ids": 1,
            "chat_response": "<think> 456. </think><answer> love ; you </answer><think> w*n *t </think><answer> lxxx ; you </answer>",
        }
    ]
    ctx_manager.action_sep_lookup={
        0: "|",
        1: ";"
    }
    for item in batch_list:
        item["responses"]=tokenizer.encode(item["chat_response"], return_tensors="pt",max_length=512, truncation=True,padding="max_length")[0]
    batch_dict = collate_fn(batch_list)
    batch = DataProto.from_single_dict(batch_dict)
    env_inputs = ctx_manager.get_env_inputs(batch)
    print(env_inputs)
    
