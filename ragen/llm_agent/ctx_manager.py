"""
This is the context manager for the LLM agent.
author: Kangrui Wang, Zihan Wang
date: 2025-03-30
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
register_resolvers()


def get_loss_mask(input_ids: torch.Tensor, tokenizer: AutoTokenizer):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    """
    special_token = tokenizer.encode("<|im_start|>")[0]
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
    return loss_mask    



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
        self.action_sep = self.config.agent_proxy.action_sep
        self._init_prefix_lookup()
    
    def _init_prefix_lookup(self):
        self.prefix_lookup = {"val": {}, "train": {}}
        prefixes = {}
        for env_tag, env_config in self.config.custom_envs.items():
            env_instruction = env_config.get("env_instruction", "")
            prefixes[env_tag] = env_instruction

        # Training
        for split in ["train", "val"]:
            tags = self.config.es_manager[split].env_configs.tags
            n_groups = self.config.es_manager[split].env_configs.n_groups
            group_size = self.config.es_manager[split].group_size

            cur_group = 0
            for env_tag, n_group in zip(tags, n_groups):
                env_instruction = prefixes[env_tag]
                start_idx = cur_group
                end_idx = cur_group + n_group * group_size
                for i in range(start_idx, end_idx):
                    self.prefix_lookup[split][i] = env_instruction
                cur_group += n_group

        
    def _parse_response(self, response: str, action_sep=' || ',special_token_list=None) -> List:
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
        return llm_response, actions
        
    
    def get_lm_inputs(self, env_outputs: List[Dict], is_final_turn: bool, val: bool = False) -> Dict:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """
        llm_input_texts = []
        for env_output in env_outputs:
            if 'state' in env_output['history'][-1] and is_final_turn:
                env_output['history'] = env_output['history'][:-1] # for final (n-th) turn, we do not learn the state from the n+1 turn

            messages = [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": self.prefix_lookup["val" if val else "train"][env_output["env_id"]]}
            ]

            for idx, content in enumerate(env_output["history"]):
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    messages[-1]["content"] += f"State:\n{content['state']}\n"
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content:
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})

            # NOTE: this assertion is important for loss mask computation        
            assert all(msg["role"] == "assistant" for msg in messages[2::2])

            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not is_final_turn), tokenize=False)
            llm_input_texts.append(text)
        

        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        loss_mask = get_loss_mask(input_ids, self.tokenizer)

        llm_inputs = DataProto()
        llm_inputs.batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids.clone(),
            "loss_mask": loss_mask
        }

        return llm_inputs

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, reponse in zip(env_ids, responses):
            llm_response, actions = self._parse_response(reponse, action_sep=self.action_sep)
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

    



@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    import json
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)

    batch_list = [
        {
            "env_ids": 0,
            "chat_response": "<think><think></answer> 123. </think><answer> <answer> say | hi </answer></answer>",
        },
        {
            "env_ids": 1,
            "chat_response": "<think> 456. </think><answer> love ; you </answer><think> mlll nb </think><answer> lxxx ; you </answer>",
        }
    ]
    ctx_manager.action_sep_lookup={
        0: "|",
        1: ";"
    }
    for item in batch_list:
        item["responses"] = tokenizer.encode(item["chat_response"], return_tensors="pt",max_length=512, truncation=True,padding="max_length")[0]
    batch_dict = collate_fn(batch_list)
    batch = DataProto.from_single_dict(batch_dict)
    env_inputs = ctx_manager.get_env_inputs(batch)
    print(env_inputs)
    


    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 1", "reward": 0.5},
                {"state": "###\n#x_#<image>", "llm_response": "Response 2", "reward": 0.8},
                {"state": "###\n#x_#<image>"}
            ]
        },
        {
            "env_id": 2,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 3", "reward": 0.3},
                {"state": "###\n#x_#<image>"}
            ]
        }
    ]
    
    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, is_final_turn=False)
    print(env_prompt)

if __name__=="__main__":
    main()
    