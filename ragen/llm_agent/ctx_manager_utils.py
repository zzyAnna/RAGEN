import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import PIL
from PIL import Image
import torch

@torch.no_grad()
def env_outputs_to_llm_inputs(env_outputs: Dict, prefix_lookup: Dict, tokenizer, is_final_turn: bool):
    """
    env_outputs - example be like:
    [
		{"env_id": xxx, "history": [
			("state", "###\n#x_#<image>")
            ("llm_response", "Response 1")
            ("reward", xxx)
            ("state", "###\n#x_#<image>")
        ],
        "images": [Image.new('RGB', (100, 100)), Image.new('RGB', (100, 100))]
    ]
    prefix_lookup - from env_id to initial prompt

    """
    llm_input_texts = []


    for env_output in env_outputs:
        if env_output['history'][-1][0] == "state" and is_final_turn:
            env_output['history'] = env_output['history'][:-1] # for final n-th turn, we do not learn the state from the n+1 turn

        messages = [{"role": "user", "content": prefix_lookup[env_output["env_id"]]}]

        for type_, content in env_output["history"]:
            if type_ == "llm_response":
                messages.append({"role": "assistant", "content": content})
            elif type_ == "reward":
                messages.append({"role": "user", "content": f"Reward:\n{content}\n"})
            elif type_ == "state":
                assert messages[-1]["role"] == "user"
                messages[-1]["content"] += f"State:\n{content}\n" # ensure that the state is appended after the prev-step reward            
           
        chat = tokenizer.apply_chat_template(chat, add_generation_prompt=(not is_final_turn), tokenize=False)
        

        ""

        llm_input_texts.append({
            "text": chat,
            "env_id": env_output["env_id"],
        })
    breakpoint()
    # for item in llm_input_texts:
    #     text = item["text"]
    #     env_id = item["env_id"]

    # llm_inputs = DataProto()
    # llm_inputs.batch = {
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "position_ids": position_ids,
    #     "responses": input_ids.copy()
    # }


    return env_prompt


    
    
if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    import json
    env_outputs = [
        {
            "env_id": 1,
            "history": [
                ("state", "###\n#x_#<image>"),
                ("llm_response", "Response 1"),
                ("reward", 0.5),
                ("state", "###\n#x_#<image>"),
                ("llm_response", "Response 2"),
                ("reward", 0.8),
                ("state", "###\n#x_#<image>"),
            ]
        },
        {
            "env_id": 2,
            "history": [
                ("state", "###\n#x_#<image>"),
                ("llm_response", "Response 3"),
                ("reward", 0.3),
                ("state", "###\n#x_#<image>"),
            ]
        }
    ]
    
    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = env_outputs_to_llm_inputs(env_outputs, prefix_lookup, tokenizer, is_final_turn=False)
    
    print(json.dumps(env_prompt, indent=4))







    # env_outputs = [
    #     {
    #         "env_id": 1,
    #         "history": [
    #             {"llm_response": "Response 1", "reward": 0.5, "state": "State 1 <image>", "images": [Image.new('RGB', (100, 100))]},
    #             {"llm_response": "Response 2", "reward": 0.8, "state": "State 2 <image>", "images": [Image.new('RGB', (100, 100))]}
    #         ]
    #     },
    #     {
    #         "env_id": 2,
    #         "history": [
    #             {"llm_response": "Response 3", "reward": 0.3, "state": "State 3 <image>", "images": [Image.new('RGB', (100, 100))]}
    #         ]
    #     }
    # ]


# @torch.no_grad()      
# def handle_multi_modal_data(
#         processor,
#         raw_prompt: str, 
#         row_dict: Dict,
#         image_data: List[Image.Image],
#         do_embedding: bool = True,
#     ) -> Dict:
#     """
#     vllm: do_embedding=False -> processd_prompt: <|vision_start|><|image_pad|><|vision_end|>
#     actor: do_embedding=True -> processd_prompt: <|vision_start|><|placeholder|>...<|placeholder|><|vision_end|>
#     """
#     processed_prompt = raw_prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
#     row_dict['multi_modal_data'] = {'image': image_data}
#     image_grid_thw = None
#     if do_embedding:
#         image_inputs = processor.image_processor(image_data, return_tensors='pt')
#         image_grid_thw = image_inputs['image_grid_thw']
#         row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
#     if image_grid_thw is not None:
#         merge_length = processor.image_processor.merge_size**2
#         index = 0
#         while '<image>' in raw_prompt:
#             raw_prompt = raw_prompt.replace(
#                 '<image>',
#                 '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
#                 '<|vision_end|>',
#                 1,
#             )
#             index += 1
#         processed_prompt = raw_prompt.replace('<|placeholder|>',
#                                                     processor.image_token)
#     return {
#         'processed_prompt': processed_prompt,
#         'row_dict': row_dict,
#         'image_grid_thw': image_grid_thw
#     }

# def _generate_input_for_uptate(
#             self, 
#             recording: List[Dict], 
#             step: int, 
#             window_size: int = None,
#         ):
#         """
#         Given a recording, generate the final input for MLLM
        
#         Args:
#             recording: List of dictionaries containing recorded environment interactions
#             step: Current step to generate input for
#             window_size: Number of past steps to include in the context
        
#         Returns:
#             Dictionary containing properly formatted inputs for the MLLM
#             - prompts: task instruction
#             - responses: responses generated from prompts
#             - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
#             - position_ids: 
#                 - position_ids for prompts: rope
#                 - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute

#         """



#         # handle prompt, prompt=pad_token since we now have everything in response and compute a loss mask for them
#         prompt_with_chat_template=self.tokenizer.pad_token 
        
#         # handle response
#         response_rst=self._single_recording_to_prompt(recording, step, window_size, is_final=True, prep_for_loss_mask=True)
#         response_with_chat_template=response_rst['prompt']
#         image_data=response_rst['image_data']
#         rewards=response_rst['rewards']
       
#         has_images = len(image_data) > 0
#         row_dict = {}
#         if has_images:  # expand image token
#             response_with_chat_template, row_dict, image_grid_thw, _ = self._handle_multi_modal_data(
#                 response_with_chat_template, row_dict, image_data, do_embedding=True)

        
#         input_ids_response, attention_mask_response = verl_F.tokenize_and_postprocess_data(prompt=response_with_chat_template,
#                                                                          tokenizer=self.tokenizer,
#                                                                          max_length=self.config.max_trajectory_length-1, # -1 for the prompt padding token
#                                                                          pad_token_id=self.tokenizer.pad_token_id,
#                                                                          left_pad=False,
#                                                                          truncation=self.config.truncation)
#         input_ids_prompt, attention_mask_prompt = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
#                                                                          tokenizer=self.tokenizer,
#                                                                          max_length=1,
#                                                                          pad_token_id=self.tokenizer.pad_token_id,
#                                                                          left_pad=True,
#                                                                          truncation=self.config.truncation)
#         attention_mask_prompt=torch.zeros_like(input_ids_prompt) # All prompt will be masked
        
        
#         input_ids_response, attention_mask_response, loss_mask_response,end_of_response_position_mask_response = self._compute_loss_mask(input_ids_response, attention_mask_response)
        
#         input_ids_prompt=input_ids_prompt[0]
#         attention_mask_prompt=attention_mask_prompt[0]
#         loss_mask_prompt = torch.zeros_like(attention_mask_prompt)
#         end_of_response_position_mask_prompt = torch.zeros_like(attention_mask_prompt)
        
#         input_ids_response=input_ids_response[0]
#         attention_mask_response=attention_mask_response[0]
#         loss_mask_response=loss_mask_response[0]
#         end_of_response_position_mask_response=end_of_response_position_mask_response[0]
        
    
        
#         loss_mask = torch.cat([loss_mask_prompt, loss_mask_response], dim=-1)
#         end_of_response_position_mask = torch.cat([end_of_response_position_mask_prompt, end_of_response_position_mask_response], dim=-1)
#         input_ids = torch.cat([input_ids_prompt, input_ids_response], dim=-1)
#         attention_mask = torch.cat([attention_mask_prompt, attention_mask_response], dim=-1)

        
        
#         position_ids_prompt = compute_position_id_with_mask(attention_mask_prompt)
#         # if self.image_key in row_dict:
#         if has_images:
#             from verl.models.transformers.qwen2_vl import get_rope_index
#             position_ids_response = get_rope_index(
#                 self.processor,
#                 image_grid_thw=image_grid_thw,
#                 input_ids=input_ids_response,
#                 attention_mask=attention_mask_response,
#             )  # (3, seq_len)
#             position_ids_prompt=position_ids_prompt.view(1, -1).expand(3, -1)
#         else:
#             response_length = input_ids_response.shape[0]
#             delta_position_id = torch.arange(1, response_length + 1, device=position_ids_prompt.device)
#             position_ids_response = position_ids_prompt[-1:] + delta_position_id
        
#         if self.config.use_multi_turn_reward:
#             reward_positions = torch.nonzero(end_of_response_position_mask).squeeze(-1)
#             multi_turn_token_level_rewards = torch.zeros_like(end_of_response_position_mask, dtype=torch.float)
#             assert len(reward_positions) == len(rewards), "Number of rewards does not match number of reward positions"
#             for idx,reward in enumerate(rewards):
#                 multi_turn_token_level_rewards[reward_positions[idx]] = reward
#             row_dict["multi_turn_token_level_rewards"] = multi_turn_token_level_rewards # (seq_len,) 
#         if self.config.use_loss_mask:
#             row_dict['loss_mask'] = loss_mask
#         if self.config.use_gae_mask:
#             row_dict['gae_mask'] = loss_mask
#         row_dict["end_of_response_position_mask"] = end_of_response_position_mask # 
#         position_ids = torch.cat([position_ids_prompt, position_ids_response], dim=-1)
#         row_dict['prompts'] = input_ids_prompt
#         row_dict['responses'] = input_ids_response
#         row_dict['input_ids'] = input_ids
#         row_dict['attention_mask'] = attention_mask
#         row_dict['position_ids'] = position_ids
#         index = row_dict.get("extra_info", {}).get("index", 0)
#         row_dict["index"] = index
#         return row_dict