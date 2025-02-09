import torch
import numpy as np
from collections import defaultdict
import os
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from ragen.utils import set_seed
from ragen.utils.plot_utils import (
    save_trajectory_to_output,
    parse_llm_output
)
from verl import DataProto

import shutil

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    logging: dict

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        env_class,
        config: GenerationConfig
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.env_class = env_class
        self.config = config

    def _cut_to_effective_len(self, protocol, keys, cut_left=True):
        assert 'attention_mask' in protocol.batch, 'attention_mask is required'
        effective_len = protocol.batch['attention_mask'].sum(dim=1).max()
        for key in keys:
            if cut_left:
                protocol.batch[key] = protocol.batch[key][:, -effective_len:]
            else:
                protocol.batch[key] = protocol.batch[key][:, :effective_len]
        return protocol

    def _batch_tokenize(self, responses):
        return self.tokenizer(responses, add_special_tokens=False, return_tensors='pt', padding="longest")['input_ids']

    def _convert_pad_structure(self, tensor, pad_token_id, pad_to_left=True):
        """originally: [pad_left, content, pad_right]
        want to: [pad_right, pad_left, content] / [content, pad_right, pad_left]
        """
        if pad_to_left:
            mask = tensor != pad_token_id
        else:
            mask = tensor == pad_token_id

        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        converted_tensor = tensor.gather(1, sorted_indices)

        return converted_tensor, sorted_indices
        

    def _process_responses(self, responses: List[str]) -> List[str]:
        """Process generated responses to remove reward hacking attempts."""
        # Remove everything after </answer> but keep the tag
        responses = [resp.split('</answer>')[0] + '</answer>' 
                    if '</answer>' in resp else resp 
                    for resp in responses]
        
        # Remove reward hacking patterns
        hack_pattern = r'reward: \d+\.\d+\n|done: (True|False)\n'
        hacked = [resp for resp in responses if re.search(hack_pattern, resp)]
        if hacked:
            print(f"[WARNING] HACKED RESPONSES: {hacked}")
            responses = [re.sub(hack_pattern, '', resp) for resp in responses]
            
        return responses

    def _update_trajectory(self, 
                          trajectory: List[Dict], 
                          envs: List[Any],
                          responses: List[str],
                          n_visualize: int):
        """Update visualization trajectory."""
        if trajectory is None:
            return
            
        for idx, env in enumerate(envs[:n_visualize]):
            trajectory[idx]['img_before_action'].append(env.render('rgb_array'))
            
        for idx, (response, env) in enumerate(zip(responses[:n_visualize], 
                                                envs[:n_visualize])):
            img_after = env.render('rgb_array')
            parsed = parse_llm_output(response, strategy="raw")
            
            trajectory[idx]['img_after_action'].append(img_after)
            trajectory[idx]['answer'].append(response)
            trajectory[idx]['parsed_response'].append(parsed)

    def run_llm_loop(self, 
                     gen_batch, 
                     envs: List[Any],
                     initial_input_ids: torch.Tensor,
                     output_dir: str,
                     global_steps: int) -> Tuple[Dict, Dict]:
        """Run the main LLM generation loop."""
        
        # Initialize states
        rollings = gen_batch
        original_left_side = {
            'input_ids': initial_input_ids[:, -self.config.max_start_length:].clone(),
        }
        original_right_side = {
            'responses': initial_input_ids[:, []].clone(),
        }
        meta_info = {}

        # Setup visualization if needed
        trajectory = None
        if self.config.logging.log_images:
            n_visualize = self.config.logging.log_n_image_per_batch
            trajectory = [defaultdict(list) for _ in range(n_visualize)]

        # Main generation loop
        for step in range(self.config.max_turns):
            # Generate responses
            rollings = self._cut_to_effective_len(
                rollings, 
                keys=['input_ids', 'attention_mask', 'position_ids'],
                cut_left=True
            )
            gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            meta_info.update(gen_output.meta_info)

            # Process responses
            responses = self.tokenizer.batch_decode(
                gen_output.batch['responses'], 
                skip_special_tokens=False
            )
            responses = self._process_responses(responses)
            gen_output.batch['responses'] = self._batch_tokenize(responses)

            # Update visualization
            self._update_trajectory(
                trajectory,
                envs,
                responses,
                n_visualize if trajectory else 0
            )

            # Execute in environment
            next_obs = self.env_class.execute_predictions(
                envs, 
                responses,
                self.tokenizer.pad_token
            )
            
            # Process next observations
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                gen_output.batch['responses'],
                next_obs_ids
            )
            
            original_right_side = self._update_right_side(
                original_right_side,
                gen_output.batch['responses'],
                next_obs_ids
            )

        # Save trajectory if needed
        self._save_trajectory(trajectory, output_dir, global_steps)

        return self._compose_final_output(
            original_left_side,
            original_right_side,
            meta_info
        )

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print("[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
            
        return next_obs_ids

    def _update_rolling_state(self,
                            rollings,
                            cur_responses: torch.Tensor,
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update the rolling state with new responses and observations."""
        # Concatenate new tokens
        new_input_ids = torch.cat([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ], dim=1)

        # Update padding and position IDs
        new_input_ids, _ = self._convert_pad_structure(
            new_input_ids,
            self.tokenizer.pad_token_id,
            pad_to_left=True
        )
        
        new_attention_mask = torch.where(
            new_input_ids != self.tokenizer.pad_token_id,
            1, 0
        )
        new_position_ids = (
            torch.cumsum(new_attention_mask, dim=1) - 1
        ) * new_attention_mask

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
                            'input_ids': new_input_ids,
                            'position_ids': new_position_ids,
                            'attention_mask': new_attention_mask
                        })

    def _update_right_side(self,
                          right_side: Dict,
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor) -> Dict:
        """Update the right side state."""
        responses = torch.cat([
            right_side['responses'],
            cur_responses,
            next_obs_ids
        ], dim=1)
        
        responses, _ = self._convert_pad_structure(
            responses,
            self.tokenizer.pad_token_id,
            pad_to_left=False
        )
        
        effective_len = torch.where(
            responses != self.tokenizer.pad_token_id,
            1, 0
        ).sum(dim=1).max()
        
        max_len = min(self.config.max_prompt_length, effective_len)
        return {'responses': responses[:, :max_len]}

    def _save_trajectory(self, trajectory, output_dir, global_steps):
        """Save trajectory visualization if enabled."""
        if not trajectory:
            return
            
        save_step_size = self.config.logging.log_image_step_size
        if not global_steps % save_step_size:
            os.makedirs(output_dir, exist_ok=True)
            save_trajectory_to_output(trajectory, save_dir=output_dir)

    def _compose_final_output(self,
                            left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.concat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask
        final_output['attention_mask'] = torch.concat([
            torch.where(
                left_side['input_ids'] != self.tokenizer.pad_token_id,
                1, 0
            ),
            torch.where(
                final_output['responses'] != self.tokenizer.pad_token_id,
                1, 0
            ),
        ], dim=1)
        
        # Create position IDs
        final_output['position_ids'] = (
            torch.cumsum(final_output['attention_mask'], dim=1) - 1
        ) * final_output['attention_mask']
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output