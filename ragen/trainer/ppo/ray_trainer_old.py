# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

import re
import json
from collections import defaultdict
from ragen.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager, apply_kl_penalty, reduce_metrics, compute_throughout_metrics, _timer
from verl.trainer.ppo.ray_trainer import compute_advantage as verl_compute_advantage
from verl.trainer.ppo.ray_trainer import compute_data_metrics as verl_compute_data_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'
    ARPO = 'arpo'
    BRPO = 'brpo'




import torch
from verl.utils.torch_functional import masked_mean



def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    if adv_estimator in [AdvantageEstimator.GRPO, AdvantageEstimator.BRPO]:
        raise NotImplementedError
    else:
        return verl_compute_advantage(data, adv_estimator, gamma, lam, num_repeat)

def normalize_reward(reward, uid, reward_norm_type):
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    for r, u in zip(reward, uid):
        id2score[u].append(r)
    for u in id2score:
        if len(id2score[u]) == 1:
            id2mean[u] = torch.tensor(0.0)
            id2std[u] = torch.tensor(1.0)
        elif len(id2score[u]) > 1:
            id2mean[u] = torch.mean(torch.tensor(id2score[u], dtype=torch.float32))
            id2std[u] = torch.std(torch.tensor([id2score[u]], dtype=torch.float32))
        else:
            raise ValueError(f"no score in prompt index: {u}")
    normalized_reward = [(r - id2mean[u]) / (id2std[u] + 1e-6) for r, u in zip(reward, uid)] # NOTE: +1e-6, maybe +1!
    # transform to the same dtype as reward
    return np.array(normalized_reward, dtype=reward.dtype)

def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    metrics = verl_compute_data_metrics(batch, use_critic)
    metrics.update({
        # metrics for actions
        'metric/total_env':
            int(np.array(batch.non_tensor_batch['total_env'], dtype=np.int16).sum()),
        'metric/finished_env':
            int(np.array(batch.non_tensor_batch['finished_env'], dtype=np.int16).sum()),
        'metric/success_env':
            int(np.array(batch.non_tensor_batch['success_env'], dtype=np.int16).sum()),
        'metric/traj_length':
            float(np.array(batch.non_tensor_batch['traj_length'], dtype=np.int16).mean()),
        'metric/valid_action':
            float(np.array(batch.non_tensor_batch['valid_action'], dtype=np.int16).mean()),
        'metric/effective_action':
            float(np.array(batch.non_tensor_batch['effective_action'], dtype=np.int16).mean()),
        'metric/effective_action_ratio':
            float(np.array(batch.non_tensor_batch['effective_action_ratio'], dtype=np.float32).mean()),
    })

    # metric for two-armed bandit
    if batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
        batch_action = np.array(batch.non_tensor_batch['bandit_metrics'], dtype=np.int16)
        metrics['metric/n_low_arm'] = int(np.sum(batch_action == 1))
        metrics['metric/n_high_arm'] = int(np.sum(batch_action == 2))
        metrics['metric/n_invalid'] = int(np.sum(batch_action == 0))

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }



class RayPPOTrainer(VerlRayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)

        self.train_data_num = config.data.train_data_num or config.trainer.total_training_steps * config.data.train_batch_size
        self.val_data_num = config.data.val_data_num or 100
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))


    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from ragen.utils.dataset.agent_dataset import AgentDataset, collate_fn
        self.train_dataset = AgentDataset(
            data_num=self.train_data_num,
            tokenizer=self.tokenizer,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error'
        )
        if self.train_data_num is not None:
            if self.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = AgentDataset(
            data_num=self.val_data_num,
            tokenizer=self.tokenizer,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error'
        )
        if self.val_data_num is not None:
            if self.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps


    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        import pdb; pdb.set_trace()
        logger = self.logger
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None


        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            logging=self.config.logging,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.no_think_rl,
            state_masking=self.config.actor_rollout_ref.actor.state_masking,
            start_state_marker=self.config.algorithm.state_masking.start_state_marker,
            end_state_marker=self.config.algorithm.state_masking.end_state_marker,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            logger=logger,
        )

        
        envs = [self.env.copy() for _ in range(self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n_agent)] 



        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')

                # update ref_policy_wg
                if self.config.trainer.ref_update_steps is not None and self.global_steps % self.config.trainer.ref_update_steps == 0:
                    self.actor_rollout_wg.save_checkpoint(
                        local_path=f'./log/temp/actor_rollout_wg_global_step_{self.global_steps}',
                        hdfs_path=None
                    )
                    self.ref_policy_wg.load_model_parameters(
                        source_model_path=f'./log/temp/actor_rollout_wg_global_step_{self.global_steps}',
                        strict=True
                    )
                    print(f"load parameters from ./log/temp/actor_rollout_wg_global_step_{self.global_steps} to ref_policy_wg")

                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                env_seeds = [i['index'] for i in batch.non_tensor_batch['extra_info']]
                print("env_seeds:", env_seeds)
                for env, seed in zip(envs, env_seeds):
                    env.reset(seed=seed)


                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                # with _timer('gen', timing_raw):
                #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                #     batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                #                                              dtype=object)
                #     # repeat to align with repeated responses in rollout
                #     batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                #     batch = batch.union(gen_batch_output)

                #     # output batch to file
                #     self._record_batch(batch, path=f'.log/{self.config.trainer.experiment_name}/gen_batch.txt')

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################

                with _timer('step', timing_raw):
                    """
                    keep rolling to generate K turns of responses.
                    when doing this, update the "original right side" when new responses are generated.
                    finally, concatenate the "original left side" and "original right side" to get the final thing to feed to train the model.

                    Left-pad prompts, right-gen flow, Tensors dance like stardust glow.
                    Errors swarm? Stay calm, don't fret- Code with coffee, debug sunset.
                    """

                    first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    output_dir = (f"{self.config.logging.log_image_dir}/"
                                 f"{self.config.trainer.experiment_name}/"
                                 f"train/"
                                 f"step_{self.global_steps}")

                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            envs=envs,
                            initial_input_ids=first_input_ids,
                            output_dir=output_dir,
                            global_steps=self.global_steps,
                        )

                    with torch.no_grad():
                        output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                        final_gen_batch_output = final_gen_batch_output.union(output)
                    
                    if self.config.algorithm.adv_estimator == 'grpo' or self.config.algorithm.reward_norm_type == 'grpo': # NOTE we currently use seed to group, better use prompt (hash) to group
                        batch.non_tensor_batch['uid'] = np.array([str(i) for i in env_seeds], dtype=object)
                    elif self.config.algorithm.adv_estimator == 'brpo' or self.config.algorithm.reward_norm_type == 'brpo':
                        batch.non_tensor_batch['uid'] = np.array(["" for _ in range(len(batch.batch))], dtype=object)
                    elif self.config.algorithm.adv_estimator == 'arpo' or self.config.algorithm.reward_norm_type == 'arpo':
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object) # No Relative normalization

                    # reward
                    batch.non_tensor_batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        batch.non_tensor_batch['reward'][idx] = env.reward
                    
                    # normalize reward
                    if self.config.algorithm.reward_norm_type is not None:
                        batch.non_tensor_batch['reward'] = normalize_reward(batch.non_tensor_batch['reward'], batch.non_tensor_batch['uid'], self.config.algorithm.reward_norm_type)

                    # metric for two-armed bandit
                    # NOTE here we assume invalid action is 0, low arm is 1, high arm is 2
                    if batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
                        batch.non_tensor_batch['bandit_metrics'] = np.array([0 for _ in range(len(envs))], dtype=object)
                        for idx, env in enumerate(envs):
                            batch.non_tensor_batch['bandit_metrics'][idx] = env.get_last_action()

                    # metrics for actions
                    batch.non_tensor_batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        batch.non_tensor_batch['finished_env'][idx] = int(env.finished())
                        batch.non_tensor_batch['success_env'][idx] = int(env.success())
                        tracking_vars = env.get_tracking_variables()
                        batch.non_tensor_batch['traj_length'][idx] = len(tracking_vars['actions'])
                        batch.non_tensor_batch['valid_action'][idx] = sum(1 for x in tracking_vars['actions_valid'] if x is not None)
                        batch.non_tensor_batch['effective_action'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None)
                        batch.non_tensor_batch['effective_action_ratio'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None) / len(tracking_vars['actions'])
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################


                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available, no kl_loss or kl_penalty for GAE
                        if not self.config.actor_rollout_ref.actor.use_kl_loss or self.config.algorithm.adv_estimator != 'gae':
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.actor_rollout_ref.actor.state_masking:
                                batch,metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                    return
                    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        # Initialize state mask
        state_mask = torch.zeros_like(response_mask)
        
        responses = [self.tokenizer.decode(resp, skip_special_tokens=False) for resp in batch.batch['responses']]
    
        for i, response in enumerate(responses):
            # Find all pairs of start and end marker positions
            start_marker = self.config.algorithm.state_masking.start_state_marker
            end_marker = self.config.algorithm.state_masking.end_state_marker   
            
            # Get all start and end positions
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), response)]
            end_positions = [m.start() + len(end_marker) for m in re.finditer(re.escape(end_marker), response)]
            
            
            # Convert character positions to token positions
            for start, end in zip(start_positions, end_positions):
                prefix_to_start = response[:start]
                state_section = response[start:end]
                
                start_tokens = self.tokenizer.encode(prefix_to_start, add_special_tokens=False)
                state_tokens = self.tokenizer.encode(state_section, add_special_tokens=False)
                
                start_token_pos = len(start_tokens)
                end_token_pos = start_token_pos + len(state_tokens)
                
                state_mask[i, start_token_pos:end_token_pos] = 1
        
        loss_mask = state_mask * response_mask
        batch.batch['loss_mask'] = loss_mask
        
        # Debug print
        print("\nRaw batch[0] (before masking):\n", self.tokenizer.decode(batch.batch['responses'][0]))
        response_ids = batch.batch['responses'][0]
        unmasked_ids = response_ids[loss_mask[0] == 0]
        print("\nUnmasked batch[0] (after masking):\n", self.tokenizer.decode(unmasked_ids))
        
        masked_ids = response_ids[loss_mask[0] == 1]
        print("\nMasked batch[0] (masked parts):\n", self.tokenizer.decode(masked_ids))
        
        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        # Initialize global metric storage
        global_token_scores = []
        global_metrics = {}
        metrics = defaultdict(list)

        self.val_num += 1

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            logging=self.config.logging,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.no_think_rl,
            state_masking=self.config.actor_rollout_ref.actor.state_masking,
            start_state_marker=self.config.algorithm.state_masking.start_state_marker,
            end_state_marker=self.config.algorithm.state_masking.end_state_marker,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            logger=self.logger,
            is_validation=True,
        )

        envs = [self.val_env.copy() for _ in range(self.config.data.val_batch_size)] # do not repeat
        # envs = [self.val_env.copy() for _ in range(self.config.data.val_batch_size * self.config.actor_rollout_ref.rollout.n_agent)]
        val_global_steps = 1

        for batch_dict in self.val_dataloader:
            timing_raw = {}
            test_batch: DataProto = DataProto.from_single_dict(batch_dict)
            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

            env_seeds = [i['index'] for i in test_batch.non_tensor_batch['extra_info']]
            print("env_seeds:", env_seeds)
            for env, seed in zip(envs, env_seeds):
                env.reset(seed=seed)
            
            test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            with _timer('step', timing_raw):
                first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                output_dir = (f"{self.config.logging.log_image_dir}/"
                                f"{self.config.trainer.experiment_name}/"
                                f"validation_{self.val_num}/"
                                f"step_{val_global_steps}")
                with _timer('gen', timing_raw):
                    generation_manager.timing_raw = timing_raw
                    final_gen_batch_output = generation_manager.run_llm_loop(
                        gen_batch=test_gen_batch,
                        envs=envs,
                        initial_input_ids=first_input_ids,
                        output_dir=output_dir,
                        global_steps=val_global_steps,
                    )
                with torch.no_grad():
                    output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                    final_gen_batch_output = final_gen_batch_output.union(output)

                test_batch.non_tensor_batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch.non_tensor_batch['reward'][idx] = env.reward

                if test_batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
                    # metric for two-armed bandit
                    # NOTE here we assume invalid action is 0, low arm is 1, high arm is 2
                    test_batch.non_tensor_batch['bandit_metrics'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        test_batch.non_tensor_batch['bandit_metrics'][idx] = env.get_last_action()
                    metrics['bandit_metrics'].append(test_batch.non_tensor_batch['bandit_metrics'])
                
                test_batch.non_tensor_batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch.non_tensor_batch['finished_env'][idx] = int(env.finished())
                    test_batch.non_tensor_batch['success_env'][idx] = int(env.success())
                    tracking_vars = env.get_tracking_variables()
                    test_batch.non_tensor_batch['traj_length'][idx] = len(tracking_vars['actions'])
                    test_batch.non_tensor_batch['valid_action'][idx] = sum(1 for x in tracking_vars['actions_valid'] if x is not None)
                    test_batch.non_tensor_batch['effective_action'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None)
                    test_batch.non_tensor_batch['effective_action_ratio'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None) / len(tracking_vars['actions'])

                # action metrics
                metrics['total_env'].append(test_batch.non_tensor_batch['total_env'])
                metrics['finished_env'].append(test_batch.non_tensor_batch['finished_env'])
                metrics['success_env'].append(test_batch.non_tensor_batch['success_env'])
                metrics['traj_length'].append(test_batch.non_tensor_batch['traj_length'])
                metrics['valid_action'].append(test_batch.non_tensor_batch['valid_action'])
                metrics['effective_action'].append(test_batch.non_tensor_batch['effective_action'])
                metrics['effective_action_ratio'].append(test_batch.non_tensor_batch['effective_action_ratio'])

                # Accumulate batch metrics into global storage
                global_token_scores.append(test_batch.non_tensor_batch['reward'])


        global_scores = np.concatenate(global_token_scores, axis=0)
        global_metrics = {
            'global_score/mean': float(global_scores.mean()),
            'global_score/max': float(global_scores.max()),
            'global_score/min': float(global_scores.min()),
            'global_score/std': float(global_scores.std()),
            'validate_metric/total_env': int(np.array(metrics['total_env'], dtype=np.int16).sum()),
            'validate_metric/finished_env': int(np.array(metrics['finished_env'], dtype=np.int16).sum()),
            'validate_metric/success_env': int(np.array(metrics['success_env'], dtype=np.int16).sum()),
            'validate_metric/traj_length': float(np.array(metrics['traj_length'], dtype=np.int16).mean()),
            'validate_metric/valid_action': float(np.array(metrics['valid_action'], dtype=np.int16).mean()),
            'validate_metric/effective_action': float(np.array(metrics['effective_action'], dtype=np.int16).mean()),
            'validate_metric/effective_action_ratio': float(np.array(metrics['effective_action_ratio'], dtype=np.float32).mean()),
        }
        if 'bandit_metrics' in metrics: # NOTE hard code for two-armed bandit
            batch_action = np.array(metrics['bandit_metrics'], dtype=np.int16)
            global_metrics['validate_metric/n_low_arm'] = int(np.sum(batch_action == 1))
            global_metrics['validate_metric/n_high_arm'] = int(np.sum(batch_action == 2))
            global_metrics['validate_metric/n_invalid'] = int(np.sum(batch_action == 0))
        print("global_metrics", global_metrics)
        self.logger.log(data=global_metrics, step=global_steps)
        return global_metrics
    
    