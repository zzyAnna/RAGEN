"""
FSDP PPO Trainer with Ray-based single controller.
Adapted from the excellently written verl implementation.
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from tqdm import tqdm

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
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

WorkerType = Type[Worker]

from verl.trainer.ppo.ray_trainer import Role, AdvantageEstimator, ResourcePoolManager, apply_kl_penalty, compute_advantage, compute_response_mask, _timer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer

import torch
from verl.utils.torch_functional import masked_mean

from ragen.llm_agent import LLMAgentProxy


class RayAgentTrainer(VerlRayPPOTrainer):
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

        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)
        
        
    def _create_dataloader(self):
        assert self.config.trainer.total_training_steps is not None, "must determine total training steps"
        total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
        # val_start = 100000
        # self.train_seeds = [seed for seed in range(0, self.config.trainer.total_training_steps * 1000, 1000)]
        # self.val_seeds = [seed for seed in range(val_start, val_start + self.config.trainer.validation_steps)]

    def init_agent_proxy(self):
        self.agent_proxy = LLMAgentProxy(
            config=self.config,
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer
        )

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        env_metric_dict = {}
        for step in range(self.config.trainer.validation_steps):
            # Store original inputs
            input_texts = ["" for _ in range(self.config.es_manager.val.env_groups * self.config.es_manager.val.group_size)]
            sample_inputs.extend(input_texts)
            
            meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            import time
            start_time = time.time()
            test_batch = self.agent_proxy.rollout(test_gen_batch)
            end_time = time.time()
            print(f'validation generation time: {end_time - start_time} seconds')
            for key, value in test_batch.meta_info['metrics'].items():
                if "val/" + key not in env_metric_dict:
                    env_metric_dict["val/" + key] = []
                env_metric_dict["val/" + key].append(value)

            # Store generated outputs
            output_ids = test_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = reduce_metrics(env_metric_dict)
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

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

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for step in range(self.total_training_steps):
            # metrics = {}
            timing_raw = {}

            batch: DataProto = DataProto()
            is_last_step = self.global_steps >= self.total_training_steps

            with _timer('step', timing_raw):
                # generate a batch
                with _timer('gen', timing_raw):
                    batch = self.agent_proxy.rollout(batch)
                    metrics = {"train/" + key: value for key, value in batch.meta_info['metrics'].items()}

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # TODO: check if this is correct. Not tested yer
                    logger.log("[WARNING] REMAX implementation is not tested yet in RAGEN.")
                    with _timer('gen_max', timing_raw):
                        gen_baseline_batch = deepcopy(batch)
                        gen_baseline_batch.meta_info['do_sample'] = False
                        gen_baseline_output = self.agent_proxy.rollout(gen_baseline_batch)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch['reward_baselines'] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            # dtype=object)
                # repeat to align with repeated responses in rollout
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                # batch = batch.union(gen_batch_output)
                batch.non_tensor_batch['uid'] = batch.non_tensor_batch['group_ids']

                batch.batch['response_mask'] = compute_response_mask(batch)
                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                # recompute old_log_probs
                with _timer('old_log_prob', timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch = batch.union(old_log_prob)

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

                    # compute rewards. apply_kl_penalty if available
                    if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
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
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                    (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer('testing', timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and ( is_last_step or \
                        self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f'Final validation metrics: {last_val_metrics}')
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1

    def _save_checkpoint(self):
        """ 
        Different from VerlRayPPOTrainer, we have no dataloader so we won't save it. Other logic is the same.
        """
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')

        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))
