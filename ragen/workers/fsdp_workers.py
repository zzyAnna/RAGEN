"""
This file basically adapts the implementation of the ActorRolloutRefWorker of verl.workers.fsdp_workers.py
NOTE: Currently not used
"""
import logging
import os

import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import DictConfig
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_to_local

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy, CriticWorker, RewardModelWorker
from verl.workers.fsdp_workers import ActorRolloutRefWorker as VerlActorRolloutRefWorker

class ActorRolloutRefWorker(VerlActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__(config, role)


    def _build_rollout(self):
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])
        rollout_name = self.config.rollout.name
        assert rollout_name == 'vllm', f"rollout_name not supported in RAGEN: {rollout_name}"

        from ragen.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
        from verl.workers.sharding_manager import FSDPVLLMShardingManager
        log_gpu_memory_usage(f'Before building {rollout_name} rollout', logger=None)
        local_path = copy_to_local(self.config.model.path)
        if vllm_mode == 'customized':
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                    config=self.config.rollout,
                                    tokenizer=self.tokenizer,
                                    model_hf_config=self.actor_model_config)
        elif vllm_mode == 'spmd':
            rollout = vLLMRollout(model_path=local_path,
                                    config=self.config.rollout,
                                    tokenizer=self.tokenizer,
                                    model_hf_config=self.actor_model_config,
                                    device_mesh=rollout_device_mesh)
        else:
            raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")
        log_gpu_memory_usage(f'After building {rollout_name} rollout', logger=None)
        if torch.distributed.get_world_size() == 1:
            self.config.rollout.load_format = 'dummy_hf'
        rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                            inference_engine=rollout.inference_engine,
                                                            model_config=self.actor_model_config,
                                                            full_params='hf' in self.config.rollout.load_format,
                                                            device_mesh=rollout_device_mesh)
        log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager
