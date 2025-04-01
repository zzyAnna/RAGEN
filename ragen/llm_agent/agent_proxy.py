
from .ctx_manager import ContextManager
from .es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_name = config.actor_rollout_ref.model.path
		ro_config = config.actor_rollout_ref.rollout
		self.llm = LLM(
			model_name, 
			enable_prefix_caching=config.actor_rollout_ref.rollout.enable_kv_cache, 
			enforce_eager=config.actor_rollout_ref.rollout.enforce_eager,
            dtype=config.actor_rollout_ref.rollout.dtype,
            gpu_memory_utilization=config.actor_rollout_ref.rollout.gpu_memory_utilization,
            disable_log_stats=config.actor_rollout_ref.rollout.disable_log_stats,
            enable_chunked_prefill=config.actor_rollout_ref.rollout.enable_chunked_prefill,
			max_num_batched_tokens=config.actor_rollout_ref.rollout.max_num_batched_tokens
		)
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			# temperature=ro_config.temperature,
			temperature=0,
			top_p=ro_config.top_p,
			top_k=ro_config.top_k,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		cache_action = lm_inputs.meta_info.get('cache_action', None)

		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
		texts = ["<think>" + output.outputs[0].text for output in outputs] # The LLM generation does not include <think> and <answer> tags. Add them back here.
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs

class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, actor_rollout_wg, tokenizer):
		self.config = config
		self.ctx_manager = ContextManager(config, tokenizer)
		self.es_manager = EnvStateManager(config)

		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer

	def generate_sequences(self, lm_inputs: DataProto):
		# TODO: add kv cache both for the vllm wrapper here and for verl vllm.
		if isinstance(self.actor_wg, RayWorkerGroup):
			padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
			padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
			lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
			lm_outputs.meta_info = lm_inputs.meta_info
			lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
		elif isinstance(self.actor_wg, VllmWrapperWg):
			lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

		return lm_outputs


	def rollout(self, dataproto: DataProto, val=False):
		env_outputs = self.es_manager.reset(val=val)

		for i in range(self.config.agent_proxy.max_turn):
			lm_inputs: DataProto = self.ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
			lm_inputs.meta_info = dataproto.meta_info # TODO: setup vllm early stop. make sure this can be done
			cache_action = None
			if i == 0:
				cache_action = 'init'
			elif i == self.config.agent_proxy.max_turn - 1:
				cache_action = 'clear'
			lm_inputs.meta_info['cache_action'] = cache_action
			lm_outputs: DataProto = self.generate_sequences(lm_inputs)
			env_inputs: List[Dict] = self.ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = self.es_manager.step(env_inputs, val=val)

		rollout_states = self.es_manager.get_rollout_states(val=val) 
		rollouts = self.ctx_manager.formulate_rollouts(rollout_states, val=val)
		return rollouts

	def _reset(self):
		self.es_manager.reset_envs()
		self.es_manager.reset_env_status() # make all status to be initialized again

@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	actor_wg = VllmWrapperWg(config, tokenizer)
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	import time
	start_time = time.time()
	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}))
	end_time = time.time()
	print(f'rollout time: {end_time - start_time} seconds')
	# print rollout rewards from the rm_scores
	rm_scores = rollouts.batch["rm_scores"]
	metrics = rollouts.meta_info["metrics"]
	avg_reward = rm_scores.sum(-1).mean().item()
	print(f'rollout rewards: {avg_reward}')
	print(f'metrics:')
	for k, v in metrics.items():
		print(f'{k}: {v}')



if __name__ == "__main__":
	main()
