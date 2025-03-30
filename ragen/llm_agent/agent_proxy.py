
from .config import LLMAgentProxyConfig, ContextManagerConfig, EnvStateManagerConfig
from vllm import LLM
from transformers import AutoTokenizer
from verl import DataProto
import hydra


class VllmWrapperWg:
	def __init__(self, config):
		self.config = config
		model_name = config.actor_rollout_ref.model.path
		self.llm = LLM(model=model_name, enable_prefix_caching=True)

	def generate_sequences(self, lm_inputs: DataProto):
		input_ids = lm_inputs.input_ids
		sequences = self.llm.generate(input_ids)
		return sequences



class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""

	def __init__(self, config, actor_rollout_wg, tokenizer):
		self.ctx_manager = ContextManager(ContextManagerConfig(config.context), tokenizer)
		self.es_manager = EnvStateManager(EnvStateManagerConfig(config.envs))

		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer

	def rollout(self, val=False):
		self._reset()
		env_outputs = self.es_manager.reset(val=val)

		# # This would look like something as 
		# 	对于ctx: 第0步有sys input，最后一步要考虑放不放s_{t+1}(感觉不用放，model学了没啥用)
		# [
		# 		{“env_id”: xxx, “history”: [
		# 			{“state”: “###\n#x_#”, “actions”: [“xx”, “yy”], “reward”: xxx}
		# 			{“state”, “###\n#x_#”}
		# ], 

		for i in range(self.config.max_turn):
			lm_inputs: DataProto = self.ctx_manager.get_lm_inputs(env_outputs)
			lm_outputs: DataProto = self.actor_wg.generate_sequence(lm_inputs)
			env_inputs: List[Dict] = self.ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = self.es_manager.step(env_inputs, val=val)


		rollout_states = self.es_manager.get_rollout_states(env_outputs) 
		rollouts = self.ctx_manager.formulate_rollouts(rollout_states) 
		return rollouts

	def _reset(self):
		self.es_manager.reset_envs()
		self.es_manager.reset_env_status() # make all status to be initialized again

@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
	actor_wg = VllmWrapperWg(config)
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	proxy.rollout()


if __name__ == "__main__":
	main()
