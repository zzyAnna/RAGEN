
from .config import LLMAgentProxyConfig, ContextManagerConfig, EnvStateManagerConfig
from vllm import LLM



class VllmWrapperWg:
	def __init__(self, config: VllmWrapperWgConfig):
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

	Def __init__(self, config, actor_wg, tokenizer):
		self.ctx_manager = ContextManager(ContextManagerConfig(config.context), tokenizer)
		self.es_manager = EnvStateManager(EnvStateManagerConfig(config.envs))

		Self.actor_wg = actor_wg
		self.tokenizer = tokenizer

	def rollout(self, val=False):
		self._reset()
		Env_outputs = Self.es_manager.reset(val=val)

	# # This would look like something as 
	# 	对于ctx: 第0步有sys input，最后一步要考虑放不放s_{t+1}(感觉不用放，model学了没啥用)
	# [
	# 		{“env_id”: xxx, “history”: [
	# 			{“state”: “###\n#x_#”, “actions”: [“xx”, “yy”], “reward”: xxx}
	# 			{“state”, “###\n#x_#”}
	# ], 
]


		for i in range(self.config.max_turn):
			lm_inputs: DataProto = self.ctx_manager.get_lm_inputs(env_outputs)
			lm_outputs: DataProto = self.actor_wg.generate_sequence(lm_inputs)
			env_inputs: List[Dict] = self.ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = self.es_manager.step(env_inputs, val=val)


			rollouts = self.ctx_manager.formulate_rollouts(env_outputs) 
			return rollouts

	def _reset(self):
		self.es_manager.reset_envs()
		self.es_manager.reset_env_status() # make all status to be initialized again

If __name__ == "__main__":
	from hydra import compose, core, initialize
	from omegaconf import DictConfig, OmegaConf
	from ragen.llm_agent.ctx_manager import ContextManagerConfig
	from ragen.llm_agent.es_manager import EnvStateManagerConfig

	# config = # load hydra yaml at RAGEN/config/ppo_trainer.yaml
	config = OmegaConf.load("RAGEN/config/ppo_trainer.yaml")
	breakpoint()
	config.actor_rollout_ref.model.path = "Qwen/Qwen2.5-0.5B-Instruct"

	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	proxy.rollout()
