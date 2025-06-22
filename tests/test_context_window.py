import pytest
from ragen.llm_agent.ctx_manager import ContextManager
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from verl import DataProto

class DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return " ".join([msg["content"] for msg in messages])
    def __call__(self, texts, return_tensors, padding, padding_side, truncation):
        import torch
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

@pytest.fixture
def dummy_config():
    cfg = OmegaConf.create({
        "agent_proxy": {
            "max_context_window": 2,
            "enable_think": False,
            "use_turn_scores": False
        },
        "enable_response_mask": False
    })
    return cfg

def test_context_window_truncation(dummy_config):
    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=dummy_config, tokenizer=tokenizer, mode="train")
    ctx.prefix_lookup = {0: "Initial prompt"}
    ctx.env_config_lookup = {0: {"max_tokens": 128}}
    ctx.env_nums = {"": 1}  # For metrics

    env_outputs = [{
        "env_id": 0,
        "group_id": 0,
        "history": [
            {"state": "S1", "llm_response": "R1", "reward": 0.1, "actions_left": 5},
            {"state": "S2", "llm_response": "R2", "reward": 0.2, "actions_left": 4},
            {"state": "S3", "llm_response": "R3", "reward": 0.3, "actions_left": 3},
        ],
        "metrics": {},
    }]

    lm_inputs: DataProto = ctx.get_lm_inputs(env_outputs, prepare_for_update=True)
    messages = lm_inputs.non_tensor_batch["messages_list"][0]

    # Ensure only last 2 turns are present
    assert "S1" not in str(messages)
    assert "S2" in str(messages)
    assert "S3" in str(messages)
