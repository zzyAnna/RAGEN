import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
def env_output_to_prompt(self, env_output: Dict,include_last_state: bool,init_id_to_prompt:Dict,tokenizer) -> str:
        """
        Convert environment output to a prompt string.
        
        Args:
            env_outputs: List of dictionaries containing environment outputs.
                Each dict has format:
                {
                    "env_id": int,
                    "history": [
                        {"state": str, "actions": List[str], "reward": float,"images": List[images]},
                        {"state": str}
                    ],
                    "kwargs": {
                    }
                }
        
        Returns:
            env_prompts: List of dictionaries containing environment outputs.
                Each dict has format:
                {
                    "env_id": int,
                    "chat_text": str,
                    "images": List[images]
                }
        """