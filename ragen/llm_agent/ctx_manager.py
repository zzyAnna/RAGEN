import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class ContextManagerConfig:
    action_sep: str  
    start_prompt: str  


class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 actor_rollout_wg,
                 tokenizer,
                 processor = None,
                 ):
        """
        Initialize the ContextManager.
        
        Args:
            config: Configuration for the ContextManager.
            tokenizer: Tokenizer to use for encoding text.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg=actor_rollout_wg
        self.processor = processor
        self.id_to_prompt = {}
        self._init_id_to_prompt()
    
    def _init_id_to_prompt(self):
        """Initialize the mapping from environment IDs to specific prompts."""
        pass
        
        
    
    def get_lm_inputs(self, env_outputs: List[Dict]) -> Dict:
       pass

    def get_env_inputs(self, lm_outputs: Dict) -> List[Dict]:
        pass


    def formulate_rollouts(self, env_outputs: List[Dict]) -> Dict:
       pass

    
   